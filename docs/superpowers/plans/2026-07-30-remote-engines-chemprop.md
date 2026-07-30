# Remote Engines & Chemprop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an engine where its hardware is — without any code knowing where that is — and add Chemprop D-MPNN as the first GPU-lane engine.

**Architecture:** An engine's manifest declares a `lane`; the enqueuer routes the job to that lane's arq queue; a deployment satisfies a lane by running a worker process with `STUDIO_WORKER_LANE` set. Long fits become stoppable through a cooperative `report` callback on `TrainContext`, which is the only mechanism that can interrupt work already running on a `to_thread` worker thread. Blob storage moves from a local path to any fsspec-addressable object store, so a worker container needs no volumes at all.

**Tech Stack:** Python 3.13, arq 0.28 (named queues), fsspec + s3fs/adlfs, chemprop 2.2.x, lightning 2.x, polars, pytest.

**Spec:** `docs/superpowers/specs/2026-07-30-remote-engines-chemprop-design.md`

## Global Constraints

- All commands run from `backend/`. Tests: `uv run pytest`. Lint: `uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src`. Layering: `uv run lint-imports`.
- Import-linter enforces `interface > infrastructure > application > domain`. `application` must never import `infrastructure`. `domain` must never import `polars`, `sklearn`, `numpy`, `arq`, or any of the forbidden list in `pyproject.toml`.
- `daikonstudio.settings` sits outside the layer contract; `infrastructure` may import it, `application` may not.
- Python floor stays `>=3.13`. The codebase uses PEP 695 generics (`class PageResult[T]`, `def result_to_response[T]`), so no base image or dependency may force ≤3.12.
- The queue message stays a bare `run_id`. `lane` is routing metadata passed as a function argument, never serialized into the job payload and never stored on the Run row.
- `EngineManifest` must stay free of infrastructure imports. `tests/unit/engines/test_engine_contract.py` round-trips every registered manifest through `json.dumps(dataclasses.asdict(m))` — if it fails, fix the field, not the test.
- Engine `train`/`predict` stay synchronous. Do not make them `async def`.
- Metric names are a fixed vocabulary: regression is `rmse`, `mae`, `r2`; classification is `mcc`, `balanced_accuracy`, `auroc`, `auprc`. Plain accuracy is never computed, not even as an unused local.
- `predict()` must return exactly `row_id` (`pl.Int64`), `value` (`pl.Float64`), `uncertainty` (`pl.Float64`), with dtypes declared explicitly. An all-`None` uncertainty list inferred as polars `Null` makes engines' outputs schema-incompatible.
- Commit after every task. Conventional commit prefixes (`feat:`, `fix:`, `refactor:`, `test:`, `chore:`).

---

## File Structure

**Create:**
- `backend/src/daikonstudio/infrastructure/engines/chemprop_dmpnn.py` — the engine adapter, sole owner of every chemprop/torch/lightning import.
- `backend/Dockerfile` — API + default-lane worker (slim, no CUDA).
- `backend/Dockerfile.gpu` — GPU-lane worker (CUDA runtime + chemprop).
- `backend/.dockerignore`
- `backend/tests/unit/engines/test_chemprop_dmpnn.py`
- `backend/tests/unit/engines/test_metrics.py`
- `backend/tests/unit/execution/test_lanes.py`

**Modify:**
- `backend/pyproject.toml` — optional dependency extras.
- `backend/src/daikonstudio/settings.py` — worker + blob settings.
- `backend/src/daikonstudio/infrastructure/storage/fsspec_blob_store.py` — forward storage options.
- `backend/src/daikonstudio/application/engines/manifest.py` — `DEFAULT_LANE`, `EngineManifest.lane`.
- `backend/src/daikonstudio/application/engines/context.py` — `ProgressReporter`, `RunInterrupted`, `TrainContext.report`.
- `backend/src/daikonstudio/application/engines/protocol.py` — document the `report` contract.
- `backend/src/daikonstudio/application/execution/enqueue.py` — `enqueue(run_id, lane)`.
- `backend/src/daikonstudio/infrastructure/worker.py` — `queue_for`, lane-aware enqueuers, settings-driven `WorkerSettings`, `RunInterrupted` handling in `run_job`.
- `backend/src/daikonstudio/application/execution/train_protocol.py` — reporter construction, deadline, progress spans.
- `backend/src/daikonstudio/application/execution/predict_with_protocol.py` — lane lookup via `EngineRegistry`.
- `backend/src/daikonstudio/infrastructure/di/container.py` — new constructor arguments.
- `backend/src/daikonstudio/infrastructure/engines/_scoring.py` — extract pure metric functions.
- `backend/src/daikonstudio/infrastructure/engines/registry.py` — register the chemprop engine.
- `backend/tests/unit/engines/test_engine_contract.py` — default `report` is a no-op.
- `backend/tests/unit/execution/test_worker.py` — interruption handling.

---

## Task 1: Swappable blob storage

Points `BlobStore` at any fsspec-addressable backend (MinIO, S3, Azure Blob) through configuration alone, and declares the optional dependency extras the rest of the plan installs.

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/src/daikonstudio/settings.py`
- Modify: `backend/src/daikonstudio/infrastructure/storage/fsspec_blob_store.py`
- Modify: `backend/src/daikonstudio/infrastructure/di/container.py:66-69`
- Modify: `backend/src/daikonstudio/infrastructure/worker.py` (in `_on_startup`)
- Test: `backend/tests/integration/test_blob_store.py` (add to the existing file — despite
  the name it uses only `tmp_path`, and there is no `tests/unit/infrastructure/test_blob_store.py`)

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings.blob_storage_options: dict[str, Any]`; `FsspecBlobStore(base_url: str, storage_options: dict[str, Any] | None = None)`. Extras named `s3`, `azure`, `gpu` in `pyproject.toml`.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/unit/infrastructure/test_blob_store.py`:

```python
def test_storage_options_are_forwarded_to_fsspec(monkeypatch: pytest.MonkeyPatch) -> None:
    """A MinIO or Azure endpoint reaches fsspec, or the store silently talks to the
    wrong backend -- which on S3 means a confusing 403 rather than a clear failure."""
    seen: dict[str, object] = {}

    def fake_url_to_fs(url: str, **options: object) -> tuple[object, str]:
        seen["url"] = url
        seen["options"] = options
        return (object(), url)

    monkeypatch.setattr(fsspec.core, "url_to_fs", fake_url_to_fs)

    FsspecBlobStore("s3://bucket/prefix", {"endpoint_url": "https://minio.example.edu"})

    assert seen["url"] == "s3://bucket/prefix"
    assert seen["options"] == {"endpoint_url": "https://minio.example.edu"}


def test_no_storage_options_still_round_trips(tmp_path: Path) -> None:
    """The default path must be untouched: omitting options behaves exactly as before."""
    store = FsspecBlobStore(f"file://{tmp_path}")

    uri = store.put_bytes("a/b.bin", b"payload")

    assert store.get_bytes("a/b.bin") == b"payload"
    assert store.exists(uri)
```

Add `import fsspec.core` and `import pytest` to the file's imports if not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/infrastructure/test_blob_store.py -v`
Expected: FAIL — `TypeError: FsspecBlobStore.__init__() takes 2 positional arguments but 3 were given`

- [ ] **Step 3: Forward storage options in the store**

In `backend/src/daikonstudio/infrastructure/storage/fsspec_blob_store.py`, replace the module docstring and `__init__`:

```python
"""One implementation, driven by BLOB_BASE_URL. file:// in dev, s3:// or abfs:// in prod.

`storage_options` is what makes the backend swappable without a code change: an
S3-compatible endpoint (MinIO) needs `endpoint_url`, AWS needs `key`/`secret`, Azure
needs `account_name`/`connection_string`. Passing them explicitly rather than relying on
ambient environment variables means a misconfiguration fails where it is configured,
not three layers down inside botocore.
"""

from typing import Any

import fsspec  # type: ignore[import-untyped]


class FsspecBlobStore:
    def __init__(self, base_url: str, storage_options: dict[str, Any] | None = None) -> None:
        self._base = base_url.rstrip("/")
        self._fs, _ = fsspec.core.url_to_fs(self._base, **(storage_options or {}))
```

Leave `_path`, `put_bytes`, `get_bytes`, `exists` and `delete` unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/infrastructure/test_blob_store.py -v`
Expected: PASS

- [ ] **Step 5: Add the setting**

In `backend/src/daikonstudio/settings.py`, add `from typing import Any` at the top and this field after `blob_base_url`:

```python
    # Forwarded to fsspec's url_to_fs, which is what makes the blob backend swappable
    # with no code change: `endpoint_url` for MinIO or any S3-compatible store,
    # `key`/`secret` for AWS, `account_name`/`connection_string` for Azure Blob. Supplied
    # as JSON in the environment, e.g.
    #   STUDIO_BLOB_STORAGE_OPTIONS='{"endpoint_url": "https://minio-api.example.edu"}'
    blob_storage_options: dict[str, Any] = {}
```

- [ ] **Step 6: Pass it at both composition roots**

In `backend/src/daikonstudio/infrastructure/di/container.py`, change the `BlobStore` binding:

```python
    container.define(
        BlobStore,  # type: ignore[type-abstract]
        Singleton(lambda: FsspecBlobStore(resolved.blob_base_url, resolved.blob_storage_options)),
    )
```

In `backend/src/daikonstudio/infrastructure/worker.py`, inside `_on_startup`:

```python
    ctx["store"] = FsspecBlobStore(settings.blob_base_url, settings.blob_storage_options)
```

- [ ] **Step 7: Declare the extras**

In `backend/pyproject.toml`, add after the `[project]` block's `dependencies` list:

```toml
[project.optional-dependencies]
# Networked blob backends. `BLOB_BASE_URL=file://...` needs neither.
s3 = ["s3fs>=2024.10"]
azure = ["adlfs>=2024.7"]
# The GPU lane. chemprop pulls torch, lightning and rdkit itself, so listing them
# again here would only create a second place for their floors to drift. The CUDA
# build of torch is selected in Dockerfile.gpu, deliberately NOT in [tool.uv.sources]:
# pinning a cu124 index here would break `uv sync --extra gpu` on a developer's Mac,
# and running the chemprop tests locally on CPU torch is what makes this engine testable.
gpu = ["chemprop>=2.2,<3"]
```

- [ ] **Step 8: Refresh the lock and verify the extras resolve**

Run:
```bash
uv lock
uv sync --extra s3
uv run pytest tests/unit/infrastructure/test_blob_store.py -v
```
Expected: lock updates, sync succeeds, tests PASS.

- [ ] **Step 9: Full check**

Run: `uv run pytest tests/unit -q && uv run lint-imports && uv run ruff check src tests && uv run mypy src`
Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml uv.lock src/daikonstudio/settings.py \
        src/daikonstudio/infrastructure/storage/fsspec_blob_store.py \
        src/daikonstudio/infrastructure/di/container.py \
        src/daikonstudio/infrastructure/worker.py \
        tests/unit/infrastructure/test_blob_store.py
git commit -m "feat: swappable blob backends via fsspec storage options"
```

---

## Task 2: Lane routing

An engine declares which lane it needs; the enqueuer routes to that lane's arq queue; a worker pulls exactly one lane. This is the whole of "placement is deployment configuration, not code".

**Files:**
- Modify: `backend/src/daikonstudio/application/engines/manifest.py`
- Modify: `backend/src/daikonstudio/application/execution/enqueue.py`
- Modify: `backend/src/daikonstudio/infrastructure/worker.py`
- Modify: `backend/src/daikonstudio/settings.py`
- Modify: `backend/src/daikonstudio/application/execution/train_protocol.py` (in `TrainProtocol.__call__`)
- Modify: `backend/src/daikonstudio/application/execution/predict_with_protocol.py` (`PredictWithProtocol`)
- Modify: `backend/src/daikonstudio/infrastructure/di/container.py`
- Test: `backend/tests/unit/execution/test_lanes.py` (create)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `DEFAULT_LANE: str = "default"` and `EngineManifest.lane: str` in `application/engines/manifest.py`; `queue_for(lane: str) -> str` in `infrastructure/worker.py`; `JobEnqueuer.enqueue(run_id: uuid.UUID, lane: str = DEFAULT_LANE) -> None`; `PredictWithProtocol(protocols, runs, store, enqueuer, engines)` — a fifth constructor argument of type `EngineRegistry`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/unit/execution/test_lanes.py`:

```python
"""Lane routing: an engine says what it needs, a deployment says where that runs.

These tests pin the seam, not the queue. What matters is that the lane on the
manifest is the lane the job is enqueued to -- for both run kinds -- and that the
default lane keeps arq's own queue name so adding lanes needs no drain-and-migrate.
"""

from __future__ import annotations

import uuid
from typing import Any

import polars as pl
import pytest

from daikonstudio.application.engines.context import PredictContext, TrainContext, TrainResult
from daikonstudio.application.engines.manifest import DEFAULT_LANE, EngineManifest, TaskType
from daikonstudio.application.engines.registry import EngineRegistry
from daikonstudio.infrastructure.worker import ArqEnqueuer, queue_for


class _StubEngine:
    def __init__(self, engine_id: str, lane: str) -> None:
        self._manifest = EngineManifest(
            id=engine_id,
            version="1.0.0",
            name=engine_id,
            description="",
            tasks=(TaskType.REGRESSION,),
            lane=lane,
        )

    def manifest(self) -> EngineManifest:
        return self._manifest

    def train(self, ctx: TrainContext) -> TrainResult:  # pragma: no cover - never called
        raise NotImplementedError

    def predict(self, ctx: PredictContext) -> pl.DataFrame:  # pragma: no cover
        raise NotImplementedError


class _RecordingPool:
    """Stands in for an ArqRedis pool; records the queue each job was routed to."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def enqueue_job(self, function: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append({"function": function, "args": args, "kwargs": kwargs})


def test_default_lane_keeps_arqs_own_queue_name() -> None:
    """Renaming the default queue would strand every job already sitting in it."""
    assert queue_for(DEFAULT_LANE) == "arq:queue"


def test_a_named_lane_gets_its_own_queue() -> None:
    assert queue_for("gpu") == "arq:queue:gpu"


async def test_arq_enqueuer_routes_to_the_lanes_queue() -> None:
    enqueuer = ArqEnqueuer("redis://unused")
    pool = _RecordingPool()
    enqueuer._pool = pool  # type: ignore[assignment]
    run_id = uuid.uuid4()

    await enqueuer.enqueue(run_id, lane="gpu")

    assert pool.calls == [
        {"function": "run_job", "args": (run_id,), "kwargs": {"_queue_name": "arq:queue:gpu"}}
    ]


async def test_arq_enqueuer_carries_only_the_run_id() -> None:
    """The load-bearing invariant: all job state lives on the Run row. A payload in
    the queue message is what makes the orchestrator un-swappable."""
    enqueuer = ArqEnqueuer("redis://unused")
    pool = _RecordingPool()
    enqueuer._pool = pool  # type: ignore[assignment]
    run_id = uuid.uuid4()

    await enqueuer.enqueue(run_id)

    assert pool.calls[0]["args"] == (run_id,)


def test_registry_exposes_the_lane_for_routing() -> None:
    registry = EngineRegistry({"heavy": _StubEngine("heavy", "gpu")})

    assert registry.get("heavy").manifest().lane == "gpu"


def test_manifest_lane_defaults_to_the_default_lane() -> None:
    """Existing engines must be untouched by this change."""
    manifest = EngineManifest(
        id="e", version="1", name="e", description="", tasks=(TaskType.REGRESSION,)
    )

    assert manifest.lane == DEFAULT_LANE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/execution/test_lanes.py -v`
Expected: FAIL — `ImportError: cannot import name 'DEFAULT_LANE'`

- [ ] **Step 3: Add the lane to the manifest**

In `backend/src/daikonstudio/application/engines/manifest.py`, add above `class TaskType`:

```python
# The lane an engine runs in when it does not ask for anything special. A lane is a
# requirement ("this needs a GPU"), not a machine: a deployment satisfies it by running
# a worker with STUDIO_WORKER_LANE set to that name, on whatever hardware it has.
DEFAULT_LANE = "default"
```

and add the field to `EngineManifest`, after `conditions`:

```python
    lane: str = DEFAULT_LANE
```

Keep `is_baseline: bool = False` last. `lane` is a plain `str`, so the manifest JSON round-trip tripwire keeps passing.

- [ ] **Step 4: Widen the enqueuer port**

Replace the `JobEnqueuer` protocol in `backend/src/daikonstudio/application/execution/enqueue.py`:

```python
import uuid
from typing import Protocol

from daikonstudio.application.engines.manifest import DEFAULT_LANE


class JobEnqueuer(Protocol):
    async def enqueue(self, run_id: uuid.UUID, lane: str = DEFAULT_LANE) -> None:
        """`lane` is routing metadata and nothing else.

        The queue message stays a bare `run_id` -- all job state lives on the Run row,
        which is what lets any orchestrator sit behind this port. A lane names which
        pool of workers should pick the job up; it is never state, and is never stored.
        """
        ...
```

- [ ] **Step 5: Route in the enqueuers, and drive WorkerSettings from Settings**

In `backend/src/daikonstudio/infrastructure/worker.py`:

Add to the imports:
```python
from arq.constants import default_queue_name

from daikonstudio.application.engines.manifest import DEFAULT_LANE
```

Add above `class WorkerSettings`:
```python
# arq's own hard timeout is a backstop, not the mechanism. The cooperative deadline in
# train_protocol.py is what actually stops a fit; this margin exists so arq only fires
# when an engine ignores `report` entirely -- see the module docstring there for why a
# firing arq timeout is expensive.
_HARD_TIMEOUT_MARGIN_SECONDS = 600


def queue_for(lane: str) -> str:
    """The arq queue a lane's jobs land on.

    The default lane deliberately keeps arq's own default queue name, so introducing
    lanes needs no drain-and-migrate of jobs already queued under the old one.
    """
    return default_queue_name if lane == DEFAULT_LANE else f"{default_queue_name}:{lane}"
```

Replace `ArqEnqueuer.enqueue`:
```python
    async def enqueue(self, run_id: uuid.UUID, lane: str = DEFAULT_LANE) -> None:
        pool = await self._get_pool()
        await pool.enqueue_job("run_job", run_id, _queue_name=queue_for(lane))
```

Replace `InlineEnqueuer.enqueue`:
```python
    async def enqueue(self, run_id: uuid.UUID, lane: str = DEFAULT_LANE) -> None:
        # `lane` is ignored on purpose: running the job in the caller's own process
        # has no queue to route it to. Accepting the argument is what keeps the two
        # implementations interchangeable from a caller's point of view.
        with contextlib.suppress(Exception, SystemExit):
            await run_job(self._ctx, run_id)
```

Replace `class WorkerSettings` entirely:
```python
_settings = Settings()


class WorkerSettings:
    """arq WorkerSettings -- mirrors the lifespan wiring in `interface/app.py`.

    One worker process serves exactly one lane. A deployment with a GPU runs a second
    process with STUDIO_WORKER_LANE=gpu and STUDIO_WORKER_MAX_JOBS=1; a deployment with
    several GPUs runs one process per device with CUDA_VISIBLE_DEVICES pinned; a
    deployment with a GPU cluster runs one per node. All of them pull the same queue,
    and Redis distributes. None of that is code.
    """

    functions: ClassVar[list[Any]] = [run_job]
    redis_settings = RedisSettings.from_dsn(_settings.redis_url)
    on_startup = _on_startup
    on_shutdown = _on_shutdown
    queue_name = queue_for(_settings.worker_lane)
    max_jobs = _settings.worker_max_jobs
    job_timeout = _settings.worker_job_timeout + _HARD_TIMEOUT_MARGIN_SECONDS
```

- [ ] **Step 6: Add the worker settings**

In `backend/src/daikonstudio/settings.py`, after `inline_jobs`:

```python
    # Which lane's queue this worker process pulls. Engines declare a lane on their
    # manifest and a deployment satisfies it by running a worker here -- nothing in the
    # codebase names a host. See
    # docs/superpowers/specs/2026-07-30-remote-engines-chemprop-design.md.
    worker_lane: str = "default"
    # arq's own default is 10. A GPU worker MUST set this to 1 (or run one process per
    # device with CUDA_VISIBLE_DEVICES pinned): concurrent fits on one device exhaust
    # its memory, and arq will happily start ten.
    worker_max_jobs: int = 10
    # The SOFT deadline, in seconds, enforced cooperatively inside the engine through
    # TrainContext.report. arq's hard job_timeout is derived from this with a margin;
    # see infrastructure/worker.py. Raise this on a GPU lane, not the hard timeout.
    worker_job_timeout: int = 1800
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/unit/execution/test_lanes.py -v`
Expected: PASS

- [ ] **Step 8: Write the failing call-site tests**

Append to `backend/tests/unit/execution/test_lanes.py`:

```python
class _RecordingEnqueuer:
    def __init__(self) -> None:
        self.lanes: list[str] = []

    async def enqueue(self, run_id: uuid.UUID, lane: str = DEFAULT_LANE) -> None:
        self.lanes.append(lane)


class _StubRuns:
    def __init__(self) -> None:
        self.added: list[Any] = []

    async def add(self, run: Any) -> None:
        self.added.append(run)

    async def find_by_cache_key(self, workspace_id: uuid.UUID, cache_key: str) -> None:
        return None


async def test_training_is_enqueued_to_its_engines_lane(
    training_setup: Any,
) -> None:
    """A GPU engine's training job must not land on the default queue, where a CPU
    worker would pick it up and fail on a missing CUDA install."""
    enqueuer, command, auth, use_case = training_setup

    result = await use_case(command, auth)

    assert result.unwrap() is not None
    assert enqueuer.lanes == ["gpu"]
```

Add this fixture to the same file:

```python
@pytest.fixture
def training_setup() -> Any:
    """Builds TrainProtocol against stubs, with one engine declaring lane='gpu'."""
    from daikonstudio.application.auth import AuthContext
    from daikonstudio.application.execution.train_protocol import (
        TrainProtocol,
        TrainProtocolCommand,
    )
    from daikonstudio.domain.data.dataset import Dataset

    workspace_id = uuid.uuid4()
    dataset_id = uuid.uuid4()

    class _StubDatasets:
        async def get(self, ws: uuid.UUID, ds: uuid.UUID) -> Any:
            return _stub_dataset

    _stub_dataset = _make_dataset(workspace_id, dataset_id)
    enqueuer = _RecordingEnqueuer()
    registry = EngineRegistry({"heavy": _StubEngine("heavy", "gpu")})
    use_case = TrainProtocol(_StubDatasets(), _StubRuns(), enqueuer, registry)  # type: ignore[arg-type]
    command = TrainProtocolCommand(
        name="run", dataset_id=dataset_id, engine_id="heavy", conditions={}
    )
    auth = AuthContext(
        user_id=uuid.uuid4(), workspace_id=workspace_id, roles=frozenset({"editor"})
    )
    return enqueuer, command, auth, use_case
```

> **Note for the implementer:** `_make_dataset` and the exact `AuthContext` constructor are
> not invented here — read `tests/integration/test_train_protocol.py` and
> `tests/fakes/auth.py` for how this repository already builds a `Dataset` and an
> `AuthContext`, and reuse those helpers rather than duplicating them. If a shared
> dataset factory already exists, import it; if not, build the `Dataset` inline exactly
> as the integration test does. The assertion above is the part that matters.

- [ ] **Step 9: Route both call sites**

In `backend/src/daikonstudio/application/execution/train_protocol.py`, inside `TrainProtocol.__call__`, replace the engine lookup and the enqueue:

```python
        try:
            engine = self._engines.get(command.engine_id)
        except UnknownEngineError:
            return Failure(NotFoundError("Engine", command.engine_id))
```

and at the end:

```python
        await self._runs.add(run)
        # Routed by the engine's own declared lane. The engine is already in hand from
        # the membership check above, so this costs nothing.
        await self._enqueuer.enqueue(run.id, lane=engine.manifest().lane)
        return Success(run)
```

In `backend/src/daikonstudio/application/execution/predict_with_protocol.py`:

Add `EngineRegistry` and `UnknownEngineError` to the imports (`EngineRegistry` is already imported; add `UnknownEngineError` from the same module), then change `PredictWithProtocol.__init__`:

```python
    def __init__(
        self,
        protocols: ProtocolRepository,
        runs: RunRepository,
        store: BlobStore,
        enqueuer: JobEnqueuer,
        engines: EngineRegistry,
    ) -> None:
        self._protocols = protocols
        self._runs = runs
        self._store = store
        self._enqueuer = enqueuer
        self._engines = engines
```

In `__call__`, immediately after the cache-hit early return and **before** `run = Run(...)`:

```python
        # Resolved before the Run row exists, so a Protocol whose engine this deployment
        # no longer ships fails as a clean 404 rather than leaving an orphan PENDING Run
        # that no worker can ever serve. Deliberately after the cache check: an already
        # READY result stays reusable even if the engine has since been removed.
        try:
            lane = self._engines.get(protocol.engine_id).manifest().lane
        except UnknownEngineError:
            return Failure(NotFoundError("Engine", protocol.engine_id))
```

and change the enqueue at the end of `__call__`:

```python
        await self._runs.add(run)
        await self._enqueuer.enqueue(run.id, lane=lane)
        return Success(run)
```

- [ ] **Step 10: Update the container wiring**

In `backend/src/daikonstudio/infrastructure/di/container.py`:

```python
    container.define(
        PredictWithProtocol,
        lambda c: PredictWithProtocol(
            _protocols(c), _runs(c), c[BlobStore], c[JobEnqueuer], c[EngineRegistry]
        ),
    )
```

- [ ] **Step 11: Run the full suite**

Run: `uv run pytest -q && uv run lint-imports && uv run ruff check src tests && uv run mypy src`
Expected: all pass. If an API or integration test constructs `PredictWithProtocol` directly, add the registry argument there too — `default_registry()` is the right value in a test.

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "feat: route runs to an engine's declared lane"
```

---

## Task 3: Cooperative progress and cancellation

The only mechanism that can stop a fit already running on a `to_thread` worker thread. Without it, a GPU lane's `job_timeout` produces orphaned, unkillable training threads — see the spec's "finding that shapes this design".

**Files:**
- Modify: `backend/src/daikonstudio/application/engines/context.py`
- Modify: `backend/src/daikonstudio/application/engines/protocol.py`
- Modify: `backend/src/daikonstudio/application/execution/train_protocol.py`
- Modify: `backend/src/daikonstudio/infrastructure/worker.py`
- Test: `backend/tests/unit/execution/test_worker.py` (extend)
- Test: `backend/tests/unit/engines/test_engine_contract.py` (extend)

**Interfaces:**
- Consumes: `DEFAULT_LANE` from Task 2 (indirectly, via unchanged imports).
- Produces: `ProgressReporter = Callable[[float, str], None]`, `RunInterrupted(reason: str, *, cancelled: bool)` and `TrainContext.report: ProgressReporter` in `application/engines/context.py`; `RunTraining(datasets, protocols, runs, store, engines, normalizer, deadline_seconds: float | None = None)`.

- [ ] **Step 1: Write the failing contract test**

Append to `backend/tests/unit/engines/test_engine_contract.py`:

```python
def test_report_defaults_to_a_no_op() -> None:
    """An engine that never calls `report` must still train. Both ECFP4 engines are
    exactly that: a single `.fit()` offers no yield point, so they are honestly not
    interruptible, and the contract must not force them to pretend otherwise."""
    ctx = TrainContext(
        frame=pl.DataFrame({"smiles": ["CCO"], "y": [1.0], "split": ["train"]}),
        task=TaskType.REGRESSION,
        structure_column="smiles",
        target_column="y",
        conditions={},
        seed=7,
    )

    assert ctx.report(0.5, "anything") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/engines/test_engine_contract.py::test_report_defaults_to_a_no_op -v`
Expected: FAIL — `AttributeError: 'TrainContext' object has no attribute 'report'`

- [ ] **Step 3: Add the reporter to the engine contract**

In `backend/src/daikonstudio/application/engines/context.py`, add above `TrainContext`:

```python
from collections.abc import Callable

ProgressReporter = Callable[[float, str], None]


class RunInterrupted(Exception):
    """Raised by `TrainContext.report` to stop a fit that must not continue.

    Deliberately not a `DomainError`: this is a control signal from the worker to the
    engine, not a violated invariant, and mapping it to an HTTP status would be
    meaningless -- nothing serves it over HTTP.

    **Engines must not catch this.** Python cannot kill a thread from outside, and
    engines run on a worker thread via `asyncio.to_thread`, so raising inside the
    engine's own call stack is the only way anything can stop work already in flight.
    Swallowing it turns a cancelled run into a run that keeps burning a GPU.

    `cancelled` distinguishes the two reasons, because they end the Run differently:
    a user cancellation leaves a row that is already CANCELLED, while a deadline is a
    failure that still needs recording.
    """

    def __init__(self, reason: str, *, cancelled: bool) -> None:
        super().__init__(reason)
        self.reason = reason
        self.cancelled = cancelled


def _no_op(fraction: float, phase: str) -> None:
    """The default reporter: an engine that ignores `report` is simply not
    interruptible, which is the honest description of any engine whose work happens
    inside one opaque library call."""
```

Add the field to `TrainContext`, after `seed`:

```python
    report: ProgressReporter = _no_op
    """Called periodically during long work to publish progress and to check whether
    the run is still wanted. `fraction` is progress within *this fit*, 0.0 to 1.0; the
    worker maps it onto the overall run. May raise `RunInterrupted` -- do not catch it.
    Calling it is optional; calling it often is what makes an engine stoppable."""
```

In `backend/src/daikonstudio/application/engines/protocol.py`, extend the module docstring with a final paragraph:

```
An engine that does long work SHOULD call `ctx.report(fraction, phase)` periodically.
That single callback is what publishes progress to the Run row a client is polling,
and -- because it may raise `RunInterrupted` -- is the only thing that can stop a fit
already running on a worker thread. An engine that never calls it still works; it is
simply not interruptible, and its progress bar will not move.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/engines/test_engine_contract.py -v`
Expected: PASS

- [ ] **Step 5: Write the failing worker tests**

Append to `backend/tests/unit/execution/test_worker.py`:

```python
async def test_a_cancelled_run_is_not_retried() -> None:
    """`run_job` must not re-raise RunInterrupted. arq treats a propagating exception
    as a job to retry (retry_jobs=True, max_tries=5), and at GPU durations retrying a
    deliberately-stopped fit is how one cancelled run becomes five training threads on
    one device."""
    run = _pending_run()
    ctx = _ctx_with(run)

    async def _interrupt(_ctx: dict[str, Any], _run: Run) -> str:
        raise RunInterrupted("the run was cancelled", cancelled=True)

    monkeypatch_handler(ctx, run.kind, _interrupt)

    await worker.run_job(ctx, run.id)  # must not raise

    assert _reload(ctx, run.id).status is RunStatus.CANCELLED


async def test_a_deadline_fails_the_run_with_its_reason() -> None:
    run = _pending_run()
    ctx = _ctx_with(run)

    async def _interrupt(_ctx: dict[str, Any], _run: Run) -> str:
        raise RunInterrupted("exceeded the 60s deadline for this worker lane", cancelled=False)

    monkeypatch_handler(ctx, run.kind, _interrupt)

    await worker.run_job(ctx, run.id)  # must not raise

    reloaded = _reload(ctx, run.id)
    assert reloaded.status is RunStatus.FAILED
    assert "deadline" in (reloaded.error_message or "")


async def test_a_run_cancelled_during_the_last_throttle_window_is_not_marked_ready() -> None:
    """The reporter only checks the row every 10 seconds. A cancellation landing inside
    that window would otherwise be overwritten with READY by a handler that finished."""
    run = _pending_run()
    ctx = _ctx_with(run)

    async def _succeed_after_cancellation(_ctx: dict[str, Any], inflight: Run) -> str:
        await _cancel_in_the_database(ctx, inflight.id)
        return "blob://result"

    monkeypatch_handler(ctx, run.kind, _succeed_after_cancellation)

    await worker.run_job(ctx, run.id)

    assert _reload(ctx, run.id).status is RunStatus.CANCELLED
```

> **Note for the implementer:** `test_worker.py` already has a working pattern for
> building a `ctx`, seeding a Run and swapping `_HANDLERS` — read the existing tests in
> that file first and express `_pending_run`, `_ctx_with`, `_reload`,
> `monkeypatch_handler` and `_cancel_in_the_database` in whatever style it already uses
> (it may use a real session factory or an in-memory double). Do not introduce a second
> convention. The three assertions above are the contract.

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/unit/execution/test_worker.py -v`
Expected: FAIL — `ImportError: cannot import name 'RunInterrupted'` or the runs end `FAILED`/`READY`.

- [ ] **Step 7: Handle interruption in run_job**

In `backend/src/daikonstudio/infrastructure/worker.py`, add to the imports:

```python
from daikonstudio.application.engines.context import RunInterrupted
from daikonstudio.domain.execution.run import Run, RunKind, RunStatus
```

Replace the body of `run_job` after `await _save(ctx, run)`:

```python
    try:
        handler = _HANDLERS[run.kind]
        result_uri = await handler(ctx, run)
    except RunInterrupted as interrupted:
        # Deliberately NOT re-raised, unlike a handler failure below. arq treats a
        # propagating exception as a retry (retry_jobs=True, max_tries=5 by default),
        # and this exception means the work was stopped on purpose. Retrying it would
        # restart a fit the user cancelled -- five times, on the same GPU.
        if not interrupted.cancelled:
            run.fail(interrupted.reason)
            await _save(ctx, run)
        # A cancellation needs no write: the row is already CANCELLED, which is
        # precisely why `report` raised.
        return
    except (Exception, SystemExit) as exc:
        run.fail(repr(exc))
        await _save(ctx, run)
        raise  # re-raise: FAILED is persisted above regardless of what happens next

    # Re-read before claiming success. The reporter only consults the row every
    # `_PROGRESS_INTERVAL_SECONDS`, so a cancellation landing inside the final window
    # would otherwise be silently overwritten with READY -- the exact lie
    # `Run.cancel()`'s old ponytail comment admitted to.
    current = await _load(ctx, run.id)
    if current.status is not RunStatus.RUNNING:
        return
    run.succeed(result_uri)
    await _save(ctx, run)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/unit/execution/test_worker.py -v`
Expected: PASS

- [ ] **Step 9: Build the reporter in RunTraining**

In `backend/src/daikonstudio/application/execution/train_protocol.py`:

Add to the imports:
```python
import time

from daikonstudio.application.engines.context import (
    PredictContext,
    ProgressReporter,
    RunInterrupted,
    TrainContext,
    TrainResult,
)
from daikonstudio.domain.execution.run import Run, RunKind, RunStatus, compute_cache_key
```

Add module-level constants below `scorecard_inputs_key`:

```python
# How often the reporter is allowed a database round-trip. Cancellation latency is
# bounded by this; against a fit measured in minutes that is immaterial, and it keeps a
# 500-epoch run from writing 500 rows.
_PROGRESS_INTERVAL_SECONDS = 10.0
# How long the training thread will wait for the event loop to service one checkpoint.
# Generous: the loop is otherwise idle while the fit runs.
_CHECKPOINT_TIMEOUT_SECONDS = 30.0
# Progress spans per leg, so a long fit's own per-epoch progress has somewhere to move
# instead of the bar sitting at a single number for its whole duration.
_CHOSEN_SPAN = (0.0, 0.6)
_BASELINE_SPAN = (0.6, 0.7)
_RANDOM_SPLIT_SPAN = (0.7, 0.95)
```

Change `RunTraining.__init__` to accept the deadline:

```python
    def __init__(
        self,
        datasets: DatasetRepository,
        protocols: ProtocolRepository,
        runs: RunRepository,
        store: BlobStore,
        engines: EngineRegistry,
        normalizer: StructureNormalizer,
        deadline_seconds: float | None = None,
    ) -> None:
        self._datasets = datasets
        self._protocols = protocols
        self._runs = runs
        self._store = store
        self._engines = engines
        self._normalizer = normalizer
        self._deadline_seconds = deadline_seconds
        self._deadline_at: float | None = None
```

At the top of `RunTraining.__call__`, before anything else:

```python
        self._deadline_at = (
            time.monotonic() + self._deadline_seconds
            if self._deadline_seconds is not None
            else None
        )
```

Add these two methods to `RunTraining`:

```python
    def _reporter(self, run: Run, span: tuple[float, float]) -> ProgressReporter:
        """A callback the engine invokes from the worker thread.

        Three things happen per call, in this order and for this reason:

        1. The deadline is checked. It needs no I/O, so it runs on every call -- which
           is what makes an overrunning fit stop promptly rather than at the next
           throttled checkpoint.
        2. The database round-trip is throttled to `_PROGRESS_INTERVAL_SECONDS`.
        3. The Run's status is re-read and progress written. Reading is the point: a
           cancellation happens in the API process against a different row instance
           entirely, so this worker's in-memory aggregate would never see it.

        `asyncio.run_coroutine_threadsafe` is how a synchronous engine reaches the
        event loop that owns the repositories. Blocking on the result is deliberate:
        the engine must not proceed past a checkpoint that says the run was cancelled.
        """
        loop = asyncio.get_running_loop()
        low, high = span
        last_written = 0.0

        def report(fraction: float, phase: str) -> None:
            nonlocal last_written
            now = time.monotonic()
            if self._deadline_at is not None and now > self._deadline_at:
                raise RunInterrupted(
                    f"exceeded this worker lane's {self._deadline_seconds:.0f}s deadline; "
                    "raise STUDIO_WORKER_JOB_TIMEOUT on the lane if the work is legitimate",
                    cancelled=False,
                )
            if now - last_written < _PROGRESS_INTERVAL_SECONDS:
                return
            last_written = now
            clamped = min(max(fraction, 0.0), 1.0)
            future = asyncio.run_coroutine_threadsafe(
                self._checkpoint(run, low + (high - low) * clamped, phase), loop
            )
            if not future.result(timeout=_CHECKPOINT_TIMEOUT_SECONDS):
                raise RunInterrupted("the run was cancelled", cancelled=True)

        return report

    async def _checkpoint(self, run: Run, fraction: float, phase: str) -> bool:
        """Write progress; report whether the run is still wanted."""
        current = await self._runs.get_by_id(run.id)
        if current is None or current.status is not RunStatus.RUNNING:
            return False
        run.report_progress(fraction, phase=phase)
        await self._runs.update(run)
        return True
```

Change `_fit` and `_train_off_thread` to take and thread the span:

```python
    async def _fit(
        self,
        run: Run,
        engine: Engine,
        dataset: Dataset,
        task: TaskType,
        conditions: dict[str, object],
        frame: pl.DataFrame,
        span: tuple[float, float],
        phase: str | None = None,
    ) -> TrainResult:
        """Report the phase, then fit off the event loop.

        Progress is set *before* the work, naming what is about to run rather than what
        just finished: the row is the only channel the client has, and a phase that
        describes the completed step would leave the UI reading "training baseline"
        while the random-split fit is what is actually holding it up. Inside the span,
        an engine that calls `ctx.report` moves the bar itself.
        """
        resolved_phase = phase or f"training {engine.manifest().id}"
        await self._progress(run, span[0], resolved_phase)
        return await self._train_off_thread(engine, dataset, task, conditions, frame, span)

    async def _train_off_thread(
        self,
        run: Run,
        engine: Engine,
        dataset: Dataset,
        task: TaskType,
        conditions: dict[str, object],
        frame: pl.DataFrame,
        span: tuple[float, float],
    ) -> TrainResult:
        # train() is synchronous and CPU-bound by contract (see engines/protocol.py):
        # the worker offloads it so engine authors never have to think about threads.
        # `run` is threaded through only so the reporter can reach the row -- the
        # engine never sees it.
        return await asyncio.to_thread(
            engine.train,
            TrainContext(
                frame=frame,
                task=task,
                structure_column=dataset.structure_column,
                target_column=dataset.target.column,
                conditions=conditions,
                seed=dataset.split.seed,
                report=self._reporter(run, span),
            ),
        )
```

Note that `_train_off_thread` gains `run` as its first parameter, because building a
reporter needs the row. `_fit` already holds it; `_optimism_gap` passes its own.

Update the call sites in `__call__` and `_optimism_gap` to pass spans instead of the old float fractions:

```python
        chosen = await self._fit(run, engine, dataset, task, conditions, frame, _CHOSEN_SPAN)
        ...
            baseline_result = await self._fit(
                run, baseline, dataset, task, baseline_conditions, frame,
                _BASELINE_SPAN, "training baseline",
            )
```

and inside `_optimism_gap`, replace `await self._progress(run, 0.9, ...)` with
`await self._progress(run, _RANDOM_SPLIT_SPAN[0], "training random-split comparison")`
and the fit call with
`await self._train_off_thread(run, engine, dataset, task, conditions, random_frame, _RANDOM_SPLIT_SPAN)`.

`_fit`'s own body already calls `self._train_off_thread(...)`; add `run` as its first
argument there too.

- [ ] **Step 10: Stop the optimism gap from swallowing an interruption**

This is a real bug the new exception introduces. `_optimism_gap` catches bare `Exception`,
which would turn a user's cancellation into a recorded "random split unavailable" and let
the run finish successfully. In `_optimism_gap`, add this clause **before** the existing
`except Exception`:

```python
        except RunInterrupted:
            # Not degradable, unlike every other failure in this leg. A cancellation or
            # a deadline means stop, and recording it as an unavailable comparison would
            # let the run succeed after the user asked it not to.
            raise
```

- [ ] **Step 11: Pass the deadline from the worker**

In `backend/src/daikonstudio/infrastructure/worker.py`, change `_train`:

```python
async def _train(ctx: dict[str, Any], run: Run) -> str:
    sessions = ctx["sessions"]
    return await RunTraining(
        SqlAlchemyDatasetRepository(sessions),
        SqlAlchemyProtocolRepository(sessions),
        SqlAlchemyRunRepository(sessions),
        ctx["store"],
        default_registry(),
        RdkitStructureNormalizer(),
        # `.get`, not `[...]`: InlineEnqueuer builds its own ctx with only the two
        # entries a handler needs, and dev-mode jobs have no lane deadline to enforce.
        deadline_seconds=ctx.get("job_deadline_seconds"),
    )(run)
```

and in `_on_startup`:

```python
    ctx["job_deadline_seconds"] = settings.worker_job_timeout
```

- [ ] **Step 12: Run the full suite**

Run: `uv run pytest -q && uv run lint-imports && uv run ruff check src tests && uv run mypy src`
Expected: all pass. `tests/integration/test_train_protocol.py` exercises the real path through `InlineEnqueuer` and must stay green — if a progress assertion there pins `0.33`/`0.66`, update it to the new spans.

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "feat: cooperative progress and cancellation for long engine fits"
```

---

## Task 4: Extract shared metric functions

Chemprop must report metrics under identical names and definitions to the baseline, or the Scorecard's central comparison compares unlike things. Pure refactor — behaviour must not change.

**Files:**
- Modify: `backend/src/daikonstudio/infrastructure/engines/_scoring.py`
- Test: `backend/tests/unit/engines/test_metrics.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: in `infrastructure/engines/_scoring.py` —
  `regression_metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]` and
  `classification_metrics(y_true: np.ndarray, labels: np.ndarray, probabilities: np.ndarray, *, train_has_both_classes: bool) -> dict[str, float]`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/engines/test_metrics.py`:

```python
"""The metric vocabulary every engine shares.

These functions exist so a chemprop model and the ECFP4 baseline are measured by
literally the same code. A Scorecard comparing "your model" against "the baseline" is
the product's central claim; two engines computing "auroc" slightly differently would
make that claim false while looking correct.
"""

from __future__ import annotations

import math

import numpy as np

from daikonstudio.infrastructure.engines._scoring import (
    classification_metrics,
    regression_metrics,
)


def test_regression_reports_exactly_rmse_mae_and_r2() -> None:
    metrics = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))

    assert sorted(metrics) == ["mae", "r2", "rmse"]
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_regression_never_reports_accuracy() -> None:
    """A 99.9%-negative dataset yields a 99.9%-accurate useless model. The only
    reliable way to keep that number off a Scorecard is to never compute it."""
    assert "accuracy" not in regression_metrics(np.array([1.0]), np.array([1.0]))


def test_classification_reports_exactly_the_four_defined_metrics() -> None:
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = classification_metrics(
        y_true, (probabilities >= 0.5).astype(float), probabilities,
        train_has_both_classes=True,
    )

    assert sorted(metrics) == ["auprc", "auroc", "balanced_accuracy", "mcc"]
    assert metrics["auroc"] == 1.0
    assert "accuracy" not in metrics


def test_a_single_class_test_split_makes_every_metric_undefined() -> None:
    """Balanced accuracy silently collapses to plain accuracy when y_true has one
    class -- exactly the number this module exists never to report. All four go
    undefined together, uniformly."""
    y_true = np.array([1.0, 1.0, 1.0])
    probabilities = np.array([0.6, 0.7, 0.8])

    metrics = classification_metrics(
        y_true, np.ones(3), probabilities, train_has_both_classes=True
    )

    assert sorted(metrics) == ["auprc", "auroc", "balanced_accuracy", "mcc"]
    assert all(math.isnan(value) for value in metrics.values())


def test_a_single_class_training_split_makes_every_metric_undefined() -> None:
    y_true = np.array([0.0, 1.0])
    probabilities = np.array([0.4, 0.6])

    metrics = classification_metrics(
        y_true, np.array([0.0, 1.0]), probabilities, train_has_both_classes=False
    )

    assert all(math.isnan(value) for value in metrics.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/engines/test_metrics.py -v`
Expected: FAIL — `ImportError: cannot import name 'regression_metrics'`

- [ ] **Step 3: Extract the two pure functions**

In `backend/src/daikonstudio/infrastructure/engines/_scoring.py`, add above `_score`:

```python
def _undefined_classification_metrics() -> dict[str, float]:
    """All four together, uniformly. MCC has no defined value on a single-class split,
    and balanced accuracy silently collapses to plain accuracy -- reporting three of
    four would imply the missing one was the only problem."""
    return {
        "mcc": float("nan"),
        "balanced_accuracy": float("nan"),
        "auroc": float("nan"),
        "auprc": float("nan"),
    }


def regression_metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """RMSE, MAE and R2 -- the regression half of the shared vocabulary.

    Engine-agnostic on purpose: this is the code a chemprop model and the ECFP4
    baseline are both measured by, which is what makes a Scorecard's comparison mean
    anything.
    """
    return {
        "rmse": float(root_mean_squared_error(y_true, predicted)),
        "mae": float(mean_absolute_error(y_true, predicted)),
        "r2": float(r2_score(y_true, predicted)),
    }


def classification_metrics(
    y_true: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    train_has_both_classes: bool,
) -> dict[str, float]:
    """MCC, balanced accuracy, AUROC and AUPRC. Never plain accuracy.

    `labels` are hard 0/1 predictions and `probabilities` is P(class=1); both are passed
    rather than derived, because sklearn's `predict` and a 0.5 threshold on
    `predict_proba` are the same thing for these estimators and an engine that only has
    probabilities (chemprop) should threshold them explicitly rather than have this
    function guess.

    `train_has_both_classes` generalises what used to be a `model.classes_` check, so an
    engine with no such attribute can answer the same question.
    """
    if len(np.unique(y_true)) < 2 or not train_has_both_classes:
        return _undefined_classification_metrics()
    return {
        "mcc": float(matthews_corrcoef(y_true, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, labels)),
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
    }
```

Replace the body of `_score` with a thin caller that preserves the existing early return
exactly:

```python
def _score(
    model: Any, test_rows: pl.DataFrame, ctx: TrainContext, is_classification: bool
) -> dict[str, float]:
    """RMSE/MAE/R2 for regression; MCC/balanced accuracy/AUROC/AUPRC for classification."""
    x_test = ecfp4(test_rows[ctx.structure_column].to_list())
    y_test = test_rows[ctx.target_column].to_numpy()

    if not is_classification:
        return regression_metrics(y_test, model.predict(x_test))

    # Both single-class checks stay HERE, before any sklearn call -- not delegated to
    # `classification_metrics` -- because short-circuiting is what keeps sklearn's
    # "y_pred contains classes not in y_true" warning from firing at all.
    if len(np.unique(y_test)) < 2 or len(model.classes_) < 2:
        return _undefined_classification_metrics()

    return classification_metrics(
        y_test,
        model.predict(x_test),
        _positive_class_probability(model, x_test),
        train_has_both_classes=True,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/engines/ -v`
Expected: PASS, including the pre-existing `test_ecfp4_engines.py` unchanged — that suite passing untouched is the assertion that this refactor preserved behaviour.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q && uv run ruff check src tests && uv run mypy src`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: extract engine-agnostic metric functions from _scoring"
```

---

## Task 5: Chemprop D-MPNN engine

**Files:**
- Create: `backend/src/daikonstudio/infrastructure/engines/chemprop_dmpnn.py`
- Modify: `backend/src/daikonstudio/infrastructure/engines/registry.py`
- Test: `backend/tests/unit/engines/test_chemprop_dmpnn.py` (create)

**Interfaces:**
- Consumes: `regression_metrics`, `classification_metrics` (Task 4); `TrainContext.report` (Task 3); `EngineManifest.lane` (Task 2).
- Produces: `ChempropDMPNN` with `manifest()`, `train(ctx) -> TrainResult`, `predict(ctx) -> pl.DataFrame`, registered in `default_registry()` under id `chemprop-dmpnn`.

**Verified chemprop 2.2.3 API** (read from source; do not substitute from memory):
- `MoleculeDatapoint.from_smi(smi, *args, **kwargs)`; `y` is a keyword field and must be a **1-D `np.ndarray` of shape `(1,)`, float** — a scalar breaks `dataset.t` and `StandardScaler.fit`.
- `MoleculeDataset(data, featurizer=SimpleMoleculeMolGraphFeaturizer(), n_workers=0)` — the featurizer default is correct; do not pass one.
- `build_dataloader(dataset, batch_size=64, num_workers=0, class_balance=False, seed=None, shuffle=True, drop_last=None)` — **`shuffle` defaults to `True`** (the docstring says otherwise and is wrong). Pass `shuffle=False` for validation, test and predict loaders.
- `MoleculeDataset.normalize_targets(scaler=None) -> StandardScaler` — apply an existing scaler by passing it positionally. There is no separate apply method. **Never normalize the test split.**
- `BondMessagePassing(d_v, d_e, d_h=300, bias=False, depth=3, dropout=0.0, ...)` — hidden dim is `d_h`.
- `MeanAggregation(dim=0)`.
- `RegressionFFN` / `BinaryClassificationFFN` share `(n_tasks=1, input_dim=300, hidden_dim=300, n_layers=1, dropout=0.0, activation="relu", criterion=None, task_weights=None, threshold=None, output_transform=None)`. `input_dim` **must equal** the message passing `d_h`.
- `UnscaleTransform.from_standard_scaler(scaler, pad=0)`. It is a no-op in train mode by design, so the loss stays in scaled space and predictions come back in real units.
- `MPNN(message_passing, agg, predictor, batch_norm=False, metrics=None, warmup_epochs=2, init_lr=1e-4, max_lr=1e-3, final_lr=1e-4, X_d_transform=None)`. It is a `lightning.pytorch.LightningModule` and calls `save_hyperparameters()`, so `MPNN.load_from_checkpoint(path)` reconstructs the architecture standalone.
- `predict_step` returns `forward(...)`. `BinaryClassificationFFN.forward` ends in `.sigmoid()` — **predictions are probabilities, not logits; do not apply another sigmoid.** `RegressionFFN.forward` applies `output_transform`, and `trainer.predict()` puts the model in eval mode, so regression predictions arrive in original units. Shape is `(batch, 1)`.
- Imports: `from chemprop.data import ...`, `from chemprop.models import MPNN`, `from chemprop.nn import ...`, `from chemprop.nn.transforms import UnscaleTransform`, `from lightning import pytorch as lightning`, `from lightning.pytorch.callbacks import LambdaCallback`. `from chemprop import MPNN` does **not** work.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/engines/test_chemprop_dmpnn.py`:

```python
"""The chemprop engine's contract, exercised on CPU.

Skipped unless chemprop imports, because the default-lane worker and the API tier
deliberately install without it. Two epochs on forty molecules: this is not a test of
whether a D-MPNN learns anything, it is a test that the adapter honours the Engine
contract -- metric names the Scorecard can compare, and a predict frame whose dtypes
line up with every other engine's.
"""

from __future__ import annotations

import polars as pl
import pytest

pytest.importorskip("chemprop")

from daikonstudio.application.engines.context import PredictContext, TrainContext
from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.infrastructure.engines.chemprop_dmpnn import ChempropDMPNN

_SMILES = [
    "CCO", "CCC", "CCCC", "c1ccccc1", "CC(=O)O", "CCN", "CCOCC", "CC(C)O",
    "c1ccncc1", "CCCCO", "CC(C)(C)O", "CCCCCC", "c1ccc(cc1)O", "CCS", "CCCl",
    "CC=O", "CCC(=O)O", "c1ccc(cc1)N", "CCCCCCC", "CC(C)C",
]
_SPLITS = ["train"] * 12 + ["validation"] * 4 + ["test"] * 4
_FAST = {"epochs": 2, "depth": 2, "message_hidden_dim": 64, "batch_size": 8}


def _frame(targets: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"smiles": _SMILES, "y": targets, "split": _SPLITS})


def _train_context(frame: pl.DataFrame, task: TaskType) -> TrainContext:
    return TrainContext(
        frame=frame,
        task=task,
        structure_column="smiles",
        target_column="y",
        conditions=_FAST,
        seed=13,
    )


def test_manifest_declares_the_gpu_lane() -> None:
    """The whole point of this engine: it must not land on the default queue, where a
    worker without CUDA would pick it up."""
    assert ChempropDMPNN.manifest().lane == "gpu"


def test_regression_reports_the_shared_metric_vocabulary() -> None:
    frame = _frame([float(i) for i in range(20)])

    result = ChempropDMPNN().train(_train_context(frame, TaskType.REGRESSION))

    assert sorted(result.metrics) == ["mae", "r2", "rmse"]
    assert result.artifact  # a loadable checkpoint, not an empty blob


def test_classification_reports_the_shared_metric_vocabulary() -> None:
    frame = _frame([float(i % 2) for i in range(20)])

    result = ChempropDMPNN().train(_train_context(frame, TaskType.BINARY_CLASSIFICATION))

    assert sorted(result.metrics) == ["auprc", "auroc", "balanced_accuracy", "mcc"]


def test_predict_returns_the_contracted_schema_and_dtypes() -> None:
    """An all-None uncertainty column inferred as polars Null would make this engine's
    output schema-incompatible with every other engine's."""
    frame = _frame([float(i) for i in range(20)])
    trained = ChempropDMPNN().train(_train_context(frame, TaskType.REGRESSION))

    predictions = ChempropDMPNN().predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": ["CCO", "CCC"]}),
            structure_column="smiles",
            artifact=trained.artifact,
            conditions={},
        )
    )

    assert predictions.columns == ["row_id", "value", "uncertainty"]
    assert predictions.schema["row_id"] == pl.Int64
    assert predictions.schema["value"] == pl.Float64
    assert predictions.schema["uncertainty"] == pl.Float64
    assert predictions.height == 2


def test_classification_predictions_are_probabilities() -> None:
    """chemprop's BinaryClassificationFFN already applies sigmoid in forward(). A
    second one here would squash every prediction into [0.5, 0.73]."""
    frame = _frame([float(i % 2) for i in range(20)])
    trained = ChempropDMPNN().train(_train_context(frame, TaskType.BINARY_CLASSIFICATION))

    predictions = ChempropDMPNN().predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": _SMILES}),
            structure_column="smiles",
            artifact=trained.artifact,
            conditions={},
        )
    )

    values = predictions["value"].to_list()
    assert all(0.0 <= value <= 1.0 for value in values)


def test_report_is_called_once_per_epoch() -> None:
    """This callback is the only thing that can stop a GPU fit already in flight."""
    calls: list[tuple[float, str]] = []
    frame = _frame([float(i) for i in range(20)])
    ctx = TrainContext(
        frame=frame,
        task=TaskType.REGRESSION,
        structure_column="smiles",
        target_column="y",
        conditions=_FAST,
        seed=13,
        report=lambda fraction, phase: calls.append((fraction, phase)),
    )

    ChempropDMPNN().train(ctx)

    assert len(calls) == 2
    assert calls[-1][0] == pytest.approx(1.0)
    assert calls[-1][1] == "training chemprop-dmpnn"
```

- [ ] **Step 2: Install chemprop locally and run the test to verify it fails**

Run:
```bash
uv sync --extra gpu
uv run pytest tests/unit/engines/test_chemprop_dmpnn.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'daikonstudio.infrastructure.engines.chemprop_dmpnn'`.

If `uv sync --extra gpu` cannot resolve torch for this machine, the tests are skipped by
`importorskip` and this task's verification must happen in the GPU image built in Task 6.
Say so explicitly rather than marking the task done on a skipped suite.

- [ ] **Step 3: Write the engine**

Create `backend/src/daikonstudio/infrastructure/engines/chemprop_dmpnn.py`:

```python
"""Chemprop D-MPNN: a directed message-passing neural network over the molecular graph.

The first engine that does not run wherever the API runs. Its manifest declares
`lane="gpu"`, and a deployment satisfies that by running a worker with
STUDIO_WORKER_LANE=gpu on a machine that has a GPU. Nothing here names a host.

**Every chemprop, torch and lightning import lives inside a method, never at module
scope.** That is what lets the API tier and the default-lane worker install without CUDA
and still serve this manifest, validate its conditions and render it in the engine
picker. `default_registry()` instantiates this class at import time, so the constructor
must stay import-free too.

Reproducibility here is honest, not exact. `seed_everything` fixes the Python, NumPy and
torch seeds, but GPU kernels are not bit-deterministic. The optimism-gap comparison in
`train_protocol.py` assumes the split strategy is the only variable between its two
numbers; for this engine there is also kernel-level noise. The comparison stays
directionally valid, and this paragraph is why nobody should read it as exact.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import polars as pl

from daikonstudio.application.engines.context import PredictContext, TrainContext, TrainResult
from daikonstudio.application.engines.manifest import (
    ConditionSpec,
    ConditionType,
    EngineManifest,
    TaskType,
    validate_conditions,
)
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.engines._scoring import (
    classification_metrics,
    regression_metrics,
)

_MANIFEST = EngineManifest(
    id="chemprop-dmpnn",
    version="1.0.0",
    name="Chemprop D-MPNN",
    description=(
        "A directed message-passing neural network that learns its own representation "
        "from the molecular graph instead of using a fixed fingerprint. Trains on a "
        "GPU and takes minutes to hours depending on dataset size."
    ),
    tasks=(TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION),
    conditions=(
        ConditionSpec(
            key="epochs",
            label="Training epochs",
            type=ConditionType.INTEGER,
            default=50,
            minimum=1,
            maximum=500,
            help="How many passes over the training set. More epochs fit the training "
            "data more closely, at growing risk of memorising it. 50 is a good default.",
        ),
        ConditionSpec(
            key="depth",
            label="Message passing steps",
            type=ConditionType.INTEGER,
            default=3,
            minimum=2,
            maximum=6,
            help="How far information travels across the molecule. Each step lets an "
            "atom see one bond further. 3 covers most local chemistry.",
        ),
        ConditionSpec(
            key="message_hidden_dim",
            label="Hidden size",
            type=ConditionType.INTEGER,
            default=300,
            minimum=64,
            maximum=2400,
            help="How much the network can represent about each atom. Larger needs "
            "more data to be worth it.",
        ),
        ConditionSpec(
            key="batch_size",
            label="Batch size",
            type=ConditionType.INTEGER,
            default=64,
            minimum=8,
            maximum=512,
            help="How many molecules are scored before the weights update. Lower it "
            "if training runs out of GPU memory.",
        ),
    ),
    lane="gpu",
)

# Prediction is a forward pass with no gradients, so this only trades memory against
# kernel-launch overhead. It is not the training `batch_size` condition and does not
# change any result.
_PREDICT_BATCH_SIZE = 64


def _require_chemprop() -> None:
    """Fail with a cause a human can act on, not a bare ModuleNotFoundError.

    The worker records `repr(exc)` on the Run, so this message is exactly what a
    scientist sees when their training run failed -- which makes "you are running this
    engine on a worker that cannot" the single most valuable thing it can say.
    """
    try:
        import chemprop  # noqa: F401
    except ImportError as exc:
        raise ValidationError(
            "The chemprop-dmpnn engine needs the 'gpu' extra, which this worker does "
            "not have installed. This engine declares lane 'gpu': run it on a worker "
            "started with STUDIO_WORKER_LANE=gpu (see Dockerfile.gpu), or install the "
            "extra locally with `uv sync --extra gpu`."
        ) from exc


def _datapoints(
    structures: list[str], targets: list[float] | None = None
) -> list[Any]:
    """chemprop wants one datapoint per molecule, with `y` a 1-D array of length
    n_tasks. A scalar y silently breaks both `MoleculeDataset.t` and the target
    scaler, so the `(1,)` shape here is load-bearing rather than stylistic."""
    import numpy as np
    from chemprop.data import MoleculeDatapoint

    if targets is None:
        return [MoleculeDatapoint.from_smi(smiles) for smiles in structures]
    return [
        MoleculeDatapoint.from_smi(smiles, y=np.array([float(target)]))
        for smiles, target in zip(structures, targets, strict=True)
    ]


class ChempropDMPNN:
    @staticmethod
    def manifest() -> EngineManifest:
        return _MANIFEST

    def train(self, ctx: TrainContext) -> TrainResult:
        _require_chemprop()

        import torch
        from chemprop.data import MoleculeDataset, build_dataloader
        from chemprop.models import MPNN
        from chemprop.nn import (
            BinaryClassificationFFN,
            BondMessagePassing,
            MeanAggregation,
            RegressionFFN,
        )
        from chemprop.nn.transforms import UnscaleTransform
        from lightning import pytorch as lightning
        from lightning.pytorch.callbacks import LambdaCallback

        conditions = validate_conditions(_MANIFEST, ctx.conditions)
        epochs = int(conditions["epochs"])
        hidden = int(conditions["message_hidden_dim"])
        depth = int(conditions["depth"])
        batch_size = int(conditions["batch_size"])
        is_classification = ctx.task is TaskType.BINARY_CLASSIFICATION

        lightning.seed_everything(ctx.seed, workers=True)

        train_rows = ctx.frame.filter(pl.col("split") == "train")
        validation_rows = ctx.frame.filter(pl.col("split") == "validation")
        test_rows = ctx.frame.filter(pl.col("split") == "test")

        train_set = MoleculeDataset(
            _datapoints(
                train_rows[ctx.structure_column].to_list(),
                train_rows[ctx.target_column].to_list(),
            )
        )
        validation_set = MoleculeDataset(
            _datapoints(
                validation_rows[ctx.structure_column].to_list(),
                validation_rows[ctx.target_column].to_list(),
            )
        )
        test_set = MoleculeDataset(
            _datapoints(test_rows[ctx.structure_column].to_list())
        )

        output_transform = None
        if not is_classification:
            # Fit the scaler on the training split only, and hand the model its
            # inverse. UnscaleTransform is a no-op in train mode by design, so the loss
            # is computed in scaled space while predictions come back in the target's
            # own unit -- which is what lets the Scorecard compare them against
            # `actual` without rescaling anything itself.
            scaler = train_set.normalize_targets()
            if len(validation_set) > 0:
                validation_set.normalize_targets(scaler)
            output_transform = UnscaleTransform.from_standard_scaler(scaler)
        # The test split is deliberately never normalised: `output_transform` is what
        # puts predictions back into real units, and scaling the truth as well would
        # cancel out silently.

        predictor = (
            BinaryClassificationFFN(input_dim=hidden)
            if is_classification
            # input_dim must equal the message-passing d_h, or the FFN's first layer
            # is built for the wrong width.
            else RegressionFFN(input_dim=hidden, output_transform=output_transform)
        )
        model = MPNN(
            message_passing=BondMessagePassing(d_h=hidden, depth=depth),
            agg=MeanAggregation(),
            predictor=predictor,
            batch_norm=True,
        )

        def _report_epoch(trainer: Any, _module: Any) -> None:
            ctx.report((trainer.current_epoch + 1) / epochs, f"training {_MANIFEST.id}")

        trainer = lightning.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=epochs,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            # The interruption point. `report` may raise RunInterrupted, which
            # propagates out of `fit` and out of `train` -- the only way to stop work
            # already running on the worker thread.
            callbacks=[LambdaCallback(on_train_epoch_end=_report_epoch)],
        )
        trainer.fit(
            model,
            build_dataloader(train_set, batch_size=batch_size, seed=ctx.seed),
            build_dataloader(validation_set, batch_size=batch_size, shuffle=False)
            if len(validation_set) > 0
            else None,
        )

        predicted = _forward(trainer, model, test_set, torch, build_dataloader)
        actual = test_rows[ctx.target_column].to_numpy()
        if is_classification:
            metrics = classification_metrics(
                actual,
                (predicted >= 0.5).astype(float),
                predicted,
                train_has_both_classes=train_rows[ctx.target_column].n_unique() >= 2,
            )
        else:
            metrics = regression_metrics(actual, predicted)

        # Lightning writes checkpoints to a path, so this round-trips through the
        # filesystem. `tempfile` honours TMPDIR, which is how a deployment points
        # scratch at a fast local NVMe without a setting of our own.
        with tempfile.TemporaryDirectory() as scratch:
            checkpoint = Path(scratch) / "model.ckpt"
            trainer.save_checkpoint(checkpoint)
            artifact = checkpoint.read_bytes()

        return TrainResult(artifact=artifact, metrics=metrics)

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        _require_chemprop()

        import torch
        from chemprop.data import MoleculeDataset, build_dataloader
        from chemprop.models import MPNN
        from lightning import pytorch as lightning

        with tempfile.TemporaryDirectory() as scratch:
            checkpoint = Path(scratch) / "model.ckpt"
            checkpoint.write_bytes(ctx.artifact)
            # Loaded inside the block, used outside it: the weights are in memory by
            # the time the directory is removed. MPNN.save_hyperparameters() is what
            # makes the architecture recoverable from the checkpoint alone.
            model = MPNN.load_from_checkpoint(checkpoint)

        dataset = MoleculeDataset(
            _datapoints(ctx.frame[ctx.structure_column].to_list())
        )
        trainer = lightning.Trainer(
            accelerator="auto",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )
        values = _forward(trainer, model, dataset, torch, build_dataloader)
        row_ids = list(range(len(values)))

        # Explicit dtypes, matching `_predict_with_tree_ensemble`. An all-None
        # uncertainty list would otherwise infer as polars' Null dtype and make this
        # engine's output schema-incompatible with the ECFP4 engines' for any caller
        # that concatenates or persists results across engines.
        #
        # ponytail: uncertainty is always null. chemprop can produce it through an MVE
        # head or an ensemble; a fabricated number would be plotted by a triage grid as
        # "the model is confident here", which is worse than an admitted absent one.
        # Upgrade path: an `uncertainty` condition selecting MveFFN for regression.
        return pl.DataFrame(
            {
                "row_id": pl.Series(row_ids, dtype=pl.Int64),
                "value": pl.Series([float(value) for value in values], dtype=pl.Float64),
                "uncertainty": pl.Series([None] * len(row_ids), dtype=pl.Float64),
            }
        )


def _forward(trainer: Any, model: Any, dataset: Any, torch: Any, build_dataloader: Any) -> Any:
    """Run the model over a dataset and flatten to one value per molecule.

    `trainer.predict` is what puts the model in eval mode, which is what activates
    `UnscaleTransform` -- calling `model(...)` directly would silently return scaled
    values. Each batch comes back shaped (batch, 1) because n_tasks and n_targets are
    both 1; for binary classification these are already sigmoid probabilities, so
    nothing further is applied to them here.
    """
    batches = trainer.predict(
        model, build_dataloader(dataset, batch_size=_PREDICT_BATCH_SIZE, shuffle=False)
    )
    return torch.cat(batches).cpu().numpy().reshape(-1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/engines/test_chemprop_dmpnn.py -v`
Expected: PASS (or SKIPPED if chemprop could not be installed on this machine — in which case defer verification to Task 6 and say so).

- [ ] **Step 5: Register the engine**

In `backend/src/daikonstudio/infrastructure/engines/registry.py`, extend the docstring's first paragraph and the tuple:

```python
from daikonstudio.infrastructure.engines.chemprop_dmpnn import ChempropDMPNN

_ENGINES: tuple[Engine, ...] = (Ecfp4RandomForest(), Ecfp4XGBoost(), ChempropDMPNN())
```

Add to the module docstring:

```
Every engine is registered in every deployment, including ones with no GPU. That is
deliberate: the manifest is plain data and costs nothing to serve, so the engine picker
stays identical everywhere, and a deployment that cannot run one finds out through a Run
that sits PENDING with no worker on its lane -- which is visible and diagnosable -- rather
than through an engine that mysteriously does not appear.
```

- [ ] **Step 6: Verify the manifest tripwire and the whole suite**

Run: `uv run pytest -q && uv run lint-imports && uv run ruff check src tests && uv run mypy src`
Expected: all pass. `test_engine_contract.py`'s JSON round-trip now covers the chemprop manifest; if it fails, fix the manifest field, never the test.

- [ ] **Step 7: Regenerate the API snapshot**

The engines endpoint response is unchanged (`lane` is deliberately not exposed), but the
new engine appears in `GET /api/v1/engines`. From the repo root:

```bash
make generate-api
```

Expected: `frontend/openapi.json` unchanged in shape; commit it only if it actually differs.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: chemprop D-MPNN engine on the gpu lane"
```

---

## Task 6: Container images

**Files:**
- Create: `backend/Dockerfile`
- Create: `backend/Dockerfile.gpu`
- Create: `backend/.dockerignore`

**Interfaces:**
- Consumes: the `s3` and `gpu` extras from Task 1; `WorkerSettings` from Task 2; the chemprop engine from Task 5.
- Produces: two images. The slim one runs the API (default command) or the default-lane worker (command override). The GPU one runs a `gpu`-lane worker.

- [ ] **Step 1: Write the dockerignore**

Create `backend/.dockerignore`:

```
.venv
.env
.pytest_cache
.mypy_cache
.ruff_cache
.import_linter_cache
**/__pycache__
tests
```

- [ ] **Step 2: Write the slim image**

Create `backend/Dockerfile`:

```dockerfile
# API and default-lane worker. No CUDA, no torch -- the chemprop engine's imports are
# all inside its methods, so this image serves its manifest and renders it in the engine
# picker without being able to run it.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Dependencies before source, so a code change does not re-resolve the world.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra s3 --no-install-project

COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
RUN uv sync --frozen --no-dev --extra s3

ENV PATH="/app/.venv/bin:$PATH"
RUN useradd --create-home --uid 10001 studio && chown -R studio /app
USER studio

# Overridden to `arq daikonstudio.infrastructure.worker.WorkerSettings` for a worker.
CMD ["uvicorn", "daikonstudio.interface.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Write the GPU image**

Create `backend/Dockerfile.gpu`:

```dockerfile
# GPU-lane worker.
#
# pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime is the obvious base and is ruled out: it
# ships Python 3.11, and this codebase uses PEP 695 generics (`class PageResult[T]` in
# domain/shared/pagination.py) which need 3.12+. Lowering requires-python to fit a base
# image would be the tail wagging the dog; uv installs its own interpreter, so the plain
# CUDA runtime is the cleaner floor.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install 3.13

WORKDIR /app

# The CUDA build of torch is selected HERE, not in pyproject's [tool.uv.sources]:
# pinning a cu124 index in the project file would break `uv sync --extra gpu` on a
# developer's Mac, and running the chemprop tests locally on CPU torch is exactly what
# makes this engine testable without a GPU.
ENV UV_INDEX="pytorch-cu124=https://download.pytorch.org/whl/cu124" \
    UV_INDEX_STRATEGY=unsafe-best-match

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra gpu --extra s3 --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --extra gpu --extra s3

# TMPDIR is where chemprop's checkpoint round-trip lands. Bind a fast local NVMe at
# /scratch on a node that has one; everything written there is transient.
ENV PATH="/app/.venv/bin:$PATH" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TMPDIR=/scratch \
    STUDIO_WORKER_LANE=gpu \
    STUDIO_WORKER_MAX_JOBS=1

RUN mkdir -p /scratch \
 && useradd --create-home --uid 10001 studio \
 && chown -R studio /app /scratch
USER studio

CMD ["arq", "daikonstudio.infrastructure.worker.WorkerSettings"]
```

- [ ] **Step 4: Build both images**

Run, from `backend/`:
```bash
docker build -t daikon-studio:slim -f Dockerfile .
docker build -t daikon-studio:gpu -f Dockerfile.gpu .
```
Expected: both build. The GPU image is large (CUDA + torch, roughly 7 GB) and slow on a
cold cache — that is expected, not a fault.

- [ ] **Step 5: Verify the slim image genuinely has no torch**

This is the property the lazy imports exist to buy. Run:
```bash
docker run --rm daikon-studio:slim python -c "
import daikonstudio.infrastructure.engines.registry as r
ids = [e.manifest().id for e in r._ENGINES]
assert 'chemprop-dmpnn' in ids, ids
import importlib.util
assert importlib.util.find_spec('torch') is None, 'torch leaked into the slim image'
print('ok:', ids)
"
```
Expected: `ok: ['ecfp4-randomforest', 'ecfp4-xgboost', 'chemprop-dmpnn']`

- [ ] **Step 6: Verify the GPU image can run the engine**

Run:
```bash
docker run --rm daikon-studio:gpu python -c "
from daikonstudio.infrastructure.engines.chemprop_dmpnn import ChempropDMPNN
import chemprop, torch
print('chemprop', chemprop.__version__, 'torch', torch.__version__)
print('lane', ChempropDMPNN.manifest().lane)
"
```
Expected: prints versions and `lane gpu`. On a machine with an NVIDIA GPU and the
container toolkit, add `--gpus all` and check `torch.cuda.is_available()` is `True`.

If the chemprop tests were skipped in Task 5, run them here now:
```bash
docker run --rm -v "$PWD/tests:/app/tests" daikon-studio:gpu \
  python -m pytest tests/unit/engines/test_chemprop_dmpnn.py -v
```

- [ ] **Step 7: Commit**

```bash
git add Dockerfile Dockerfile.gpu .dockerignore
git commit -m "chore: container images for the api and both worker lanes"
```

---

## Deployment notes (not code — for whoever writes the swarm stack)

The stack file lands in `snet2-infrastructure/docker-swarm-core/1XX-daikon-studio/`,
alongside every other daikon service's deploy configuration. What it must provide:

- **api** — slim image, default command.
- **worker** — slim image, command `arq daikonstudio.infrastructure.worker.WorkerSettings`,
  no lane override (default lane).
- **worker-gpu** — GPU image, `deploy.placement.constraints: [node.labels.gpu == true]`,
  `NVIDIA_VISIBLE_DEVICES=all`, `STUDIO_WORKER_MAX_JOBS=1`,
  `STUDIO_WORKER_JOB_TIMEOUT` raised to whatever a real run needs.
- **All three** get `STUDIO_DATABASE_URL`, `STUDIO_REDIS_URL`, `STUDIO_BLOB_BASE_URL`
  and `STUDIO_BLOB_STORAGE_OPTIONS`. No volumes on any of them.

The existing MinIO at `https://minio-api.snet.biobio.tamu.edu`
(`docker-swarm-core/116-minio/minio.yml`) is the blob backend; create a `daikon-studio`
bucket and set:

```
STUDIO_BLOB_BASE_URL=s3://daikon-studio/blobs
STUDIO_BLOB_STORAGE_OPTIONS={"endpoint_url":"https://minio-api.snet.biobio.tamu.edu","key":"...","secret":"..."}
```

Several GPUs on one node: one `worker-gpu` replica per device with `CUDA_VISIBLE_DEVICES`
pinned. Several GPU nodes: raise the replica count. Both pull the same queue.

---

## Self-Review

**Spec coverage.** Thesis (four env vars, no volumes) → Tasks 1, 2, 6. Lanes → Task 2.
Cooperative progress and cancellation, including the `run_job` re-read → Task 3. Chemprop
engine → Task 5. Shared metrics → Task 4. Blob storage and extras → Task 1. Container
images and the ruled-out PyTorch base → Task 6. Testing section → distributed across each
task's test steps. The spec's `Settings.scratch_dir` was amended to `TMPDIR` before this
plan was written and Task 5 implements the amended form.

**Deliberately not implemented, per the spec's deferred list:** fit-result caching,
artifact-as-URI, chemprop uncertainty, per-engine optimism-gap opt-out, Temporal.

**Type consistency.** `DEFAULT_LANE` (Task 2) is used by Tasks 2 and 5. `queue_for` is
defined once in Task 2 and asserted in Task 2's tests only. `RunInterrupted(reason, *,
cancelled)` is defined in Task 3 and raised in Tasks 3 and 5 — always with the keyword.
`regression_metrics` / `classification_metrics` signatures in Task 4 match their call
sites in Task 5 exactly, including the keyword-only `train_has_both_classes`.
`FsspecBlobStore(base_url, storage_options)` in Task 1 matches both composition roots.

**Two places the plan asks the implementer to read existing code rather than giving them
a literal snippet**, both because inventing a second convention alongside the repo's own
would be worse than the small ambiguity: Task 2 Step 8's `training_setup` fixture (build
the `Dataset` and `AuthContext` the way `tests/integration/test_train_protocol.py` and
`tests/fakes/auth.py` already do) and Task 3 Step 5's worker-test helpers (follow
`tests/unit/execution/test_worker.py`'s existing ctx/handler-swap pattern). In both cases
the assertions — which are the contract — are written out in full.
