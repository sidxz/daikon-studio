# Resumable Training Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A training run stopped by its time limit, a runner crash or a cancel resumes from saved progress instead of starting over.

**Architecture:**
- A `Checkpoints` store (application layer, over the existing `BlobStore` port) holds a run's saved progress under the dataset folder.
- `RunTraining` saves each completed fit (model, baseline, random-split) and `FanOut` saves each per-target fit; both skip saved fits on the next attempt.
- chemprop and MoLFormer save Lightning training state every N minutes and at a time-limit stop, and resume with `ckpt_path`.
- Retry resumes by default; `fresh` (Start over) clears saved progress first.

**Tech Stack:** Python 3.13, Lightning 2.6.5, torch, chemprop 2.3, transformers, FastAPI, SQLAlchemy, httpx; Next.js, React Query, Vitest.

**Spec:** `docs/superpowers/specs/2026-10-03-resumable-training-design.md`. Read it in full; this plan argues from it.

## Global Constraints

- **Branch:** `resumable-runs` (cut from `training-options`). Do not switch branches.
- **Checkpoint root:** exactly `{workspace_id}/datasets/{dataset_id}/runs/{run_id}/checkpoints/`. It sits under the dataset folder so `DeleteDataset`'s `delete_prefix(dataset_folder)` removes it.
- **Defaults:**
  - in-progress save interval `600` seconds, from the runner env `STUDIO_CHECKPOINT_INTERVAL_SECONDS`;
  - `RetryRunCommand.fresh=False`.
- **Saving is best effort.** A failed save or load is logged and never fails or stops a run.
- **No save on cancel** (`RunInterrupted.cancelled=True`): the runner API refuses writes to a cancelled run.
- **Lightning facts verified on 2.6.5** (the controller probed both):
  - a checkpoint saved in `on_train_epoch_end`, or in `on_exception` after an exception raised from `on_train_epoch_end`, resumes via `trainer.fit(..., ckpt_path=...)` at the NEXT epoch;
  - a Callback's `state_dict`/`load_state_dict` round-trips through it.
- **Layers** (`lint-imports`, 3 contracts):
  - interface → infrastructure → application → domain.
  - `application/engines/checkpoints.py` imports only stdlib, `application.ports.blob_store` and `application.engines.context`.
  - `context.py` refers to `Checkpoints` only under `TYPE_CHECKING`, which avoids an import cycle.
- **Import discipline:** torch, lightning, chemprop and transformers are imported only inside functions in `infrastructure/engines/*`. API workers have none of them.
- **Copy:** American spelling, academic and plain, no em-dash clause chains. Use the strings given here.
- **Tests per task:** run only the test files you touch, plus these gates:
  ```
  uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src && uv run lint-imports
  ```
  Use `OMP_NUM_THREADS=1`. Run torch-loading tests (chemprop, molformer) in their own pytest invocation, never with `test_ecfp4_lightgbm.py` (GitHub issue #1). **No full suites.**
- **Do NOT restart dev processes** (`make dev*`). A live CheMeleon run is training on the GPU runner.
- **Commits:**
  - conventional subject;
  - end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`;
  - **never** a `Claude-Session:` trailer;
  - stage explicit paths;
  - no `git stash`.

## Review Focus

1. **A restored fit must be indistinguishable from a fresh one.** That includes NaN metrics, `validation_metrics=None` and `cutoffs=None`. Pinned in T1: a pack/unpack round-trip with NaN, None and empty values.
2. **A target column named `a/b` or `..`** must not become a blob path segment. Pinned in T1 (`scoped` rejects unsafe names) and T2 (FanOut scopes by target index).
3. **A save that fails mid-training must not kill the run.** Examples: a 413 from the upload cap, or a network error. Pinned in T1 (`save` swallows store errors) and T4 (the Lightning callback over a store whose `put_bytes` raises).
4. **An interrupted save must not destroy the previous good save.** Pinned in T1: content-addressed blobs plus a marker written last. A save whose marker write fails leaves the old save loadable.
5. **Start over must leave nothing for the requeued attempt to load.** Pinned in T3: after `fresh` retry, no key remains under the root, and the clear happens before the enqueue.

---

### Task 1: The checkpoint store

**Files:**
- Create: `backend/src/daikonstudio/application/engines/checkpoints.py`
- Modify: `backend/src/daikonstudio/application/engines/context.py` (`TrainContext.checkpoints`)
- Create: `backend/tests/fakes/blob_store.py`
- Test: `backend/tests/unit/engines/test_checkpoints.py` (create)

**Interfaces produced:**
- `checkpoint_root(workspace_id: uuid.UUID, dataset_id: uuid.UUID, run_id: uuid.UUID) -> str`
- `DEFAULT_INTERVAL_SECONDS: float = 600.0`
- `RESULT_FORMAT: str = "1"`
- `class Checkpoints(store: BlobStore, root: str, *, interval_seconds: float = DEFAULT_INTERVAL_SECONDS, fingerprint: dict[str, str] | None = None)`, with:
  - `.root: str`;
  - `.interval_seconds: float`;
  - `.scoped(name: str, **fingerprint: str) -> Checkpoints`;
  - `.save(name: str, data: bytes) -> None`;
  - `.load(name: str) -> bytes | None`;
  - `.clear() -> None`.
- `pack_result(result: TrainResult) -> bytes`; `unpack_result(data: bytes) -> TrainResult`
- `TrainContext.checkpoints: Checkpoints | None = None`
- `tests.fakes.blob_store.InMemoryBlobStore`, a full `BlobStore`: missing keys raise `FileNotFoundError`, and `delete_prefix` is supported.

- [ ] **Step 1: The fake**

Create `backend/tests/fakes/blob_store.py`:

```python
"""An in-memory `BlobStore` with the real stores' missing-key signal (FileNotFoundError)
and `delete_prefix`, shared by the checkpoint tests."""

from __future__ import annotations


class InMemoryBlobStore:
    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def put_bytes(self, key: str, data: bytes) -> str:
        self.blobs[key] = data
        return f"memory://{key}"

    def get_bytes(self, key: str) -> bytes:
        try:
            return self.blobs[key]
        except KeyError:
            raise FileNotFoundError(key) from None

    def exists(self, key: str) -> bool:
        return key in self.blobs

    def delete(self, key: str) -> None:
        self.blobs.pop(key, None)

    def delete_prefix(self, prefix: str) -> None:
        for key in [key for key in self.blobs if key.startswith(prefix)]:
            del self.blobs[key]
```

- [ ] **Step 2: Failing tests**

Create `backend/tests/unit/engines/test_checkpoints.py`:

```python
import math
import uuid

import pytest

from daikonstudio.application.data.delete_dataset import dataset_folder
from daikonstudio.application.engines.checkpoints import (
    Checkpoints,
    checkpoint_root,
    pack_result,
    unpack_result,
)
from daikonstudio.application.engines.context import TrainResult
from tests.fakes.blob_store import InMemoryBlobStore

ROOT = "ws/datasets/d/runs/r/checkpoints/"


def test_the_root_sits_under_the_dataset_folder_so_deleting_the_dataset_deletes_it():
    ws, dataset, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    root = checkpoint_root(ws, dataset, run)
    assert root.startswith(dataset_folder(ws, dataset))
    assert root.endswith(f"runs/{run}/checkpoints/")


def test_save_then_load_round_trips_in_a_scope():
    store = InMemoryBlobStore()
    stage = Checkpoints(store, ROOT).scoped("model", engine="x", engine_version="1")
    stage.save("result", b"payload")
    assert stage.load("result") == b"payload"
    assert all(key.startswith(ROOT + "model/") for key in store.blobs)


def test_a_different_fingerprint_reads_as_nothing_saved():
    store = InMemoryBlobStore()
    Checkpoints(store, ROOT).scoped("model", engine_version="1").save("result", b"old")
    assert Checkpoints(store, ROOT).scoped("model", engine_version="2").load("result") is None


def test_a_corrupted_blob_reads_as_nothing_saved():
    store = InMemoryBlobStore()
    checkpoints = Checkpoints(store, ROOT)
    checkpoints.save("state", b"0123456789")
    (blob,) = [key for key in store.blobs if not key.endswith(".json")]
    store.blobs[blob] = b"01234"  # truncated mid-write
    assert checkpoints.load("state") is None


def test_a_save_whose_marker_write_fails_keeps_the_previous_save_loadable():
    store = InMemoryBlobStore()
    checkpoints = Checkpoints(store, ROOT)
    checkpoints.save("state", b"first")
    real_put = store.put_bytes

    def fail_on_marker(key: str, data: bytes) -> str:
        if key.endswith(".json"):
            raise OSError("disk full")
        return real_put(key, data)

    store.put_bytes = fail_on_marker  # type: ignore[method-assign]
    checkpoints.save("state", b"second")  # must not raise
    store.put_bytes = real_put  # type: ignore[method-assign]
    assert checkpoints.load("state") == b"first"


def test_a_failing_store_never_raises_from_save_or_load():
    class Broken(InMemoryBlobStore):
        def put_bytes(self, key: str, data: bytes) -> str:
            raise OSError("413: upload exceeds runner_upload_max_bytes")

        def get_bytes(self, key: str) -> bytes:
            raise OSError("connection reset")

    checkpoints = Checkpoints(Broken(), ROOT)
    checkpoints.save("state", b"x")
    assert checkpoints.load("state") is None


def test_saving_again_replaces_the_previous_blob():
    store = InMemoryBlobStore()
    checkpoints = Checkpoints(store, ROOT)
    checkpoints.save("state", b"first")
    checkpoints.save("state", b"second")
    assert checkpoints.load("state") == b"second"
    assert len([key for key in store.blobs if not key.endswith(".json")]) == 1


@pytest.mark.parametrize("name", ["a/b", "..", ".", "", "x y"])
def test_unsafe_names_are_refused(name):
    with pytest.raises(ValueError):
        Checkpoints(InMemoryBlobStore(), ROOT).scoped(name)


def test_clear_deletes_everything_under_the_root_only():
    store = InMemoryBlobStore()
    store.put_bytes("ws/datasets/d/snapshot.parquet", b"keep")
    checkpoints = Checkpoints(store, ROOT)
    checkpoints.scoped("model").save("result", b"x")
    checkpoints.clear()
    assert list(store.blobs) == ["ws/datasets/d/snapshot.parquet"]


def test_a_train_result_round_trips_exactly():
    result = TrainResult(
        artifact=b"\x00model\xff",
        metrics={"y": {"mcc": 0.5, "auroc": float("nan")}},
        validation_metrics=None,
        cutoffs={"y": 0.31},
    )
    restored = unpack_result(pack_result(result))
    assert restored.artifact == result.artifact
    assert restored.metrics["y"]["mcc"] == 0.5 and math.isnan(restored.metrics["y"]["auroc"])
    assert restored.validation_metrics is None
    assert restored.cutoffs == {"y": 0.31}
    assert unpack_result(pack_result(TrainResult(artifact=b"", metrics={}))).cutoffs is None
```

- [ ] **Step 3: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_checkpoints.py -q
```

Expected: FAIL with ImportError.

- [ ] **Step 4: Implement**

Create `backend/src/daikonstudio/application/engines/checkpoints.py`:

```python
"""A training run's saved progress: what a stopped run resumes from.

Spec: `docs/superpowers/specs/2026-10-03-resumable-training-design.md`. Each completed
fit is saved as a packed `TrainResult`; chemprop and MoLFormer also save their Lightning
training state while a fit runs. All of it lives under one root per run, so a successful
run, a Start over and a deleted dataset can each remove it in one call.

Saving is best effort: every failure here is logged and swallowed. A save that fails
costs a resume some progress; a save that raised would cost the run itself.

Integrity: each save writes its data to a content-addressed blob, then a small JSON
marker naming that blob, its length, its sha256 and the fingerprint it was saved under.
The marker is written last, so a save interrupted before it leaves the previous save
loadable; a marker whose fingerprint or checksum disagrees reads as nothing saved.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import struct
import uuid
from typing import Any

from daikonstudio.application.engines.context import TrainResult
from daikonstudio.application.ports.blob_store import BlobStore

logger = logging.getLogger(__name__)

#: How often an in-progress neural fit saves its training state. A runner overrides it
#: with STUDIO_CHECKPOINT_INTERVAL_SECONDS.
DEFAULT_INTERVAL_SECONDS = 600.0

#: The packed-`TrainResult` layout. Part of every fit's fingerprint: bump it when
#: `pack_result` changes and older saves read as absent instead of misreading.
RESULT_FORMAT = "1"

_SAFE_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")


def checkpoint_root(workspace_id: uuid.UUID, dataset_id: uuid.UUID, run_id: uuid.UUID) -> str:
    """Under the dataset's folder on purpose: `DeleteDataset` removes that folder with
    `delete_prefix`, which takes its runs' saved progress with it."""
    return f"{workspace_id}/datasets/{dataset_id}/runs/{run_id}/checkpoints/"


def _require_safe(name: str) -> None:
    # Names become blob key segments; the runner API refuses `..` and the like, and a
    # user-chosen target column could contain anything. Scopes are code-chosen names.
    if not _SAFE_NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError(f"Not a safe checkpoint name: {name!r}")


class Checkpoints:
    def __init__(
        self,
        store: BlobStore,
        root: str,
        *,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        fingerprint: dict[str, str] | None = None,
    ) -> None:
        if not root.endswith("/"):
            raise ValueError(f"A checkpoint root is a folder key ending in '/', got {root!r}")
        self._store = store
        self.root = root
        self.interval_seconds = interval_seconds
        self._fingerprint = dict(fingerprint or {})

    def scoped(self, name: str, **fingerprint: str) -> Checkpoints:
        """A sub-store under `{root}{name}/` whose saves also record `fingerprint`."""
        _require_safe(name)
        return Checkpoints(
            self._store,
            f"{self.root}{name}/",
            interval_seconds=self.interval_seconds,
            fingerprint={**self._fingerprint, **fingerprint},
        )

    def save(self, name: str, data: bytes) -> None:
        _require_safe(name)
        digest = hashlib.sha256(data).hexdigest()
        blob = f"{name}.{digest[:16]}"
        marker_key = f"{self.root}{name}.json"
        try:
            previous = self._marker(marker_key)
            self._store.put_bytes(self.root + blob, data)
            self._store.put_bytes(
                marker_key,
                json.dumps(
                    {
                        "fingerprint": self._fingerprint,
                        "blob": blob,
                        "length": len(data),
                        "sha256": digest,
                    },
                    sort_keys=True,
                ).encode(),
            )
        except Exception:
            logger.warning(
                "Could not save %s%s; a resume falls back to the previous save.",
                self.root,
                name,
                exc_info=True,
            )
            return
        if previous is not None and previous.get("blob") not in {None, blob}:
            try:
                self._store.delete(self.root + str(previous["blob"]))
            except Exception:
                logger.info("Left a superseded checkpoint blob in %s", self.root, exc_info=True)

    def load(self, name: str) -> bytes | None:
        _require_safe(name)
        try:
            marker = self._marker(f"{self.root}{name}.json")
            if marker is None or marker.get("fingerprint") != self._fingerprint:
                return None
            data = self._store.get_bytes(self.root + str(marker["blob"]))
        except Exception:
            logger.warning("Could not read %s%s; treating it as not saved.", self.root, name, exc_info=True)
            return None
        if len(data) != marker.get("length") or hashlib.sha256(data).hexdigest() != marker.get("sha256"):
            logger.warning("%s%s failed its integrity check; treating it as not saved.", self.root, name)
            return None
        return data

    def clear(self) -> None:
        try:
            self._store.delete_prefix(self.root)
        except Exception:
            logger.warning("Could not delete saved progress under %s", self.root, exc_info=True)

    def _marker(self, key: str) -> dict[str, Any] | None:
        try:
            raw = self._store.get_bytes(key)
        except FileNotFoundError:
            return None
        value = json.loads(raw)
        return value if isinstance(value, dict) else None


def pack_result(result: TrainResult) -> bytes:
    """A completed fit as one blob: a length-prefixed JSON header, then the artifact."""
    header = json.dumps(
        {
            "metrics": result.metrics,
            "validation_metrics": result.validation_metrics,
            "cutoffs": result.cutoffs,
        }
    ).encode()
    return struct.pack(">I", len(header)) + header + result.artifact


def unpack_result(data: bytes) -> TrainResult:
    (length,) = struct.unpack(">I", data[:4])
    header = json.loads(data[4 : 4 + length])
    return TrainResult(
        artifact=data[4 + length :],
        metrics=header["metrics"],
        validation_metrics=header["validation_metrics"],
        cutoffs=header["cutoffs"],
    )
```

In `context.py`:
- add `from typing import TYPE_CHECKING`, plus `if TYPE_CHECKING: from daikonstudio.application.engines.checkpoints import Checkpoints`;
- `TrainContext` gains, after `tune_cutoffs`:

```python
    # Where this fit's saved progress lives, already scoped to it; `None` when nothing
    # is saved (tests, and any caller without a run). An engine that trains for long
    # saves its in-progress state here and resumes from it -- see `_lightning.py`.
    checkpoints: Checkpoints | None = None
```

Check that `context.py` already has `from __future__ import annotations`, so the annotation stays a string. If it doesn't, add it.

- [ ] **Step 5: Run tests and gates, then commit**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_checkpoints.py tests/unit/engines/test_fan_out.py -q && uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src && uv run lint-imports
```

Expected: PASS.

```bash
git add backend/src/daikonstudio/application/engines/checkpoints.py backend/src/daikonstudio/application/engines/context.py backend/tests/fakes/blob_store.py backend/tests/unit/engines/test_checkpoints.py
git commit -m "feat(training): a checkpoint store for a run's saved progress

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Completed fits are saved and skipped

**Files:**
- Modify:
  - `backend/src/daikonstudio/application/execution/train_protocol.py`: `RunTraining`. Covers `__init__`, `__call__`, `_fit`, `_train_off_thread`, `_optimism_gap`, `_check_deadline`, and the success cleanup.
  - `backend/src/daikonstudio/application/engines/fan_out.py`: `FanOut.train`.
  - `backend/src/daikonstudio/infrastructure/jobs.py`: `_train` passes the interval.
  - `backend/src/daikonstudio/infrastructure/runner/agent.py`: `AgentSettings.checkpoint_interval_seconds` and the `build_http_ctx` call.
  - `backend/src/daikonstudio/infrastructure/runner/ports.py`: `build_http_ctx` and `HttpBlobStore.delete_prefix`.
  - `backend/src/daikonstudio/interface/routes/runner_api.py`: `DELETE /runs/{run_id}/checkpoints`.
- Test:
  - `backend/tests/unit/engines/test_fan_out.py`;
  - `backend/tests/integration/test_train_protocol.py`;
  - `backend/tests/unit/execution/test_reporter.py`, for the deadline text;
  - `backend/tests/api/test_runner_protocol.py`, or the existing runner-API test file that covers blob routes.

**Interfaces:**
- Consumes (T1): `Checkpoints`, `checkpoint_root`, `pack_result`, `unpack_result`, `RESULT_FORMAT`, `DEFAULT_INTERVAL_SECONDS`, `TrainContext.checkpoints`.
- Produces:
  - `RunTraining(..., checkpoint_interval_seconds: float = DEFAULT_INTERVAL_SECONDS)`;
  - ctx key `"checkpoint_interval_seconds"`;
  - `build_http_ctx(..., checkpoint_interval_seconds: float = DEFAULT_INTERVAL_SECONDS)`;
  - runner route `DELETE /api/v1/runner/runs/{run_id}/checkpoints` (204).

- [ ] **Step 1: Failing tests**

**FanOut**, in `tests/unit/engines/test_fan_out.py`. Reuse `_Recorder` and `_ctx`, and add:

```python
def test_a_saved_target_is_restored_not_refitted():
    store = InMemoryBlobStore()
    checkpoints = Checkpoints(store, "ws/datasets/d/runs/r/checkpoints/").scoped("model")
    targets = {"a": TaskType.REGRESSION, "a/b": TaskType.REGRESSION}  # unsafe as a path

    first = _Recorder()
    FanOut(first).train(replace(_ctx(targets), checkpoints=checkpoints))
    assert len(first.contexts) == 2

    again = _Recorder()
    result = FanOut(again).train(replace(_ctx(targets), checkpoints=checkpoints))
    assert again.contexts == []  # both targets restored
    assert set(result.metrics) == {"a", "a/b"}
```

Add the imports: `InMemoryBlobStore`, `Checkpoints`, `replace`. If `_Recorder.train` returns metrics keyed by its single target, the restored metrics carry the same keys.

**RunTraining**, in `tests/integration/test_train_protocol.py`. Train `ecfp4-xgboost` on a **scaffold-split** dataset, so all three fits run. Use the file's existing scaffold fixture, or pass the split spec the way other tests do.
1. First attempt: monkeypatch the random-split fit to raise `RunInterrupted("limit", cancelled=False)`. Monkeypatch `module.assign_split` so the comparison's `assign_split` call raises `RunInterrupted`; the existing tests show how to patch `assign_split`. The run ends FAILED.
2. Count fits: wrap `Ecfp4XGBoost.train` in a counter (`monkeypatch.setattr(Ecfp4XGBoost, "train", counting)`).
3. Retry with `RetryRun`, or by setting the row back to pending and re-running the job the way `Studio` executes jobs; read the file's helpers first. Undo the `assign_split` patch first.
4. Assert:
   - the second attempt called `train` exactly once, for the random-split fit;
   - the model and baseline were restored;
   - the run is READY;
   - after success, no blob remains under `checkpoint_root(ws, dataset.id, run.id)` (`studio.store` is the real fsspec store; check that the folder doesn't exist on disk under the blob base path).

`baseline_is_self` holds when the chosen engine is the baseline at its defaults, so choose `ecfp4-xgboost` (the baseline is RF): there will be three distinct fits.

**Deadline text**, in `tests/unit/execution/test_reporter.py:137`. Keep the `"time limit"` assertion, and add an assertion that the reason contains `STUDIO_WORKER_JOB_TIMEOUT_BY_LANE`.

**Runner route.** Add tests to the API test file that already exercises runner blob routes (grep `blobs/` under `tests/api`):
- with a claimed training run and two blobs, one under its checkpoint root and one elsewhere in the workspace, `DELETE /api/v1/runner/runs/{id}/checkpoints` returns 204, deletes only the first, and leaves the snapshot;
- without the claim, the same request returns 403.

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_fan_out.py tests/integration/test_train_protocol.py -k "saved or restored or resum" tests/unit/execution/test_reporter.py -q
```

Expected: FAIL.

- [ ] **Step 3: FanOut**

In `FanOut.train`, inside the loop over targets:

```python
            # A multi-target fit saves each target's result as it completes, scoped by
            # position (a column name could be anything) with the column in the
            # fingerprint, so a resumed run refits only the targets it had not finished.
            saved_scope = (
                ctx.checkpoints.scoped(f"target-{index}", column=column)
                if ctx.checkpoints is not None and count > 1
                else None
            )
            restored = _restore(saved_scope)
            if restored is not None:
                ctx.report((index + 1) / count, f"Restored the fit for {column} from saved progress")
                results.append(restored)
                continue
            result = self._inner.train(
                replace(
                    ctx,
                    targets={column: task},
                    report=_slice(ctx.report, index, count),
                    checkpoints=saved_scope if saved_scope is not None else ctx.checkpoints,
                )
            )
            if saved_scope is not None:
                saved_scope.save("result", pack_result(result))
            results.append(result)
```

Keep the existing `if count > 1: ctx.report(...)` before each sub-fit. Add the module-level helper:

```python
def _restore(scope: Checkpoints | None) -> TrainResult | None:
    data = scope.load("result") if scope is not None else None
    if data is None:
        return None
    try:
        return unpack_result(data)
    except Exception:
        logger.warning("A saved fit in %s did not unpack; refitting it.", scope.root, exc_info=True)
        return None
```

Add a module `logger = logging.getLogger(__name__)` if absent.

- [ ] **Step 4: RunTraining**

- **`__init__`** gains `checkpoint_interval_seconds: float = DEFAULT_INTERVAL_SECONDS`, stored, plus `self._checkpoints: Checkpoints | None = None`.
- **`__call__`**: right after the dataset is loaded, before any fit:

```python
        # The run's saved progress. A retry of this same run (same id) finds what an
        # earlier attempt saved and skips it; success clears it below.
        self._checkpoints = Checkpoints(
            self._store,
            checkpoint_root(run.workspace_id, dataset.id, run.id),
            interval_seconds=self._checkpoint_interval_seconds,
        )
```

- **`_fit`** gains a keyword argument `scope: str`. The chosen fit passes `scope="model"` and the baseline fit `scope="baseline"`. `_fit` forwards `scope` to `_train_off_thread`.
- **`_optimism_gap`** passes `scope="random-split"` to its `_train_off_thread` call.
- **`_train_off_thread`** gains `scope: str` and becomes:

```python
        manifest = engine.manifest()
        stage = (
            self._checkpoints.scoped(
                scope,
                engine=manifest.id,
                engine_version=manifest.version,
                result_format=RESULT_FORMAT,
            )
            if self._checkpoints is not None
            else None
        )
        if stage is not None:
            saved = await asyncio.to_thread(stage.load, "result")
            if saved is not None:
                try:
                    restored = unpack_result(saved)
                except Exception:
                    logger.warning("The saved %s fit did not unpack; refitting it.", scope, exc_info=True)
                else:
                    await self._progress(
                        run, span[1], f"Restored the {manifest.name} fit from saved progress"
                    )
                    return restored
        result = await asyncio.to_thread(
            engine.train,
            TrainContext(
                frame=frame,
                targets=targets,
                structure_column=dataset.structure_column,
                conditions=conditions,
                seed=dataset.split.seed,
                tune_cutoffs=self._tune_cutoffs,
                checkpoints=stage,
                report=self._reporter(run, span),
            ),
        )
        if stage is not None:
            # Off the event loop: on a runner this is an HTTP upload, and the loop also
            # carries the run's heartbeat.
            await asyncio.to_thread(stage.save, "result", pack_result(result))
        return result
```

Keep the existing comments in `_train_off_thread`.

- **Success cleanup.** Right after `run.record_metrics(headlines)`, before `_map_chemical_space`:

```python
        # The Protocol exists and the run is about to be READY: nothing here will be
        # resumed again. Best effort -- `clear` logs and swallows a failure.
        if self._checkpoints is not None:
            await asyncio.to_thread(self._checkpoints.clear)
```

- **`_check_deadline`** message becomes:

```python
            saved = (
                " Its progress is saved: Resume continues from where it stopped."
                if self._checkpoints is not None
                else ""
            )
            raise RunInterrupted(
                f"The run exceeded its {self._deadline_seconds:.0f} s time limit and was "
                f"stopped.{saved} An administrator can raise the limit "
                "(STUDIO_WORKER_JOB_TIMEOUT, or STUDIO_WORKER_JOB_TIMEOUT_BY_LANE for one lane).",
                cancelled=False,
            )
```

- [ ] **Step 5: Interval plumbing and the runner-side clear**

**`infrastructure/jobs.py` `_train`:** pass `checkpoint_interval_seconds=ctx.get("checkpoint_interval_seconds", DEFAULT_INTERVAL_SECONDS)` to `RunTraining`.

**`infrastructure/runner/ports.py`:**
- `build_http_ctx` gains `checkpoint_interval_seconds: float = DEFAULT_INTERVAL_SECONDS` and puts it in the ctx under that key.
- `HttpBlobStore` gains:

```python
    def delete_prefix(self, prefix: str) -> None:
        # The only delete a runner may make: its own run's saved progress, when the run
        # succeeds. The server derives the folder from the run; this check just keeps a
        # mistaken caller from believing some other folder was deleted.
        if not prefix.endswith(f"/runs/{self._client.run_id}/checkpoints/"):
            raise NotImplementedError("A runner can delete only its own run's saved progress.")
        response = self._client._blobs.delete(f"/runs/{self._client.run_id}/checkpoints")
        response.raise_for_status()
```

**`infrastructure/runner/agent.py`:**
- `AgentSettings` gains:

```python
    # How often a long neural fit saves its training state (STUDIO_CHECKPOINT_INTERVAL_SECONDS).
    checkpoint_interval_seconds: float = 600.0
```

- The `build_http_ctx(...)` call passes `checkpoint_interval_seconds=settings.checkpoint_interval_seconds`.

**`interface/routes/runner_api.py`** gains a route next to the blob routes:

```python
@router.delete("/runs/{run_id}/checkpoints", status_code=204)
async def delete_checkpoints(run: ClaimedRunWrite, store: BlobStoreDep) -> Response:
    """Delete this run's saved training progress, and nothing else: the folder is
    derived from the run itself, never from the request."""
    dataset_id = run.params.get("dataset_id")
    if run.kind is RunKind.TRAINING and dataset_id:
        store.delete_prefix(checkpoint_root(run.workspace_id, uuid.UUID(str(dataset_id)), run.id))
    return Response(status_code=204)
```

Add the imports (`RunKind`, `checkpoint_root`, `Response`) if absent.

- [ ] **Step 6: Run tests and gates, then commit**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_fan_out.py tests/unit/engines/test_checkpoints.py tests/unit/execution tests/integration/test_train_protocol.py -q && OMP_NUM_THREADS=1 uv run pytest <the runner API test file> -q && uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src && uv run lint-imports
```

Expected: PASS.

```bash
git add <explicit paths of every file above>
git commit -m "feat(training): save each completed fit and skip it when the run resumes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Start over

**Files:**
- Modify: `backend/src/daikonstudio/application/execution/retry_run.py`
- Modify: `backend/src/daikonstudio/infrastructure/di/container.py` (RetryRun gets `c[BlobStore]`)
- Modify: `backend/src/daikonstudio/interface/routes/runs.py` (optional body)
- Test: `backend/tests/api/test_runs.py`

**Interfaces:**
- Produces:
  - `RetryRunCommand.fresh: bool = False`;
  - `RetryRun(runs, protocols, enqueuer, engines, store)`;
  - `POST /api/v1/runs/{id}/retry` with optional body `{"fresh": bool}`.

- [ ] **Step 1: Failing tests**

In `tests/api/test_runs.py`, next to `test_retrying_a_failed_prediction_reenqueues_and_runs_it`, use the file's training helpers. Read how a training run is created and failed there, for example by setting the row to FAILED through the repository.

**Start over clears.** Put two blobs under the run's `checkpoint_root` (workspace id, `dataset_id` from run params, run id) in the app's blob store, then:
- `POST /retry` with `{"fresh": true}` returns 204;
- no key remains under the root;
- the run is pending, or has run, depending on the enqueuer in the test app.

**Plain retry keeps progress.** Same setup, then:
- `POST /retry` with no body returns 204;
- the blobs are still there, unless the inline enqueuer already ran the job to success and cleared them. If the test app runs jobs inline, assert the run's phase history or that the run reached READY without refitting. Otherwise assert the blobs remain.

  Choose based on how the test app enqueues; read the fixture.

**Prediction runs ignore the flag.** `POST /retry` with `{"fresh": true}` on a failed prediction run returns 204 and changes nothing else.

**Unknown fields are refused.** `{"fresh": "yes"}` and `{"other": 1}` both return 422.

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/api/test_runs.py -k retry -q
```

Expected: FAIL.

- [ ] **Step 3: Implement**

**`retry_run.py`.**

`RetryRunCommand` gains:

```python
    # Start over: discard the run's saved training progress before requeueing it, so
    # the next attempt fits everything again. A prediction run keeps none; ignored there.
    fresh: bool = False
```

`RetryRun.__init__` gains `store: BlobStore`. In `__call__`, after `run.retry()` and BEFORE `self._runs.update(run)`:

```python
        if command.fresh and run.kind is RunKind.TRAINING:
            # Before the update and the enqueue, so the requeued attempt can never load
            # what it was asked to forget.
            Checkpoints(
                self._store,
                checkpoint_root(run.workspace_id, uuid.UUID(str(run.params["dataset_id"])), run.id),
            ).clear()
```

**`container.py`.** `RetryRun(_runs(c), _protocols(c), c[JobEnqueuer], c[EngineRegistry], c[BlobStore])`.

**`routes/runs.py`.**

```python
class RetryRunBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Start over instead of resuming: discards the run's saved training progress.
    fresh: bool = False


@router.post("/{run_id}/retry", status_code=204)
async def retry_run(
    run_id: uuid.UUID, auth: AuthDep, service: RetryRunDep, body: RetryRunBody | None = None
) -> Response:
    """Re-execute a failed or cancelled run in place, resuming a training run from its
    saved progress unless `fresh` is set; 409 for any other status."""
    fresh = body.fresh if body is not None else False
    result_to_response(await service(RetryRunCommand(run_id=run_id, fresh=fresh), auth=auth))
    return Response(status_code=204)
```

Pydantic's default `bool` coerces `"yes"`. Use `fresh: StrictBool = False` (`from pydantic import StrictBool`) so `"yes"` is a 422.

- [ ] **Step 4: Run tests and gates, then commit**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/api/test_runs.py -q && uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src && uv run lint-imports
```

Expected: PASS.

```bash
git add backend/src/daikonstudio/application/execution/retry_run.py backend/src/daikonstudio/infrastructure/di/container.py backend/src/daikonstudio/interface/routes/runs.py backend/tests/api/test_runs.py
git commit -m "feat(runs): Start over discards a training run's saved progress before requeueing

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Neural fits save and resume their training state

**Files:**
- Modify: `backend/src/daikonstudio/infrastructure/engines/_lightning.py`
- Modify: `backend/src/daikonstudio/infrastructure/engines/chemprop_dmpnn.py`
- Modify: `backend/src/daikonstudio/infrastructure/engines/molformer_xl.py`
- Test:
  - `backend/tests/unit/engines/test_chemprop_dmpnn.py`;
  - `backend/tests/unit/engines/test_molformer_xl.py`;
  - `backend/tests/unit/engines/test_lightning_helpers.py` (create if no `_lightning` test file exists).

**Interfaces:**
- Consumes (T1): `Checkpoints`, `TrainContext.checkpoints`, and `InMemoryBlobStore` for tests.
- Produces, in `_lightning.py`:
  - `training_state_scope(checkpoints: Checkpoints | None, *libraries: str) -> Checkpoints | None`;
  - `save_training_state(checkpoints: Checkpoints, scratch: Path) -> Callback`;
  - `saved_training_state(checkpoints: Checkpoints | None, scratch: Path, module: Any) -> str | None`;
  - `keep_best_by_validation_loss()`, whose callback now has `state_dict`/`load_state_dict`.

- [ ] **Step 1: Failing tests**

`tests/unit/engines/test_chemprop_dmpnn.py`, reusing its chemprop skip guard and a small frame:

```python
def test_a_fit_stopped_by_its_time_limit_resumes_at_the_next_epoch():
    store = InMemoryBlobStore()
    checkpoints = Checkpoints(store, "ws/datasets/d/runs/r/checkpoints/", interval_seconds=1e9).scoped("model")
    frame = _small_regression_frame()  # whatever this file already uses; >= 40 rows with validation
    reported: list[tuple[float, str]] = []
    resuming = False

    def stop_after_epoch_two(fraction: float, phase: str) -> None:
        reported.append((fraction, phase))
        if phase.startswith("Training") and fraction >= 2 / 4 and not resuming:
            raise RunInterrupted("time limit", cancelled=False)

    ctx = TrainContext(
        frame=frame, targets={"y": TaskType.REGRESSION}, structure_column="smiles",
        conditions={"epochs": 4}, seed=1, checkpoints=checkpoints, report=stop_after_epoch_two,
    )
    with pytest.raises(RunInterrupted):
        ChempropDMPNN().train(ctx)
    assert training_state_scope(checkpoints, "chemprop").load("training-state") is not None

    reported.clear()
    resuming = True
    result = ChempropDMPNN().train(ctx)
    training = [f for f, p in reported if p.startswith("Training")]
    assert training[0] == pytest.approx(3 / 4)  # epoch 3 of 4 is the first one run
    assert any(p.startswith("Resuming") for _, p in reported)
    assert result.metrics["y"]
```

The closure only reads `resuming`, so the test body can reassign it without `nonlocal`. Import `training_state_scope` from `daikonstudio.infrastructure.engines._lightning`.

More tests:

- **Cancel does not save.** Same as above, but with `RunInterrupted("cancelled", cancelled=True)` and `interval_seconds=1e9`. After the raise, `training_state_scope(...).load("training-state") is None`.
- **A failing store never fails the fit.** Use a store whose `put_bytes` raises `OSError`, with `interval_seconds=0`, so every epoch tries to save. `train` completes and returns metrics.
- **Best epoch carries over.** After the first, interrupted attempt, `torch.load` the saved training state from a temp file. One entry of `state["callbacks"]` has a finite `best_loss`.
- **A corrupt but checksum-valid state is discarded.** Save garbage bytes through `training_state_scope(...).save("training-state", b"not a checkpoint")`, then train. It completes from epoch 1: the first reported training fraction is `1/4`, and the run reports no "Resuming" phase.
- **MoLFormer.** In `test_molformer_xl.py`, the time-limit resume test with the file's tiny-model fixture, `epochs=3` and a stop after epoch 1. Expect the first resumed fraction to be `2/3`. If the fixture cannot run two short fits fast, assert only that `save_training_state` is among the trainer callbacks (spy on `lightning.Trainer`), and record that in the report.

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_chemprop_dmpnn.py -q -rs -k "resume or cancel or failing_store or best_epoch or corrupt"
```

Expected: FAIL, not skipped.

- [ ] **Step 3: `_lightning.py`**

The best-epoch callback class gains:

```python
        # Saved with the training state and restored on resume (Lightning calls these),
        # so best-epoch selection survives a stopped run: without them a resumed fit
        # would forget its best epoch and keep the last one.
        def state_dict(self) -> dict[str, Any]:
            return {"best_loss": self.best_loss, "best_state": self.best_state}

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            self.best_loss = state_dict["best_loss"]
            self.best_state = state_dict["best_state"]
```

Add:

```python
def training_state_scope(checkpoints: Any, *libraries: str) -> Any:
    """Where a fit's Lightning training state is saved, fingerprinted with the library
    versions that can read it back: a runner upgraded between attempts starts the fit
    over rather than loading state its torch or lightning cannot."""
    if checkpoints is None:
        return None
    from importlib.metadata import version

    return checkpoints.scoped(
        "lightning", **{name: version(name) for name in ("torch", "lightning", *libraries)}
    )


def save_training_state(checkpoints: Any, scratch: Path) -> Any:
    """A callback saving the trainer's full state at an epoch's end: every
    `checkpoints.interval_seconds`, and once more when the run's time limit stops the
    fit. Verified on Lightning 2.6.5: either save resumes at the next epoch.

    Not on a cancel: the run row is already CANCELLED and the runner API refuses the
    write, so Resume after a cancel continues from the last periodic save.
    """
    import time

    from lightning.pytorch.callbacks import Callback

    from daikonstudio.application.engines.context import RunInterrupted

    class _SaveTrainingState(Callback):
        def __init__(self) -> None:
            self._last = time.monotonic()

        def _save(self, trainer: Any) -> None:
            path = scratch / "training-state.ckpt"
            try:
                trainer.save_checkpoint(path)
                checkpoints.save("training-state", path.read_bytes())
            except Exception:  # best effort: a failed save must not stop the fit
                logger.warning("Could not save training state", exc_info=True)
            self._last = time.monotonic()

        def on_train_epoch_end(self, trainer: Any, module: Any) -> None:
            if time.monotonic() - self._last >= checkpoints.interval_seconds:
                self._save(trainer)

        def on_exception(self, trainer: Any, module: Any, exception: BaseException) -> None:
            if isinstance(exception, RunInterrupted) and not exception.cancelled:
                self._save(trainer)

    return _SaveTrainingState()


def saved_training_state(checkpoints: Any, scratch: Path, module: Any) -> str | None:
    """A path to pass as `ckpt_path`, or None to train from the start.

    Loads the weights into `module` first, strictly, as a check: a state saved by a
    differently built model (shapes changed between attempts) is discarded here
    rather than failing `fit` halfway through restoring. Loading the same weights
    `fit` is about to restore is harmless.
    """
    if checkpoints is None:
        return None
    data = checkpoints.load("training-state")
    if data is None:
        return None
    path = scratch / "resume.ckpt"
    path.write_bytes(data)
    try:
        import torch

        # weights_only=False: a Lightning training state pickles chemprop objects (the
        # criterion, the descriptor transform), which weights-only loading refuses. It
        # is the same trust boundary as every saved model this engine loads -- a blob
        # written by this deployment's own runners into the run's workspace -- and
        # Lightning's own `ckpt_path` restore loads it the same way.
        state = torch.load(path, map_location="cpu", weights_only=False)
        module.load_state_dict(state["state_dict"])
    except Exception:
        logger.warning("Saved training state did not fit this model; training from the start", exc_info=True)
        return None
    return str(path)
```

Add `import logging`, `from pathlib import Path` and `logger = logging.getLogger(__name__)` at module top. Those are stdlib only. Keep the module's no-torch-at-import rule.

- [ ] **Step 4: Wire chemprop and MoLFormer**

In each engine's `train`, wrap the trainer construction and `trainer.fit` in `with tempfile.TemporaryDirectory() as scratch_dir:` and set `scratch = Path(scratch_dir)`. Keep everything after `fit` that needs the trainer inside the block, or keep the trainer object alive past it (it doesn't need the directory). Then:

```python
            state = training_state_scope(ctx.checkpoints, "chemprop")  # "transformers" in molformer
            resume_from = saved_training_state(state, scratch, model)  # `module` in molformer
            if state is not None:
                callbacks.append(save_training_state(state, scratch))
            if resume_from is not None:

                def _report_resume(trainer: Any, _module: Any) -> None:
                    ctx.report(
                        trainer.current_epoch / epochs,
                        f"Resuming {_MANIFEST.name} from epoch {trainer.current_epoch + 1} of {epochs}",
                    )

                callbacks.append(LambdaCallback(on_train_start=_report_resume))
            trainer = lightning.Trainer(...)  # unchanged arguments
            trainer.fit(model, <train loader>, <validation loader>, ckpt_path=resume_from)
```

Append the save callback AFTER the reporter `LambdaCallback`. The reporter raises in `on_train_epoch_end`, and `on_exception` then saves, as verified. The best-epoch restore (`load_state_dict(keep_best.best_state)`) and all scoring stay exactly as they are.

- [ ] **Step 5: Run tests and gates, then commit**

Run (separate invocations):
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_chemprop_dmpnn.py -q -rs && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_molformer_xl.py -q -rs && uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src && uv run lint-imports
```

Expected: PASS, none skipped. The existing tests pass unchanged.

```bash
git add backend/src/daikonstudio/infrastructure/engines/_lightning.py backend/src/daikonstudio/infrastructure/engines/chemprop_dmpnn.py backend/src/daikonstudio/infrastructure/engines/molformer_xl.py backend/tests/unit/engines
git commit -m "feat(engines): chemprop and MoLFormer save training state and resume from it

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Resume and Start over in the UI

**Files:**
- Modify: `frontend/src/features/runs/hooks/use-runs.ts` (`useRetryRun` takes `{ id, fresh }`)
- Modify: `frontend/src/features/runs/index.ts` (export `useRetryRun`)
- Modify: `frontend/src/features/runs/components/run-detail.tsx`
- Modify: `frontend/src/features/sweeps/components/sweep-detail.tsx` (Resume on a failed or cancelled row)
- Modify: `frontend/openapi.json` and the generated client (`make generate-api`, for the new `RetryRunBody`)
- Test: the existing `run-detail` and sweep tests beside these files

- [ ] **Step 1: Regenerate the client**

Run `make generate-api` from the repo root. The diff must contain only `RetryRunBody` and the retry route's optional body.

- [ ] **Step 2: Failing tests**

**Run detail.**
- A failed **training** run shows a "Resume" button and a "Start over" button. Clicking "Start over" posts `{"fresh": true}` to `/runs/{id}/retry`; clicking "Resume" posts no body, or `{}`.
- A failed **prediction** run shows "Retry", and neither of the other two.

**Sweep detail.** A failed row shows "Resume", and clicking it posts to `/runs/{id}/retry`. A ready row shows no Resume.

Follow each file's existing mock pattern for `customInstance` or the hooks.

- [ ] **Step 3: Implement**

**`useRetryRun`:**

```ts
export function useRetryRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, fresh = false }: { id: string; fresh?: boolean }) =>
      customInstance<void>({
        url: `${API_V1}/runs/${id}/retry`,
        method: "POST",
        ...(fresh ? { data: { fresh: true } } : {}),
      }),
    onSuccess: (_data, { id, fresh }) => {
      queryClient.invalidateQueries({ queryKey: [...RUN_KEY, id] });
      queryClient.invalidateQueries({ queryKey: RUNS_KEY });
      showSuccess(fresh ? "Starting over" : "Run requeued");
    },
  });
}
```

Update every caller. `grep -rn "useRetryRun\|retry.mutate" frontend/src` lists them.

**`run-detail.tsx`.** Replace the Retry button block:

```tsx
        {(run.status === "failed" || run.status === "cancelled") &&
          (run.kind === "training" ? (
            <>
              <Button onClick={() => retry.mutate({ id: runId })} disabled={retry.isPending}>
                Resume
              </Button>
              <Button
                variant="ghost"
                onClick={() => retry.mutate({ id: runId, fresh: true })}
                disabled={retry.isPending}
                title="Discards saved progress and trains from the beginning"
              >
                Start over
              </Button>
            </>
          ) : (
            <Button variant="outline" onClick={() => retry.mutate({ id: runId })} disabled={retry.isPending}>
              Retry
            </Button>
          ))}
```

**`sweep-detail.tsx`.** In the Status cell, after the badge, for `run.status === "failed" || run.status === "cancelled"`:

```tsx
                    {(run.status === "failed" || run.status === "cancelled") && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="mt-1.5 h-6 px-2 text-xs"
                        onClick={() => retry.mutate({ id: run.id })}
                        disabled={retry.isPending}
                      >
                        Resume
                      </Button>
                    )}
```

Here `const retry = useRetryRun();` is imported from `@/features/runs`. Sweep runs are always training runs.

- [ ] **Step 4: Gates, then commit**

Run:
```
cd frontend && pnpm vitest run src/features/runs src/features/sweeps && pnpm lint && pnpm exec tsc --noEmit
```

Expected: PASS.

```bash
git add frontend/openapi.json frontend/src/shared/lib/api frontend/src/features/runs frontend/src/features/sweeps
git commit -m "feat(frontend): Resume and Start over for stopped training runs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Verify (controller)

- [ ] **Step 1: Run the full suites once**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest -q
cd ../frontend && pnpm test
```

- [ ] **Step 2: Live check**

Only after the CheMeleon rerun has finished.
1. Set `STUDIO_WORKER_JOB_TIMEOUT_BY_LANE='{"gpu": 300}'` in `backend/.env`, set `STUDIO_CHECKPOINT_INTERVAL_SECONDS=120` for the runners, then run `make dev-be dev-worker dev-worker-gpu`.
2. Start a CheMeleon chemprop run on `nuisance_sample_10k`. It stops at 300 s, and its message says progress is saved.
3. Click **Resume**. The phases show "Restored the … fit" for completed fits and "Resuming … from epoch N", and the run finishes READY. Check that no folder remains under the run's checkpoint root.
4. Click **Start over** on another stopped run. It retrains from epoch 1.
5. Restore the `.env` values (gpu 432000; interval default) and restart the dev processes.

- [ ] **Step 3: Final audit**

One whole-branch audit (most capable model), then one fix wave, then one scoped re-review.
