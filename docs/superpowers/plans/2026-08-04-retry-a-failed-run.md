# Retry a Failed Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user re-execute a run that failed or was cancelled, without rebuilding the request.

**Architecture:** One new edge out of the terminal statuses — `Run.retry()` moves `failed`/`cancelled` back to `pending` — plus a `RetryRun` use case that re-enqueues on the correct lane. Because `run.params` is write-once and `update()` never persists it, the retried run re-executes its original instructions by construction; nothing is copied or rebuilt.

**Tech Stack:** Python 3.14, FastAPI, SQLAlchemy (async), arq, pytest. Frontend: Next.js, React, TypeScript, TanStack Query, shadcn/radix.

**Spec:** `docs/superpowers/specs/2026-08-04-retry-a-failed-run-design.md`

**Depends on:** `2026-08-04-choosable-baseline-and-pretrained-weights.md` Task 1, which adds `lane_for()`. Land that plan first. If it has not landed, Task 2 below cannot resolve a training run's lane correctly and would reintroduce the routing bug that plan fixes.

## Global Constraints

- **`running` must never be retryable.** A crashed worker leaves a run RUNNING with no error recorded; retrying from there starts a second fit beside one that may still be alive. arq's `job_timeout` cancels a coroutine, not the `to_thread` OS thread, so Python cannot kill the first one (handoff §3).
- **`run.params` is write-once.** `SqlAlchemyRunRepository.update()` never persists it. The retry path must not attempt to modify it.
- **Update the row before enqueuing.** `run_job` drops redeliveries for terminal runs (`infrastructure/worker.py:138-141`), so a job enqueued while the row still reads `failed` is silently discarded.
- **`cache_key` does not change on retry.** The inputs are identical; recomputing or altering it would break the prediction cache's identity.
- Run `make test`, `ruff check`, `ruff format --check` and `mypy` before each commit.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `backend/src/daikonstudio/domain/execution/run.py` | The Run aggregate and its status lattice | Add `retry()` |
| `backend/src/daikonstudio/application/execution/retry_run.py` | **New.** Load, transition, re-enqueue on the right lane | Create |
| `backend/src/daikonstudio/interface/routes/runs.py` | HTTP surface | One endpoint |
| `frontend/src/features/runs/hooks/use-runs.ts` | Run mutations | `useRetryRun` |
| `frontend/src/features/runs/components/run-detail.tsx` | Run page | Retry button |

`retry_run.py` is its own module rather than more lines in `predict_with_protocol.py` (where `CancelRun` currently lives). `CancelRun` needs only a `RunRepository`; `RetryRun` needs the registry, the protocol repository and the enqueuer to resolve a lane, and it serves both run kinds — putting that inside the prediction module would make training depend on prediction.

---

### Task 1: `Run.retry()`

**Files:**
- Modify: `backend/src/daikonstudio/domain/execution/run.py` (add after `cancel()`, `:177-196`)
- Test: `backend/tests/unit/execution/test_run.py` (extend). It already has the `_pending(**overrides) -> Run` helper at `:15` and covers every other transition — follow its naming.

**Interfaces:**
- Produces: `Run.retry() -> None`, consumed by Task 2.

- [ ] **Step 1: Write the failing test**

```python
def test_retry_returns_a_failed_run_to_pending_and_clears_the_failure():
    run = _pending()
    run.start()
    run.report_progress(0.5, phase="training chemprop-dmpnn")
    run.fail("FileNotFoundError(2, 'No such file or directory')")

    run.retry()

    assert run.status is RunStatus.PENDING
    assert run.progress == 0.0
    assert run.phase is None
    # A stale message would render on a run that is queued again and has not
    # failed this time.
    assert run.error_message is None


def test_retry_is_allowed_from_cancelled():
    """This is what makes a worker crash recoverable. A crashed worker leaves the
    run RUNNING with no error recorded, so retry-from-failed alone cannot reach
    it; the user cancels first -- legal from running -- and then retries."""
    run = _pending()
    run.start()
    run.cancel()

    run.retry()

    assert run.status is RunStatus.PENDING


@pytest.mark.parametrize("setup", ["pending", "running", "ready"])
def test_retry_refuses_every_status_that_is_not_a_finished_failure(setup):
    """`running` is the important one. Retrying a run that may still be executing
    starts a second fit beside it, and arq's timeout cannot kill the first --
    it cancels a coroutine, not the OS thread the fit runs on."""
    run = _pending()
    if setup in {"running", "ready"}:
        run.start()
    if setup == "ready":
        run.succeed("blob://result")

    with pytest.raises(ConflictError):
        run.retry()


def test_retry_preserves_params_and_cache_key():
    """What makes retry-in-place cheap: params are write-once, so the re-enqueued
    job re-reads the same instructions with nothing rebuilt."""
    run = _pending()
    params_before, key_before = dict(run.params), run.cache_key
    run.start()
    run.fail("boom")

    run.retry()

    assert run.params == params_before
    assert run.cache_key == key_before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/execution/test_run.py -v -k retry`
Expected: FAIL with `AttributeError: 'Run' object has no attribute 'retry'`

- [ ] **Step 3: Write minimal implementation**

Add to `run.py` after `cancel()`:

```python
    def retry(self) -> None:
        """`failed -> pending` and `cancelled -> pending`. The only edge that
        leaves `_TERMINAL`, and deliberately narrow.

        `params` is write-once, so the re-enqueued job re-reads exactly the
        instructions that failed -- there is nothing to rebuild, which is what
        makes retry-in-place cheaper than creating a replacement Run. Clearing
        `error_message` matters: a stale one would render on a Run that is
        queued again and has not failed this time.

        `running` is excluded on purpose. A crashed worker leaves a Run RUNNING
        with no error recorded, and retrying from there would start a second fit
        beside one that may still be alive -- arq's `job_timeout` cancels a
        coroutine, not the `to_thread` OS thread, so nothing can kill the first.
        Cancel first: that flips the row, which is the only channel a live
        worker checkpoints against, and then retry.
        """
        if self.status not in {RunStatus.FAILED, RunStatus.CANCELLED}:
            raise ConflictError(f"Cannot retry run '{self.id}' in status '{self.status}'")
        self.status = RunStatus.PENDING
        self.progress = 0.0
        self.phase = None
        self.error_message = None
        self._touch()
```

Update the module docstring's lattice sentence (`:3-6`) to record the new edge:

```
Status is a one-way lattice: `pending -> running -> {ready, failed,
cancelled}`, with `pending -> cancelled` as the only shortcut,
`running -> running` allowed as a restart (at-least-once redelivery after a
worker crash -- see `start()`), and `{failed, cancelled} -> pending` as the
one deliberate way back (see `retry()`).
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/unit/ -v && uv run mypy src`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/daikonstudio/domain/execution/run.py backend/tests/unit/execution/test_run.py
git commit -m "feat: a failed or cancelled run can return to pending"
```

---

### Task 2: The `RetryRun` use case

**Files:**
- Create: `backend/src/daikonstudio/application/execution/retry_run.py`
- Test: `backend/tests/unit/execution/test_retry_run.py` (create)

**Interfaces:**
- Consumes: `Run.retry()` (Task 1), `lane_for` from the baseline plan's Task 1.
- Produces: `RetryRunCommand(run_id: uuid.UUID)` and `RetryRun(runs, protocols, engines, enqueuer)`, returning `Result[Run, DomainError]`. Consumed by Task 3.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/execution/test_retry_run.py`. Self-contained stubs, modelled on the ones in `test_lanes.py` (`_StubEngine` at `:34`, `_RecordingEnqueuer` at `:115` recording into `.lanes`) — do not import private names across test modules.

```python
"""Retrying a Run: which statuses may come back, and where the job goes."""

from __future__ import annotations

import uuid
from typing import Any

import polars as pl
import pytest

from daikonstudio.application.engines.context import PredictContext, TrainContext, TrainResult
from daikonstudio.application.engines.manifest import DEFAULT_LANE, EngineManifest, TaskType
from daikonstudio.application.engines.registry import EngineRegistry
from daikonstudio.application.execution.retry_run import RetryRun, RetryRunCommand
from daikonstudio.domain.catalog.protocol import InSilicoProtocol, ProtocolStatus
from daikonstudio.domain.execution.run import Run, RunKind, RunStatus, compute_cache_key
from daikonstudio.domain.shared.errors import ConflictError, NotFoundError
from tests.fakes.auth import FakeAuth


class _StubEngine:
    def __init__(self, engine_id: str, lane: str, *, is_baseline: bool = False) -> None:
        self._manifest = EngineManifest(
            id=engine_id,
            version="1.0.0",
            name=engine_id,
            description="",
            tasks=(TaskType.REGRESSION,),
            lane=lane,
            is_baseline=is_baseline,
        )

    def manifest(self) -> EngineManifest:
        return self._manifest

    def train(self, ctx: TrainContext) -> TrainResult:  # pragma: no cover
        raise NotImplementedError

    def predict(self, ctx: PredictContext) -> pl.DataFrame:  # pragma: no cover
        raise NotImplementedError


REGISTRY = EngineRegistry(
    {
        "heavy": _StubEngine("heavy", "gpu"),
        "plain": _StubEngine("plain", DEFAULT_LANE, is_baseline=True),
    }
)


class _StubRuns:
    """Records the status the row held at each enqueue, which is what the
    ordering assertion below inspects."""

    def __init__(self, run: Run, workspace_id: uuid.UUID) -> None:
        self._run = run
        self._workspace_id = workspace_id
        self.updated: list[RunStatus] = []

    async def get(self, workspace_id: uuid.UUID, run_id: uuid.UUID) -> Run | None:
        if workspace_id != self._workspace_id or run_id != self._run.id:
            return None
        return self._run

    async def update(self, run: Run) -> None:
        self.updated.append(run.status)


class _StubProtocols:
    def __init__(self, protocol: InSilicoProtocol | None) -> None:
        self._protocol = protocol

    async def get(self, ws: uuid.UUID, protocol_id: uuid.UUID) -> Any:
        return self._protocol


class _RecordingEnqueuer:
    def __init__(self, runs: _StubRuns | None = None) -> None:
        self.lanes: list[str] = []
        self.statuses_at_enqueue: list[RunStatus] = []
        self._runs = runs

    async def enqueue(self, run_id: uuid.UUID, lane: str = DEFAULT_LANE) -> None:
        self.lanes.append(lane)
        if self._runs is not None:
            self.statuses_at_enqueue.append(self._runs._run.status)


def _training_run(auth: FakeAuth, *, engine_id: str, baseline_engine_id: str) -> Run:
    return Run(
        kind=RunKind.TRAINING,
        workspace_id=auth.workspace_id,
        requested_by=auth.user_id,
        cache_key=compute_cache_key(kind="training"),
        params={
            "name": "a run",
            "dataset_id": str(uuid.uuid4()),
            "engine_id": engine_id,
            "conditions": {},
            "baseline_engine_id": baseline_engine_id,
            "baseline_conditions": {},
        },
    )


def _build(run: Run, auth: FakeAuth, protocol: InSilicoProtocol | None = None) -> Any:
    runs = _StubRuns(run, auth.workspace_id)
    enqueuer = _RecordingEnqueuer(runs)
    use_case = RetryRun(runs, _StubProtocols(protocol), REGISTRY, enqueuer)  # type: ignore[arg-type]
    return runs, enqueuer, use_case


async def test_a_failed_training_run_is_re_enqueued_on_its_engines_lane() -> None:
    auth = FakeAuth()
    run = _training_run(auth, engine_id="heavy", baseline_engine_id="plain")
    run.start()
    run.fail("boom")
    _runs, enqueuer, use_case = _build(run, auth)

    result = await use_case(RetryRunCommand(run_id=run.id), auth)

    assert result.unwrap().status is RunStatus.PENDING
    assert enqueuer.lanes == ["gpu"]


async def test_a_default_lane_run_with_a_gpu_baseline_still_goes_to_gpu() -> None:
    """Retry must use the same lane rule as the original enqueue, not a second
    copy of it -- otherwise a run routed correctly the first time lands on a
    worker that cannot serve its baseline the second."""
    auth = FakeAuth()
    run = _training_run(auth, engine_id="plain", baseline_engine_id="heavy")
    run.start()
    run.fail("boom")
    _runs, enqueuer, use_case = _build(run, auth)

    await use_case(RetryRunCommand(run_id=run.id), auth)

    assert enqueuer.lanes == ["gpu"]


async def test_a_failed_prediction_run_resolves_its_lane_from_the_protocol() -> None:
    auth = FakeAuth()
    protocol = InSilicoProtocol(
        workspace_id=auth.workspace_id,
        name="p",
        dataset_id=uuid.uuid4(),
        engine_id="heavy",
        artifact_uri="file:///artifact.joblib",
        readouts=(),
        conditions={},
        status=ProtocolStatus.PUBLISHED,
    )
    run = Run(
        kind=RunKind.PREDICTION,
        workspace_id=auth.workspace_id,
        requested_by=auth.user_id,
        cache_key=compute_cache_key(kind="prediction"),
        params={},
        protocol_id=protocol.id,
    )
    run.start()
    run.fail("boom")
    _runs, enqueuer, use_case = _build(run, auth, protocol)

    await use_case(RetryRunCommand(run_id=run.id), auth)

    assert enqueuer.lanes == ["gpu"]


async def test_the_row_is_pending_before_anything_is_enqueued() -> None:
    """`run_job` drops redeliveries for terminal runs, so a job made visible while
    the row still read `failed` would be discarded and the retry would silently
    do nothing."""
    auth = FakeAuth()
    run = _training_run(auth, engine_id="plain", baseline_engine_id="plain")
    run.start()
    run.fail("boom")
    runs, enqueuer, use_case = _build(run, auth)

    await use_case(RetryRunCommand(run_id=run.id), auth)

    assert runs.updated == [RunStatus.PENDING]
    assert enqueuer.statuses_at_enqueue == [RunStatus.PENDING]


async def test_retrying_a_running_run_is_a_conflict_and_enqueues_nothing() -> None:
    auth = FakeAuth()
    run = _training_run(auth, engine_id="plain", baseline_engine_id="plain")
    run.start()
    _runs, enqueuer, use_case = _build(run, auth)

    result = await use_case(RetryRunCommand(run_id=run.id), auth)

    assert isinstance(result.failure(), ConflictError)
    assert enqueuer.lanes == []


async def test_retrying_a_run_in_another_workspace_is_a_404() -> None:
    auth = FakeAuth()
    run = _training_run(auth, engine_id="plain", baseline_engine_id="plain")
    run.start()
    run.fail("boom")
    _runs, enqueuer, use_case = _build(run, auth)

    result = await use_case(RetryRunCommand(run_id=run.id), FakeAuth())

    assert isinstance(result.failure(), NotFoundError)
    assert enqueuer.lanes == []
```

Check `tests/fakes/auth.py` before relying on it: the last test needs two `FakeAuth()` instances with **different** `workspace_id`s. If `FakeAuth` uses a fixed id, construct the second with an explicit distinct workspace instead.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/execution/test_retry_run.py -v`
Expected: FAIL with `ModuleNotFoundError: daikonstudio.application.execution.retry_run`

- [ ] **Step 3: Write minimal implementation**

Create `backend/src/daikonstudio/application/execution/retry_run.py`:

```python
"""Re-execute a Run that failed or was cancelled.

Its own module rather than more lines beside `CancelRun`: cancelling needs only
a RunRepository, while retrying has to resolve a lane, which pulls in the engine
registry and -- for a prediction -- the Protocol. Putting that in the prediction
module would make training depend on prediction.

`Run.retry()` owns the rule about which statuses may come back. This use case's
job is the two things the aggregate cannot do: find the queue that can serve the
work, and get the row persisted before the job is visible to a worker.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from daikonstudio.application.engines.manifest import lane_for
from daikonstudio.application.engines.registry import EngineRegistry, UnknownEngineError
from daikonstudio.application.execution.enqueue import JobEnqueuer
from daikonstudio.application.execution.train_protocol import TrainProtocolCommand
from daikonstudio.application.ports.repositories import ProtocolRepository, RunRepository
from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.domain.shared.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.domain.shared.errors import DomainError, NotFoundError
from daikonstudio.domain.shared.result import Failure, Result, Success


@dataclass(frozen=True, kw_only=True)
class RetryRunCommand:
    run_id: uuid.UUID


class RetryRun:
    def __init__(
        self,
        runs: RunRepository,
        protocols: ProtocolRepository,
        engines: EngineRegistry,
        enqueuer: JobEnqueuer,
    ) -> None:
        self._runs = runs
        self._protocols = protocols
        self._engines = engines
        self._enqueuer = enqueuer

    async def __call__(
        self, command: RetryRunCommand, auth: AuthContext | None = None
    ) -> Result[Run, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None

        run = await self._runs.get(auth.workspace_id, command.run_id)
        if run is None:
            return Failure(NotFoundError("Run", str(command.run_id)))

        # Before the transition: an engine this deployment no longer ships makes
        # the Run unservable, and leaving it FAILED with its message intact is
        # more useful than moving it to PENDING for a worker that can never take
        # it.
        lane = await self._lane_for_run(run)
        if lane is None:
            return Failure(NotFoundError("Engine", str(run.params.get("engine_id", ""))))

        try:
            run.retry()
        except DomainError as error:
            return Failure(error)

        # Persist first. `run_job` drops redeliveries for terminal runs, so a job
        # made visible while the row still read FAILED would be discarded and the
        # retry would silently do nothing.
        await self._runs.update(run)
        await self._enqueuer.enqueue(run.id, lane=lane)
        return Success(run)

    async def _lane_for_run(self, run: Run) -> str | None:
        """The queue that can serve this Run, resolved exactly as its original
        enqueue did -- `TrainProtocol` for training, `PredictWithProtocol` for
        prediction. Returns None when an engine is no longer registered."""
        try:
            if run.kind is RunKind.TRAINING:
                command = TrainProtocolCommand.from_params(run.params)
                engine = self._engines.get(command.engine_id)
                baseline = (
                    self._engines.get(command.baseline_engine_id)
                    if command.baseline_engine_id
                    else self._engines.baseline()
                )
                return lane_for(engine.manifest(), baseline.manifest())

            if run.protocol_id is None:
                return None
            protocol = await self._protocols.get(run.workspace_id, run.protocol_id)
            if protocol is None:
                return None
            return self._engines.get(protocol.engine_id).manifest().lane
        except UnknownEngineError:
            return None
```

Wire `RetryRun` into the DI container beside the other execution use cases in `backend/src/daikonstudio/infrastructure/di/container.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/ -v && uv run mypy src && uv run lint-imports`
Expected: PASS, 3 import-linter contracts kept

- [ ] **Step 5: Commit**

```bash
git add backend/src/daikonstudio/application/execution/retry_run.py backend/src/daikonstudio/infrastructure/di/container.py backend/tests/unit/execution/test_retry_run.py
git commit -m "feat: RetryRun re-enqueues a failed run on a lane that can serve it"
```

---

### Task 3: The retry endpoint

**Files:**
- Modify: `backend/src/daikonstudio/interface/routes/runs.py` (add after `cancel_run`, `:253-256`)
- Test: `backend/tests/integration/test_run_lifecycle.py` (extend)

**Interfaces:**
- Consumes: `RetryRun`, `RetryRunCommand` (Task 2).
- Produces: `POST /api/v1/runs/{run_id}/retry` → 204. Consumed by Task 4.

- [ ] **Step 1: Write the failing test**

**Read `tests/integration/test_run_lifecycle.py` first and reuse its existing fixtures and helper style — do not invent a `client` fixture.** The integration suite's `conftest.py` re-exports fixtures by import, and the sibling files build a per-test `Studio`-style helper rather than a bare HTTP client, so the exact entry point differs by file. Mirror whatever that file already does to reach the runs router; the three behaviours to assert are:

1. **204, and the run comes back `pending` with `error_message` cleared.** Fail a run, POST `/api/v1/runs/{id}/retry`, then GET `/api/v1/runs/{id}` and assert `status == "pending"` and `error_message is None`.
2. **A running run is rejected, and nothing is enqueued.** Assert the status `ConflictError` maps to — check `_error_to_status` in `interface/` and assert the real value rather than assuming 409.
3. **An unknown run id is 404.**

Write them in the file's established naming style (`test_<what happens>_<under what condition>`), with a docstring on the second one recording *why* a running run is refused: retrying it would start a second fit that nothing can kill.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/integration/test_run_lifecycle.py -v -k retry`
Expected: FAIL with 404 — the route does not exist

- [ ] **Step 3: Write minimal implementation**

Add the dependency alias beside the others near `:55`:

```python
RetryRunDep = Annotated[RetryRun, Depends(use_case(RetryRun))]
```

And the endpoint after `cancel_run`:

```python
@router.post("/{run_id}/retry", status_code=204)
async def retry_run(run_id: uuid.UUID, auth: AuthDep, service: RetryRunDep) -> Response:
    result_to_response(await service(RetryRunCommand(run_id=run_id), auth=auth))
    return Response(status_code=204)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/ -v && uv run ruff check && uv run mypy src`
Expected: PASS

- [ ] **Step 5: Regenerate the contract and commit**

```bash
make generate-api
git add backend/src/daikonstudio/interface/routes/runs.py backend/tests/ frontend/openapi.json frontend/src/shared/lib/api/
git commit -m "feat: POST /runs/{id}/retry"
```

---

### Task 4: The Retry button

**Files:**
- Modify: `frontend/src/features/runs/hooks/use-runs.ts` (add after `useCancelRun`, `:80-90`)
- Modify: `frontend/src/features/runs/components/run-detail.tsx:112-117` (the failure block)
- Test: `frontend/src/features/runs/components/run-detail.test.tsx` (create if absent)

**Interfaces:**
- Consumes: `POST /api/v1/runs/{id}/retry` (Task 3).

- [ ] **Step 1: Write the failing test**

```tsx
import { describe, expect, it } from "vitest";
import { canRetry } from "../types";

describe("canRetry", () => {
  it("offers retry on a finished failure", () => {
    expect(canRetry("failed")).toBe(true);
    expect(canRetry("cancelled")).toBe(true);
  });

  it("never offers retry while work may still be running", () => {
    // Retrying a live run starts a second fit beside it and nothing can kill
    // the first.
    expect(canRetry("running")).toBe(false);
    expect(canRetry("pending")).toBe(false);
    expect(canRetry("ready")).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && pnpm test run-detail`
Expected: FAIL — `canRetry` is not exported

- [ ] **Step 3: Write minimal implementation**

Add to `frontend/src/features/runs/types/index.ts`, beside `RUN_STATUS_COPY`:

```ts
/** Mirrors `Run.retry()`: only a run that has finished unsuccessfully. */
export function canRetry(status: string): boolean {
  return status === "failed" || status === "cancelled";
}
```

Add to `use-runs.ts` after `useCancelRun`, following its shape exactly:

```ts
export function useRetryRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      customInstance<void>({ url: `${API_V1}/runs/${id}/retry`, method: "POST" }),
    onSuccess: (_data, id) => {
      // Back to `pending` restarts the existing poll on its own.
      queryClient.invalidateQueries({ queryKey: [...RUN_KEY, id] });
      showSuccess("Run queued again");
    },
  });
}
```

In `run-detail.tsx`, render a `<Button variant="outline">` beside the error message at `:112-117`, shown when `canRetry(run.status)`, calling `retry.mutate(run.id)` and disabled while `retry.isPending`. No confirm dialog — matching Cancel, and retrying is neither destructive nor expensive to undo.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && pnpm test && pnpm typecheck && pnpm lint`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/features/runs/
git commit -m "feat: retry a failed run from its detail page"
```

---

### Task 5: End-to-end verification

**Files:** none modified — this task produces evidence, not code.

- [ ] **Step 1: Full suites**

```bash
cd backend && uv run pytest -v && uv run ruff check && uv run ruff format --check && uv run mypy src && uv run lint-imports
cd ../frontend && pnpm test && pnpm typecheck && pnpm lint
```

- [ ] **Step 2: Force a real failure and recover it**

With `make dev` running, submit a training run against an engine the target worker cannot serve — the cleanest trigger is `chemprop-dmpnn` with the gpu-lane worker stopped, or a dataset whose structure column is missing. Wait for the run to reach `failed`, then press Retry.

Expected: the row returns to `pending`, the error message clears, the progress bar restarts from zero, and the poll resumes without a page reload.

- [ ] **Step 3: Verify the crashed-worker path**

Start a training run, `kill -9` the worker mid-fit, and confirm the run sits at `running` with a frozen bar. Press Cancel, then Retry.

Expected: Cancel is offered and succeeds, Retry then becomes available, and the run completes on a fresh worker. This is the recovery that lets the staleness reaper stay deferred — confirm it works, because the spec's argument for excluding `running` from `retry()` depends on it.

- [ ] **Step 4: Confirm retry does not double-fit**

Start a run on the ECFP4 baseline engine, which never calls `ctx.report` and so cannot be interrupted mid-fit. Cancel and immediately Retry.

Expected: the second fit completes and produces one Protocol. Note the observed overlap in the PR — the spec accepts a brief double-fit here, and this step is what turns that from an assumption into a measurement.

- [ ] **Step 5: Open the PR**

Record in the description: which failure was forced in Step 2, whether Step 3 recovered cleanly, and what Step 4 measured.
