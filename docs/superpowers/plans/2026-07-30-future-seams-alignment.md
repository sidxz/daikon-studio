# Future-Seams Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the stuck-RUNNING redelivery gap and add the manifest JSON round-trip tripwire, per `docs/superpowers/specs/2026-07-30-future-seams-alignment-design.md`.

**Architecture:** Two small changes. (1) `Run.start()` additionally allows RUNNING→RUNNING — an at-least-once queue redelivering after a worker crash is a legitimate restart-from-zero; `run_job` then treats `ConflictError` from `start()` as "run went terminal while queued" and drops the job cleanly. (2) One unit test asserting every registered engine manifest survives a JSON round-trip, mechanically enforcing the Phase 5 out-of-process exit.

**Tech Stack:** Python 3.12+, pytest (asyncio auto mode — async tests need no decorator), uv.

## Global Constraints

- All commands run from `/Users/sidx/workspace/daikon-studio/backend/`.
- Test runner: `uv run pytest <path> -v`. No database or Valkey needed for any test in this plan (all unit tests).
- Work on the existing branch `future-seams-alignment`.
- Comment style: this codebase marks deliberate shortcuts with `ponytail:` comments naming the ceiling and upgrade path. Preserve that idiom; do not delete unrelated comments.
- Do not change any public signature. `Run.start()` keeps its zero-argument shape; `run_job(ctx, run_id)` keeps its shape.

---

### Task 1: `Run.start()` allows RUNNING→RUNNING restart

**Files:**
- Modify: `src/daikonstudio/domain/execution/run.py:130-137` (the `start()` method) and the module docstring lines 1-11
- Test: `tests/unit/execution/test_run.py`

**Interfaces:**
- Produces: `Run.start()` — raises `ConflictError` only from terminal states (READY/FAILED/CANCELLED); from PENDING or RUNNING it sets `status=RUNNING`, `progress=0.0`, `phase=None`. Task 2 relies on exactly this: `ConflictError` from `start()` now means "terminal", nothing else.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/execution/test_run.py` (imports already present: `pytest`, `Run`, `RunKind`, `RunStatus`, `ConflictError`, `_pending` helper):

```python
def test_start_on_a_running_run_restarts_it():
    """arq is at-least-once: a worker crash mid-job redelivers the same run_id
    to a fresh process, which finds the row already RUNNING. With no
    checkpoints, restart-from-zero is the designed recovery -- so the
    redelivery is a legitimate restart, and stale progress from the dead
    attempt is wiped."""
    run = _pending()
    run.start()
    run.report_progress(0.66, phase="training baseline")

    run.start()  # redelivery after a worker crash

    assert run.status is RunStatus.RUNNING
    assert run.progress == 0.0
    assert run.phase is None


def test_start_on_a_terminal_run_still_raises():
    """Cancelled-while-queued (or already finished) runs must not restart --
    ConflictError from start() is how the worker knows to drop a redelivery."""
    run = _pending()
    run.cancel()
    with pytest.raises(ConflictError):
        run.start()
```

- [ ] **Step 2: Run the new tests to verify the first fails**

Run: `uv run pytest tests/unit/execution/test_run.py -v -k "start_on"`
Expected: `test_start_on_a_running_run_restarts_it` FAILS with `ConflictError: Cannot start run ... in status 'running'`; `test_start_on_a_terminal_run_still_raises` PASSES (already-true behavior, kept as a guard).

- [ ] **Step 3: Relax `start()`**

In `src/daikonstudio/domain/execution/run.py`, replace the whole `start()` method (currently lines 130-137) with:

```python
    def start(self) -> None:
        """`pending -> running` normally. `running -> running` is also legal:
        arq is at-least-once, so a worker crash mid-job redelivers the same
        run_id to a fresh process, which lands here with the row already
        RUNNING. With no checkpoints, restart-from-zero is the designed
        recovery, so the redelivery restarts the run and wipes the dead
        attempt's stale progress. Terminal runs still refuse -- a redelivery
        for a run that was cancelled (or somehow finished) while queued must
        be dropped by the caller, not restarted."""
        if self.status in _TERMINAL:
            raise ConflictError(f"Cannot start run '{self.id}' in status '{self.status}'")
        self.status = RunStatus.RUNNING
        self.progress = 0.0
        self.phase = None
        self._touch()
```

Then update the module docstring's second sentence (lines 3-6) from:

```
Status is a strict one-way lattice: `pending -> running -> {ready, failed,
cancelled}`, with `pending -> cancelled` as the only shortcut. `_TERMINAL`
```

to:

```
Status is a one-way lattice: `pending -> running -> {ready, failed,
cancelled}`, with `pending -> cancelled` as the only shortcut and
`running -> running` allowed as a restart (at-least-once redelivery after a
worker crash -- see `start()`). `_TERMINAL`
```

- [ ] **Step 4: Run the execution unit tests plus the run-lifecycle integration tests**

Run: `uv run pytest tests/unit/execution/ tests/integration/test_run_lifecycle.py -v`
Expected: all PASS — including the pre-existing `test_a_terminal_run_cannot_restart` (terminal guard unchanged) and the two new tests.

- [ ] **Step 5: Commit**

```bash
git add src/daikonstudio/domain/execution/run.py tests/unit/execution/test_run.py
git commit -m "fix: allow RUNNING->RUNNING restart on Run.start for queue redelivery"
```

---

### Task 2: `run_job` drops redeliveries for terminal runs

**Files:**
- Modify: `src/daikonstudio/infrastructure/worker.py:124-137` (the known-gap `ponytail:` comment and the bare `run.start()` call) plus one import
- Test: `tests/unit/execution/test_worker.py`

**Interfaces:**
- Consumes: Task 1's `Run.start()` contract — `ConflictError` from `start()` now means "run is terminal", full stop.
- Produces: `run_job(ctx, run_id)` — on a terminal run, returns normally without saving; on a RUNNING run, restarts and completes it. No signature change.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/execution/test_worker.py` (the `_stub_load_and_save` fixture and imports already exist in that file):

```python
async def test_run_job_restarts_a_running_row_after_redelivery(
    monkeypatch: pytest.MonkeyPatch, _stub_load_and_save: tuple[Run, list[str]]
) -> None:
    """A worker crash mid-job leaves the row RUNNING; arq's at-least-once
    delivery hands the same run_id to a fresh process. That redelivery must
    restart the run and complete it -- not raise outside the try/except and
    strand the row at RUNNING forever."""
    run, saved = _stub_load_and_save
    run.start()  # the dead attempt got this far before its process died

    async def ok(ctx: dict[str, Any], run: Run) -> str:
        return "blob://result"

    monkeypatch.setitem(worker._HANDLERS, RunKind.TRAINING, ok)

    await worker.run_job({}, run.id)

    assert run.status.value == "ready"
    assert saved == ["running", "ready"]


async def test_run_job_drops_a_redelivery_for_a_terminal_run(
    _stub_load_and_save: tuple[Run, list[str]],
) -> None:
    """Cancelled while queued: the redelivered job is nobody's work anymore.
    run_job must return normally (so arq does not retry) without touching the
    row -- no save, no status change, no exception."""
    run, saved = _stub_load_and_save
    run.cancel()

    await worker.run_job({}, run.id)

    assert run.status.value == "cancelled"
    assert saved == []
```

- [ ] **Step 2: Run the new tests to verify the second fails**

Run: `uv run pytest tests/unit/execution/test_worker.py -v -k "redelivery or terminal_run"`
Expected: the first test PASSES already (Task 1's `start()` relaxation covers the restart path; it stays as an end-to-end guard through `run_job`). The second FAILS with `ConflictError: Cannot start run ... in status 'cancelled'` escaping `run_job` — that failure is the behavior this task changes.

- [ ] **Step 3: Implement the drop**

In `src/daikonstudio/infrastructure/worker.py`, add to the existing import block:

```python
from daikonstudio.domain.shared.errors import ConflictError
```

Then replace lines 124-137 (from `run = await _load(ctx, run_id)` through `await _save(ctx, run)`, including the whole `ponytail:` known-gap comment) with:

```python
    run = await _load(ctx, run_id)
    # arq is at-least-once: a worker crash mid-job redelivers this run_id with
    # the row already RUNNING, and start() treats that as a restart-from-zero
    # (see Run.start). A ConflictError here therefore means the run went
    # terminal while queued -- cancelled, most likely -- so the redelivered
    # job is nobody's work anymore: return without saving, and without
    # raising, so arq marks the job done instead of retrying it.
    try:
        run.start()
    except ConflictError:
        return
    await _save(ctx, run)
```

Finally, the `run_job` docstring's `SystemExit` paragraph cross-references the comment being deleted. In that docstring (around line 117-119), replace:

```
    guarantee the row is persisted as `FAILED` before the process dies, which
    is strictly better than the alternative (see the note on `run.start()`
    below for what "not catching it" would leave behind instead). Upgrade
```

with:

```
    guarantee the row is persisted as `FAILED` before the process dies, rather
    than leaving it RUNNING until arq redelivers and restarts the whole job
    from zero (see the comment on `run.start()` below). Upgrade
```

- [ ] **Step 4: Run the worker tests**

Run: `uv run pytest tests/unit/execution/test_worker.py -v`
Expected: all 8 PASS (6 pre-existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/daikonstudio/infrastructure/worker.py tests/unit/execution/test_worker.py
git commit -m "fix: run_job drops redeliveries for terminal runs instead of stranding them"
```

---

### Task 3: Manifest JSON round-trip tripwire

**Files:**
- Test: `tests/unit/engines/test_engine_contract.py` (append one test; no source change)

**Interfaces:**
- Consumes: `default_registry()` from `daikonstudio.infrastructure.engines.registry`, `EngineRegistry.manifests() -> list[EngineManifest]`. `EngineManifest` and `ConditionSpec` are frozen dataclasses; `TaskType`/`ConditionType` are `StrEnum`s.

- [ ] **Step 1: Write the tripwire test**

Append to `tests/unit/engines/test_engine_contract.py`:

```python
def test_every_registered_manifest_round_trips_through_json():
    """The manifest is the Phase 5 HTTP envelope: engines move out of process
    by *serving* their manifest instead of being imported (manifest.py's
    docstring). That exit stays open only while every manifest field is plain
    data. This trips the moment someone adds a non-serializable field -- an
    infrastructure object, a callable, a custom type -- to EngineManifest or
    ConditionSpec. If it fails, fix the field, not this test.
    (Spec: docs/superpowers/specs/2026-07-30-future-seams-alignment-design.md)
    """
    import json
    from dataclasses import asdict

    from daikonstudio.infrastructure.engines.registry import default_registry

    manifests = default_registry().manifests()
    assert manifests, "registry unexpectedly empty"
    for manifest in manifests:
        payload = json.loads(json.dumps(asdict(manifest)))
        assert payload["id"] == manifest.id
        assert payload["tasks"] == [task.value for task in manifest.tasks]
        for spec, raw in zip(manifest.conditions, payload["conditions"], strict=True):
            assert raw["key"] == spec.key
            assert raw["type"] == spec.type.value
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/unit/engines/test_engine_contract.py -v -k round_trips`
Expected: PASS. (This is a tripwire over an already-true property, not red-green TDD: it exists to fail *later*, when someone breaks the property.)

- [ ] **Step 3: Run the full backend suite**

Run: `uv run pytest`
Expected: everything green. If anything unrelated fails, stop and report — do not fix drive-by failures in this branch.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/engines/test_engine_contract.py
git commit -m "test: tripwire - every registered engine manifest must round-trip through JSON"
```
