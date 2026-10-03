"""Unit coverage for `run_job`'s exception-handling contract: a caught
failure must be recorded on the Run before re-raising (`Exception`,
`SystemExit`), and `CancelledError`/`KeyboardInterrupt` must reach the caller
untouched so process shutdown still works -- both directly (`run_job`) and in
`InlineEnqueuer`'s dev-mode path.

Uses a fake Run/save-log rather than a real database: what's under test here
is control flow (which branch runs, does it re-raise), not persistence, which
tests/integration/test_run_repository.py already covers.
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from typing import Any

import pytest

from daikonstudio.application.engines.context import RunInterrupted
from daikonstudio.domain.execution.run import Run, RunKind, RunStatus
from daikonstudio.infrastructure import jobs


def _pending_run() -> Run:
    return Run(
        kind=RunKind.TRAINING,
        workspace_id=uuid.uuid4(),
        requested_by=uuid.uuid4(),
        cache_key="k",
    )


@pytest.fixture
def _stub_load_and_save(monkeypatch: pytest.MonkeyPatch) -> tuple[Run, list[str]]:
    """Route `_load`/`_save` through an in-memory holder instead of a real
    repository -- `run_job` only needs *some* Run object to mutate."""
    run = _pending_run()
    saved: list[str] = []

    async def fake_load(ctx: dict[str, Any], run_id: uuid.UUID) -> Run:
        return run

    async def fake_save(ctx: dict[str, Any], saved_run: Run) -> None:
        saved.append(saved_run.status.value)

    monkeypatch.setattr(jobs, "_load", fake_load)
    monkeypatch.setattr(jobs, "_save", fake_save)
    return run, saved


async def test_run_job_records_failure_and_reraises(
    monkeypatch: pytest.MonkeyPatch, _stub_load_and_save: tuple[Run, list[str]]
) -> None:
    run, saved = _stub_load_and_save

    async def boom(ctx: dict[str, Any], run: Run) -> str:
        raise ValueError("engine exploded")

    monkeypatch.setitem(jobs._HANDLERS, RunKind.TRAINING, boom)

    with pytest.raises(ValueError, match="engine exploded"):
        await jobs.run_job({}, run.id)

    assert run.status.value == "failed"
    # Class and text, never repr(): the text is what a scientist can act on, and
    # a repr would carry a library's internal paths to every viewer of the run.
    assert run.error_message == "engine exploded"
    assert saved == ["running", "failed"]


async def test_run_job_catches_system_exit_and_records_failure(
    monkeypatch: pytest.MonkeyPatch, _stub_load_and_save: tuple[Run, list[str]]
) -> None:
    run, saved = _stub_load_and_save

    async def boom(ctx: dict[str, Any], run: Run) -> str:
        raise SystemExit("no organism resolved")

    monkeypatch.setitem(jobs._HANDLERS, RunKind.TRAINING, boom)

    with pytest.raises(SystemExit):
        await jobs.run_job({}, run.id)

    assert run.status.value == "failed"
    assert saved == ["running", "failed"]


async def test_run_job_lets_cancelled_error_propagate_uncaught(
    monkeypatch: pytest.MonkeyPatch, _stub_load_and_save: tuple[Run, list[str]]
) -> None:
    run, saved = _stub_load_and_save

    async def boom(ctx: dict[str, Any], run: Run) -> str:
        raise asyncio.CancelledError()

    monkeypatch.setitem(jobs._HANDLERS, RunKind.TRAINING, boom)

    with pytest.raises(asyncio.CancelledError):
        await jobs.run_job({}, run.id)

    # fail() was never called: the run is still RUNNING, and _save only ran
    # once (the initial start()), not a second time for a recorded failure.
    assert run.status.value == "running"
    assert saved == ["running"]


async def test_run_job_lets_keyboard_interrupt_propagate_uncaught(
    monkeypatch: pytest.MonkeyPatch, _stub_load_and_save: tuple[Run, list[str]]
) -> None:
    run, saved = _stub_load_and_save

    async def boom(ctx: dict[str, Any], run: Run) -> str:
        raise KeyboardInterrupt()

    monkeypatch.setitem(jobs._HANDLERS, RunKind.TRAINING, boom)

    with pytest.raises(KeyboardInterrupt):
        await jobs.run_job({}, run.id)

    assert run.status.value == "running"
    assert saved == ["running"]


async def test_inline_enqueuer_swallows_a_handler_failure_after_recording_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`run_job` (stubbed here) already recorded the failure on the Run and
    re-raised; `InlineEnqueuer.enqueue()` must not let that reach an HTTP
    caller that only ever expects the async contract (202 now, poll later)."""

    async def failing_run_job(ctx: dict[str, Any], run_id: uuid.UUID) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(jobs, "run_job", failing_run_job)
    enqueuer = jobs.InlineEnqueuer(sessions=None, store=None)  # type: ignore[arg-type]

    await enqueuer.enqueue(uuid.uuid4())  # must not raise


async def test_inline_enqueuer_lets_cancelled_error_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def cancelled_run_job(ctx: dict[str, Any], run_id: uuid.UUID) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(jobs, "run_job", cancelled_run_job)
    enqueuer = jobs.InlineEnqueuer(sessions=None, store=None)  # type: ignore[arg-type]

    with pytest.raises(asyncio.CancelledError):
        await enqueuer.enqueue(uuid.uuid4())


async def test_db_enqueuer_sets_lane() -> None:
    calls: list[tuple[uuid.UUID, str]] = []

    class FakeQueue:
        async def set_lane(self, run_id: uuid.UUID, lane: str) -> None:
            calls.append((run_id, lane))

    run_id = uuid.uuid4()
    await jobs.DbEnqueuer(FakeQueue()).enqueue(run_id, lane="gpu")  # type: ignore[arg-type]

    assert calls == [(run_id, "gpu")]


async def test_run_job_restarts_a_running_row_after_redelivery(
    monkeypatch: pytest.MonkeyPatch, _stub_load_and_save: tuple[Run, list[str]]
) -> None:
    """A process crash mid-job leaves the row RUNNING; at-least-once delivery
    hands the same run_id to a fresh process. That redelivery must restart the
    run and complete it -- not raise outside the try/except and strand the
    row at RUNNING forever."""
    run, saved = _stub_load_and_save
    run.start()  # the dead attempt got this far before its process died

    async def ok(ctx: dict[str, Any], run: Run) -> str:
        return "blob://result"

    monkeypatch.setitem(jobs._HANDLERS, RunKind.TRAINING, ok)

    await jobs.run_job({}, run.id)

    assert run.status.value == "ready"
    assert saved == ["running", "ready"]


async def test_run_job_drops_a_redelivery_for_a_terminal_run(
    _stub_load_and_save: tuple[Run, list[str]],
) -> None:
    """Cancelled while queued: the redelivered job is nobody's work anymore.
    run_job must return normally (so the caller does not retry it) without
    touching the row -- no save, no status change, no exception."""
    run, saved = _stub_load_and_save
    run.cancel()

    await jobs.run_job({}, run.id)

    assert run.status.value == "cancelled"
    assert saved == []


def _ctx_with(run: Run, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A ctx whose `_load`/`_save` round-trip through an in-memory row store.

    Unlike `_stub_load_and_save` above, every `_load` builds a *fresh* Run from the
    stored row the way the real repository does. That is the whole point of the
    cancellation tests below: a cancellation happens in the API process against a
    different instance of the row entirely, so `run_job`'s own aggregate can never
    see it and re-reading is the only thing that can.
    """
    ctx: dict[str, Any] = {"rows": {run.id: copy.deepcopy(run)}}

    async def fake_load(ctx: dict[str, Any], run_id: uuid.UUID) -> Run:
        return copy.deepcopy(ctx["rows"][run_id])  # type: ignore[no-any-return]

    async def fake_save(ctx: dict[str, Any], saved_run: Run) -> None:
        ctx["rows"][saved_run.id] = copy.deepcopy(saved_run)

    monkeypatch.setattr(jobs, "_load", fake_load)
    monkeypatch.setattr(jobs, "_save", fake_save)
    return ctx


def _reload(ctx: dict[str, Any], run_id: uuid.UUID) -> Run:
    return ctx["rows"][run_id]  # type: ignore[no-any-return]


async def _cancel_in_the_database(ctx: dict[str, Any], run_id: uuid.UUID) -> None:
    """What `POST /runs/{id}/cancel` does: flips the row, in another process, on an
    instance of the aggregate `run_job` is not holding."""
    ctx["rows"][run_id].cancel()


async def test_a_cancelled_run_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_job` must not re-raise RunInterrupted. A propagating exception is how a
    redelivering caller would come to retry a job, and at GPU durations retrying a
    deliberately-stopped fit is how one cancelled run becomes several training threads
    on one device."""
    run = _pending_run()
    ctx = _ctx_with(run, monkeypatch)

    async def _interrupt(_ctx: dict[str, Any], _run: Run) -> str:
        # The order the real path takes: the row is cancelled first, and `report`
        # noticing that is exactly why it raises.
        await _cancel_in_the_database(ctx, run.id)
        raise RunInterrupted("the run was cancelled", cancelled=True)

    monkeypatch.setitem(jobs._HANDLERS, run.kind, _interrupt)

    await jobs.run_job(ctx, run.id)  # must not raise

    assert _reload(ctx, run.id).status is RunStatus.CANCELLED


async def test_a_deadline_fails_the_run_with_its_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _pending_run()
    ctx = _ctx_with(run, monkeypatch)

    async def _interrupt(_ctx: dict[str, Any], _run: Run) -> str:
        raise RunInterrupted("exceeded the 60s deadline for this worker lane", cancelled=False)

    monkeypatch.setitem(jobs._HANDLERS, run.kind, _interrupt)

    await jobs.run_job(ctx, run.id)  # must not raise

    reloaded = _reload(ctx, run.id)
    assert reloaded.status is RunStatus.FAILED
    assert "deadline" in (reloaded.error_message or "")


async def test_a_run_cancelled_during_the_last_throttle_window_is_not_marked_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reporter only checks the row every 10 seconds. A cancellation landing inside
    that window would otherwise be overwritten with READY by a handler that finished."""
    run = _pending_run()
    ctx = _ctx_with(run, monkeypatch)

    async def _succeed_after_cancellation(_ctx: dict[str, Any], inflight: Run) -> str:
        await _cancel_in_the_database(ctx, inflight.id)
        return "blob://result"

    monkeypatch.setitem(jobs._HANDLERS, run.kind, _succeed_after_cancellation)

    await jobs.run_job(ctx, run.id)

    assert _reload(ctx, run.id).status is RunStatus.CANCELLED
