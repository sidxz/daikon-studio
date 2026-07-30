"""Unit coverage for the worker's exception-handling contract: a caught
failure must be recorded on the Run before re-raising (`Exception`,
`SystemExit`), and `CancelledError`/`KeyboardInterrupt` must reach the caller
untouched so process shutdown still works -- both in a real arq worker
(`run_job`) and in `InlineEnqueuer`'s dev-mode path.

Uses a fake Run/save-log rather than a real database: what's under test here
is control flow (which branch runs, does it re-raise), not persistence, which
tests/integration/test_run_repository.py already covers.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.infrastructure import worker


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

    monkeypatch.setattr(worker, "_load", fake_load)
    monkeypatch.setattr(worker, "_save", fake_save)
    return run, saved


async def test_run_job_records_failure_and_reraises(
    monkeypatch: pytest.MonkeyPatch, _stub_load_and_save: tuple[Run, list[str]]
) -> None:
    run, saved = _stub_load_and_save

    async def boom(ctx: dict[str, Any], run: Run) -> str:
        raise ValueError("engine exploded")

    monkeypatch.setitem(worker._HANDLERS, RunKind.TRAINING, boom)

    with pytest.raises(ValueError, match="engine exploded"):
        await worker.run_job({}, run.id)

    assert run.status.value == "failed"
    assert run.error_message == "ValueError('engine exploded')"
    assert saved == ["running", "failed"]


async def test_run_job_catches_system_exit_and_records_failure(
    monkeypatch: pytest.MonkeyPatch, _stub_load_and_save: tuple[Run, list[str]]
) -> None:
    run, saved = _stub_load_and_save

    async def boom(ctx: dict[str, Any], run: Run) -> str:
        raise SystemExit("no organism resolved")

    monkeypatch.setitem(worker._HANDLERS, RunKind.TRAINING, boom)

    with pytest.raises(SystemExit):
        await worker.run_job({}, run.id)

    assert run.status.value == "failed"
    assert saved == ["running", "failed"]


async def test_run_job_lets_cancelled_error_propagate_uncaught(
    monkeypatch: pytest.MonkeyPatch, _stub_load_and_save: tuple[Run, list[str]]
) -> None:
    run, saved = _stub_load_and_save

    async def boom(ctx: dict[str, Any], run: Run) -> str:
        raise asyncio.CancelledError()

    monkeypatch.setitem(worker._HANDLERS, RunKind.TRAINING, boom)

    with pytest.raises(asyncio.CancelledError):
        await worker.run_job({}, run.id)

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

    monkeypatch.setitem(worker._HANDLERS, RunKind.TRAINING, boom)

    with pytest.raises(KeyboardInterrupt):
        await worker.run_job({}, run.id)

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

    monkeypatch.setattr(worker, "run_job", failing_run_job)
    enqueuer = worker.InlineEnqueuer(sessions=None, store=None)  # type: ignore[arg-type]

    await enqueuer.enqueue(uuid.uuid4())  # must not raise


async def test_inline_enqueuer_lets_cancelled_error_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def cancelled_run_job(ctx: dict[str, Any], run_id: uuid.UUID) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(worker, "run_job", cancelled_run_job)
    enqueuer = worker.InlineEnqueuer(sessions=None, store=None)  # type: ignore[arg-type]

    with pytest.raises(asyncio.CancelledError):
        await enqueuer.enqueue(uuid.uuid4())


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
