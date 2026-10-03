"""`RunTraining`'s cooperative reporter -- the only thing that can stop a fit.

An engine runs on a `to_thread` worker thread, and Python cannot kill a thread from
outside. So everything that stops work in this codebase happens inside `report`, called
by the engine, on the engine's own stack. These tests pin the three properties that
make that work: the deadline is checked without I/O so it fires promptly, a cancelled
row raises rather than returning, and progress lands inside the leg's span.
"""

from __future__ import annotations

import asyncio
import copy
import time
import uuid

import pytest

from daikonstudio.application.engines.context import RunInterrupted
from daikonstudio.application.execution.train_protocol import RunTraining
from daikonstudio.domain.execution.run import Run, RunKind


class _Rows:
    """A RunRepository stand-in that records what the checkpoint wrote."""

    def __init__(self, run: Run) -> None:
        self.row = run
        self.updates: list[tuple[float, str | None]] = []

    async def get_by_id(self, run_id: uuid.UUID) -> Run | None:
        return self.row

    async def update(self, run: Run) -> None:
        self.updates.append((run.progress, run.phase))


def _running_run() -> Run:
    run = Run(
        kind=RunKind.TRAINING,
        workspace_id=uuid.uuid4(),
        requested_by=uuid.uuid4(),
        cache_key="k",
    )
    run.start()
    return run


def _training(rows: _Rows) -> RunTraining:
    # Only the run repository is reachable from the reporter; the rest of the
    # collaborators belong to legs these tests never run.
    return RunTraining(None, None, rows, None, None, None)  # type: ignore[arg-type]


async def test_an_overrun_deadline_stops_the_fit_inside_the_throttle_window() -> None:
    """The deadline needs no I/O, so it is checked *before* the throttle. Checking it
    after would leave an overrunning fit going until the next 10s checkpoint."""
    run = _running_run()
    rows = _Rows(run)
    training = _training(rows)
    report = training._reporter(run, (0.0, 0.6))

    await asyncio.to_thread(report, 0.0, "epoch 1")  # writes, closing the window
    training._deadline_seconds = 1800.0
    training._deadline_at = time.monotonic() - 1

    with pytest.raises(RunInterrupted) as raised:
        await asyncio.to_thread(report, 0.1, "epoch 2")

    assert raised.value.cancelled is False
    # The throttle would have skipped the second call entirely; the deadline did not.
    assert rows.updates == [(0.0, "epoch 1")]


async def test_a_cancelled_row_raises_in_the_engines_own_call_stack() -> None:
    """Cancellation happens in the API process, against a different instance of the
    row. Re-reading it is the only way this worker can ever find out."""
    run = _running_run()
    cancelled_elsewhere = copy.deepcopy(run)
    cancelled_elsewhere.cancel()
    rows = _Rows(cancelled_elsewhere)
    report = _training(rows)._reporter(run, (0.0, 0.6))

    with pytest.raises(RunInterrupted) as raised:
        await asyncio.to_thread(report, 0.5, "epoch 1")

    assert raised.value.cancelled is True
    assert rows.updates == []  # a run nobody wants gets no progress written


async def test_a_vanished_row_also_stops_the_fit() -> None:
    run = _running_run()
    rows = _Rows(run)
    rows.row = None  # type: ignore[assignment]

    with pytest.raises(RunInterrupted):
        await asyncio.to_thread(_training(rows)._reporter(run, (0.0, 0.6)), 0.5, "epoch 1")


async def test_progress_is_mapped_onto_the_legs_own_span() -> None:
    """An engine reports 0..1 within *its* fit; the worker owns where that sits in the
    overall run, so a second leg cannot rewind the bar to zero."""
    run = _running_run()
    rows = _Rows(run)
    report = _training(rows)._reporter(run, (0.6, 0.7))

    await asyncio.to_thread(report, 0.5, "epoch 1")

    assert rows.updates == [(pytest.approx(0.65), "epoch 1")]


async def test_an_out_of_range_fraction_is_clamped_to_the_span() -> None:
    run = _running_run()
    rows = _Rows(run)
    report = _training(rows)._reporter(run, (0.0, 0.6))

    await asyncio.to_thread(report, 7.0, "an engine that miscounted its epochs")

    assert rows.updates == [(0.6, "an engine that miscounted its epochs")]


async def test_progress_between_fits_honours_the_deadline():
    """Tree and GP engines never call ctx.report, so the only deadline check they
    can ever hit is the one between fits. Before this, a hung ECFP4 fit followed
    by a baseline fit ran both to completion against an expired deadline."""
    training = RunTraining(*([None] * 6), deadline_seconds=1)
    training._deadline_at = time.monotonic() - 1
    run = Run(
        kind=RunKind.TRAINING, workspace_id=uuid.uuid4(), requested_by=uuid.uuid4(), cache_key="k"
    )
    run.start()
    with pytest.raises(RunInterrupted) as raised:
        await training._progress(run, 0.6, "training baseline")
    assert raised.value.cancelled is False
    assert "time limit" in raised.value.reason
