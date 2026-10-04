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
from types import SimpleNamespace

import polars as pl
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


@pytest.mark.parametrize(
    ("scope", "phase"),
    [
        ("model", "Training Chemprop D-MPNN on mps:0"),
        ("baseline", "Baseline: Training Chemprop D-MPNN on mps:0"),
        ("random-split", "Random-split comparison: Training Chemprop D-MPNN on mps:0"),
    ],
)
async def test_the_progress_text_names_the_stage_that_is_not_the_model(
    scope: str, phase: str
) -> None:
    """With the baseline the same engine as the model, the engine's own text cannot
    say which of the three fits is running."""
    run = _running_run()
    rows = _Rows(run)
    report = _training(rows)._reporter(run, (0.6, 0.7), scope)

    await asyncio.to_thread(report, 0.5, "Training Chemprop D-MPNN on mps:0")

    assert rows.updates == [(pytest.approx(0.65), phase)]


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
    assert "STUDIO_WORKER_JOB_TIMEOUT_BY_LANE" in raised.value.reason


class _Layout:
    """A layout that records whether it was asked for a map."""

    def __init__(self) -> None:
        self.calls = 0

    def layout(self, structures: list[str], seed: int) -> tuple[list[float], list[float]]:
        self.calls += 1
        return [0.5] * len(structures), [0.5] * len(structures)

    def describe(self) -> dict[str, object]:
        return {"method": "test", "params": {}}


class _Store:
    def __init__(self) -> None:
        self.keys: list[str] = []

    def put_bytes(self, key: str, data: bytes) -> str:
        self.keys.append(key)
        return key


def _mapping_training(rows: _Rows, store: _Store, layout: _Layout) -> RunTraining:
    return RunTraining(None, None, rows, store, None, None, deadline_seconds=1, layout=layout)  # type: ignore[arg-type]


_FRAME = pl.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCCl", "CCBr"], "split": ["train"] * 5})
_DATASET = SimpleNamespace(structure_column="smiles", split=SimpleNamespace(seed=1))


async def test_a_deadline_past_during_the_last_fit_skips_the_map_instead_of_failing_the_run():
    """The map is drawn after the Protocol row exists. A tree or GP fit that ran past
    the soft deadline used to finish READY; checking the deadline again at the map
    would now fail a run whose Protocol is already listed. Skip the map instead."""
    run = _running_run()
    layout = _Layout()
    store = _Store()
    training = _mapping_training(_Rows(run), store, layout)
    training._deadline_at = time.monotonic() - 1

    await training._map_chemical_space(run, uuid.uuid4(), _FRAME, _DATASET)

    assert layout.calls == 0
    assert store.keys == []


async def test_within_the_deadline_the_map_is_drawn_and_its_phase_reported():
    run = _running_run()
    rows = _Rows(run)
    layout = _Layout()
    store = _Store()
    training = _mapping_training(rows, store, layout)
    training._deadline_at = time.monotonic() + 60

    await training._map_chemical_space(run, uuid.uuid4(), _FRAME, _DATASET)

    assert layout.calls == 1
    assert [key.rsplit("/", 1)[-1] for key in store.keys] == [
        "chemical-space.parquet",
        "chemical-space.json",
    ]
    assert rows.updates[-1] == (0.97, "Mapping chemical space")


async def test_a_failed_epoch_save_never_stops_the_training_run() -> None:
    """A chart is not worth a training run: the points are dropped, the fit goes on.
    `_Rows` has no `append_epochs`, which is as failed as a save gets."""
    from daikonstudio.application.engines.context import EpochPoint
    from daikonstudio.application.execution.train_protocol import _EpochBuffer

    run = _running_run()
    buffer = _EpochBuffer("model")
    buffer.record(EpochPoint(epoch=1, epochs=2, train_loss=0.5, val_loss=0.4, scores={}))

    await _training(_Rows(run))._flush_epochs(run, buffer)

    assert buffer.take() == []  # taken, then dropped


async def test_a_progress_write_that_never_answers_lets_the_fit_go_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API stalled (prod, 2026-10-04: a blocking dataset build) and the write
    waited past its limit. That cost a 26-minute fit; it must cost one progress write."""
    import daikonstudio.application.execution.train_protocol as module

    class _Stalled(_Rows):
        async def get_by_id(self, run_id: uuid.UUID) -> Run | None:
            await asyncio.sleep(60)
            return self.row

    monkeypatch.setattr(module, "_CHECKPOINT_TIMEOUT_SECONDS", 0.05)
    run = _running_run()
    report = _training(_Stalled(run))._reporter(run, (0.0, 0.6))

    await asyncio.to_thread(report, 0.5, "epoch 1")  # must not raise


async def test_a_studio_that_cannot_be_reached_lets_the_fit_go_on() -> None:
    """Mid-deploy: the runner's repository reports the studio as unavailable."""
    from daikonstudio.domain.shared.errors import ServiceUnavailableError

    class _Down(_Rows):
        async def get_by_id(self, run_id: uuid.UUID) -> Run | None:
            raise ServiceUnavailableError("The studio could not be reached")

    run = _running_run()
    report = _training(_Down(run))._reporter(run, (0.0, 0.6))

    await asyncio.to_thread(report, 0.5, "epoch 1")  # must not raise


async def test_a_refusal_still_ends_the_fit() -> None:
    """Not every failed write is passing: a version conflict means another writer
    holds the run, and training on would be wasted work."""
    from daikonstudio.domain.shared.errors import ConcurrencyConflictError

    class _Conflicted(_Rows):
        async def update(self, run: Run) -> None:
            raise ConcurrencyConflictError("Run", str(run.id))

    run = _running_run()
    report = _training(_Conflicted(run))._reporter(run, (0.0, 0.6))

    with pytest.raises(ConcurrencyConflictError):
        await asyncio.to_thread(report, 0.5, "epoch 1")
