from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from daikonstudio.application.engines.manifest import TaskType

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


@dataclass(frozen=True, kw_only=True)
class TrainContext:
    """`frame` carries the dataset columns plus a `split` column of train/validation/test.

    `task` is passed explicitly and is authoritative. An engine must NEVER infer
    regression-vs-classification from the target values: a regression target whose
    values happen to all be 0.0 or 1.0 would silently train a classifier. The
    Dataset's TargetSpec is the only source of truth for what is being predicted.
    """

    frame: pl.DataFrame
    task: TaskType
    structure_column: str
    target_column: str
    conditions: dict[str, object]
    seed: int
    report: ProgressReporter = _no_op
    """Called periodically during long work to publish progress and to check whether
    the run is still wanted. `fraction` is progress within *this fit*, 0.0 to 1.0; the
    worker maps it onto the overall run. May raise `RunInterrupted` -- do not catch it.
    Calling it is optional; calling it often is what makes an engine stoppable."""


@dataclass(frozen=True, kw_only=True)
class TrainResult:
    """`metrics` are scored on the test partition; `validation_metrics` on the
    validation one, by the identical scoring code.

    The validation partition existed from the first split and, until this field,
    no engine read it and nothing downstream reported it -- ten percent of every
    dataset held out and spent on nothing. That is not merely wasteful, it is the
    hole that makes model selection dishonest: with no validation number to tune
    against, the only feedback a scientist has when choosing between two sets of
    conditions is the *test* score, and a test set consulted once per retrain
    stops being held out at all.

    `None` means the partition was empty (a split declared with a zero validation
    fraction), never that scoring was skipped -- an absent number and a zero are
    different claims and only one of them is true here.
    """

    artifact: bytes
    metrics: dict[str, float]
    validation_metrics: dict[str, float] | None = None


@dataclass(frozen=True, kw_only=True)
class PredictContext:
    frame: pl.DataFrame
    structure_column: str
    artifact: bytes
    conditions: dict[str, object]
