from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from daikonstudio.application.engines.manifest import TaskType

ProgressReporter = Callable[[float, str], None]

#: Fewer validation positives or negatives than this for a label and a cutoff tuned on
#: them fits noise: five actives pick whichever cutoff happens to separate those five.
#: Defined here, not beside the search in `_scoring.py`, because the training run quotes
#: it when it explains an untuned cutoff, and application code may not import engines.
#: ponytail: fixed at 10 per class; make it a training setting if a rarer label needs it.
MIN_CUTOFF_CLASS_COUNT = 10


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

    `targets` maps each target column to its task, in the Dataset's order, and is
    authoritative. An engine must NEVER infer regression-vs-classification from the
    target values: a regression target whose values happen to all be 0.0 or 1.0
    would silently train a classifier. The Dataset's TargetSpecs are the only source
    of truth for what is being predicted.

    `target_column` and `task` are the single-target reading every engine without
    `supports_multitask` uses, and they raise rather than guess when the context
    holds more than they can describe. Such an engine is never handed one: the
    registry wraps it in `FanOut`, which splits the context per target first. The
    raise is the loud form of that guarantee -- a first-element answer here would
    train on one target and silently drop the rest.
    """

    frame: pl.DataFrame
    targets: dict[str, TaskType]
    structure_column: str
    conditions: dict[str, object]
    seed: int
    # Choose each binary target's decision cutoff to maximize MCC on the validation
    # partition, and report the threshold-dependent metrics at it. The training run sets
    # the same value for the model, its baseline and the random-split comparison, so a
    # tuned model is never measured against an untuned baseline.
    tune_cutoffs: bool = False
    report: ProgressReporter = _no_op
    """Called periodically during long work to publish progress and to check whether
    the run is still wanted. `fraction` is progress within *this fit*, 0.0 to 1.0; the
    worker maps it onto the overall run. May raise `RunInterrupted` -- do not catch it.
    Calling it is optional; calling it often is what makes an engine stoppable."""

    @property
    def target_columns(self) -> tuple[str, ...]:
        return tuple(self.targets)

    @property
    def target_column(self) -> str:
        if len(self.targets) != 1:
            raise ValueError(
                f"This training context carries {len(self.targets)} targets "
                f"({', '.join(self.targets)}); `target_column` describes exactly one. An "
                "engine without `supports_multitask` must be wrapped in FanOut."
            )
        return next(iter(self.targets))

    @property
    def task(self) -> TaskType:
        tasks = set(self.targets.values())
        if len(tasks) != 1:
            raise ValueError(
                "The targets in this training context are of different kinds; `task` "
                "describes one. Only a uniform-kind dataset reaches a joint engine."
            )
        return tasks.pop()


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
    # Keyed by target column: {"solubility": {"rmse": 0.61, ...}}. A one-target fit
    # has exactly one key, so the shape is uniform.
    metrics: dict[str, dict[str, float]]
    validation_metrics: dict[str, dict[str, float]] | None = None
    # Each binary target's tuned cutoff, keyed by column: only targets whose cutoff was
    # tuned (`tune_cutoffs` on, and validation held enough of both classes).
    cutoffs: dict[str, float] | None = None


@dataclass(frozen=True, kw_only=True)
class PredictContext:
    frame: pl.DataFrame
    structure_column: str
    artifact: bytes
    conditions: dict[str, object]
    # The Protocol's targets, in order (`target_columns_of(protocol.readouts)`). A
    # joint engine labels its output rows with them; `FanOut` uses them to pick the
    # artifact for each target and to tag every row.
    target_columns: tuple[str, ...]
