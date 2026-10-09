"""FanOut already rebuilds the context per target; it must rebuild the frame too.

This is the whole of the sparse-label mask for the seven engines that do not declare
`supports_multitask`. They never see a null, so neither do the five `_scoring.py`
call sites inside them, and none of them needed changing.
"""

from __future__ import annotations

import polars as pl

from daikonstudio.application.engines.context import TrainContext, TrainResult
from daikonstudio.application.engines.fan_out import FanOut
from daikonstudio.application.engines.manifest import EngineManifest, TaskType


class _Recorder:
    """An engine that records the frame it was handed, per target."""

    def __init__(self) -> None:
        self.seen: dict[str, pl.DataFrame] = {}

    def manifest(self) -> EngineManifest:
        return EngineManifest(
            id="recorder",
            version="1",
            name="Recorder",
            description="",
            tasks=(TaskType.BINARY_CLASSIFICATION,),
        )

    def train(self, ctx: TrainContext) -> TrainResult:
        self.seen[ctx.target_column] = ctx.frame
        return TrainResult(artifact=b"", metrics={ctx.target_column: {"mcc": 1.0}})


def _context(frame: pl.DataFrame, targets: dict[str, TaskType]) -> TrainContext:
    return TrainContext(
        frame=frame,
        targets=targets,
        structure_column="smiles",
        conditions={},
        seed=42,
    )


def test_each_target_sees_only_its_labelled_rows() -> None:
    frame = pl.DataFrame(
        {
            "smiles": ["CCO", "c1ccccc1", "CCN", "CCC"],
            "a": [1.0, 0.0, None, None],
            "b": [None, 1.0, 0.0, 1.0],
            "split": ["train"] * 4,
        }
    )
    inner = _Recorder()

    FanOut(inner).train(
        _context(
            frame,
            {"a": TaskType.BINARY_CLASSIFICATION, "b": TaskType.BINARY_CLASSIFICATION},
        )
    )

    assert inner.seen["a"].height == 2
    assert inner.seen["b"].height == 3
    assert inner.seen["a"]["a"].null_count() == 0
    assert inner.seen["b"]["b"].null_count() == 0


def test_a_target_keeps_the_other_targets_nulls() -> None:
    """Only the fitted target's nulls decide which rows go: a row measured for `a` is
    kept for `a`'s fit even though `b` is blank on it, because `b` is not what is being
    learned. Filtering on any null would restore the intersection the ingestion gate
    just stopped doing."""
    frame = pl.DataFrame(
        {
            "smiles": ["CCO", "c1ccccc1"],
            "a": [1.0, 0.0],
            "b": [None, None],
            "split": ["train"] * 2,
        }
    )
    inner = _Recorder()

    FanOut(inner).train(
        _context(
            frame,
            {"a": TaskType.BINARY_CLASSIFICATION, "b": TaskType.BINARY_CLASSIFICATION},
        )
    )

    assert inner.seen["a"].height == 2


def test_a_dense_frame_is_passed_through_unchanged() -> None:
    """Every dataset frozen before this feature is dense; none of them may lose a row
    to a filter that has nothing to remove."""
    frame = pl.DataFrame({"smiles": ["CCO", "CCN"], "a": [1.0, 0.0], "split": ["train", "train"]})
    inner = _Recorder()

    FanOut(inner).train(_context(frame, {"a": TaskType.BINARY_CLASSIFICATION}))

    assert inner.seen["a"].height == 2
