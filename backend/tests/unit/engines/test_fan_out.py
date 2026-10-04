import pickle
import zipfile
from dataclasses import replace
from io import BytesIO

import polars as pl
import pytest

from daikonstudio.application.engines.checkpoints import Checkpoints
from daikonstudio.application.engines.context import (
    PredictContext,
    RunInterrupted,
    TrainContext,
    TrainResult,
)
from daikonstudio.application.engines.fan_out import FanOut
from daikonstudio.application.engines.manifest import EngineManifest, TaskType
from daikonstudio.application.engines.registry import EngineRegistry
from tests.fakes.blob_store import InMemoryBlobStore

_SINGLE = EngineManifest(
    id="single", version="1", name="Single", description="d", tasks=tuple(TaskType)
)
_JOINT = EngineManifest(
    id="joint",
    version="1",
    name="Joint",
    description="d",
    tasks=tuple(TaskType),
    supports_multitask=True,
)


class _Recorder:
    """Trains by remembering what it was asked; its artifact names the target."""

    def __init__(self, interrupt_on: int | None = None) -> None:
        self.contexts: list[TrainContext] = []
        self._interrupt_on = interrupt_on

    @staticmethod
    def manifest() -> EngineManifest:
        return _SINGLE

    def train(self, ctx: TrainContext) -> TrainResult:
        self.contexts.append(ctx)
        if self._interrupt_on == len(self.contexts):
            raise RunInterrupted("cancelled", cancelled=True)
        for fraction in (0.0, 0.5, 1.0):
            ctx.report(fraction, "fitting")
        column = ctx.target_column  # raises if FanOut handed it several
        return TrainResult(
            artifact=pickle.dumps({"column": column, "task": ctx.task.value}),
            metrics={column: {"rmse": float(len(self.contexts))}},
            validation_metrics={column: {"rmse": 0.0}},
            cutoffs={column: 0.3} if ctx.tune_cutoffs else None,
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        trained_on = pickle.loads(ctx.artifact)["column"]
        return pl.DataFrame(
            {
                "row_id": pl.Series(range(ctx.frame.height), dtype=pl.Int64),
                "value": pl.Series([float(len(trained_on))] * ctx.frame.height),
                "uncertainty": pl.Series([None] * ctx.frame.height, dtype=pl.Float64),
            }
        )


def _ctx(targets: dict[str, TaskType], report=lambda fraction, phase: None) -> TrainContext:
    return TrainContext(
        frame=pl.DataFrame({"smiles": ["C"]}),
        targets=targets,
        structure_column="smiles",
        conditions={},
        seed=1,
        report=report,
    )


def _predict(engine: FanOut, artifact: bytes, columns: tuple[str, ...]) -> pl.DataFrame:
    return engine.predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": ["C", "CC"]}),
            structure_column="smiles",
            artifact=artifact,
            conditions={},
            target_columns=columns,
        )
    )


def test_one_target_stores_the_inner_artifact_verbatim_and_still_tags_rows():
    inner = _Recorder()
    result = FanOut(inner).train(_ctx({"y": TaskType.REGRESSION}))
    assert pickle.loads(result.artifact)["column"] == "y"  # bare bytes, no container
    predictions = _predict(FanOut(inner), result.artifact, ("y",))
    assert predictions["target"].to_list() == ["y", "y"]


def test_several_targets_round_trip_through_one_container():
    columns = ("aggregator", "/odd {name}", "reactive")
    inner = _Recorder()
    result = FanOut(inner).train(_ctx(dict.fromkeys(columns, TaskType.BINARY_CLASSIFICATION)))
    assert zipfile.is_zipfile(BytesIO(result.artifact))
    predictions = _predict(FanOut(inner), result.artifact, columns)
    for column in columns:
        rows = predictions.filter(pl.col("target") == column)
        # each target's rows came from its own sub-model
        assert rows["value"].to_list() == [float(len(column))] * 2


def test_metrics_nest_per_target_in_order():
    result = FanOut(_Recorder()).train(
        _ctx({"a": TaskType.REGRESSION, "b": TaskType.BINARY_CLASSIFICATION})
    )
    assert list(result.metrics) == ["a", "b"]
    assert result.validation_metrics == {"a": {"rmse": 0.0}, "b": {"rmse": 0.0}}


def test_each_sub_fit_sees_one_target_and_that_targets_own_task():
    inner = _Recorder()
    FanOut(inner).train(_ctx({"a": TaskType.REGRESSION, "b": TaskType.BINARY_CLASSIFICATION}))
    assert [ctx.targets for ctx in inner.contexts] == [
        {"a": TaskType.REGRESSION},
        {"b": TaskType.BINARY_CLASSIFICATION},
    ]


def test_progress_spans_zero_to_one_once_across_all_sub_fits():
    seen: list[float] = []
    FanOut(_Recorder()).train(
        _ctx(dict.fromkeys("abcd", TaskType.REGRESSION), report=lambda f, _: seen.append(f))
    )
    assert seen[0] == 0.0 and seen[-1] == 1.0
    assert seen == sorted(seen)


def test_an_interruption_in_the_second_of_four_fits_propagates_and_stops_the_rest():
    inner = _Recorder(interrupt_on=2)
    with pytest.raises(RunInterrupted):
        FanOut(inner).train(_ctx(dict.fromkeys("abcd", TaskType.REGRESSION)))
    assert len(inner.contexts) == 2


def test_predicting_with_targets_the_container_was_not_built_for_fails_loudly():
    result = FanOut(_Recorder()).train(_ctx(dict.fromkeys("ab", TaskType.REGRESSION)))
    with pytest.raises(ValueError, match="trained on"):
        _predict(FanOut(_Recorder()), result.artifact, ("a", "c"))


def test_the_registry_wraps_only_engines_that_cannot_learn_targets_jointly():
    class _Joint(_Recorder):
        @staticmethod
        def manifest() -> EngineManifest:
            return _JOINT

    joint = _Joint()
    registry = EngineRegistry({"single": _Recorder(), "joint": joint})
    assert isinstance(registry.get("single"), FanOut)
    assert registry.get("joint") is joint
    assert registry.get("single").manifest() is _SINGLE


def test_cutoffs_merge_per_target_and_tune_cutoffs_reaches_every_sub_fit():
    inner = _Recorder()
    result = FanOut(inner).train(
        replace(
            _ctx({"a": TaskType.BINARY_CLASSIFICATION, "b": TaskType.BINARY_CLASSIFICATION}),
            tune_cutoffs=True,
        )
    )
    assert result.cutoffs == {"a": 0.3, "b": 0.3}
    assert all(ctx.tune_cutoffs for ctx in inner.contexts)


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
