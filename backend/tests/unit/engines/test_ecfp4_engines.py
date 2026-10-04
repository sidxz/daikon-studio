import math

import polars as pl
import pytest

from daikonstudio.application.engines.context import PredictContext, TrainContext
from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.infrastructure.engines.ecfp4_randomforest import Ecfp4RandomForest
from daikonstudio.infrastructure.engines.ecfp4_xgboost import Ecfp4XGBoost

SMILES = [
    "CCO",
    "CCCO",
    "CCCCO",
    "c1ccccc1",
    "Cc1ccccc1",
    "CCc1ccccc1",
    "CCN",
    "CCCN",
    "CCCCN",
    "c1ccncc1",
    "Cc1ccncc1",
    "CCc1ccncc1",
]
VALUES = [1.0, 1.2, 1.4, 5.0, 5.2, 5.4, 2.0, 2.2, 2.4, 6.0, 6.2, 6.4]
SPLIT = ["train"] * 8 + ["test"] * 4


def frame() -> pl.DataFrame:
    return pl.DataFrame({"smiles": SMILES, "y": VALUES, "split": SPLIT})


@pytest.mark.parametrize("engine", [Ecfp4XGBoost(), Ecfp4RandomForest()])
def test_train_returns_artifact_and_metrics(engine):
    ctx = TrainContext(
        frame=frame(),
        targets={"y": TaskType.REGRESSION},
        structure_column="smiles",
        conditions={},
        seed=42,
    )
    result = engine.train(ctx)
    assert isinstance(result.artifact, bytes) and len(result.artifact) > 0
    assert "rmse" in result.metrics["y"]


@pytest.mark.parametrize("engine", [Ecfp4XGBoost(), Ecfp4RandomForest()])
def test_predict_returns_one_row_per_input(engine):
    ctx = TrainContext(
        frame=frame(),
        targets={"y": TaskType.REGRESSION},
        structure_column="smiles",
        conditions={},
        seed=42,
    )
    artifact = engine.train(ctx).artifact
    predictions = engine.predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": ["CCO", "c1ccccc1"]}),
            structure_column="smiles",
            artifact=artifact,
            conditions={},
            target_columns=("y",),
        )
    )
    assert predictions.height == 2
    assert set(predictions.columns) == {"row_id", "value", "uncertainty"}


def test_training_is_reproducible_from_the_seed():
    ctx = TrainContext(
        frame=frame(),
        targets={"y": TaskType.REGRESSION},
        structure_column="smiles",
        conditions={},
        seed=42,
    )
    first = Ecfp4RandomForest().train(ctx).metrics["y"]["rmse"]
    second = Ecfp4RandomForest().train(ctx).metrics["y"]["rmse"]
    assert first == second


def test_a_regression_target_of_only_zeros_and_ones_still_trains_a_regressor():
    """Guards the sniffing bug: task comes from TargetSpec, never from the values."""
    binary_looking = pl.DataFrame(
        {
            "smiles": SMILES,
            "y": [0.0, 1.0] * 6,
            "split": SPLIT,
        }
    )
    ctx = TrainContext(
        frame=binary_looking,
        targets={"y": TaskType.REGRESSION},
        structure_column="smiles",
        conditions={},
        seed=42,
    )
    assert "rmse" in Ecfp4RandomForest().train(ctx).metrics["y"]


def test_random_forest_is_flagged_as_the_baseline():
    assert Ecfp4RandomForest.manifest().is_baseline is True
    assert Ecfp4XGBoost.manifest().is_baseline is False


@pytest.mark.parametrize("engine", [Ecfp4XGBoost(), Ecfp4RandomForest()])
def test_single_class_train_split_reports_undefined_metrics_not_a_crash(engine):
    """A single-class TRAIN split makes predict_proba return one column, not two --
    guards the IndexError this used to raise in both _score and predict(), and
    checks that every classification metric comes back uniformly undefined rather
    than a mix of NaN and a misleadingly confident number."""
    single_class_train = pl.DataFrame(
        {
            "smiles": SMILES[:10],
            "y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "split": ["train"] * 8 + ["test"] * 2,
        }
    )
    ctx = TrainContext(
        frame=single_class_train,
        targets={"y": TaskType.BINARY_CLASSIFICATION},
        structure_column="smiles",
        conditions={},
        seed=42,
    )
    result = engine.train(ctx)
    assert set(result.metrics["y"]) == {"mcc", "balanced_accuracy", "auroc", "auprc"}
    assert all(math.isnan(value) for value in result.metrics["y"].values())
    assert "accuracy" not in result.metrics["y"]

    predictions = engine.predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": SMILES[:3]}),
            structure_column="smiles",
            artifact=result.artifact,
            conditions={},
            target_columns=("y",),
        )
    )
    assert predictions.height == 3
