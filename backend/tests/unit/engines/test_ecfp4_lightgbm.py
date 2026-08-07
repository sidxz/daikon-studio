"""The LightGBM engine's contract, and the one setting that is not decoration.

The shared metric vocabulary, the artifact bundle and the predict schema are covered
the same way as the other tree engines. What is specific here is `deterministic=True`:
LightGBM's multithreaded histogram build is not reproducible by default, so the
reproducibility test below is the only thing standing between a Scorecard and numbers
that drift between two runs of the same configuration.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from daikonstudio.application.engines.context import PredictContext, TrainContext
from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.infrastructure.engines.ecfp4_lightgbm import Ecfp4LightGBM

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

# LightGBM's default of 20 compounds per leaf cannot split an 8-row training set at
# all, which would make every test below pass against a constant model -- including
# the reproducibility one, vacuously. 1 forces real trees on a fixture this size.
CONDITIONS = {"min_child_samples": 1, "n_estimators": 20}


def frame(values: list[float] | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        {"smiles": SMILES, "y": VALUES if values is None else values, "split": SPLIT}
    )


def context(
    task: TaskType = TaskType.REGRESSION, rows: pl.DataFrame | None = None
) -> TrainContext:
    return TrainContext(
        frame=frame() if rows is None else rows,
        task=task,
        structure_column="smiles",
        target_column="y",
        conditions=CONDITIONS,
        seed=42,
    )


def test_regression_reports_the_shared_metric_vocabulary() -> None:
    result = Ecfp4LightGBM().train(context())

    assert sorted(result.metrics) == ["mae", "r2", "rmse"]
    assert isinstance(result.artifact, bytes) and len(result.artifact) > 0


def test_classification_reports_the_shared_metric_vocabulary() -> None:
    result = Ecfp4LightGBM().train(
        context(TaskType.BINARY_CLASSIFICATION, frame([float(i % 2) for i in range(12)]))
    )

    assert sorted(result.metrics) == ["auprc", "auroc", "balanced_accuracy", "mcc"]
    assert "accuracy" not in result.metrics


def test_predict_returns_the_contracted_schema_and_dtypes() -> None:
    """An all-None uncertainty column inferred as polars Null would make this engine's
    output schema-incompatible with every other engine's."""
    artifact = Ecfp4LightGBM().train(context()).artifact

    predictions = Ecfp4LightGBM().predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": ["CCO", "c1ccccc1"]}),
            structure_column="smiles",
            artifact=artifact,
            conditions={},
        )
    )

    assert predictions.columns == ["row_id", "value", "uncertainty"]
    assert predictions.schema["row_id"] == pl.Int64
    assert predictions.schema["value"] == pl.Float64
    assert predictions.schema["uncertainty"] == pl.Float64
    assert predictions.height == 2


def test_uncertainty_is_null_rather_than_a_fabricated_number() -> None:
    """Boosted trees are additive corrections to one running prediction, not
    independent estimates of it, so their spread is not an uncertainty. XGBoost
    reports None for the same reason; a triage grid plotting a fabricated value as
    confidence is worse than an admitted absent one."""
    artifact = Ecfp4LightGBM().train(context()).artifact

    predictions = Ecfp4LightGBM().predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": SMILES}),
            structure_column="smiles",
            artifact=artifact,
            conditions={},
        )
    )

    assert predictions["uncertainty"].null_count() == predictions.height


def test_training_is_reproducible_from_the_seed() -> None:
    """The reason `deterministic=True` and `force_row_wise=True` are set. Without
    them LightGBM sums histogram bins in thread-completion order, and two fits of the
    same configuration on the same data return different trees -- so a Scorecard
    changes when nothing changed. Fails if either setting is dropped."""
    first = Ecfp4LightGBM().train(context()).metrics
    second = Ecfp4LightGBM().train(context()).metrics

    assert first["rmse"] == second["rmse"]
    assert first["r2"] == second["r2"]


def test_predictions_are_reproducible_from_the_artifact() -> None:
    artifact = Ecfp4LightGBM().train(context()).artifact
    predict_context = PredictContext(
        frame=pl.DataFrame({"smiles": SMILES}),
        structure_column="smiles",
        artifact=artifact,
        conditions={},
    )

    first = Ecfp4LightGBM().predict(predict_context)["value"].to_list()
    second = Ecfp4LightGBM().predict(predict_context)["value"].to_list()

    assert first == second


def test_a_regression_target_of_only_zeros_and_ones_still_trains_a_regressor() -> None:
    """Task comes from the TargetSpec, never from sniffing the values."""
    result = Ecfp4LightGBM().train(context(rows=frame([0.0, 1.0] * 6)))

    assert "rmse" in result.metrics


def test_single_class_train_split_reports_undefined_metrics_not_a_crash() -> None:
    """A single-class train split makes predict_proba return one column, not two.
    Every classification metric must come back uniformly undefined rather than a mix
    of NaN and a misleadingly confident number."""
    single_class = pl.DataFrame(
        {
            "smiles": SMILES[:10],
            "y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "split": ["train"] * 8 + ["test"] * 2,
        }
    )
    ctx = TrainContext(
        frame=single_class,
        task=TaskType.BINARY_CLASSIFICATION,
        structure_column="smiles",
        target_column="y",
        conditions=CONDITIONS,
        seed=42,
    )

    result = Ecfp4LightGBM().train(ctx)

    assert set(result.metrics) == {"mcc", "balanced_accuracy", "auroc", "auprc"}
    assert all(math.isnan(value) for value in result.metrics.values())


def test_manifest_stays_on_the_default_lane() -> None:
    """LightGBM is CPU. If this ever declares the gpu lane, every fit queues behind
    a GPU runner that does not need to be involved."""
    manifest = Ecfp4LightGBM.manifest()

    assert manifest.lane == "default"
    assert manifest.is_baseline is False


@pytest.mark.parametrize("key", ["n_estimators", "num_leaves", "learning_rate"])
def test_manifest_declares_the_conditions_the_form_renders(key: str) -> None:
    assert any(condition.key == key for condition in Ecfp4LightGBM.manifest().conditions)
