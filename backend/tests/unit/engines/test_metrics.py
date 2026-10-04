"""The metric vocabulary every engine shares.

These functions exist so a chemprop model and the ECFP4 baseline are measured by
literally the same code. A Scorecard comparing "your model" against "the baseline" is
the product's central claim; two engines computing "auroc" slightly differently would
make that claim false while looking correct.
"""

from __future__ import annotations

import math

import numpy as np

from daikonstudio.infrastructure.engines._scoring import (
    classification_metrics,
    regression_metrics,
)


def test_regression_reports_exactly_rmse_mae_and_r2() -> None:
    metrics = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))

    assert sorted(metrics) == ["mae", "r2", "rmse"]
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_regression_never_reports_accuracy() -> None:
    """A 99.9%-negative dataset yields a 99.9%-accurate useless model. The only
    reliable way to keep that number off a Scorecard is to never compute it."""
    assert "accuracy" not in regression_metrics(np.array([1.0]), np.array([1.0]))


def test_classification_reports_exactly_the_four_defined_metrics() -> None:
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = classification_metrics(
        y_true,
        (probabilities >= 0.5).astype(float),
        probabilities,
        train_has_both_classes=True,
    )

    assert sorted(metrics) == ["auprc", "auroc", "balanced_accuracy", "mcc"]
    assert metrics["auroc"] == 1.0
    assert "accuracy" not in metrics


def test_a_single_class_test_split_makes_every_metric_undefined() -> None:
    """Balanced accuracy silently collapses to plain accuracy when y_true has one
    class -- exactly the number this module exists never to report. All four go
    undefined together, uniformly."""
    y_true = np.array([1.0, 1.0, 1.0])
    probabilities = np.array([0.6, 0.7, 0.8])

    metrics = classification_metrics(
        y_true, np.ones(3), probabilities, train_has_both_classes=True
    )

    assert sorted(metrics) == ["auprc", "auroc", "balanced_accuracy", "mcc"]
    assert all(math.isnan(value) for value in metrics.values())


def test_a_single_class_training_split_makes_every_metric_undefined() -> None:
    y_true = np.array([0.0, 1.0])
    probabilities = np.array([0.4, 0.6])

    metrics = classification_metrics(
        y_true, np.array([0.0, 1.0]), probabilities, train_has_both_classes=False
    )

    assert all(math.isnan(value) for value in metrics.values())


# --- The validation partition ------------------------------------------------
#
# It was assigned by every split from the beginning and read by nothing: both
# fingerprint engines ignored it outright, and chemprop computed a validation loss
# each epoch that no callback consumed. Ten percent of every dataset held out and
# spent on nothing -- and, worse, no number for a scientist to tune conditions
# against except the test score, which is what turns a held-out test set into a
# selection set one retrain at a time. These pin the fix.


def _frame_with_validation():
    import polars as pl

    # Distinct structures per partition so a leak would be visible, and enough of
    # each class in every partition that no metric is undefined.
    actives = ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCN", "CCCN", "CCCCN", "CCCCCN"]
    inactives = ["c1ccccc1", "c1ccccc1C", "c1ccccc1CC", "c1ccncc1", "C1CCCCC1"]
    structures = actives + inactives
    labels = [1.0] * len(actives) + [0.0] * len(inactives)
    splits = (
        ["train"] * 5
        + ["validation"] * 2
        + ["test"] * 1
        + ["train"] * 3
        + ["validation"] * 1
        + ["test"] * 1
    )
    return pl.DataFrame({"smiles": structures, "y": labels, "split": splits})


def _train(frame, engine=None):
    from daikonstudio.application.engines.context import TrainContext
    from daikonstudio.application.engines.manifest import TaskType
    from daikonstudio.infrastructure.engines.ecfp4_randomforest import Ecfp4RandomForest

    return (engine or Ecfp4RandomForest()).train(
        TrainContext(
            frame=frame,
            targets={"y": TaskType.BINARY_CLASSIFICATION},
            structure_column="smiles",
            conditions={"n_estimators": 50},
            seed=42,
        )
    )


def test_the_validation_partition_is_actually_scored():
    """It is the number a scientist is supposed to tune against. If it is absent
    the only feedback available is the test score, which is the whole problem."""
    result = _train(_frame_with_validation())
    assert result.validation_metrics is not None
    # The same vocabulary as the test metrics, from the same scoring code -- a
    # validation number measured differently would optimise the wrong thing.
    assert set(result.validation_metrics["y"]) == set(result.metrics["y"])


def test_validation_metrics_are_scored_on_validation_not_test():
    """The two partitions hold different compounds, so a scoring pass pointed at
    the wrong one is silent -- both dicts would simply be identical."""
    import polars as pl

    frame = _frame_with_validation()
    # Make the validation rows unmistakably different from the test rows: flip
    # every validation label, so a model fit on train cannot score the same on both.
    flipped = frame.with_columns(
        pl.when(pl.col("split") == "validation")
        .then(1.0 - pl.col("y"))
        .otherwise(pl.col("y"))
        .alias("y")
    )
    result = _train(flipped)
    assert result.validation_metrics is not None
    assert result.validation_metrics["y"] != result.metrics["y"]


def test_an_empty_validation_partition_reports_none_not_zero():
    """A split declared with a zero validation fraction is a legitimate choice.
    Reporting 0.0 there would render as a measured, catastrophically bad score."""
    import polars as pl

    frame = _frame_with_validation()
    no_validation = frame.with_columns(
        pl.when(pl.col("split") == "validation")
        .then(pl.lit("train"))
        .otherwise(pl.col("split"))
        .alias("split")
    )
    assert _train(no_validation).validation_metrics is None
