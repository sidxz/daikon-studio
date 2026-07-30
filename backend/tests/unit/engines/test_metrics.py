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
