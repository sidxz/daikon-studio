"""The bootstrap interval under the headline metric.

Pins three properties: the interval brackets the point estimate, it is computed
with the metric's own definition on the classification path, and a test set too
small or too skewed to support one yields None rather than a misleading pair.
"""

import numpy as np
import pytest

from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.application.execution.build_scorecard import primary_metric_ci


def test_the_regression_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    actual = rng.normal(size=200)
    predicted = actual + rng.normal(scale=0.5, size=200)
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    interval = primary_metric_ci(TaskType.REGRESSION, actual.tolist(), predicted.tolist())

    assert interval is not None
    low, high = interval
    assert low < rmse < high
    assert high - low < 0.3


def test_the_classification_interval_is_over_mcc_at_the_half_threshold():
    actual = [1.0, 0.0] * 50
    # 80 of 100 right: a comfortably positive MCC with room on both sides.
    predicted = [0.9, 0.1] * 40 + [0.1, 0.9] * 10

    interval = primary_metric_ci(TaskType.BINARY_CLASSIFICATION, actual, predicted)

    assert interval is not None
    low, high = interval
    assert 0.4 < low < 0.6 < high <= 1.0


def test_the_interval_is_deterministic_across_calls():
    actual = list(np.linspace(0, 1, 60))
    predicted = [value + 0.1 for value in actual]
    assert primary_metric_ci(TaskType.REGRESSION, actual, predicted) == primary_metric_ci(
        TaskType.REGRESSION, actual, predicted
    )


def test_too_few_rows_means_no_interval_rather_than_a_misleading_one():
    assert primary_metric_ci(TaskType.REGRESSION, [1.0] * 10, [1.1] * 10) is None


def test_a_single_class_test_set_yields_no_classification_interval():
    actual = [1.0] * 40
    predicted = [0.9] * 40
    assert primary_metric_ci(TaskType.BINARY_CLASSIFICATION, actual, predicted) is None


def test_the_classification_interval_follows_the_models_decision_cutoff():
    # Positives score 0.6 to 0.95 and negatives 0.05 to 0.7: the two overlap, so
    # moving the cutoff from 0.5 to 0.9 changes which compounds count as predicted
    # positive in every resample, and so the interval itself.
    actual = [1.0] * 30 + [0.0] * 30
    predicted = [float(value) for value in np.linspace(0.6, 0.95, 30)] + [
        float(value) for value in np.linspace(0.05, 0.7, 30)
    ]

    at_half = primary_metric_ci(TaskType.BINARY_CLASSIFICATION, actual, predicted)
    at_default = primary_metric_ci(TaskType.BINARY_CLASSIFICATION, actual, predicted, cutoff=0.5)
    at_nine_tenths = primary_metric_ci(
        TaskType.BINARY_CLASSIFICATION, actual, predicted, cutoff=0.9
    )

    assert at_half is not None and at_nine_tenths is not None
    assert at_default == at_half
    assert at_nine_tenths != at_half


# --- The metric over a flagged part of the test set ------------------------------------
#
# A published benchmark often reports a number over a named subset of its test set:
# MoleculeACE's headline is RMSE restricted to activity-cliff compounds. Comparing
# against it means computing exactly that, not the overall number.


def test_a_subset_metric_uses_only_the_flagged_rows():
    from daikonstudio.application.execution.build_scorecard import subset_metric

    # All the error is in the two flagged rows.
    actual = [1.0, 2.0, 3.0, 4.0]
    predicted = [1.0, 2.0, 4.0, 5.0]
    value = subset_metric(
        TaskType.REGRESSION, actual, predicted, subset=[False, False, True, True]
    )
    assert value == pytest.approx(1.0)


def test_an_empty_subset_is_no_measurement_rather_than_zero():
    """A cliff flag whose compounds all landed in training is an ordinary outcome of a
    split we did not choose. Zero would read as a perfect score."""
    from daikonstudio.application.execution.build_scorecard import subset_metric

    assert (
        subset_metric(TaskType.REGRESSION, [1.0, 2.0], [1.1, 2.1], subset=[False, False]) is None
    )


def test_a_subset_covering_every_row_equals_the_overall_metric():
    from daikonstudio.application.execution.build_scorecard import subset_metric

    actual = [1.0, 2.0, 3.0]
    predicted = [1.5, 2.5, 3.5]
    assert subset_metric(
        TaskType.REGRESSION, actual, predicted, subset=[True, True, True]
    ) == pytest.approx(0.5)


def test_a_binary_subset_metric_is_undefined_when_the_flagged_rows_are_one_class():
    from daikonstudio.application.execution.build_scorecard import subset_metric

    assert (
        subset_metric(
            TaskType.BINARY_CLASSIFICATION,
            [1.0, 1.0, 0.0],
            [0.9, 0.8, 0.1],
            subset=[True, True, False],
        )
        is None
    )
