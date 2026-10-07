from itertools import pairwise

import pytest

from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.application.execution.build_scorecard import (
    HeldOutChemistry,
    build_scorecard,
    similarity_groups,
)
from daikonstudio.interface.routes.protocols import ScorecardResponse


def card(
    actual, predicted, *, binary=False, cutoff=None, engine="chemprop-dmpnn", similarities=None
):
    return build_scorecard(
        target="y",
        task=TaskType.BINARY_CLASSIFICATION if binary else TaskType.REGRESSION,
        metrics={},
        engine_id=engine,
        conditions={},
        baseline_engine_id=engine,
        baseline_metrics={},
        baseline_is_self=True,
        actual=actual,
        predicted=predicted,
        structures=["CCO"] * len(actual),
        chemistry=HeldOutChemistry(similarities=similarities, scaffolds=[""] * len(actual)),
        target_unit=None,
        target_direction=None,
        split_strategy="scaffold",
        cutoff=cutoff,
    )


def test_regression_summary_uses_all_rows_even_when_plot_is_sampled():
    # The plot's stride skips every large error; summary statistics must not.
    result = card([0.0] * 4002, [0.0, 10.0] * 2001)
    assert len(result.parity) < result.test_count == 4002
    assert all(point.predicted == 0 for point in result.parity)
    assert result.regression_summary.mean_signed_error == 5
    assert result.regression_summary.absolute_error_p90 == 10
    assert result.classification_summary is None
    response = ScorecardResponse.from_domain(result)
    assert response.regression_summary.absolute_error_p90 == 10
    assert response.test_count == 4002


def test_p90_is_an_observed_bound_and_bias_is_signed():
    result = card([10.0] * 10, [9.0] * 9 + [0.0])
    assert result.regression_summary.absolute_error_p90 == 1
    assert result.regression_summary.mean_signed_error == pytest.approx(-1.9)
    assert card([], []).regression_summary is None


def test_tuned_cutoff_counts_inclusive_boundary_and_exposes_precision_recall():
    result = card([1, 1, 0, 0], [0.3, 0.1, 0.3, 0.05], binary=True, cutoff=0.3)
    summary = result.classification_summary
    assert (
        summary.true_positive,
        summary.false_negative,
        summary.false_positive,
        summary.true_negative,
    ) == (1, 1, 1, 1)
    assert summary.precision == summary.recall == 0.5
    assert summary.cutoff_inclusive
    assert result.regression_summary is None
    assert ScorecardResponse.from_domain(result).classification_summary.true_positive == 1


@pytest.mark.parametrize(
    "engine",
    [
        "ecfp4-randomforest",
        "ecfp4-xgboost",
        "ecfp4-lightgbm",
        "descriptors-xgboost",
        "tanimoto-gp",
    ],
)
def test_default_estimator_ties_match_predict_but_tuned_cutoffs_are_inclusive(engine):
    default = card([1, 0], [0.5, 0.5], binary=True, engine=engine).classification_summary
    tuned = card([1, 0], [0.5, 0.5], binary=True, engine=engine, cutoff=0.5).classification_summary
    assert default.true_positive == default.false_positive == 0
    assert not default.cutoff_inclusive
    assert tuned.true_positive == tuned.false_positive == 1
    assert tuned.cutoff_inclusive


def test_undefined_rates_are_not_fabricated_zeroes():
    no_predictions = card([1, 0], [0, 0], binary=True).classification_summary
    assert no_predictions.precision is None
    assert no_predictions.recall == 0
    no_actives = card([0, 0], [1, 1], binary=True).classification_summary
    assert no_actives.precision == 0
    assert no_actives.recall is None
    empty = card([], [], binary=True).classification_summary
    assert empty.precision is empty.recall is None


def test_confusion_matrix_uses_full_population():
    result = card([0, 1] * 2001, [0.0, 0.9] * 2001, binary=True)
    assert len(result.parity) < result.test_count
    assert result.classification_summary.true_positive == 2001
    assert result.classification_summary.true_negative == 2001


def test_ranking_uses_full_population_and_never_measured_outcomes_to_break_ties():
    result = card([float(i) for i in range(4002)], [0.0, 10.0] * 2001)
    assert all(point.predicted == 0 for point in result.parity)
    assert [row.test_index for row in result.ranked_high] == list(range(1, 40, 2))
    assert [row.test_index for row in result.ranked_low] == list(range(0, 40, 2))
    reversed_actual = card(list(reversed(range(4002))), [0.0, 10.0] * 2001)
    assert [row.test_index for row in reversed_actual.ranked_high] == [
        row.test_index for row in result.ranked_high
    ]
    assert ScorecardResponse.from_domain(result).ranked_high[0].predicted == 10


def test_ranked_rows_carry_their_own_results_and_handle_small_or_empty_sets():
    result = card([7, 8, 9], [30, 10, 20], similarities=[0.1, 0.2, 0.9])
    assert [r.test_index for r in result.ranked_high] == [0, 2, 1]
    assert [r.test_index for r in result.ranked_low] == [1, 2, 0]
    row = result.ranked_low[0]
    assert (row.actual, row.predicted, row.similarity) == (8, 10, 0.2)
    assert card([], []).ranked_high == card([], []).ranked_low == []


@pytest.mark.parametrize("cutoff,inclusive", [(None, False), (0.5, True)])
def test_binary_similarity_uses_saved_cutoff_and_class_counts(cutoff, inclusive):
    # First group has no actives; last has no inactives. Neither denominator
    # may silently become the whole bin or a fabricated zero-percent error.
    actual = [0, 0] + [0, 1] * 6 + [1, 1]
    result = card(
        actual,
        [0.5] * 16,
        binary=True,
        cutoff=cutoff,
        engine="ecfp4-xgboost",
        similarities=[i / 16 for i in range(16)],
    )
    bins = result.classification_by_similarity
    assert len(bins) == 8
    assert sum(b.count for b in bins) == result.test_count
    assert bins[0].summary.recall is None
    assert bins[-1].summary.true_negative + bins[-1].summary.false_positive == 0
    for name in ["true_positive", "true_negative", "false_positive", "false_negative"]:
        assert sum(getattr(b.summary, name) for b in bins) == getattr(
            result.classification_summary, name
        )
    assert all(b.summary.cutoff_inclusive == inclusive for b in bins)
    assert bins[-1].summary.false_negative == (0 if inclusive else 2)
    assert ScorecardResponse.from_domain(result).classification_by_similarity[0].count == 2


def test_similarity_groups_preserve_ties_and_do_not_lose_or_duplicate_rows():
    similarities = [0.2] * 7 + [0.4] * 11 + [0.9] * 3
    groups = similarity_groups(similarities)
    assert [i for group in groups for i in group] == list(range(len(similarities)))
    assert all(similarities[a[-1]] < similarities[b[0]] for a, b in pairwise(groups))
    assert similarity_groups([0.4] * 24) == [list(range(24))]
    assert similarity_groups(None) == similarity_groups([0.4] * 15) == []
