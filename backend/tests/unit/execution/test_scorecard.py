"""Task 15: the Scorecard. `build_scorecard` reads Task 14's already-computed
metrics as measured -- it never recomputes them -- and uses `actual`/`predicted`
only for the worst-20 residual list and the applicability distribution.
"""

from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.application.execution.build_scorecard import (
    _PAIRED_CI_RESAMPLES,
    build_scorecard,
    held_out_chemistry,
)
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer

NORMALIZER = RdkitStructureNormalizer()


def regression_card(**overrides):
    # `metrics`/`baseline_metrics`/`baseline_engine_id`/`baseline_is_self` are
    # mandatory on `build_scorecard` (never recomputed, per the corrected brief) but
    # the brief's own fixture omits them -- the same gap Task 9/10 hit with their
    # own test fixtures needing a `normalizer`. Filled in here with plausible
    # values; the brief's literal assertions are otherwise unchanged.
    kwargs = {
        "task": TaskType.REGRESSION,
        "metrics": {"rmse": 0.5, "mae": 0.4, "r2": 0.8},
        "engine_id": "ecfp4-randomforest",
        "conditions": {},
        "baseline_engine_id": "ecfp4-randomforest",
        "baseline_metrics": {"rmse": 0.9, "mae": 0.7, "r2": 0.3},
        "baseline_is_self": False,
        "actual": [1.0, 2.0, 3.0],
        "predicted": [1.1, 2.1, 2.9],
        "structures": ["CCO", "CCN", "CCCO"],
        "target_unit": "nM",
        "target_direction": "low",
        "split_strategy": "random",
    }
    kwargs.update(overrides)
    # `train_structures` is not a `build_scorecard` argument any more -- the
    # chemistry is computed once per Protocol and handed in -- but tests still
    # vary it, so it stays an override here and is folded into the chemistry.
    train_structures = kwargs.pop("train_structures", ["CCO"])
    return build_scorecard(
        target="y",
        chemistry=held_out_chemistry(kwargs["structures"], train_structures, NORMALIZER),
        **kwargs,
    )


def test_unit_direction_and_split_strategy_are_carried_through_unchanged():
    """Task 15 review, Important 2 (whole-branch review): the fourth and last
    place a predicted number reaches a consumer without its unit and
    direction, and the only way to know which split produced a metric short
    of a two-null inference. `build_scorecard` must pass these straight
    through, not derive or drop them."""
    card = regression_card(target_unit="uM", target_direction="low", split_strategy="scaffold")
    assert card.target_unit == "uM"
    assert card.target_direction == "low"
    assert card.split_strategy == "scaffold"


def test_regression_leads_with_rmse():
    card = regression_card()
    assert card.primary_metric == "rmse"
    assert set(card.metrics) >= {"rmse", "mae", "r2"}


def test_classification_leads_with_mcc_never_accuracy():
    """A 99.9%-negative dataset yields a 99.9%-accurate useless model."""
    card = build_scorecard(
        target="y",
        task=TaskType.BINARY_CLASSIFICATION,
        metrics={"mcc": 0.0, "balanced_accuracy": 0.5, "auroc": 0.5, "auprc": 0.5},
        engine_id="ecfp4-randomforest",
        conditions={},
        baseline_engine_id="ecfp4-randomforest",
        baseline_metrics={"mcc": 0.0, "balanced_accuracy": 0.5, "auroc": 0.5, "auprc": 0.5},
        baseline_is_self=False,
        actual=[0.0] * 99 + [1.0],
        predicted=[0.0] * 100,
        structures=["CCO"] * 100,
        chemistry=held_out_chemistry(["CCO"] * 100, ["CCO"], NORMALIZER),
        target_unit=None,
        target_direction=None,
        split_strategy="random",
    )
    assert card.primary_metric == "mcc"
    assert set(card.metrics) >= {"mcc", "balanced_accuracy", "auroc", "auprc"}
    assert card.metrics["mcc"] == 0.0
    assert "accuracy" not in card.metrics


def test_worst_rows_are_ranked_by_residual_and_carry_their_scaffold():
    card = regression_card(
        actual=[1.0, 2.0, 9.0],
        predicted=[1.0, 2.0, 2.0],
        structures=["CCO", "CCN", "Cc1ccccc1"],
    )
    assert card.worst_rows[0].structure == "Cc1ccccc1"
    assert card.worst_rows[0].residual == 7.0
    assert card.worst_rows[0].scaffold == "c1ccccc1"


def test_applicability_coverage_is_the_fraction_above_the_threshold():
    card = regression_card(
        actual=[1.0, 2.0],
        predicted=[1.0, 2.0],
        structures=["CCO", "CCCCCCCCCCCCCCCC"],
        train_structures=["CCO"],
    )
    assert card.applicability_coverage == 0.5


def test_baseline_and_optimism_gap_are_carried_through_when_supplied():
    card = regression_card(
        baseline_engine_id="ecfp4-randomforest",
        baseline_metrics={"rmse": 0.9},
        random_split_metrics={"rmse": 0.2},
    )
    assert card.baseline_engine_id == "ecfp4-randomforest"
    assert card.random_split_metrics["rmse"] == 0.2


def test_noise_floor_is_absent_when_there_were_no_duplicates():
    assert regression_card(duplicate_spread=None).noise_floor is None
    assert regression_card(duplicate_spread=0.4).noise_floor == 0.4


# --- The rest are mine, covering the three decisions the brief asks for. ---


def test_noise_floor_is_forced_none_for_classification_regardless_of_input():
    """Binary targets have no duplicate-spread equivalent -- even if a caller
    passes one through (e.g. stale data from a migration), the Scorecard must
    not present it as a meaningful floor."""
    card = build_scorecard(
        target="y",
        task=TaskType.BINARY_CLASSIFICATION,
        metrics={"mcc": 0.5, "balanced_accuracy": 0.7, "auroc": 0.8, "auprc": 0.6},
        engine_id="ecfp4-randomforest",
        conditions={},
        baseline_engine_id="ecfp4-randomforest",
        baseline_metrics={"mcc": 0.1, "balanced_accuracy": 0.5, "auroc": 0.5, "auprc": 0.5},
        baseline_is_self=False,
        actual=[0.0, 1.0],
        predicted=[0.1, 0.9],
        structures=["CCO", "CCN"],
        chemistry=held_out_chemistry(["CCO", "CCN"], ["CCO"], NORMALIZER),
        target_unit=None,
        target_direction=None,
        split_strategy="random",
        duplicate_spread=0.4,
    )
    assert card.noise_floor is None


def test_baseline_is_self_is_carried_through_so_the_scorecard_can_say_so():
    """When the chosen engine *is* the baseline, the Scorecard must be able to
    say so rather than presenting one result twice as if a comparison happened."""
    card = regression_card(baseline_is_self=True)
    assert card.baseline_is_self is True
    card = regression_card(baseline_is_self=False)
    assert card.baseline_is_self is False


def test_classification_worst_rows_are_probability_residuals_and_say_so():
    """Decision 1: a probability residual is not a regression residual. The
    Scorecard's own `prediction_kind` names which one `worst_rows` holds rather
    than leaving a consumer to infer it (or worse, assume regression units)."""
    card = build_scorecard(
        target="y",
        task=TaskType.BINARY_CLASSIFICATION,
        metrics={"mcc": 0.0, "balanced_accuracy": 0.5, "auroc": 0.5, "auprc": 0.5},
        engine_id="ecfp4-randomforest",
        conditions={},
        baseline_engine_id="ecfp4-randomforest",
        baseline_metrics={"mcc": 0.0, "balanced_accuracy": 0.5, "auroc": 0.5, "auprc": 0.5},
        baseline_is_self=False,
        actual=[1.0, 0.0, 0.0],
        predicted=[0.02, 0.4, 0.1],
        structures=["CCO", "CCN", "CCCO"],
        chemistry=held_out_chemistry(["CCO", "CCN", "CCCO"], ["CCO"], NORMALIZER),
        target_unit=None,
        target_direction=None,
        split_strategy="random",
    )
    assert card.prediction_kind == "probability"
    assert card.worst_rows[0].structure == "CCO"
    assert card.worst_rows[0].residual == 0.98


def test_regression_worst_rows_are_value_residuals():
    card = regression_card()
    assert card.prediction_kind == "value"


def test_undefined_metrics_carry_their_reason_onto_the_scorecard():
    """Decision 2: `metrics_undefined` explains *why* a metric is None. A
    scientist should never see a bare blank where a number belongs."""
    card = build_scorecard(
        target="y",
        task=TaskType.BINARY_CLASSIFICATION,
        metrics={"mcc": None, "balanced_accuracy": None, "auroc": None, "auprc": None},
        engine_id="ecfp4-randomforest",
        conditions={},
        baseline_engine_id="ecfp4-randomforest",
        baseline_metrics={"mcc": None, "balanced_accuracy": None, "auroc": None, "auprc": None},
        baseline_is_self=False,
        actual=[0.0, 0.0],
        predicted=[0.1, 0.2],
        structures=["CCO", "CCN"],
        chemistry=held_out_chemistry(["CCO", "CCN"], ["CCO"], NORMALIZER),
        target_unit=None,
        target_direction=None,
        split_strategy="random",
        metrics_undefined={
            "mcc": "every row in the test split has the same value",
            "balanced_accuracy": "every row in the test split has the same value",
            "auroc": "every row in the test split has the same value",
            "auprc": "every row in the test split has the same value",
        },
    )
    assert card.metrics["mcc"] is None
    assert card.metrics_undefined["mcc"] == "every row in the test split has the same value"


def test_metrics_undefined_defaults_to_none_when_not_supplied():
    card = regression_card()
    assert card.metrics_undefined is None


def test_worst_rows_caps_at_twenty_without_crashing_on_more_rows():
    """Decision 3a: more than twenty test rows -- only the worst twenty appear."""
    n = 25
    card = regression_card(
        actual=[float(i) for i in range(n)],
        predicted=[0.0] * n,
        structures=["CCO"] * n,
        train_structures=["CCO"],
    )
    assert len(card.worst_rows) == 20
    # the largest residuals (24, 23, ..., 5) are the ones kept
    assert sorted((row.residual for row in card.worst_rows), reverse=True) == [
        float(i) for i in range(24, 4, -1)
    ]


def test_worst_rows_handles_fewer_than_twenty_rows_without_padding():
    """Decision 3a: fewer than twenty test rows -- no crash, no fake padding."""
    card = regression_card()  # the default fixture has exactly 3 rows
    assert len(card.worst_rows) == 3


def test_empty_train_structures_yields_no_coverage_not_a_misleading_zero():
    """Decision 3b: an empty training set means applicability truly cannot be
    computed -- reporting 0.0 would misread as "everything is out of
    distribution" rather than "unmeasurable"."""
    card = regression_card(train_structures=[])
    assert card.applicability_coverage is None
    assert all(row.similarity is None for row in card.worst_rows)


def test_no_test_rows_yields_no_coverage_and_no_worst_rows():
    card = regression_card(actual=[], predicted=[], structures=[])
    assert card.applicability_coverage is None
    assert card.worst_rows == []


def test_the_baseline_conditions_reach_the_rendered_scorecard():
    """Without this the page cannot tell a pretrained model from an untrained one:
    both sides carry the same engine id, and only the conditions differ."""
    card = regression_card(
        baseline_engine_id="chemprop-dmpnn",
        baseline_conditions={"pretrained": "none"},
    )
    assert card.baseline_conditions == {"pretrained": "none"}


def test_baseline_conditions_default_to_empty_for_a_scorecard_written_earlier():
    """Every existing call site omits them, including scorecards read back off
    blobs that predate the field."""
    assert regression_card().baseline_conditions == {}


def test_the_cutoffs_and_their_note_reach_the_scorecard():
    card = regression_card(cutoff=0.31, baseline_cutoff=0.42, cutoff_note="Not tuned: why.")
    assert (card.cutoff, card.baseline_cutoff, card.cutoff_note) == (0.31, 0.42, "Not tuned: why.")


def test_the_cutoffs_default_to_none_for_a_scorecard_written_earlier():
    """None means 0.5: every blob written before cutoffs could be tuned has none."""
    card = regression_card()
    assert (card.cutoff, card.baseline_cutoff, card.cutoff_note) == (None, None, None)


# --- Diagnostics ---------------------------------------------------------
#
# These are derived from the same `actual`/`predicted`/`structures` the fields
# above are, so nothing new is measured and nothing extra is persisted. What can
# still go wrong is a diagnostic that quietly misrepresents the population it was
# drawn from, and that is what these cover.


def test_parity_carries_every_test_compound_when_it_can():
    card = regression_card()
    assert len(card.parity) == 3
    assert card.parity_sampled_from is None
    assert [point.actual for point in card.parity] == [1.0, 2.0, 3.0]


def test_a_subsampled_parity_scatter_says_that_it_was_subsampled():
    """A scatter that silently drops points reads as the whole test set. The
    count it was drawn from is what keeps it honest."""
    size = 9000
    card = regression_card(
        actual=[float(i) for i in range(size)],
        predicted=[float(i) + 0.1 for i in range(size)],
        structures=["CCO"] * size,
    )
    assert card.parity_sampled_from == size
    assert 0 < len(card.parity) <= 4000


def test_the_interval_figure_gets_the_redraws_the_interval_was_read_from():
    """The figure draws these bins under the interval, so the two must be one
    set of redraws: every redraw counted once, and the interval inside the bins."""
    size = 60
    card = regression_card(
        actual=[float(i % 7) for i in range(size)],
        predicted=[float(i % 7) + (0.5 if i % 3 else -0.8) for i in range(size)],
        structures=["CCO"] * size,
    )
    histogram, interval = card.primary_metric_bootstrap, card.primary_metric_ci
    assert histogram is not None and interval is not None
    assert len(histogram.edges) == len(histogram.counts) + 1
    assert sum(histogram.counts) == 1000
    assert histogram.edges[0] <= interval[0] < interval[1] <= histogram.edges[-1]


def test_no_interval_means_no_redraws_to_draw():
    card = regression_card()
    assert card.primary_metric_ci is None
    assert card.primary_metric_bootstrap is None


def test_residual_histogram_is_regression_only():
    """For classification `actual` is a 0/1 label and `predicted` a probability,
    so their difference is bimodal by construction and says nothing."""
    assert regression_card().residual_histogram is not None
    card = build_scorecard(
        target="y",
        task=TaskType.BINARY_CLASSIFICATION,
        metrics={"mcc": 0.5},
        engine_id="ecfp4-randomforest",
        conditions={},
        baseline_engine_id="ecfp4-randomforest",
        baseline_metrics={"mcc": 0.1},
        baseline_is_self=False,
        actual=[1.0, 0.0, 1.0, 0.0],
        predicted=[0.9, 0.2, 0.7, 0.1],
        structures=["CCO", "CCN", "CCCO", "CCCN"],
        chemistry=held_out_chemistry(["CCO", "CCN", "CCCO", "CCCN"], ["CCO"], NORMALIZER),
        target_unit=None,
        target_direction=None,
        split_strategy="random",
    )
    assert card.residual_histogram is None
    assert card.calibration, "classification gets a calibration curve instead"
    assert all(0.0 <= entry.value <= 1.0 for entry in card.calibration)
    assert all(entry.count > 0 for entry in card.calibration)


def test_residual_histogram_edges_are_one_longer_than_its_counts():
    histogram = regression_card().residual_histogram
    assert histogram is not None
    assert len(histogram.edges) == len(histogram.counts) + 1
    assert sum(histogram.counts) == 3


def test_error_by_similarity_needs_enough_rows_to_be_a_curve():
    """Eight bins over three compounds is not a trend, it is eight noisy points
    with a line through them."""
    assert regression_card().error_by_similarity == []


def test_error_by_similarity_keeps_tied_similarities_together_and_rises_with_distance():
    """The claim the applicability number makes, turned into evidence: error
    should grow as compounds get less like the training set."""
    # Twenty analogues of a training compound predicted well, twenty unrelated
    # compounds predicted badly.
    near = ["CCO", "CCCO", "CCCCO", "CCCCCO"] * 5
    far = ["c1ccc2ccccc2c1", "c1ccc2c(c1)ccc1ccccc12", "C1CCCCC1", "c1ccncc1"] * 5
    card = regression_card(
        actual=[1.0] * 40,
        predicted=[1.05] * 20 + [4.0] * 20,
        structures=near + far,
        train_structures=["CCO", "CCCO"],
    )
    bins = card.error_by_similarity
    # There are four distinct similarity scores. Splitting the ties merely to
    # fill eight bins would imply a distance trend within identical distances.
    assert len(bins) == 4
    assert bins[0].count == 20
    assert sum(entry.count for entry in bins) == 40
    # Bins are ordered by rising similarity, so the far compounds -- the badly
    # predicted ones -- are at the low-similarity end.
    assert bins[0].value > bins[-1].value


def test_scaffold_errors_need_two_families_to_be_a_comparison():
    """This section exists only to say which families are worse than which; one
    family cannot answer that."""
    card = regression_card(
        actual=[1.0] * 6,
        predicted=[2.0] * 6,
        structures=["c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC"] * 2,
    )
    assert card.scaffold_errors == []


def test_scaffold_errors_rank_the_worst_family_first():
    benzenes = ["c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC"]
    pyridines = ["c1ccncc1C", "c1ccncc1CC", "c1ccncc1CCC"]
    card = regression_card(
        actual=[1.0] * 6,
        # The benzenes are off by 3.0, the pyridines by 0.1.
        predicted=[4.0, 4.0, 4.0, 1.1, 1.1, 1.1],
        structures=benzenes + pyridines,
    )
    assert len(card.scaffold_errors) == 2
    worst, best = card.scaffold_errors
    assert worst.median_error > best.median_error
    assert worst.count == 3
    # Named, not just ranked: the whole value of this section is that a chemist
    # can read "it is worse on the benzenes" off it.
    assert worst.scaffold == NORMALIZER.murcko_scaffold("c1ccccc1C")
    assert best.scaffold == NORMALIZER.murcko_scaffold("c1ccncc1C")


def test_a_sequence_dataset_reports_no_applicability_domain_rather_than_zero() -> None:
    """Chemistry on a sequence column does not fail -- it answers, wrongly.

    RDKit cannot read a protein, so every fingerprint comes back empty and every
    nearest-neighbour Tanimoto is 0.0. `applicability_coverage` then divides that into a
    rate and the Scorecard states that 0% of the test set is within the applicability
    domain: a fabricated number, alarming to read, and indistinguishable in shape from a
    measured one. `None` already means "there was nothing to compare" everywhere
    downstream, so that is what the sequence path returns.
    """
    from daikonstudio.application.execution.build_scorecard import held_out_chemistry
    from daikonstudio.domain.data.structure_kind import StructureKind
    from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer

    normalizer = RdkitStructureNormalizer()
    parent = "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSR"
    test = [parent, f"{parent[:-1]}A"]
    train = [f"{parent[:-2]}AA"]

    # What it used to do, and why this test exists: the numbers were not an error.
    as_molecules = held_out_chemistry(test, train, normalizer)
    assert as_molecules.similarities == [0.0, 0.0]

    sequences = held_out_chemistry(test, train, normalizer, StructureKind.SEQUENCE)
    assert sequences.similarities is None
    # Same length as the input: `_scaffold_errors` zips this against the residuals with
    # `strict=True`, and empty strings collapse to one family, which that function
    # already drops for being less than a comparison.
    assert sequences.scaffolds == ["", ""]

    molecules = held_out_chemistry(["CCO", "c1ccccc1"], ["CCN"], normalizer)
    assert molecules.similarities is not None
    assert molecules.similarities[0] > 0.0


# --- The paired comparison -------------------------------------------------------------


def _paired(baseline_predicted, *, size=60, **overrides):
    actual = [float(i % 7) for i in range(size)]
    error = [float((i * 7) % 11 - 5) for i in range(size)]
    return regression_card(
        actual=actual,
        predicted=[a + 0.95 * e for a, e in zip(actual, error, strict=True)],
        baseline_predicted=baseline_predicted(actual, error),
        structures=["CCO"] * size,
        **overrides,
    )


def _baseline_errors(actual, error):
    return [a + e for a, e in zip(actual, error, strict=True)]


def test_a_consistent_small_lead_is_resolved_by_pairing_where_the_old_check_cannot():
    """The model's errors are the baseline's, 5% smaller, on every compound. The
    baseline's RMSE (3.19) sits inside the model's own interval [2.65, 3.34], so the
    unpaired check calls it noise; paired, every redraw favors the model."""
    card = _paired(_baseline_errors)

    assert card.primary_metric_ci is not None
    low, high = card.primary_metric_ci
    assert low < 3.186 < high
    assert card.difference_ci is not None
    d_low, d_high = card.difference_ci
    assert d_low < d_high < 0
    assert card.difference_bootstrap is not None
    # Against the constant, not a literal: the claim is that every redraw lands in the
    # histogram, which must survive a change to how many redraws the paired test takes.
    assert sum(card.difference_bootstrap.counts) == _PAIRED_CI_RESAMPLES
    assert card.difference_bootstrap.edges[0] <= d_low


def test_identical_predictions_differ_by_exactly_nothing():
    card = _paired(
        lambda actual, error: [a + 0.95 * e for a, e in zip(actual, error, strict=True)]
    )
    assert card.difference_ci == (0.0, 0.0)


def test_no_paired_interval_without_a_real_second_model():
    # Written before baseline predictions were kept.
    assert _paired(lambda actual, error: None).difference_ci is None
    # The baseline is this model.
    assert _paired(_baseline_errors, baseline_is_self=True).difference_ci is None
    # No baseline was fitted.
    assert _paired(_baseline_errors, baseline_metrics=None).difference_ci is None
    # Too few compounds for any interval.
    small = _paired(_baseline_errors, size=10)
    assert small.difference_ci is None and small.difference_bootstrap is None


def test_each_side_is_thresholded_at_its_own_operating_point():
    """The model is a neural engine tuned to 0.3; the baseline is a forest left at 0.5,
    which reaches its MCC through sklearn's `predict` and so puts an exact 0.5 in class
    0. Both are perfect at their own operating point, so every redraw differs by zero.
    Thresholding the baseline at `>=` would call every negative positive instead."""
    card = regression_card(
        task=TaskType.BINARY_CLASSIFICATION,
        metrics={"mcc": 1.0},
        baseline_metrics={"mcc": 1.0},
        engine_id="chemprop-dmpnn",
        baseline_engine_id="ecfp4-randomforest",
        cutoff=0.3,
        baseline_cutoff=None,
        actual=[1.0, 0.0] * 20,
        predicted=[0.35, 0.2] * 20,
        baseline_predicted=[0.9, 0.5] * 20,
        structures=["CCO"] * 40,
    )
    assert card.difference_ci == (0.0, 0.0)


def test_a_baseline_that_never_predicts_a_positive_is_still_paired_against():
    """An untuned forest on imbalanced data often calls every test compound negative.
    The headline scores that MCC 0 (sklearn's convention), so every redraw of it must
    too: dropping those redraws as undefined would leave no paired interval at all, or
    keep only the redraws where the baseline happened to look better than 0."""
    card = regression_card(
        task=TaskType.BINARY_CLASSIFICATION,
        metrics={"mcc": 1.0},
        baseline_metrics={"mcc": 0.0},
        engine_id="chemprop-dmpnn",
        baseline_engine_id="ecfp4-randomforest",
        actual=[1.0, 0.0] * 20,
        predicted=[0.9, 0.1] * 20,
        baseline_predicted=[0.1] * 40,
        structures=["CCO"] * 40,
    )
    assert card.difference_ci == (1.0, 1.0)
