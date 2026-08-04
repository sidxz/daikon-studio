"""Task 15: the Scorecard. `build_scorecard` reads Task 14's already-computed
metrics as measured -- it never recomputes them -- and uses `actual`/`predicted`
only for the worst-20 residual list and the applicability distribution.
"""

from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.application.execution.build_scorecard import build_scorecard
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
        "train_structures": ["CCO"],
        "normalizer": NORMALIZER,
        "target_unit": "nM",
        "target_direction": "low",
        "split_strategy": "random",
    }
    return build_scorecard(**{**kwargs, **overrides})


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
        train_structures=["CCO"],
        normalizer=NORMALIZER,
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
        train_structures=["CCO"],
        normalizer=NORMALIZER,
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
        train_structures=["CCO"],
        normalizer=NORMALIZER,
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
        train_structures=["CCO"],
        normalizer=NORMALIZER,
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
