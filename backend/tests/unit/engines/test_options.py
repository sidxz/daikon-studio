import numpy as np
import polars as pl
import pytest
from sklearn.metrics import matthews_corrcoef

from daikonstudio.application.engines.context import MIN_CUTOFF_CLASS_COUNT
from daikonstudio.infrastructure.chem.featurize import DESCRIPTOR_NAMES
from daikonstudio.infrastructure.engines._options import positive_weight
from daikonstudio.infrastructure.engines._scoring import (
    _FEATURIZERS,
    bundle_features,
    classification_by_column,
    mcc_cutoff,
    tree_featurizer,
    tuned_cutoffs,
)


def _reference_cutoff(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Brute force: every distinct probability, predict positive when p >= c."""
    best = (-2.0, 0.0)
    for c in sorted(set(p.tolist()), reverse=True):
        mcc = matthews_corrcoef(y, (p >= c).astype(int))
        if mcc > best[0]:
            best = (mcc, c)
    return best


def test_mcc_cutoff_finds_the_exact_optimum():
    rng = np.random.default_rng(7)
    y = (rng.random(300) < 0.15).astype(int)
    p = np.clip(0.15 + 0.4 * y + rng.normal(0, 0.2, 300), 0, 1)
    best_mcc, best_cut = _reference_cutoff(y, p)
    cut = mcc_cutoff(y, p)
    assert cut == pytest.approx(best_cut)
    assert matthews_corrcoef(y, (p >= cut).astype(int)) == pytest.approx(best_mcc)


def test_mcc_cutoff_breaks_ties_toward_the_higher_cutoff():
    y = np.array([1] * 10 + [0] * 10)
    p = np.array([0.9] * 10 + [0.2] * 5 + [0.1] * 5)
    # Any cutoff in (0.2, 0.9] separates perfectly; the candidates are 0.9, 0.2, 0.1.
    assert mcc_cutoff(y, p) == pytest.approx(0.9)


def test_mcc_cutoff_refuses_too_few_of_either_class():
    y = np.array([1] * (MIN_CUTOFF_CLASS_COUNT - 1) + [0] * 50)
    p = np.linspace(0, 1, len(y))
    assert mcc_cutoff(y, p) is None


def test_mcc_cutoff_refuses_a_model_that_cannot_separate_anything():
    y = np.array([1] * 20 + [0] * 20)
    assert mcc_cutoff(y, np.full(len(y), 0.3)) is None


def test_positive_weight_uses_the_training_ratio():
    y = np.array([1] * 4 + [0] * 36)
    assert positive_weight(y, "none") is None
    assert positive_weight(y, "balanced") == pytest.approx(9.0)
    assert positive_weight(y, "sqrt_balanced") == pytest.approx(3.0)
    assert positive_weight(np.zeros(10), "balanced") is None


def test_tree_featurizer_appends_the_descriptors_as_float32():
    key, featurize = tree_featurizer({"rdkit_descriptors": True})
    x = featurize(["CCO", "c1ccccc1"])
    assert key == "ecfp4+rdkit_descriptors"
    assert x.shape == (2, 2048 + len(DESCRIPTOR_NAMES))
    assert x.dtype == np.float32
    assert _FEATURIZERS[key] is featurize
    assert tree_featurizer({"rdkit_descriptors": False})[0] == "ecfp4"
    assert tree_featurizer({})[0] == "ecfp4"


def test_bundle_features_names_the_descriptor_columns_and_leaves_plain_ecfp4_unkeyed():
    assert bundle_features("ecfp4") == {}
    assert bundle_features("ecfp4+rdkit_descriptors") == {
        "featurizer": "ecfp4+rdkit_descriptors",
        "feature_names": DESCRIPTOR_NAMES,
    }


def _joint_rows():
    """One label `a` whose actives score 0.30-0.44 and inactives 0.05-0.19: perfectly
    separable at a cutoff of 0.30, and every compound predicted inactive at 0.5. Label `b`
    has too few actives to tune."""
    scores = [0.30 + 0.01 * k for k in range(15)] + [0.05 + 0.01 * k for k in range(15)]
    rows = pl.DataFrame({"a": [1] * 15 + [0] * 15, "b": [1] * 3 + [0] * 27})
    return rows, np.column_stack([scores, scores])


def test_tuned_cutoffs_are_found_per_column_and_omitted_when_untunable():
    rows, probabilities = _joint_rows()
    assert tuned_cutoffs(("a", "b"), rows, probabilities) == {"a": pytest.approx(0.30)}


def test_classification_scores_use_the_column_cutoff_and_default_to_one_half():
    rows, probabilities = _joint_rows()
    tuned = classification_by_column(("a",), rows, probabilities, {"a": 0.30}, rows)
    untuned = classification_by_column(("a",), rows, probabilities, {}, rows)
    assert tuned["a"]["mcc"] == pytest.approx(1.0)
    assert untuned["a"]["mcc"] == 0.0
    # AUROC is untouched by any cutoff.
    assert tuned["a"]["auroc"] == untuned["a"]["auroc"] == 1.0


def test_a_probability_of_exactly_one_half_is_called_active_without_a_cutoff():
    rows = pl.DataFrame({"a": [1, 1, 0, 0]})
    probabilities = np.array([[0.5], [0.9], [0.49], [0.1]])
    assert classification_by_column(("a",), rows, probabilities, {}, rows)["a"]["mcc"] == 1.0
