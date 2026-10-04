import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef

from daikonstudio.application.engines.context import MIN_CUTOFF_CLASS_COUNT
from daikonstudio.infrastructure.chem.featurize import DESCRIPTOR_NAMES
from daikonstudio.infrastructure.engines._options import positive_weight
from daikonstudio.infrastructure.engines._scoring import (
    _FEATURIZERS,
    bundle_features,
    mcc_cutoff,
    tree_featurizer,
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
