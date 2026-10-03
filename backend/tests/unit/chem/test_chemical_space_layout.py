import numpy as np
import pytest

from daikonstudio.application.ports.chemical_space_layout import TooFewCompounds
from daikonstudio.infrastructure.chem.chemical_space import UmapLayout, _unit_square

ALCOHOLS = ["C" * n + "O" for n in range(1, 11)]
AROMATICS = [
    "c1ccccc1",
    "Cc1ccccc1",
    "CCc1ccccc1",
    "c1ccc2ccccc2c1",
    "Cc1ccc2ccccc2c1",
    "c1ccc(-c2ccccc2)cc1",
    "Cc1ccc(-c2ccccc2)cc1",
    "c1ccc2cc3ccccc3cc2c1",
    "Oc1ccccc1",
    "Nc1ccccc1",
]


def test_the_layout_is_reproducible_and_inside_the_unit_square():
    first = UmapLayout().layout(ALCOHOLS + AROMATICS, seed=3)
    second = UmapLayout().layout(ALCOHOLS + AROMATICS, seed=3)
    assert first == second
    xs, ys = np.array(first[0]), np.array(first[1])
    assert xs.min() >= 0 and ys.min() >= 0 and xs.max() <= 1 and ys.max() <= 1


def test_two_unrelated_series_land_in_separate_regions():
    xs, ys = UmapLayout().layout(ALCOHOLS + AROMATICS, seed=3)
    points = np.column_stack([xs, ys])
    a, b = points[:10], points[10:]
    gap = np.linalg.norm(a.mean(axis=0) - b.mean(axis=0))
    spread = max(
        np.linalg.norm(a - a.mean(axis=0), axis=1).mean(),
        np.linalg.norm(b - b.mean(axis=0), axis=1).mean(),
    )
    assert gap > spread


def test_too_few_compounds_is_refused():
    with pytest.raises(TooFewCompounds):
        UmapLayout().layout(["CCO", "CCN"], seed=1)


@pytest.mark.parametrize("odd", ["C", "O", "[Na+].[Cl-]", "not a smiles"])
def test_a_compound_sharing_no_bits_does_not_blank_the_map(odd):
    """Jaccard distance to everything is 1 for a compound with no shared ECFP4
    bit; UMAP disconnects such points and embeds them as NaN, which used to
    spread through the normalisation to every compound on the map."""
    xs, ys = UmapLayout().layout(ALCOHOLS + AROMATICS + [odd], seed=3)
    assert np.isfinite(xs).all() and np.isfinite(ys).all()


def test_non_finite_coordinates_are_refused_rather_than_spread():
    with pytest.raises(ValueError):
        _unit_square(np.array([[np.nan, 0.0], [1.0, 1.0]]))
