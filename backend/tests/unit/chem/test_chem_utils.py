import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from daikonstudio.infrastructure.chem.canonicalize import canonicalize, has_multiple_components
from daikonstudio.infrastructure.chem.featurize import ecfp4
from daikonstudio.infrastructure.chem.scaffold import murcko_scaffold
from daikonstudio.infrastructure.chem.similarity import nearest_neighbour_tanimoto

_CHOLESTEROL = "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C"


def test_canonicalize_normalises_equivalent_smiles():
    assert canonicalize("C1=CC=CC=C1") == canonicalize("c1ccccc1")


def test_canonicalize_returns_none_for_garbage():
    assert canonicalize("not-a-molecule") is None


def test_salts_are_detected_as_multi_component():
    assert has_multiple_components("CC(=O)O.[Na+]") is True
    assert has_multiple_components("CCO") is False


def test_ecfp4_shape_and_determinism():
    first = ecfp4(["CCO", "c1ccccc1"])
    second = ecfp4(["CCO", "c1ccccc1"])
    assert first.shape == (2, 2048)
    assert np.array_equal(first, second)


def test_murcko_scaffold_strips_side_chains():
    """Toluene and benzene share the benzene scaffold."""
    assert murcko_scaffold("Cc1ccccc1") == murcko_scaffold("c1ccccc1")


def test_nearest_neighbour_similarity_is_one_for_exact_match():
    scores = nearest_neighbour_tanimoto(["CCO"], ["CCO", "c1ccccc1"])
    assert scores.shape == (1,)
    assert scores[0] == 1.0


def test_nearest_neighbour_similarity_is_low_for_dissimilar():
    scores = nearest_neighbour_tanimoto(["CCCCCCCCCC"], ["c1ccc2ccccc2c1"])
    assert scores[0] < 0.3


@pytest.mark.parametrize(
    "query_smiles, reference_smiles",
    [
        ("Cc1ccccc1", "c1ccccc1"),  # toluene vs benzene: mid-range similarity
        ("c1ccccc1", _CHOLESTEROL),  # benzene vs cholesterol: size-skewed pair
    ],
)
def test_nearest_neighbour_similarity_matches_rdkit_reference(query_smiles, reference_smiles):
    """Pins the vectorised formula against RDKit's own TanimotoSimilarity at a
    mid-range value and a size-skewed pair, neither 0.0 nor 1.0. A wrong
    normalisation constant that only misbehaves away from those extremes would
    pass all the other tests in this file but fail here."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    expected = DataStructs.TanimotoSimilarity(
        generator.GetFingerprint(Chem.MolFromSmiles(query_smiles)),
        generator.GetFingerprint(Chem.MolFromSmiles(reference_smiles)),
    )
    actual = nearest_neighbour_tanimoto([query_smiles], [reference_smiles])[0]
    assert actual == pytest.approx(expected, abs=1e-6)
