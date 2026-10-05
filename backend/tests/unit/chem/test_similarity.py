import numpy as np
import pytest

from daikonstudio.infrastructure.chem import similarity
from daikonstudio.infrastructure.chem.featurize import ecfp4
from daikonstudio.infrastructure.chem.similarity import (
    nearest_neighbour_tanimoto,
    nearest_neighbours_tanimoto,
)

QUERY = ["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "not a smiles"]
REFERENCE = ["CCO", "CCCO", "c1ccccc1", "c1ccccc1N", "CC(=O)Nc1ccccc1", "CCN", "C1CCCCC1"]


def _brute(query: list[str], reference: list[str]) -> np.ndarray:
    q = ecfp4(query).astype(bool)
    r = ecfp4(reference).astype(bool)
    out = np.zeros((len(query), len(reference)))
    for i in range(len(query)):
        for j in range(len(reference)):
            union = np.logical_or(q[i], r[j]).sum()
            out[i, j] = np.logical_and(q[i], r[j]).sum() / union if union else 0.0
    return out


def test_top_k_matches_brute_force():
    expected = _brute(QUERY, REFERENCE)
    indices, sims = nearest_neighbours_tanimoto(QUERY, REFERENCE, 3)
    assert indices.shape == (4, 3)
    for i in range(len(QUERY)):
        np.testing.assert_allclose(sims[i], np.sort(expected[i])[::-1][:3], rtol=1e-6)
        np.testing.assert_allclose(expected[i, indices[i]], sims[i], rtol=1e-6)


def test_k_larger_than_the_reference_returns_every_reference():
    indices, _ = nearest_neighbours_tanimoto(QUERY, REFERENCE[:2], 5)
    assert indices.shape == (4, 2)
    assert sorted(indices[0].tolist()) == [0, 1]


def test_an_identical_structure_is_its_own_nearest_neighbour():
    indices, sims = nearest_neighbours_tanimoto(["CCO"], REFERENCE, 1)
    assert indices[0, 0] == 0
    assert sims[0, 0] == 1.0


def test_top_1_is_the_old_maximum():
    expected = _brute(QUERY, REFERENCE).max(axis=1)
    np.testing.assert_allclose(nearest_neighbour_tanimoto(QUERY, REFERENCE), expected, rtol=1e-6)


def test_banding_does_not_change_the_answer(monkeypatch):
    whole = nearest_neighbours_tanimoto(QUERY, REFERENCE, 2)
    monkeypatch.setattr(similarity, "_QUERY_BAND", 1)
    banded = similarity.nearest_neighbours_tanimoto(QUERY, REFERENCE, 2)
    np.testing.assert_array_equal(whole[1], banded[1])


def test_an_empty_reference_gives_empty_rows():
    indices, sims = nearest_neighbours_tanimoto(QUERY, [], 5)
    assert indices.shape == (4, 0)
    assert sims.shape == (4, 0)


def test_the_gpu_search_finds_the_same_neighbours_as_the_cpu_search():
    """Run on torch's CPU device, which shares every line with the CUDA path but the
    dtype (float16 there, exact for 0/1 bits): same similarities, same order, and the
    same molecules wherever the k-th place is not a tie."""
    torch = pytest.importorskip("torch")
    from daikonstudio.infrastructure.chem.similarity import _top_k_numpy, _top_k_torch

    rng = np.random.default_rng(7)
    q = (rng.random((300, 2048)) < 0.02).astype(np.float32)
    r = (rng.random((900, 2048)) < 0.02).astype(np.float32)

    cpu_indices, cpu_similarities = _top_k_numpy(q, r, 5)
    gpu_indices, gpu_similarities = _top_k_torch(q, r, 5, torch, torch.device("cpu"))

    assert np.array_equal(cpu_similarities, gpu_similarities)
    full = _top_k_numpy(q, r, 6)[1]
    untied = full[:, 4] > full[:, 5]
    assert untied.mean() > 0.5  # the comparison below is not vacuous
    assert np.array_equal(cpu_indices[untied], gpu_indices[untied])


def test_a_reference_in_many_chunks_finds_what_one_chunk_finds(monkeypatch):
    """The reference is converted a chunk at a time, keeping each query row's best so
    far; at prod's 323k compounds that is 20 chunks. Same similarities, and the same
    molecules wherever the k-th place is not a tie."""
    from daikonstudio.infrastructure.chem.similarity import _top_k_numpy

    rng = np.random.default_rng(11)
    q = (rng.random((200, 2048)) < 0.02).astype(np.uint8)
    r = (rng.random((1000, 2048)) < 0.02).astype(np.uint8)
    whole_indices, whole_similarities = _top_k_numpy(q, r, 5)
    full = _top_k_numpy(q, r, 6)[1]

    monkeypatch.setattr(similarity, "_REFERENCE_CHUNK", 37)  # 28 chunks, a ragged last one
    chunked_indices, chunked_similarities = _top_k_numpy(q, r, 5)

    assert np.array_equal(whole_similarities, chunked_similarities)
    untied = full[:, 4] > full[:, 5]
    assert untied.mean() > 0.5
    assert np.array_equal(whole_indices[untied], chunked_indices[untied])
