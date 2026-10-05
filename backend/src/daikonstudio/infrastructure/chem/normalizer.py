"""The concrete `StructureNormalizer` (application/ports) adapter, backed by RDKit.

A thin wrapper, not a reimplementation: every method delegates straight to Task 6's
module-level functions. Importing this module still imports the `chem` package, so
the RDKit log suppression in `chem/__init__.py` still applies.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from daikonstudio.infrastructure.chem.canonicalize import canonicalize, has_multiple_components
from daikonstudio.infrastructure.chem.descriptors import descriptors
from daikonstudio.infrastructure.chem.scaffold import murcko_scaffold
from daikonstudio.infrastructure.chem.similarity import (
    LARGE_SEARCH_PAIRS,
    high_similarity_pairs,
    nearest_neighbours_tanimoto,
)


class RdkitStructureNormalizer:
    def canonicalize(self, smiles: str) -> str | None:
        return canonicalize(smiles)

    def has_multiple_components(self, smiles: str) -> bool:
        return has_multiple_components(smiles)

    def murcko_scaffold(self, smiles: str) -> str:
        return murcko_scaffold(smiles)

    def nearest_neighbour_tanimoto(self, query: list[str], reference: list[str]) -> list[float]:
        if not reference:
            return [0.0] * len(query)
        return [float(v) for v in _search(query, reference, 1)[1][:, 0]]

    def nearest_neighbours_tanimoto(
        self, query: list[str], reference: list[str], k: int
    ) -> tuple[list[list[int]], list[list[float]]]:
        indices, similarities = _search(query, reference, k)
        return indices.tolist(), similarities.astype(float).tolist()

    def descriptors(self, smiles_list: list[str]) -> dict[str, list[float | None]]:
        return descriptors(smiles_list)

    def high_similarity_pairs(
        self, structures: list[str], *, threshold: float, limit: int
    ) -> list[tuple[int, int, float]]:
        return high_similarity_pairs(structures, threshold=threshold, limit=limit)


def _search(query: list[str], reference: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
    """`nearest_neighbours_tanimoto`, in a child process when the search is large.

    Featurizing and comparing hundreds of thousands of molecules holds the calling
    process's GIL for minutes, and a thread does not help: in the API, a 404k-compound
    Protocol's Scorecard (40k x 323k pairs) starved the event loop until /ready stopped
    answering and the healthcheck killed the API, again on every reload (prod,
    2026-10-05). A child has its own GIL, so the caller's other work goes on while it
    waits. Spawned, not forked: a fresh interpreter carries no torch, no event loop and
    no locks held by other threads. Small searches stay in-process, where starting a
    child would cost more than the search.
    """
    if len(query) * len(reference) < LARGE_SEARCH_PAIRS:
        return nearest_neighbours_tanimoto(query, reference, k)
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=spawn) as pool:
        return pool.submit(_search_in_child, query, reference, k).result()


def _search_in_child(
    query: list[str], reference: list[str], k: int
) -> tuple[np.ndarray, np.ndarray]:
    """The child's half: the BLAS may use several threads here. The images pin
    OMP_NUM_THREADS=1 because torch's OpenMP and another runtime in one process can
    crash; this fresh process has not loaded torch (the GPU path, when it runs, loads
    it after this limit is set, so torch's own libraries keep theirs)."""
    with threadpool_limits(limits=min(8, os.process_cpu_count() or 1), user_api="blas"):
        return nearest_neighbours_tanimoto(query, reference, k)
