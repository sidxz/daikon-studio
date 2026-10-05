from pathlib import Path
from typing import Any

import numpy as np

from daikonstudio.infrastructure.chem.featurize import ecfp4

# Rows of the similarity matrix held at once. The full matrix for n structures is
# n^2 floats -- 1.6 GB at n=20000 -- so the pair scan walks it in bands instead of
# materializing it. 256 rows is 20 MB at that same n.
_BAND_ROWS = 256


# Query rows compared against the whole reference set at once. At a 100k-compound
# training set this band is 128 x 100k x 4 B = 51 MB; the whole query at once
# (10k x 100k) was 4 GB, which is what the single-matrix version allocated.
_QUERY_BAND = 128


def nearest_neighbours_tanimoto(
    query: list[str], reference: list[str], k: int
) -> tuple[np.ndarray, np.ndarray]:
    """The `k` most Tanimoto-similar reference molecules for each query molecule.

    Returns `(indices, similarities)`, each `(len(query), min(k, len(reference)))`,
    most similar first, equal similarities by the lower reference index. One search
    serves applicability, Scorecard coverage and map placement, so the three can never
    disagree about a compound's neighbours. A large search runs on the GPU when the
    process has one (`_cuda_torch`), with the same similarities; when several molecules
    tie for the k-th place, the two paths may keep different ones of them.
    """
    width = min(k, len(reference))
    if width == 0 or not query:
        return (
            np.zeros((len(query), width), dtype=np.int32),
            np.zeros((len(query), width), dtype=np.float32),
        )
    q = ecfp4(query).astype(np.float32)
    r = ecfp4(reference).astype(np.float32)
    torch = _cuda_torch() if len(query) * len(reference) >= _GPU_MIN_PAIRS else None
    if torch is not None:
        return _top_k_torch(q, r, width, torch, torch.device("cuda"))
    return _top_k_numpy(q, r, width)


# Below this many query x reference pairs the CPU takes seconds, less than loading torch
# and starting CUDA would cost. The map that motivated the GPU path placed 254k compounds
# against a 150k fit sample: 3.8e10 pairs, 30-60 minutes on one core (prod, 2026-10-05).
_GPU_MIN_PAIRS = 100_000_000


def _cuda_torch() -> Any:
    """torch, when this process can use a CUDA GPU; otherwise None.

    /dev/nvidiactl first, so a process with no GPU never imports torch: loading it puts
    a second OpenMP runtime in the process and costs every later tree fit its threads
    (`engines._options.tree_threads`). The CPU image has no torch at all.
    """
    if not Path("/dev/nvidiactl").exists():
        return None
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def _top_k_numpy(q: np.ndarray, r: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    r_counts = r.sum(axis=1)
    indices = np.empty((len(q), width), dtype=np.int32)
    similarities = np.empty((len(q), width), dtype=np.float32)
    for start in range(0, len(q), _QUERY_BAND):
        band = q[start : start + _QUERY_BAND]
        intersection = band @ r.T
        union = band.sum(axis=1)[:, None] + r_counts[None, :] - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.where(union > 0, intersection / union, 0.0).astype(np.float32)
        if width < similarity.shape[1]:
            top = np.argpartition(-similarity, width - 1, axis=1)[:, :width]
        else:
            top = np.tile(np.arange(similarity.shape[1]), (similarity.shape[0], 1))
        top_similarity = np.take_along_axis(similarity, top, axis=1)
        stop = start + band.shape[0]
        indices[start:stop], similarities[start:stop] = _most_similar_first(top, top_similarity)
    return indices, similarities


def _top_k_torch(
    q: np.ndarray, r: np.ndarray, width: int, torch: Any, device: Any
) -> tuple[np.ndarray, np.ndarray]:
    """`_top_k_numpy` on a torch device, in bands sized to about 2 GB of working memory.

    On CUDA the fingerprints go up as float16: their bits are 0 or 1, so every partial
    sum of an intersection is a whole number no larger than 2048, which float16 holds
    exactly. Union and similarity are float32, as on the CPU, so the similarities are
    the same numbers; `_most_similar_first` orders ties the same way.
    """
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    ref = torch.from_numpy(r).to(device=device, dtype=dtype)
    ref_counts = ref.sum(dim=1, dtype=torch.float32)
    # Per query row: the float16 intersection plus three float32 rows (intersection,
    # union, similarity) against every reference molecule.
    rows = int(min(8192, max(64, 2_000_000_000 // (14 * len(r)))))
    indices = np.empty((len(q), width), dtype=np.int32)
    similarities = np.empty((len(q), width), dtype=np.float32)
    for start in range(0, len(q), rows):
        band = torch.from_numpy(q[start : start + rows]).to(device=device, dtype=dtype)
        intersection = (band @ ref.T).float()
        union = band.sum(dim=1, dtype=torch.float32)[:, None] + ref_counts[None, :] - intersection
        similarity = torch.where(union > 0, intersection / union, torch.zeros_like(union))
        top_similarity, top = torch.topk(similarity, width, dim=1)
        stop = start + band.shape[0]
        indices[start:stop], similarities[start:stop] = _most_similar_first(
            top.cpu().numpy(), top_similarity.cpu().numpy()
        )
    return indices, similarities


def _most_similar_first(
    top: np.ndarray, top_similarity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each row's candidates, most similar first and ties by the lower reference index,
    so the two search paths order a row identically."""
    order = np.lexsort((top, -top_similarity))
    return (
        np.take_along_axis(top, order, axis=1).astype(np.int32),
        np.take_along_axis(top_similarity, order, axis=1).astype(np.float32),
    )


def nearest_neighbour_tanimoto(query: list[str], reference: list[str]) -> np.ndarray:
    """Max Tanimoto from each query molecule to any reference molecule."""
    if not reference:
        return np.zeros(len(query))
    return nearest_neighbours_tanimoto(query, reference, 1)[1][:, 0]


def high_similarity_pairs(
    structures: list[str], *, threshold: float, limit: int
) -> list[tuple[int, int, float]]:
    """Index pairs whose Tanimoto is at least `threshold`, strongest first.

    Each unordered pair appears once (`i < j`), and the return is capped at
    `limit`. The cap is on *similarity*, not on whatever the caller ranks by
    afterwards -- an activity-cliff search wants the largest measured difference
    among near-identical structures, and it gets to see the `limit` most similar
    pairs to look for it in. A congeneric series can have far more than `limit`
    pairs above the threshold, in which case the caller is choosing from the most
    similar ones rather than from all of them.

    ponytail: an O(n^2) scan over fingerprint bands. Fine at the few thousand
    structures the caller subsamples to; if this ever needs the full 100k-compound
    set, an LSH/NN-descent index over the fingerprints is the upgrade path.
    """
    n = len(structures)
    if n < 2 or limit <= 0:
        return []

    fingerprints = ecfp4(structures).astype(np.float32)
    counts = fingerprints.sum(axis=1)

    pairs: list[tuple[int, int, float]] = []
    for start in range(0, n, _BAND_ROWS):
        band = fingerprints[start : start + _BAND_ROWS]
        intersection = band @ fingerprints.T
        union = counts[start : start + _BAND_ROWS, None] + counts[None, :] - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.where(union > 0, intersection / union, 0.0)

        # Only the strict upper triangle, so a pair is reported once and a
        # structure is never paired with itself (which is always 1.0).
        rows, columns = np.nonzero(similarity >= threshold)
        for row, column in zip(rows, columns, strict=True):
            left = start + int(row)
            right = int(column)
            if right > left:
                pairs.append((left, right, float(similarity[row, column])))

    pairs.sort(key=lambda pair: pair[2], reverse=True)
    return pairs[:limit]
