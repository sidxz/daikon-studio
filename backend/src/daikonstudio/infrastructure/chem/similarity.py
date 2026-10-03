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
    most similar first. One search serves applicability, Scorecard coverage and
    map placement, so the three can never disagree about a compound's neighbours.
    """
    width = min(k, len(reference))
    if width == 0 or not query:
        return (
            np.zeros((len(query), width), dtype=np.int32),
            np.zeros((len(query), width), dtype=np.float32),
        )
    q = ecfp4(query).astype(np.float32)
    r = ecfp4(reference).astype(np.float32)
    r_counts = r.sum(axis=1)
    indices = np.empty((len(query), width), dtype=np.int32)
    similarities = np.empty((len(query), width), dtype=np.float32)
    for start in range(0, len(query), _QUERY_BAND):
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
        order = np.lexsort((top, -top_similarity))
        stop = start + band.shape[0]
        indices[start:stop] = np.take_along_axis(top, order, axis=1)
        similarities[start:stop] = np.take_along_axis(top_similarity, order, axis=1)
    return indices, similarities


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
