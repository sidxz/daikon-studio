import numpy as np

from daikonstudio.infrastructure.chem.featurize import ecfp4

# Rows of the similarity matrix held at once. The full matrix for n structures is
# n^2 floats -- 1.6 GB at n=20000 -- so the pair scan walks it in bands instead of
# materializing it. 256 rows is 20 MB at that same n.
_BAND_ROWS = 256


def nearest_neighbour_tanimoto(query: list[str], reference: list[str]) -> np.ndarray:
    """Max Tanimoto from each query molecule to any reference molecule."""
    if not reference:
        return np.zeros(len(query))
    q = ecfp4(query).astype(np.float32)
    r = ecfp4(reference).astype(np.float32)
    intersection = q @ r.T
    union = q.sum(axis=1)[:, None] + r.sum(axis=1)[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.where(union > 0, intersection / union, 0.0)
    return similarity.max(axis=1)


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
