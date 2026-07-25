import numpy as np

from daikonstudio.infrastructure.chem.featurize import ecfp4


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
