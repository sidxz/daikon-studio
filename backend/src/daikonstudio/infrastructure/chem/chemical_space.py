"""UMAP layout of compounds for the chemical-space map (`ChemicalSpaceLayout`)."""

import warnings
from importlib.metadata import version

import numpy as np
import scipy.sparse as sp

from daikonstudio.application.catalog.chemical_space import NEIGHBOURS, place
from daikonstudio.application.ports.chemical_space_layout import TooFewCompounds
from daikonstudio.infrastructure.chem.featurize import ecfp4
from daikonstudio.infrastructure.chem.similarity import nearest_neighbours_tanimoto

MIN_COMPOUNDS = 5
N_NEIGHBORS = 15
MIN_DIST = 0.1
#: ponytail: above this, UMAP is fitted on a seeded sample and the rest are placed
#: by their nearest fitted neighbours, the rule runs use. Measured 71.5 s and 1.1 GB
#: at 100k, so 150k keeps the worst case near 2 minutes. Raise it, or fit on a GPU,
#: when a bigger set needs every compound in the fit.
FIT_CEILING = 150_000


class UmapLayout:
    def describe(self) -> dict[str, object]:
        return {
            "method": "umap",
            "umap_version": version("umap-learn"),
            "params": {
                "metric": "jaccard",
                "fingerprint": "ecfp4-2048",
                "n_neighbors": N_NEIGHBORS,
                "min_dist": MIN_DIST,
                "fit_ceiling": FIT_CEILING,
            },
        }

    def layout(self, structures: list[str], seed: int) -> tuple[list[float], list[float]]:
        import umap  # deferred: importing numba costs seconds, and only map builds need it

        n = len(structures)
        if n < MIN_COMPOUNDS:
            raise TooFewCompounds(f"{n} compounds; a map needs at least {MIN_COMPOUNDS}.")
        fit = np.arange(n)
        if n > FIT_CEILING:
            fit = np.sort(np.random.default_rng(seed).choice(n, FIT_CEILING, replace=False))
        bits = sp.csr_matrix(ecfp4([structures[i] for i in fit]).astype(bool))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(N_NEIGHBORS, len(fit) - 1),
            min_dist=MIN_DIST,
            metric="jaccard",
            random_state=seed,
            # A seed makes UMAP single-threaded anyway; saying so silences its warning.
            n_jobs=1,
        )
        with warnings.catch_warnings():
            # Jaccard has no gradient, so `inverse_transform` is unavailable. Nothing
            # here inverts a map; the warning would only fill every training log.
            warnings.filterwarnings("ignore", message="gradient function is not yet implemented")
            fitted = np.asarray(reducer.fit_transform(bits), dtype=float)
        xy = np.empty((n, 2))
        xy[fit] = fitted
        if len(fit) < n:
            rest = np.setdiff1d(np.arange(n), fit)
            indices, similarities = nearest_neighbours_tanimoto(
                [structures[i] for i in rest], [structures[i] for i in fit], NEIGHBOURS
            )
            xy[rest] = place(fitted, indices, similarities)
        return _unit_square(xy)


def _unit_square(xy: np.ndarray) -> tuple[list[float], list[float]]:
    """Scale into [0, 1] keeping the aspect ratio, centring the shorter axis."""
    low = xy.min(axis=0)
    extent = xy.max(axis=0) - low
    span = float(extent.max())
    if span == 0:
        return [0.5] * len(xy), [0.5] * len(xy)
    unit = (xy - low) / span + (1 - extent / span) / 2
    return unit[:, 0].tolist(), unit[:, 1].tolist()
