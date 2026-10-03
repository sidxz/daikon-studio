# Chemical Space Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A real-data WebGL map of chemical space: on a run, the protocol's training compounds and the run's compounds; on a protocol, its training, validation and test compounds. The map shows where compounds sit; their real Tanimoto similarity shows how far.

**Architecture:**
- **Training:** the training job fits a seeded UMAP over the dataset snapshot and writes a points parquet plus a meta JSON next to the protocol. This is best-effort.
- **Prediction:** the prediction job keeps each compound's five nearest training compounds, found by a chunked top-k Tanimoto search that also replaces the old top-1 search, in a `neighbors.parquet` beside the results.
- **API:** four read endpoints. Run compounds are placed server-side at the similarity-weighted center of their neighbors.
- **Backfill:** a one-off command fills in existing protocols and runs.
- **Frontend:** ChemCellar's WebGL2 point renderer, ported, with pan, zoom, picking and per-style shapes.

**Tech Stack:** Python 3.13, FastAPI, polars, numpy/scipy, RDKit, umap-learn 0.5.12 (added); Next 16, React 19, TypeScript strict, WebGL2, vitest, pytest (Postgres on :5437 for API tests).

**Spec:** `docs/superpowers/specs/2026-10-03-chemical-space-map-design.md`

## Global Constraints

- Training compounds are the `split == "train"` rows in snapshot order, and they equal `ScorecardInputs.train_structures` index for index. Neighbor indices refer to that order.
- The map is best-effort: a mapping failure must never fail training or prediction.
- Run results are immutable. Neighbors go in `{ws}/runs/{rid}/neighbors.parquet`, never into `predictions.parquet`.
- Applicability is `neighbor_similarity[0]` from the same search, and the threshold is the backend's `_APPLICABILITY_THRESHOLD` and the frontend's `IN_DOMAIN_FLOOR` (0.3).
- UMAP: `metric="jaccard"`, `n_neighbors=min(15, n-1)`, `min_dist=0.1`, `random_state=dataset.split.seed`. No map below 5 compounds. Fit ceiling 150,000.
- `MAP_VERSION = 1`. The meta JSON is written after the points parquet, so its presence means the map is complete.
- Every new route is workspace-scoped through the existing repositories and `require_authenticated`.
- No new frontend dependency. WebGL2 only, with a text fallback.
- Colors come only from tokens (`chart-1`, `chart-2`, `ds-score-fair`, `foreground`), re-read when `data-theme` changes.
- Copy is American English, sentence case, with no em-dash chains (`docs/copy-audit.md`).
- Backend commands run from `backend/`: `uv run pytest …`, `uv run ruff check src tests`, `uv run ruff format`. Frontend commands run from `frontend/`: `pnpm test`, `pnpm lint`, `pnpm exec tsc --noEmit`, `pnpm build`.

## Review Focus

1. **A protocol whose training set is smaller than 5:** no map, and both endpoints return `status: "missing"` instead of failing. Test in Task 3 (`TooFewCompounds`) and Task 6 (missing map).
2. **A run on a protocol whose map was never built, or a run predicted before neighbors existed:** `status: "missing"` and the card's "No map has been computed" state, never an exception. Test in Task 6.
3. **A query compound that RDKit cannot parse:** an all-zero fingerprint, similarity 0 to everything, so placement falls back to the plain mean of 5 neighbors. Test in Task 1 (`not a smiles`) and Task 3 (`place` with all-zero weights).
4. **WebGL2 unavailable, or context creation failing:** the card still shows the summary and a one-line fallback. Test in Task 9.
5. **Theme switch while the map is open:** the colors update without re-uploading points. Covered by the color hook's re-read and checked live in Task 11.

---

### Task 1: Chunked top-k Tanimoto search

**Files:**
- Modify: `backend/src/daikonstudio/infrastructure/chem/similarity.py`
- Modify: `backend/src/daikonstudio/application/ports/structure_normalizer.py`
- Modify: `backend/src/daikonstudio/infrastructure/chem/normalizer.py`
- Test: `backend/tests/unit/chem/test_similarity.py`

**Interfaces:**
- Produces: `nearest_neighbours_tanimoto(query: list[str], reference: list[str], k: int) -> tuple[np.ndarray, np.ndarray]` with shapes `(n, min(k, len(reference)))`, int32 indices and float32 similarities, most similar first. `StructureNormalizer.nearest_neighbours_tanimoto(query, reference, k) -> tuple[list[list[int]], list[list[float]]]`. `nearest_neighbour_tanimoto` is kept and now delegates with `k = 1`.

- [ ] **Step 1: Failing tests** (`tests/unit/chem/test_similarity.py`)

```python
import numpy as np

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
```

- [ ] **Step 2:** Run `uv run pytest tests/unit/chem/test_similarity.py -q`. Expected: FAIL, ImportError for `nearest_neighbours_tanimoto`.

- [ ] **Step 3: Implement.** In `similarity.py`, replace `nearest_neighbour_tanimoto` with:

```python
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
```

In the port (`structure_normalizer.py`), add below `nearest_neighbour_tanimoto`:
```python
    def nearest_neighbours_tanimoto(
        self, query: list[str], reference: list[str], k: int
    ) -> tuple[list[list[int]], list[list[float]]]:
        """The `k` nearest reference indices and their Tanimoto, most similar first."""
        ...
```
In the adapter (`normalizer.py`), import `nearest_neighbours_tanimoto` and add:
```python
    def nearest_neighbours_tanimoto(
        self, query: list[str], reference: list[str], k: int
    ) -> tuple[list[list[int]], list[list[float]]]:
        indices, similarities = nearest_neighbours_tanimoto(query, reference, k)
        return indices.tolist(), similarities.astype(float).tolist()
```

- [ ] **Step 4:** Run `uv run pytest tests/unit/chem -q && uv run pytest tests/unit/catalog tests/unit/data -q`. Expected: all PASS. The second command covers existing applicability callers.
- [ ] **Step 5:** Run `uv run ruff check src tests && uv run ruff format src tests`, then commit `feat(chem): chunked top-k Tanimoto search; top-1 delegates to it`.

---

### Task 2: Runner blob 404 reads as a missing file

**Files:**
- Modify: `backend/src/daikonstudio/infrastructure/runner/ports.py` (`HttpBlobStore.get_bytes`)
- Test: `backend/tests/api/test_runner_ports.py` (append)

- [ ] **Step 1: Failing test.** First read the existing `test_runner_ports.py` to learn how it builds a `RunnerApiClient`/`HttpBlobStore` against the app. Then add a test in the same style: a `get_bytes` on a key that does not exist under the run's workspace raises `FileNotFoundError`, not `httpx.HTTPStatusError`. Name it `test_a_missing_blob_is_a_missing_file`.
- [ ] **Step 2:** Run it. Expected: FAIL with `HTTPStatusError`.
- [ ] **Step 3: Implement.**
```python
    def get_bytes(self, key: str) -> bytes:
        response = self._client._blobs.get(f"/runs/{self._client.run_id}/blobs/{key}")
        if response.status_code == 404:
            # The signal FsspecBlobStore gives for the same thing, so a best-effort
            # read (`_train_structures`) degrades on a runner exactly as it does inline.
            raise FileNotFoundError(key)
        response.raise_for_status()
        return response.content
```
- [ ] **Step 4:** Run `uv run pytest tests/api/test_runner_ports.py -q`. Expected: PASS.
- [ ] **Step 5:** Commit `fix(runner): a missing blob raises FileNotFoundError on runners too`.

---

### Task 3: Map layout, storage contract and placement

**Files:**
- Create: `backend/src/daikonstudio/application/ports/chemical_space_layout.py`
- Create: `backend/src/daikonstudio/application/catalog/chemical_space.py`
- Create: `backend/src/daikonstudio/infrastructure/chem/chemical_space.py`
- Test: `backend/tests/unit/catalog/test_chemical_space.py`, `backend/tests/unit/chem/test_chemical_space_layout.py`

**Interfaces:**
- Produces:
  - `ChemicalSpaceLayout` (Protocol: `layout(structures, seed) -> tuple[list[float], list[float]]`, `describe() -> dict[str, object]`) and `TooFewCompounds(Exception)`.
  - `MAP_VERSION`, `NEIGHBOURS = 5`, `PARTITION_CODES = {"train": 0, "validation": 1, "test": 2}`.
  - Keys: `chemical_space_points_key(ws, pid)`, `chemical_space_meta_key(ws, pid)`, `neighbours_key(ws, rid)`.
  - `place(coords, indices, similarities) -> np.ndarray`.
  - `build_chemical_space(frame, structure_column, seed, layout) -> tuple[bytes, bytes]`, `write_chemical_space(store, ws, pid, frame, structure_column, seed, layout) -> None`.
  - `neighbours_parquet(indices, similarities) -> bytes`.
  - `read_meta(store, ws, pid) -> dict | None`, `read_points(store, ws, pid) -> pl.DataFrame`, `read_neighbours(store, ws, rid) -> pl.DataFrame | None`.
  - `UmapLayout`.

- [ ] **Step 1: Failing tests.**

`tests/unit/catalog/test_chemical_space.py`:
```python
import io
import json

import numpy as np
import polars as pl

from daikonstudio.application.catalog.chemical_space import (
    MAP_VERSION,
    build_chemical_space,
    neighbours_parquet,
    place,
)


class _Diagonal:
    """A layout that puts compound i at (i, i), so order is easy to read back."""

    def layout(self, structures, seed):
        return [float(i) for i in range(len(structures))], [float(i) for i in range(len(structures))]

    def describe(self):
        return {"method": "test", "params": {}}


def test_place_is_the_similarity_weighted_centre():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    xy = place(coords, np.array([[0, 1]]), np.array([[1.0, 3.0]]))
    np.testing.assert_allclose(xy, [[0.75, 0.0]])


def test_place_falls_back_to_the_plain_centre_when_nothing_is_similar():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    xy = place(coords, np.array([[0, 1, 2]]), np.zeros((1, 3)))
    np.testing.assert_allclose(xy, [[1 / 3, 1 / 3]])


def test_the_map_keeps_snapshot_order_so_train_rows_match_the_training_set():
    frame = pl.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCI"],
            "split": ["test", "train", "validation", "train", "train", "test"],
        }
    )
    points_bytes, meta_bytes = build_chemical_space(frame, "smiles", 1, _Diagonal())
    points = pl.read_parquet(io.BytesIO(points_bytes))
    train = points.filter(pl.col("partition") == 0)
    assert train["structure"].to_list() == ["CCN", "CCCl", "CCBr"]
    assert points["partition"].to_list() == [2, 0, 1, 0, 0, 2]
    meta = json.loads(meta_bytes)
    assert meta["version"] == MAP_VERSION
    assert meta["counts"] == {"train": 3, "validation": 1, "test": 2}
    assert meta["params"]["seed"] == 1


def test_neighbours_round_trip_as_lists():
    data = neighbours_parquet([[3, 1], [0, 2]], [[0.9, 0.5], [0.4, 0.1]])
    frame = pl.read_parquet(io.BytesIO(data))
    assert frame["neighbor_index"].to_list() == [[3, 1], [0, 2]]
    np.testing.assert_allclose(frame["neighbor_similarity"].to_list(), [[0.9, 0.5], [0.4, 0.1]], rtol=1e-6)
```

`tests/unit/chem/test_chemical_space_layout.py`:
```python
import numpy as np
import pytest

from daikonstudio.application.ports.chemical_space_layout import TooFewCompounds
from daikonstudio.infrastructure.chem.chemical_space import UmapLayout

ALCOHOLS = ["C" * n + "O" for n in range(1, 11)]
AROMATICS = [
    "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "c1ccc2ccccc2c1", "Cc1ccc2ccccc2c1",
    "c1ccc(-c2ccccc2)cc1", "Cc1ccc(-c2ccccc2)cc1", "c1ccc2cc3ccccc3cc2c1",
    "Oc1ccccc1", "Nc1ccccc1",
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
    spread = max(np.linalg.norm(a - a.mean(axis=0), axis=1).mean(), np.linalg.norm(b - b.mean(axis=0), axis=1).mean())
    assert gap > spread


def test_too_few_compounds_is_refused():
    with pytest.raises(TooFewCompounds):
        UmapLayout().layout(["CCO", "CCN"], seed=1)
```

- [ ] **Step 2:** Run both test files. Expected: FAIL, modules not found.

- [ ] **Step 3: Implement.**

`application/ports/chemical_space_layout.py`:
```python
from typing import Protocol


class TooFewCompounds(Exception):
    """Fewer compounds than a 2D layout can say anything about."""


class ChemicalSpaceLayout(Protocol):
    def layout(self, structures: list[str], seed: int) -> tuple[list[float], list[float]]:
        """2D coordinates in [0, 1] for each structure, deterministic for a seed."""
        ...

    def describe(self) -> dict[str, object]:
        """`method`, `params` and library version, recorded with the map."""
        ...
```

`application/catalog/chemical_space.py`:
```python
"""The chemical-space map: where it is stored, how compounds are placed, how it is built.

A map is one point per dataset compound, in snapshot order, so the train rows read
in order are exactly `ScorecardInputs.train_structures`. A run's compounds are not
in the fit; each is placed at the similarity-weighted centre of its nearest training
compounds, whose indices count in that same train order.
"""

import io
import json
import uuid

import numpy as np
import polars as pl

from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.chemical_space_layout import ChemicalSpaceLayout

MAP_VERSION = 1
NEIGHBOURS = 5
PARTITION_CODES = {"train": 0, "validation": 1, "test": 2}


def chemical_space_points_key(workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> str:
    return f"{workspace_id}/protocols/{protocol_id}/chemical-space.parquet"


def chemical_space_meta_key(workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> str:
    """Written after the points, so its presence means the map is complete."""
    return f"{workspace_id}/protocols/{protocol_id}/chemical-space.json"


def neighbours_key(workspace_id: uuid.UUID, run_id: uuid.UUID) -> str:
    """Beside, never inside, `predictions.parquet`: a run's results are written once."""
    return f"{workspace_id}/runs/{run_id}/neighbors.parquet"


def place(coords: np.ndarray, indices: np.ndarray, similarities: np.ndarray) -> np.ndarray:
    """The similarity-weighted centre of each row's neighbours.

    A compound unlike everything (all similarities 0) sits at the plain centre of
    its neighbours. It still has to be drawn somewhere, and its dashed ring and
    similarity say how far it really is.
    """
    weights = np.asarray(similarities, dtype=float)
    totals = weights.sum(axis=1, keepdims=True)
    width = max(weights.shape[1], 1)
    weights = np.where(totals > 0, weights / np.where(totals > 0, totals, 1.0), 1.0 / width)
    return np.einsum("nk,nkd->nd", weights, coords[np.asarray(indices)])


def build_chemical_space(
    frame: pl.DataFrame, structure_column: str, seed: int, layout: ChemicalSpaceLayout
) -> tuple[bytes, bytes]:
    structures = [str(s) for s in frame[structure_column].to_list()]
    xs, ys = layout.layout(structures, seed)
    partition = [PARTITION_CODES[str(s)] for s in frame["split"].to_list()]
    points = pl.DataFrame(
        {
            "x": pl.Series(xs, dtype=pl.Float32),
            "y": pl.Series(ys, dtype=pl.Float32),
            "partition": pl.Series(partition, dtype=pl.Int8),
            "structure": pl.Series(structures, dtype=pl.String),
        }
    )
    described = layout.describe()
    meta = {
        **described,
        "version": MAP_VERSION,
        "params": {**dict(described.get("params", {})), "seed": seed},  # type: ignore[arg-type]
        "counts": {name: partition.count(code) for name, code in PARTITION_CODES.items()},
    }
    buffer = io.BytesIO()
    points.write_parquet(buffer)
    return buffer.getvalue(), json.dumps(meta).encode()


def write_chemical_space(
    store: BlobStore,
    workspace_id: uuid.UUID,
    protocol_id: uuid.UUID,
    frame: pl.DataFrame,
    structure_column: str,
    seed: int,
    layout: ChemicalSpaceLayout,
) -> None:
    points, meta = build_chemical_space(frame, structure_column, seed, layout)
    store.put_bytes(chemical_space_points_key(workspace_id, protocol_id), points)
    store.put_bytes(chemical_space_meta_key(workspace_id, protocol_id), meta)


def neighbours_parquet(indices: list[list[int]], similarities: list[list[float]]) -> bytes:
    buffer = io.BytesIO()
    pl.DataFrame(
        {
            "neighbor_index": pl.Series(indices, dtype=pl.List(pl.Int32)),
            "neighbor_similarity": pl.Series(similarities, dtype=pl.List(pl.Float32)),
        }
    ).write_parquet(buffer)
    return buffer.getvalue()


def read_meta(store: BlobStore, workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> dict | None:
    try:
        meta = json.loads(store.get_bytes(chemical_space_meta_key(workspace_id, protocol_id)))
    except (FileNotFoundError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def read_points(store: BlobStore, workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> pl.DataFrame:
    return pl.read_parquet(
        io.BytesIO(store.get_bytes(chemical_space_points_key(workspace_id, protocol_id)))
    )


def read_neighbours(
    store: BlobStore, workspace_id: uuid.UUID, run_id: uuid.UUID
) -> pl.DataFrame | None:
    try:
        return pl.read_parquet(io.BytesIO(store.get_bytes(neighbours_key(workspace_id, run_id))))
    except FileNotFoundError:
        return None
```

`infrastructure/chem/chemical_space.py`:
```python
"""UMAP layout of compounds for the chemical-space map (`ChemicalSpaceLayout`)."""

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
        )
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
```

- [ ] **Step 4:** Run `uv run pytest tests/unit/catalog/test_chemical_space.py tests/unit/chem/test_chemical_space_layout.py -q`. Expected: PASS. The first UMAP call compiles numba, about 10 s.
- [ ] **Step 5:** Lint, format, then commit `feat(map): UMAP layout, storage contract and neighbour placement`.

---

### Task 4: Training builds the map

**Files:**
- Modify: `backend/src/daikonstudio/application/execution/train_protocol.py` (`RunTraining.__init__` and the end of `__call__`)
- Modify: `backend/src/daikonstudio/infrastructure/jobs.py` (`_train` passes `UmapLayout()`)
- Test: `backend/tests/api/test_chemical_space.py` (new)

**Interfaces:**
- Consumes: Task 3 `write_chemical_space`, `chemical_space_meta_key`, `UmapLayout`, `TooFewCompounds`.
- Produces: `RunTraining(..., layout: ChemicalSpaceLayout | None = None)`.

- [ ] **Step 1: Failing test** (`tests/api/test_chemical_space.py`):
```python
"""The chemical-space map, end to end through the API.

Fixtures come from the protocol and triage suites: a 20-compound random split
(16 train / 2 validation / 2 test) and a finished prediction run on it.
"""

import json
import uuid

from tests.api.test_protocols import dataset_id, trained_protocol_id  # noqa: F401
from tests.api.test_triage_round_trip import ready_run_id  # noqa: F401


def _meta_path(tmp_path, workspace_id: uuid.UUID, protocol_id: str):
    return tmp_path / str(workspace_id) / "protocols" / protocol_id / "chemical-space.json"


async def test_training_writes_a_map_of_every_compound(tmp_path, workspace_id, trained_protocol_id):
    meta = json.loads(_meta_path(tmp_path, workspace_id, trained_protocol_id).read_text())
    assert meta["version"] == 1
    assert meta["method"] == "umap"
    assert meta["counts"] == {"train": 16, "validation": 2, "test": 2}
```
Before trusting the expected counts, check that the 20-compound fixture's seeded random split really gives 16/2/2. If it doesn't, assert against `GET /datasets/{id}` partition counts instead. Record a ruling either way.

- [ ] **Step 2:** Run `uv run pytest tests/api/test_chemical_space.py -q`. Expected: FAIL, file not found.

- [ ] **Step 3: Implement.** In `RunTraining.__init__`, add a keyword parameter `layout: ChemicalSpaceLayout | None = None` stored as `self._layout`. At the end of `__call__`, after `run.record_metrics(...)` and before `return result_uri`:
```python
        if self._layout is not None:
            await self._progress(run, 0.97, "Mapping chemical space")
            try:
                await asyncio.to_thread(
                    write_chemical_space,
                    self._store,
                    run.workspace_id,
                    protocol_id,
                    frame,
                    dataset.structure_column,
                    dataset.split.seed,
                    self._layout,
                )
            except TooFewCompounds:
                pass  # a map of four compounds says nothing; the page says "no map"
            except Exception:
                # Best-effort by design: a protocol without a map is a whole, honest
                # protocol, and one that failed to train because a picture failed is not.
                logger.exception("Chemical-space map failed for protocol %s", protocol_id)
```
Add `import logging` and `logger = logging.getLogger(__name__)` at the top, plus the imports. In `jobs.py` `_train`, pass `layout=UmapLayout()`.

- [ ] **Step 4:** Run `uv run pytest tests/api/test_chemical_space.py tests/api/test_protocols.py -q`. Expected: PASS.
- [ ] **Step 5:** Commit `feat(map): training builds the protocol's chemical-space map`.

---

### Task 5: Prediction keeps each compound's five nearest training compounds

**Files:**
- Modify: `backend/src/daikonstudio/application/execution/predict_with_protocol.py` (`RunPrediction.__call__`)
- Test: `backend/tests/api/test_chemical_space.py` (append)

- [ ] **Step 1: Failing test.** Append:
```python
import io

import polars as pl


async def test_a_prediction_keeps_five_neighbours_per_compound(
    client, tmp_path, workspace_id, ready_run_id
):
    run_dir = tmp_path / str(workspace_id) / "runs" / ready_run_id
    neighbours = pl.read_parquet(run_dir / "neighbors.parquet")
    predictions = pl.read_parquet(run_dir / "predictions.parquet")
    assert neighbours.height == predictions.height
    for indices, sims, applicability in zip(
        neighbours["neighbor_index"].to_list(),
        neighbours["neighbor_similarity"].to_list(),
        predictions["applicability"].to_list(),
        strict=True,
    ):
        assert 1 <= len(indices) <= 5
        assert sims == sorted(sims, reverse=True)
        assert abs(sims[0] - applicability) < 1e-6
```
- [ ] **Step 2:** Run it. Expected: FAIL, neighbors.parquet not found.
- [ ] **Step 3: Implement.** In `RunPrediction.__call__`, replace the `similarities` block:
```python
        train_structures = self._train_structures(protocol)
        neighbours: tuple[list[list[int]], list[list[float]]] | None = None
        similarities: list[float | None]
        if structures and train_structures:
            neighbours = self._normalizer.nearest_neighbours_tanimoto(
                structures, train_structures, NEIGHBOURS
            )
            # Applicability is the nearest neighbour from the same search, so the
            # map, the triage column and the domain filter can never disagree.
            similarities = [row[0] for row in neighbours[1]]
        else:
            similarities = [None] * len(structures)
```
Keep the existing comment block about never fabricating 0.0. Replace the final `return self._store.put_bytes(...)` with:
```python
        result_uri = self._store.put_bytes(
            predictions_key(run.workspace_id, run.id), buffer.getvalue()
        )
        if neighbours is not None:
            self._store.put_bytes(
                neighbours_key(run.workspace_id, run.id), neighbours_parquet(*neighbours)
            )
        return result_uri
```
- [ ] **Step 4:** Run `uv run pytest tests/api/test_chemical_space.py tests/api/test_runs.py tests/api/test_triage_round_trip.py -q`. Expected: PASS.
- [ ] **Step 5:** Commit `feat(map): predictions keep each compound's five nearest training compounds`.

---

### Task 6: The four read endpoints

**Files:**
- Create: `backend/src/daikonstudio/application/catalog/get_chemical_space.py`
- Modify: `backend/src/daikonstudio/infrastructure/di/container.py`, `interface/routes/protocols.py`, `interface/routes/runs.py`
- Test: `backend/tests/api/test_chemical_space.py` (append)
- Regenerate: `frontend/openapi.json` and the TS models with `make generate-api` from the repo root

**Interfaces:**
- Produces (HTTP, consumed by Task 10):
  - `GET /api/v1/protocols/{id}/chemical-space` → `ChemicalSpaceResponse {status: "ready" | "missing", method?, params?, counts?, points?: {x, y, partition}}`.
  - `GET /api/v1/protocols/{id}/chemical-space/compounds?indices=…` → `MapCompoundResponse[] {index, structure, partition: "train" | "validation" | "test"}`.
  - `GET /api/v1/runs/{id}/chemical-space` → `RunChemicalSpaceResponse {status, points?: {x, y, row_id, applicability, neighbors}, summary?: {total, in_domain, threshold, nearest_min, nearest_max}}`.
  - `GET /api/v1/runs/{id}/chemical-space/compounds?rows=…` → `RunMapCompoundResponse[] {row_id, structure, compound_id, values, applicability}`.
  - At most 50 indices or rows per request (422 above that). A missing protocol or run is 404; a non-prediction run is 404; a run that is not ready is 409.

- [ ] **Step 1: Failing tests.** Append:
```python
async def test_the_protocol_map_is_served_in_unit_coordinates(client, trained_protocol_id):
    body = (await client.get(f"/api/v1/protocols/{trained_protocol_id}/chemical-space")).json()
    assert body["status"] == "ready"
    points = body["points"]
    assert len(points["x"]) == len(points["y"]) == len(points["partition"]) == 20
    assert all(0 <= v <= 1 for v in points["x"] + points["y"])
    assert sorted(set(points["partition"])) == [0, 1, 2]


async def test_a_protocol_without_a_map_says_missing(
    client, tmp_path, workspace_id, trained_protocol_id
):
    _meta_path(tmp_path, workspace_id, trained_protocol_id).unlink()
    body = (await client.get(f"/api/v1/protocols/{trained_protocol_id}/chemical-space")).json()
    assert body == {"status": "missing", "method": None, "params": None, "counts": None, "points": None}


async def test_map_compounds_are_looked_up_by_index(client, trained_protocol_id):
    response = await client.get(
        f"/api/v1/protocols/{trained_protocol_id}/chemical-space/compounds",
        params=[("indices", 0), ("indices", 3)],
    )
    assert response.status_code == 200, response.text
    items = response.json()
    assert [item["index"] for item in items] == [0, 3]
    assert all(item["partition"] in {"train", "validation", "test"} for item in items)


async def test_more_than_fifty_lookups_is_refused(client, trained_protocol_id):
    response = await client.get(
        f"/api/v1/protocols/{trained_protocol_id}/chemical-space/compounds",
        params=[("indices", i) for i in range(51)],
    )
    assert response.status_code == 422


async def test_a_run_is_placed_among_its_training_neighbours(client, ready_run_id):
    run = (await client.get(f"/api/v1/runs/{ready_run_id}")).json()
    protocol_map = (
        await client.get(f"/api/v1/protocols/{run['params']['protocol_id']}/chemical-space")
    ).json()
    body = (await client.get(f"/api/v1/runs/{ready_run_id}/chemical-space")).json()
    assert body["status"] == "ready"
    points = body["points"]
    assert len(points["x"]) == body["summary"]["total"]
    partition = protocol_map["points"]["partition"]
    for neighbours in points["neighbors"]:
        assert neighbours and all(partition[i] == 0 for i in neighbours)
    assert body["summary"]["threshold"] == 0.3


async def test_run_compounds_are_looked_up_by_row(client, ready_run_id):
    page = (await client.get(f"/api/v1/runs/{ready_run_id}/results")).json()
    first = page["items"][0]
    items = (
        await client.get(
            f"/api/v1/runs/{ready_run_id}/chemical-space/compounds",
            params=[("rows", first["row_id"])],
        )
    ).json()
    assert items[0]["structure"] == first["structure"]


async def test_a_run_without_neighbours_says_missing(client, tmp_path, workspace_id, ready_run_id):
    (tmp_path / str(workspace_id) / "runs" / ready_run_id / "neighbors.parquet").unlink()
    body = (await client.get(f"/api/v1/runs/{ready_run_id}/chemical-space")).json()
    assert body["status"] == "missing"
```
Read `PredictionResponse` and the results route first, to confirm the field names `items`, `row_id` and `structure`.

- [ ] **Step 2:** Run them. Expected: FAIL with 404 (routes missing).

- [ ] **Step 3: Implement.**
  - In `get_chemical_space.py`, write four use-case classes following `GetScorecard`'s shape (constructor-injected repositories and store; `__call__(query, auth) -> Result[...]`; `require_authenticated`; workspace-scoped `get`).
  - **Points:**
    - `GetProtocolChemicalSpace` returns `ChemicalSpaceView(status, method, params, counts, x, y, partition)`.
    - Missing or unreadable meta returns `status="missing"`.
    - Coordinates are rounded to 4 decimals.
  - **Protocol compounds:** `GetProtocolChemicalSpaceCompounds(protocol_id, indices)`.
    - More than 50 indices is a `ValidationError`.
    - Out-of-range indices are skipped.
    - The partition is returned as its name.
  - **Run map:** `GetRunChemicalSpace(run_id)` applies the same guards as `GetPredictionResults`: not found, not a prediction (404), not ready (409).
    - It reads the meta, points and neighbors. If any are missing it returns `status="missing"`.
    - `train_map = np.flatnonzero(points["partition"] == 0)`.
    - `xy = place(points[["x","y"]][train_map], idx, sims)`. Run rows are in file order, so `row_id = range(height)`.
    - Neighbors are mapped through `train_map[idx]`.
    - Applicability is read from `predictions.parquet`.
    - Summary: `total`, `in_domain` (count with applicability ≥ `_APPLICABILITY_THRESHOLD`), `nearest_min` and `nearest_max` (rounded to 2 decimals), and `threshold`.
  - **Run compounds:** `GetRunChemicalSpaceCompounds(run_id, rows)` caps at 50 and reads `predictions.parquet` rows by position.
    - It returns `structure`, `compound_id` (or None), `values` (each readout column of the protocol by name, as a float or None) and `applicability`.
  - **Wiring:**
    - Register the four classes in `container.py` like `GetScorecard`, with `_protocols(c)`, `_runs(c)` and `c[BlobStore]`.
    - Add the response models and routes to `protocols.py` and `runs.py` with `Annotated[list[int], Query(max_length=50)]`.
    - Use `result_to_response` and the `use_case(...)` dependency pattern.
- [ ] **Step 4:** Run `uv run pytest tests/api/test_chemical_space.py -q` (expected PASS), then the whole backend suite with `uv run pytest -q` (expected PASS).
- [ ] **Step 5:** Run `make generate-api` from the repo root and confirm `frontend/src/shared/lib/api/model/` gained `chemicalSpaceResponse.ts` and the others. Then commit `feat(map): read endpoints for protocol and run maps` with the openapi snapshot and generated models.

---

### Task 7: Backfill command

**Files:**
- Create: `backend/src/daikonstudio/infrastructure/backfill_maps.py`
- Modify: `Makefile` (target `backfill-maps`, listed in `.PHONY`)
- Test: `backend/tests/api/test_backfill_maps.py`

**Interfaces:**
- Produces: `async def backfill(session_factory, store, layout, normalizer, log=print) -> BackfillReport` (a dataclass with `maps_built`, `maps_current`, `neighbours_built`, `neighbours_current` and `failures: list[str]`), and `main()`.

- [ ] **Step 1: Failing test** (`tests/api/test_backfill_maps.py`):
```python
from daikonstudio.infrastructure.backfill_maps import backfill
from daikonstudio.infrastructure.chem.chemical_space import UmapLayout
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer
from daikonstudio.infrastructure.storage.fsspec_blob_store import FsspecBlobStore
from tests.api.test_triage_round_trip import ready_run_id  # noqa: F401


async def test_backfill_rebuilds_what_is_missing_and_then_has_nothing_to_do(
    client, tmp_path, workspace_id, session_factory, ready_run_id
):
    run = (await client.get(f"/api/v1/runs/{ready_run_id}")).json()
    protocol_id = run["params"]["protocol_id"]
    base = tmp_path / str(workspace_id)
    (base / "protocols" / protocol_id / "chemical-space.json").unlink()
    (base / "runs" / ready_run_id / "neighbors.parquet").unlink()
    store = FsspecBlobStore(f"file://{tmp_path}", {})

    first = await backfill(session_factory, store, UmapLayout(), RdkitStructureNormalizer(), log=lambda _: None)
    assert (first.maps_built, first.neighbours_built, first.failures) == (1, 1, [])
    assert (base / "protocols" / protocol_id / "chemical-space.json").exists()
    assert (base / "runs" / ready_run_id / "neighbors.parquet").exists()

    second = await backfill(session_factory, store, UmapLayout(), RdkitStructureNormalizer(), log=lambda _: None)
    assert (second.maps_built, second.neighbours_built) == (0, 0)
    assert (second.maps_current, second.neighbours_current) == (1, 1)
```
Check the `FsspecBlobStore` constructor's options argument name and default in `fsspec_blob_store.py` first.

- [ ] **Step 2:** Run it. Expected: FAIL, ImportError.
- [ ] **Step 3: Implement** `backfill_maps.py`, following `runner/seed.py`'s template:
  - **Listing protocols:** `select(InSilicoProtocolModel.id, InSilicoProtocolModel.workspace_id)` across all workspaces, through `session_factory`.
  - **For each protocol:**
    - Load it with `SqlAlchemyProtocolRepository(session_factory).get(ws, pid)`, and its dataset with `SqlAlchemyDatasetRepository(...).get(ws, protocol.dataset_id)`.
    - If `read_meta(...)` is missing or its `version < MAP_VERSION`, read the snapshot and call `write_chemical_space(...)` in `asyncio.to_thread`, then increment `maps_built`. Otherwise increment `maps_current`.
    - `TooFewCompounds` counts as current, with a log line.
    - Then run the run listing below.
  - **Listing runs:** `select(RunModel.id, RunModel.workspace_id, RunModel.result_uri).where(kind == "prediction", status == "ready", protocol_id == pid)`.
    - If the neighbors blob exists, increment `neighbours_current`.
    - Otherwise read the `structure` column of the results parquet and the scorecard inputs' `train_structures`, call `normalizer.nearest_neighbours_tanimoto(..., NEIGHBOURS)`, put `neighbours_parquet(...)`, and increment `neighbours_built`.
  - **Failures:** each per-item exception is caught, logged, and appended to `failures`, and the loop continues.
  - **`main()`:** builds `Settings()`, `create_async_engine`, `async_sessionmaker` and `FsspecBlobStore(settings.blob_base_url, settings.blob_storage_options)`. It prints one summary line and exits 1 if there were failures.
  - **Makefile:**
    ```
    backfill-maps: ## Build chemical-space maps (and run neighbours) for every protocol that lacks one
    	$(BACKEND) && $(BE_ENV) && uv run python -m daikonstudio.infrastructure.backfill_maps
    ```
- [ ] **Step 4:** Run `uv run pytest tests/api/test_backfill_maps.py -q`. Expected: PASS.
- [ ] **Step 5:** Commit `feat(map): one-off backfill for existing protocols and runs`.

---

### Task 8: Map camera and picking (pure)

**Files:**
- Create: `frontend/src/shared/components/chemical-space/view.ts`, `view.test.ts`
- Create: `frontend/src/shared/components/chemical-space/color.ts` (parse a CSS color to RGBA floats), tested in `view.test.ts`

**Interfaces:**
- Produces: `View {cx, cy, scale}`, `fitView(w, h, padding?)`, `toScreen(view, w, h, x, y)`, `toMap(view, w, h, sx, sy)`, `zoomAt(view, w, h, sx, sy, factor, fit)`, `pan(view, dx, dy)`, `buildPickGrid(xs, ys, size?)`, `pickNearest(grid, x, y, radius) -> number`, `parseColor(css, alpha?) -> [r, g, b, a]`.

- [ ] **Step 1: Failing tests** (`view.test.ts`):
```ts
import { describe, expect, it } from "vitest";
import { parseColor } from "./color";
import { buildPickGrid, fitView, pan, pickNearest, toMap, toScreen, zoomAt } from "./view";

describe("view", () => {
  const fit = fitView(400, 300, 10);

  it("fits the unit square into the shorter side, centred", () => {
    expect(fit).toEqual({ cx: 0.5, cy: 0.5, scale: 280 });
    expect(toScreen(fit, 400, 300, 0.5, 0.5)).toEqual([200, 150]);
  });

  it("puts map y up and screen y down", () => {
    const [, top] = toScreen(fit, 400, 300, 0.5, 1);
    expect(top).toBeLessThan(150);
  });

  it("keeps the point under the cursor fixed while zooming", () => {
    const before = toMap(fit, 400, 300, 320, 60);
    const zoomed = zoomAt(fit, 400, 300, 320, 60, 2.5, fit);
    const after = toMap(zoomed, 400, 300, 320, 60);
    expect(after[0]).toBeCloseTo(before[0]);
    expect(after[1]).toBeCloseTo(before[1]);
  });

  it("clamps zoom between half and 64 times the fit", () => {
    expect(zoomAt(fit, 400, 300, 200, 150, 1000, fit).scale).toBe(fit.scale * 64);
    expect(zoomAt(fit, 400, 300, 200, 150, 0.001, fit).scale).toBe(fit.scale * 0.5);
  });

  it("pans by screen pixels", () => {
    const moved = pan(fit, 28, 0);
    expect(moved.cx).toBeCloseTo(0.4);
  });
});

describe("pick grid", () => {
  const xs = [0.1, 0.5, 0.52, 0.9];
  const ys = [0.1, 0.5, 0.5, 0.9];
  const grid = buildPickGrid(xs, ys, 16);

  it("finds the nearest point within the radius", () => {
    expect(pickNearest(grid, 0.515, 0.5, 0.05)).toBe(2);
  });

  it("finds nothing outside the radius", () => {
    expect(pickNearest(grid, 0.3, 0.3, 0.05)).toBe(-1);
  });
});

describe("parseColor", () => {
  it("reads hex and rgb()", () => {
    expect(parseColor("#3b82f6")).toEqual([59 / 255, 130 / 255, 246 / 255, 1]);
    expect(parseColor("#fff", 0.5)).toEqual([1, 1, 1, 0.5]);
    expect(parseColor("rgb(15, 23, 42)")).toEqual([15 / 255, 23 / 255, 42 / 255, 1]);
  });
});
```
- [ ] **Step 2:** Run `pnpm test src/shared/components/chemical-space`. Expected: FAIL.
- [ ] **Step 3: Implement.**
```ts
// view.ts
/** The map's camera and hit testing. Map coordinates are the API's [0, 1] square, y up. */
export interface View {
  cx: number;
  cy: number;
  scale: number;
}

export const MIN_ZOOM = 0.5;
export const MAX_ZOOM = 64;

export function fitView(width: number, height: number, padding = 16): View {
  return { cx: 0.5, cy: 0.5, scale: Math.max(1, Math.min(width, height) - padding * 2) };
}

export function toScreen(view: View, width: number, height: number, x: number, y: number): [number, number] {
  return [(x - view.cx) * view.scale + width / 2, height / 2 - (y - view.cy) * view.scale];
}

export function toMap(view: View, width: number, height: number, sx: number, sy: number): [number, number] {
  return [(sx - width / 2) / view.scale + view.cx, view.cy - (sy - height / 2) / view.scale];
}

export function zoomAt(view: View, width: number, height: number, sx: number, sy: number, factor: number, fit: View): View {
  const [mx, my] = toMap(view, width, height, sx, sy);
  const scale = Math.min(fit.scale * MAX_ZOOM, Math.max(fit.scale * MIN_ZOOM, view.scale * factor));
  return { scale, cx: mx - (sx - width / 2) / scale, cy: my + (sy - height / 2) / scale };
}

export function pan(view: View, dx: number, dy: number): View {
  return { ...view, cx: view.cx - dx / view.scale, cy: view.cy + dy / view.scale };
}

export interface PickGrid {
  size: number;
  cells: Map<number, number[]>;
  xs: ArrayLike<number>;
  ys: ArrayLike<number>;
}

const cellOf = (v: number, size: number) => Math.min(size - 1, Math.max(0, Math.floor(v * size)));

/** A uniform grid over the unit square: hovering 100k points stays O(points near the cursor). */
export function buildPickGrid(xs: ArrayLike<number>, ys: ArrayLike<number>, size = 128): PickGrid {
  const cells = new Map<number, number[]>();
  for (let i = 0; i < xs.length; i++) {
    const key = cellOf(xs[i], size) * size + cellOf(ys[i], size);
    const cell = cells.get(key);
    if (cell) cell.push(i);
    else cells.set(key, [i]);
  }
  return { size, cells, xs, ys };
}

/** Index of the nearest point within `radius` (map units), or -1. */
export function pickNearest(grid: PickGrid, x: number, y: number, radius: number): number {
  const reach = Math.ceil(radius * grid.size);
  const col = cellOf(x, grid.size);
  const row = cellOf(y, grid.size);
  let best = -1;
  let bestDistance = radius * radius;
  for (let c = Math.max(0, col - reach); c <= Math.min(grid.size - 1, col + reach); c++) {
    for (let r = Math.max(0, row - reach); r <= Math.min(grid.size - 1, row + reach); r++) {
      for (const i of grid.cells.get(c * grid.size + r) ?? []) {
        const d = (grid.xs[i] - x) ** 2 + (grid.ys[i] - y) ** 2;
        if (d <= bestDistance) {
          bestDistance = d;
          best = i;
        }
      }
    }
  }
  return best;
}
```
```ts
// color.ts
export type Rgba = [number, number, number, number];

/** A resolved CSS color (hex or rgb[a]) as 0..1 floats for a WebGL uniform. */
export function parseColor(css: string, alpha = 1): Rgba {
  const value = css.trim();
  if (value.startsWith("#")) {
    const hex = value.length === 4 ? [...value.slice(1)].map((c) => c + c).join("") : value.slice(1, 7);
    const n = Number.parseInt(hex, 16);
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, alpha];
  }
  const parts = value.match(/[\d.]+/g)?.map(Number) ?? [0, 0, 0];
  return [parts[0] / 255, parts[1] / 255, parts[2] / 255, (parts[3] ?? 1) * alpha];
}
```
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat(map): camera, picking grid and color parsing for the map`.

---

### Task 9: WebGL renderer and map host

**Files:**
- Create: `frontend/src/shared/components/chemical-space/renderer.ts`
- Create: `frontend/src/shared/components/chemical-space/chemical-space-map.tsx`, `chemical-space-map.test.tsx`
- Modify: `frontend/src/shared/components/charts/use-chart-theme.ts` (add `held` from `--color-score-fair`)

**Interfaces:**
- Consumes: Task 8.
- Produces:
  - `STYLE = { train: 0, validation: 1, test: 2, runIn: 3, runOut: 4, neighbour: 5 }`.
  - `createMapRenderer(canvas): MapRenderer | null`, with `setLayer(0 | 1, positions: Float32Array, styles: Float32Array)`, `setColors(Rgba[])`, `resize(): [w, h]`, `draw(view, progress)` and `dispose()`.
  - `<ChemicalSpaceMap base overlay? colors label pickBase? pickOverlay? lines? tooltip? />`, where a layer is `{x: ArrayLike<number>, y: ArrayLike<number>, style: ArrayLike<number>}`, `lines(hit) -> [x, y][]` gives map-space endpoints from the hovered point, `tooltip(hit) -> ReactNode`, and `hit = {layer: "base" | "overlay", index}`.
  - `useMapColors(): Rgba[]`, indexed by `STYLE`.

- [ ] **Step 1: Failing test** (`chemical-space-map.test.tsx`): in jsdom there is no WebGL2, so the host must render the fallback and keep the accessible label.
```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ChemicalSpaceMap } from "./chemical-space-map";

describe("ChemicalSpaceMap", () => {
  it("falls back to a sentence when WebGL2 is unavailable", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    render(
      <ChemicalSpaceMap
        base={{ x: [0.1, 0.9], y: [0.2, 0.8], style: [0, 0] }}
        colors={[[0, 0, 1, 1], [0, 1, 1, 1], [1, 0.5, 0, 1], [1, 0.5, 0, 1], [1, 0.5, 0, 1], [0, 0, 0, 1]]}
        label="2 training compounds"
      />,
    );
    expect(screen.getByText("This browser cannot draw the map.")).toBeInTheDocument();
  });
});
```
- [ ] **Step 2:** Run it. Expected: FAIL, module missing.
- [ ] **Step 3: Implement.**
  - **`renderer.ts`:** a WebGL2 program, ported from ChemCellar's `renderer.ts` (blend `ONE, ONE_MINUS_SRC_ALPHA`, the dpr ≤ 2 resize, a null return on failure).
    - Two layers, each a VAO with a `vec2 a_pos` buffer and a `float a_style` buffer.
    - Uniforms: `u_center`, `u_scale`, `u_resolution`, `u_dpr`, `u_progress`, `u_sizes[6]` (CSS px `[2.5, 2.5, 3, 7, 8, 7]`) and `u_colors[6]`.
    - **Vertex shader:** the screen position from the view, matching `toScreen` exactly. Non-run styles fade in over `progress` 0 to 0.5. Run styles grow with the ease-out of `(progress - 0.35) / 0.65`.
    - **Fragment shader:** a soft disk for styles 0–3. Style 4 is a dashed ring (band radius 0.30–0.5 of the sprite, `step(0.0, sin(atan(p.y, p.x) * 5.0))`, so 5 dashes). Style 5 is a solid ring.
    - `draw` clears, then draws layer 0, then layer 1.
  - **`chemical-space-map.tsx`:**
    - On mount it creates the renderer. If that returns null, it renders `<p>This browser cannot draw the map.</p>`.
    - A ResizeObserver calls `resize()`, refits on the first size, and redraws.
    - Layers are uploaded when their props change. Colors trigger `setColors` and a redraw.
    - **Entrance:** `progress` runs 0 → 1 over 900 ms in rAF, once; it is 1 immediately under `prefers-reduced-motion`.
    - **Pointer:**
      - Drag pans with pointer capture.
      - Wheel zooms with `zoomAt(..., Math.exp(-deltaY * 0.0015))`, through a non-passive listener that calls `preventDefault`.
      - Double-click resets to the fit.
      - Hover picks the overlay first, then the base if `pickBase`. The radius is `10 / view.scale`.
    - The hovered hit, plus a `viewTick` that changes on pan or zoom, drive an absolutely positioned SVG overlay (lines from the hovered point to the `lines(hit)` endpoints, drawn with `stroke="currentColor"` in `text-foreground/60`) and a tooltip box near the cursor. Both have `pointer-events: none`.
    - The canvas has `role="img"` and `aria-label={label}`. The container is `relative h-[420px] w-full overflow-hidden rounded-lg border bg-card`, with `cursor-grab` while idle and `cursor-grabbing` while dragging.
  - **`useMapColors`** (in `chemical-space-map.tsx`): from `useChartTheme()`, return `[parseColor(blue, 0.45), parseColor(teal, 0.7), parseColor(held, 0.85), parseColor(held), parseColor(held), parseColor(text)]`. Add `held: get("--color-score-fair", "#d97706")` to `ChartTheme` and its reader.
- [ ] **Step 4:** Run `pnpm test src/shared/components && pnpm exec tsc --noEmit`. Expected: PASS and clean.
- [ ] **Step 5:** Commit `feat(map): WebGL2 renderer and the interactive map host`.

---

### Task 10: Run and protocol cards

**Files:**
- Create: `frontend/src/features/runs/components/run-chemical-space.tsx` (+ `run-chemical-space.test.ts` for `runSummary`)
- Create: `frontend/src/features/protocols/components/protocol-chemical-space.tsx`
- Modify: `frontend/src/features/runs/hooks/use-runs.ts`, `frontend/src/features/protocols/hooks/use-protocols.ts`
- Modify: `frontend/src/features/runs/components/run-detail.tsx` (the domain `Explainer` block becomes `<RunChemicalSpace …/>`), `frontend/src/features/protocols/components/protocol-detail.tsx` (`<ProtocolChemicalSpace protocolId={protocol.id} />` after `<ScorecardView …/>`)

**Interfaces:**
- Consumes: Task 6 HTTP shapes (generated types), Task 9 host.
- Produces: `runSummary(applicability: (number | null)[], threshold: number) -> {total, inDomain, min, max}`, `useProtocolChemicalSpace(id)`, `useProtocolMapCompound(id, index)`, `useRunChemicalSpace(id)`, `useRunMapCompound(id, row)`.

- [ ] **Step 1: Failing test** (`run-chemical-space.test.ts`):
```ts
import { describe, expect, it } from "vitest";
import { runSummary } from "./run-chemical-space";

describe("runSummary", () => {
  it("counts compounds inside the domain and the similarity range", () => {
    expect(runSummary([0.11, 0.45, null, 0.3], 0.3)).toEqual({ total: 4, inDomain: 2, min: 0.11, max: 0.45 });
  });
  it("has no range when nothing was measurable", () => {
    expect(runSummary([null], 0.3)).toEqual({ total: 1, inDomain: 0, min: null, max: null });
  });
});
```
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - **Hooks:** `customInstance` GETs like `useScorecard`. Use `staleTime: STALE_TIME.LONG` for the maps and `enabled` guards. The compound hooks are enabled only for an index ≥ 0 and use `paramsSerializer`, or a hand-built query string, to repeat `indices=` / `rows=`.
  - **`RunChemicalSpace({ runId, protocolId })`:**
    - A Card titled "Chemical space", with Skeleton while loading.
    - When either response is `missing`: "No map has been computed for this protocol yet."
    - Otherwise:
      - The base is the training points: partition 0, with a `baseToMap` index array.
      - The overlay is the run points, styled `runIn` when applicability ≥ `IN_DOMAIN_FLOOR` and `runOut` otherwise (null counts as out).
      - `lines` maps the neighbor map indices to coordinates.
      - The tooltip for an overlay hit shows `StructureThumbnail`, the compound id or "Row n", the readout values, and "Similarity to nearest training compound: 0.xx". For a base hit it fetches through `useProtocolMapCompound`.
      - The summary sentence comes from `runSummary`: "{inDomain} of {total} compounds are inside the applicability domain (similarity ≥ 0.3). Nearest training compound: similarity {min} to {max}."
      - The caption is the spec's text, and the legend has three swatches (training, inside domain, outside domain).
  - **`ProtocolChemicalSpace({ protocolId })`:** the base is all points, styled by partition. Hover a point to see its structure and partition. The legend gives counts per partition from `counts`. The caption is "Each dot is a compound in this protocol's dataset, placed by UMAP so that similar structures sit together. Distance on the map is approximate."
  - **`run-detail.tsx`:** remove the illustrative domain `Explainer` and its imports, and render `<RunChemicalSpace runId={runId} protocolId={protocol.id} />` before `<TriageGrid …/>`.
- [ ] **Step 4:** Run `pnpm test && pnpm lint && pnpm exec tsc --noEmit`. Expected: all green.
- [ ] **Step 5:** Commit `feat(map): chemical-space cards on the run and protocol pages`.

---

### Task 11: Verification

- [ ] **Step 1: Gates.** Backend: `uv run pytest -q && uv run ruff check src tests && uv run ruff format --check src tests`. Frontend: `pnpm lint && pnpm exec tsc --noEmit && pnpm test && pnpm build`. Expected: all exit 0.
- [ ] **Step 2: Workers.** Restart the dev workers so they load the new code (`make dev-worker`, and the gpu worker if one is running). They do not hot-reload. Do the same for the API if it is not running with `--reload`.
- [ ] **Step 3: Backfill on dev.** Run `make backfill-maps` and record the summary line.
- [ ] **Step 4: Live check.** Signed in, open:
  - the PAINS run (its compounds and the summary must match the triage grid's applicability column);
  - its protocol (train, validation and test colors);
  - pan, zoom, double-click, hover tooltips and neighbor lines;
  - dark theme;
  - reduced motion, if it can be emulated.
  Take a screenshot of each.
- [ ] **Step 5: Scale check.** Time `UmapLayout().layout` on about 2,000 real dataset compounds. For browser scale, temporarily serve a 100,000-point synthetic map: a test-only fixture in a scratch page, or a dev-only query flag. Record draw time and hover responsiveness, and remove the temporary code before the final commit.
