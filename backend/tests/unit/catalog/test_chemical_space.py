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
        positions = [float(i) for i in range(len(structures))]
        return positions, list(positions)

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
    np.testing.assert_allclose(
        frame["neighbor_similarity"].to_list(), [[0.9, 0.5], [0.4, 0.1]], rtol=1e-6
    )
