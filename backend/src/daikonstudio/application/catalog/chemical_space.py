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
