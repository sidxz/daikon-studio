"""A dataset's compound IDs, read from its snapshot when a page asks for them.

Never copied into scorecard inputs or map files, so naming or changing the
identifier column takes effect everywhere at once. Snapshot structures are
canonical and unique (duplicates are merged at upload), and the scorecard's and
the map's structures are the snapshot's own strings, so a structure is the key.
"""

from __future__ import annotations

import io
from collections.abc import Iterable

import polars as pl

from daikonstudio.application.data.snapshot import snapshot_key
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.domain.data.dataset import Dataset


def id_text(column: str) -> pl.Expr:
    """IDs as trimmed text, blanks as null. An integer column prints as "12"; a
    text column keeps leading zeros ("007")."""
    text = pl.col(column).cast(pl.String, strict=False).str.strip_chars()
    return pl.when(text == "").then(None).otherwise(text)


def snapshot_columns(store: BlobStore, dataset: Dataset) -> list[str]:
    raw = store.get_bytes(snapshot_key(dataset.workspace_id, dataset.id))
    return list(pl.read_parquet_schema(io.BytesIO(raw)).keys())


def eligible_id_columns(columns: list[str], dataset: Dataset) -> list[str]:
    reserved = {dataset.structure_column, dataset.target.column, "split"}
    return [column for column in columns if column not in reserved]


def read_compound_ids(
    store: BlobStore, dataset: Dataset, structures: Iterable[str]
) -> dict[str, str] | None:
    """Structure → ID for `structures` only, or None when the dataset names no
    identifier column.

    `structures` is required, not optional, because every caller already knows the
    handful it wants -- the map its hovered points, the scorecard its worst rows --
    and an optional parameter would let a caller keep the whole-snapshot read by
    omission. Filtering before the dict is built is what keeps a hover off the
    hundreds of thousands of rows it does not need.
    """
    if dataset.id_column is None:
        return None
    wanted = list(structures)
    if not wanted:
        return {}
    raw = store.get_bytes(snapshot_key(dataset.workspace_id, dataset.id))
    frame = (
        pl.read_parquet(io.BytesIO(raw), columns=[dataset.structure_column, dataset.id_column])
        .filter(pl.col(dataset.structure_column).is_in(wanted))
        .select(
            pl.col(dataset.structure_column).cast(pl.String).alias("structure"),
            id_text(dataset.id_column).alias("id"),
        )
    )
    return {
        structure: compound_id
        for structure, compound_id in frame.iter_rows()
        if compound_id is not None
    }
