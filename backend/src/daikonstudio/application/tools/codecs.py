"""How a tool's values travel as blobs: tables as parquet, everything else as JSON.

Convention: a tool's non-table output is a dict whose `meta` holds what a reader needs
without opening the blob; `encode` returns it as the summary. A table's summary is its
shape and, per column, what it holds, so a reader never needs the rows to say what is there.
"""

from __future__ import annotations

import io
import json
from typing import Any

import polars as pl

TABLE_KINDS: frozenset[str] = frozenset(
    {
        "compound_table",
        "rgroup_table",
        "property_table",
        "pairs",
        "effects",
        "gaps",
        "admet_table",
        "series",
    }
)


def extension(kind: str) -> str:
    return "parquet" if kind in TABLE_KINDS else "json"


def _profile(frame: pl.DataFrame) -> list[dict[str, Any]]:
    """What each column holds. Stored as JSON, so no NaN and no infinity gets in."""
    out: list[dict[str, Any]] = []
    for column in frame:
        stats: dict[str, Any] = dict.fromkeys(("min", "max", "median"))
        if column.dtype.is_float():
            column = column.fill_nan(None)  # empty, as the page shows it
        if column.dtype.is_numeric():
            numbers = column.cast(pl.Float64)
            numbers = numbers.filter(numbers.is_finite())  # an infinity is no part of a range
            stats = {"min": numbers.min(), "max": numbers.max(), "median": numbers.median()}
        out.append(
            {
                "name": column.name,
                "dtype": str(column.dtype),
                "filled": column.len() - column.null_count(),
                "distinct": column.drop_nulls().n_unique(),
                **stats,
            }
        )
    return out


def encode(kind: str, value: object) -> tuple[bytes, dict[str, Any]]:
    if kind in TABLE_KINDS:
        assert isinstance(value, pl.DataFrame)
        buffer = io.BytesIO()
        value.write_parquet(buffer)
        summary = {
            "rows": value.height,
            "columns": value.columns,
            "columns_profile": _profile(value),
        }
        return buffer.getvalue(), summary
    blob = json.dumps(value, default=str).encode()
    if isinstance(value, dict):
        meta = value.get("meta")
        return blob, dict(meta) if isinstance(meta, dict) else {}
    if isinstance(value, list):
        return blob, {"items": len(value)}
    return blob, {}


def decode(kind: str, data: bytes) -> object:
    if kind in TABLE_KINDS:
        return pl.read_parquet(io.BytesIO(data))
    return json.loads(data)
