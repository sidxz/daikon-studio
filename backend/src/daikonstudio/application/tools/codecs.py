"""How a tool's values travel as blobs: tables as parquet, everything else as JSON.

Convention: a tool's non-table output is a dict whose `meta` holds what a reader needs
without opening the blob; `encode` returns it as the summary.
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


def encode(kind: str, value: object) -> tuple[bytes, dict[str, Any]]:
    if kind in TABLE_KINDS:
        assert isinstance(value, pl.DataFrame)
        buffer = io.BytesIO()
        value.write_parquet(buffer)
        return buffer.getvalue(), {"rows": value.height, "columns": value.columns}
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
