"""How a tool's values travel as blobs: tables as parquet, everything else as JSON.

Convention: a tool's non-table output is a dict whose `meta` holds what a reader needs
without opening the blob; `encode` returns it as the summary. A table's summary is its
shape and, per column, what it holds, so a reader never needs the rows to say what is there,
and which columns hold structures or sequences, whose values no reader may show.
"""

from __future__ import annotations

import io
import json
import re
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


# Columns whose values are structures or sequences, by name. Decided here, when a table is
# encoded, because a header is the one thing an upload does not control: `Molecule`,
# `CanonicalSMILES` and `parse_error` (which quotes the SMILES it could not read) all carry
# structures under names no exact list would have.
WITHHELD_COLUMNS = re.compile(
    r"smiles|sequence|inchi|structure|molecule|^mol$|^seq$|^parse_error$", re.I
)
# And by what a column holds, without RDKit: a SMILES is one run of the SMILES alphabet with a
# bond, branch, bracket or stereo mark in it, and a sequence is twenty residues or more.
_SMILES = r"^[A-Za-z0-9@+\-\[\]\(\)\\/%=#$.:*]+$"
_SMILES_MARK = r"[=#()\[\]@]"
_SEQUENCE = r"^[ACDEFGHIKLMNPQRSTVWYXBZU*-]{20,}$"
_STRUCTURAL = 0.6  # the share of a column's filled values that must look like one


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


def _withheld(frame: pl.DataFrame) -> list[str]:
    """The columns whose values must never be shown, found by name or by what they hold."""
    out: list[str] = []
    for column in frame:
        values = column.drop_nulls()
        by_content = False
        if column.dtype == pl.String and values.len():
            # ponytail: no RDKit here, so a column of mostly bare SMILES (CCO, c1ccccc1) under a
            # header no pattern lists is not recognised; parse them here if that ever bites.
            looks = (values.str.contains(_SMILES) & values.str.contains(_SMILES_MARK)) | (
                values.str.contains(_SEQUENCE)
            )
            by_content = looks.sum() / values.len() >= _STRUCTURAL
        if by_content or WITHHELD_COLUMNS.search(column.name):
            out.append(column.name)
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
            "withheld": _withheld(value),
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
