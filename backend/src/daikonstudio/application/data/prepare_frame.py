"""The gate every uploaded dataset passes through before it can be trained on.

Deduplicating by canonical structure *before* a dataset is split is what eliminates
train/test leakage by construction, rather than warning about it after the fact: two
SMILES strings describing the same molecule can no longer land on opposite sides of a
split once they have already been merged into one row. The spread between replicate
numeric measurements is kept, not discarded, because it is a free estimate of assay
noise -- the honest floor on how accurate any model trained on this data could ever be.

Order matters: structures are canonicalized and invalid ones rejected first, then
salts/mixtures are flagged, then rows are grouped by canonical structure and collapsed.
Flagging salts before dropping invalid rows would count structures that never made it
into the data; grouping before canonicalizing would treat equivalent SMILES as distinct.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence

import polars as pl

from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.data.target import TargetKind, TargetSpec
from daikonstudio.domain.data.validation import ConflictRow, InvalidRow, ValidationReport
from daikonstudio.domain.shared.errors import ValidationError

# Called with how many rows a per-row RDKit pass has finished, every `_PROGRESS_EVERY`
# rows and once at the end. A background dataset build reports it to the wizard.
RowProgress = Callable[[int], None]
_PROGRESS_EVERY = 1000


def map_rows[T, R](
    fn: Callable[[T], R], items: Sequence[T], on_row: RowProgress | None
) -> list[R]:
    if on_row is None:
        return [fn(item) for item in items]
    out: list[R] = []
    for count, item in enumerate(items, 1):
        out.append(fn(item))
        if count % _PROGRESS_EVERY == 0:
            on_row(count)
    on_row(len(items))
    return out


def read_csv_upload(raw: bytes) -> pl.DataFrame:
    """Every CSV this app accepts comes through here.

    A UTF-8 BOM on the first header is stripped: Excel writes one, and
    `\\ufeffsmiles` is not a column a scientist can select or name.
    """
    try:
        # Every column as text. Polars otherwise infers types from the first 100
        # rows and then *raises* on row 150's "NA" (a ComputeError naming its own
        # Python API), and it also rewrites "00123" to 123 -- an identifier
        # column must come back exactly as the scientist wrote it. The target
        # gate below casts the one column that has to be numeric, with reasons.
        frame = pl.read_csv(io.BytesIO(raw), infer_schema=False)
    except pl.exceptions.PolarsError as error:
        raise ValidationError(f"The uploaded file is not readable as CSV: {error}") from error
    bom = "﻿"
    return frame.rename({c: c.lstrip(bom) for c in frame.columns if c.startswith(bom)})


def _validate_target(
    frame: pl.DataFrame, target: TargetSpec, row_numbers: list[int]
) -> tuple[pl.DataFrame, list[int], list[InvalidRow]]:
    """Rows whose target cannot be trained on, rejected here with their row
    numbers rather than as `Input y contains NaN` minutes later in a worker.

    Returns the frame with the target cast (Float64 for NUMERIC, Int64 for
    BINARY), the surviving row numbers, and one InvalidRow per rejected row.
    Three reasons, in the words a scientist needs: an empty cell, text where a
    number belongs (`NA`, `<10`, `12,5`), or a binary label that is not 0 or 1.
    Each names its column, since with several targets "missing value" alone does
    not say which cell to fill. Built with f-strings, never `str.format`: a column
    named `IC50 {nM}` would otherwise be read as a format field.
    """
    column = target.column
    raw = frame[column]
    text = raw.cast(pl.String, strict=False).fill_null("").str.strip_chars()
    numeric = (
        raw.str.strip_chars().cast(pl.Float64, strict=False)
        if raw.dtype == pl.String
        else raw.cast(pl.Float64, strict=False)
    )
    empty = text == ""
    if target.kind is TargetKind.BINARY:
        ok = numeric.is_in([0.0, 1.0]).fill_null(False) & ~empty

        def reason(value: str) -> str:
            return f"Target '{column}' must be 0 or 1 (found '{value}')"

        cast_to: pl.DataType = pl.Int64()
    else:
        # is_finite, not is_not_null: polars parses "nan" and "inf" to floats that
        # are not null, and either one is the `Input y contains NaN` failure this
        # gate exists to stop.
        ok = numeric.is_finite().fill_null(False) & ~empty

        def reason(value: str) -> str:
            return f"Target '{column}' is not numeric: '{value}'"

        cast_to = pl.Float64()
    invalid = [
        InvalidRow(
            row_number=row_numbers[index],
            value=text[index],
            reason=f"Missing value for target '{column}'" if empty[index] else reason(text[index]),
        )
        for index in range(frame.height)
        if not ok[index]
    ]
    kept = frame.filter(ok).with_columns(numeric.filter(ok).cast(cast_to).alias(column))
    kept_rows = [number for number, keep in zip(row_numbers, ok.to_list(), strict=True) if keep]
    return kept, kept_rows, invalid


def prepare_frame(
    frame: pl.DataFrame,
    structure_column: str,
    targets: Sequence[TargetSpec],
    normalizer: StructureNormalizer,
    on_row: RowProgress | None = None,
) -> tuple[pl.DataFrame, ValidationReport]:
    total_rows = frame.height
    if total_rows == 0:
        # A CSV with a header row and no data rows arrives here as height 0, not as
        # rows that fail to canonicalize -- distinct from the valid_rows == 0 guard
        # below. Caught before any per-row work: an empty boolean predicate below
        # infers polars' Null dtype rather than Boolean, and .filter() on that raises
        # instead of returning a well-formed empty report.
        return frame, ValidationReport(total_rows=0, valid_rows=0)
    raw_structures = [str(value) for value in frame[structure_column].to_list()]
    canonical = map_rows(normalizer.canonicalize, raw_structures, on_row)

    invalid = [
        InvalidRow(
            row_number=index + 1, value=raw_structures[index], reason="SMILES could not be parsed"
        )
        for index, smiles in enumerate(canonical)
        if smiles is None
    ]

    is_valid = pl.Series([smiles is not None for smiles in canonical])
    # 1-indexed positions in the *uploaded* file, the same convention
    # `InvalidRow.row_number` above already uses -- kept as a plain Python
    # list, aligned by position with `valid_frame`'s rows (both filtered by
    # this same `is_valid` mask, so they stay in lockstep), so a later
    # conflicting-duplicates group (BINARY only) can report which rows of the
    # original file it spans (I4, whole-branch review), not just the
    # canonicalized structure they collapsed to.
    #
    # Deliberately never written into `valid_frame` itself (an earlier
    # version of this fix used `with_columns` to add it as a real column,
    # which is exactly the silent-overwrite mechanism C1 exists to prevent --
    # a target column named the same as the injected column would have its
    # real values clobbered by row indices). Keeping this a bare Python list
    # means there is no column name here for a target column to ever collide
    # with, no matter what the scientist names it.
    row_numbers = [index + 1 for index, valid in enumerate(is_valid) if valid]
    valid_frame = frame.filter(is_valid).with_columns(
        pl.Series(structure_column, [smiles for smiles in canonical if smiles is not None])
    )
    # The target gate runs after the structure gate so a row that fails both is
    # reported once, for its structure -- the thing the scientist fixes first. The
    # targets are gated in the order chosen, and a row is reported for the first one
    # it fails.
    for target in targets:
        valid_frame, row_numbers, bad_targets = _validate_target(valid_frame, target, row_numbers)
        invalid.extend(bad_targets)
    invalid.sort(key=lambda row: row.row_number)
    valid_rows = valid_frame.height

    salts_flagged = sum(
        normalizer.has_multiple_components(smiles)
        for smiles in valid_frame[structure_column].to_list()
    )

    if valid_rows == 0:
        # Nothing to group. The caller decides what to do with valid_rows == 0; this
        # function's job is to hand back a well-formed report, not to guess or crash.
        return valid_frame, ValidationReport(
            total_rows=total_rows,
            valid_rows=0,
            invalid=invalid,
            salts_flagged=salts_flagged,
        )

    target_columns = {target.column for target in targets}
    other_columns = [c for c in frame.columns if c != structure_column and c not in target_columns]
    # ponytail: a duplicate group narrows extra columns to the first row's value, so a
    # column that legitimately varies across replicates (e.g. batch ID) collapses to
    # one arbitrary pick. Upgrade path: carry such columns through as a per-group list,
    # or reject on conflict the way BINARY targets already do.
    keep_others = [pl.col(c).first() for c in other_columns]

    # Row numbers per canonical structure, computed independently of the
    # group_by below (never as an aggregated column of `valid_frame` -- see
    # the comment above `row_numbers`): `valid_frame[structure_column]` and
    # `row_numbers` are aligned by position, both filtered by the same
    # `is_valid` mask, so zipping them reproduces exactly the grouping
    # `group_by(structure_column)` performs, without ever touching the frame.
    row_numbers_by_structure: dict[str, list[int]] = {}
    for structure, row_number in zip(
        valid_frame[structure_column].to_list(), row_numbers, strict=True
    ):
        row_numbers_by_structure.setdefault(structure, []).append(row_number)

    # One pass over the duplicate groups covers every target. A measured value is
    # averaged and its replicate spread kept; a binary label must agree across the
    # replicates, and a structure whose labels disagree in any binary target is a
    # data problem for the scientist to resolve, not one a majority vote papers over.
    # Output column order -- structure, the targets in order, then the rest -- is
    # what it was with one target, so a one-target upload freezes to the same bytes
    # and the same `content_hash` as before targets could be several.
    aggregations: list[pl.Expr] = [pl.len().alias("_n")]
    helpers = ["_n"]
    for index, target in enumerate(targets):
        values = pl.col(target.column)
        if target.kind is TargetKind.NUMERIC:
            aggregations += [
                (values.max() - values.min()).alias(f"_spread_{index}"),
                values.mean().alias(target.column),
            ]
            helpers.append(f"_spread_{index}")
        else:
            aggregations += [
                values.n_unique().alias(f"_n_unique_{index}"),
                values.alias(f"_values_{index}"),
                values.first().alias(target.column),
            ]
            helpers += [f"_n_unique_{index}", f"_values_{index}"]
    grouped = valid_frame.group_by(structure_column, maintain_order=True).agg(
        *aggregations, *keep_others
    )

    conflicting: list[ConflictRow] = []
    is_conflict = pl.Series([False] * grouped.height)
    for index, target in enumerate(targets):
        if target.kind is not TargetKind.BINARY:
            continue
        clash = grouped[f"_n_unique_{index}"] > 1
        conflicting += [
            ConflictRow(
                structure=str(row[structure_column]),
                column=target.column,
                values=list(row[f"_values_{index}"]),
                row_numbers=sorted(row_numbers_by_structure[str(row[structure_column])]),
            )
            for row in grouped.filter(clash).iter_rows(named=True)
        ]
        is_conflict = is_conflict | clash
    agreeing = grouped.filter(~is_conflict)

    group_sizes = agreeing["_n"].to_list()
    # Groups of size one carry no spread information and are excluded from the
    # mean entirely -- they don't count as "zero spread", they count as nothing.
    duplicate_spread: dict[str, float] = {}
    for index, target in enumerate(targets):
        if target.kind is not TargetKind.NUMERIC:
            continue
        spreads = [
            float(spread)
            for n, spread in zip(group_sizes, agreeing[f"_spread_{index}"].to_list(), strict=True)
            if n > 1
        ]
        if spreads:
            duplicate_spread[target.column] = sum(spreads) / len(spreads)

    return agreeing.drop(helpers), ValidationReport(
        total_rows=total_rows,
        valid_rows=valid_rows,
        invalid=invalid,
        conflicting=conflicting,
        duplicates_collapsed=sum(n - 1 for n in group_sizes),
        salts_flagged=salts_flagged,
        duplicate_spread=duplicate_spread,
    )
