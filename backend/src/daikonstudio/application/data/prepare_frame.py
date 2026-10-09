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
from daikonstudio.domain.data.split import normalize_partition
from daikonstudio.domain.data.structure_kind import (
    StructureKind,
    looks_like_sequence,
    normalize_sequence,
)
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
    """Cells whose target cannot be trained on, nulled here with their row
    numbers rather than reaching a worker as `Input y contains NaN`.

    **This function removes no rows.** A cell it cannot use becomes null and the row
    stays, because a bad cell in one target says nothing about the row's other targets
    -- dropping it here is what made the gate an intersection across all targets. The
    caller collects each target's rejections and removes only the rows that *every*
    target rejected. So the returned `InvalidRow`s are candidates, not verdicts, and
    `row_numbers` comes back exactly as it went in.

    Returns the frame with the target cast (Float64 for NUMERIC, Int64 for
    BINARY) and its unusable cells nulled, the row numbers unchanged, and one
    candidate InvalidRow per rejected cell.
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
    # Nulled, not filtered: a cell this target cannot use says nothing about the row's
    # other targets, and removing the row here is what made the gate an intersection.
    # The caller drops a row only when *every* target rejected it, so `invalid` here is
    # a list of candidates rather than of final rejections, and `row_numbers` comes back
    # unchanged -- this function no longer removes anything.
    kept = frame.with_columns(
        pl.when(pl.Series(ok)).then(numeric).otherwise(None).cast(cast_to).alias(column)
    )
    return kept, row_numbers, invalid


def _labelled_rows(frame: pl.DataFrame, targets: Sequence[TargetSpec]) -> dict[str, int]:
    """How many of `frame`'s rows carry a measurement for each target.

    Counted from the frame being returned, never from an earlier one: deduplication
    collapses replicate rows, so the count before it is not the count the Dataset has.
    """
    return {
        target.column: int(frame[target.column].drop_nulls().len())
        for target in targets
        if target.column in frame.columns
    }


def _validate_split_column(
    frame: pl.DataFrame, column: str, row_numbers: list[int]
) -> tuple[pl.DataFrame, list[int], list[InvalidRow]]:
    """Gate a predefined split's column, and canonicalize what survives.

    Deliberately shaped like `_validate_target`, and here rather than in `assign_split`
    for two reasons. This layer still knows each row's position in the *uploaded* file,
    so a bad cell can be reported against the number the scientist can actually find;
    by split time the frame has been filtered and deduplicated and those positions are
    gone. And normalizing the surviving cells here means the duplicate comparison below
    sees "Train " and "train" as one partition rather than two.
    """
    raw = frame[column].cast(pl.String, strict=False).fill_null("")
    text = raw.to_list()
    normalized = [normalize_partition(str(value)) for value in text]
    # `dtype=` on both Series below, not inferred: an empty list infers polars' Null
    # dtype, and `filter` refuses a Null predicate with a TypeError that falls outside
    # the PolarsError net `create_dataset` wraps this call in. The frame really can be
    # empty here -- every row is dropped first whenever the scientist picks the wrong
    # structure or target column. `_validate_target` avoids this only by building its
    # mask from polars expressions, which stay Boolean at height 0.
    ok = pl.Series([label is not None for label in normalized], dtype=pl.Boolean)
    invalid = [
        InvalidRow(
            row_number=row_numbers[index],
            value=str(text[index]),
            reason=(
                f"Missing value for column '{column}'"
                if not str(text[index]).strip()
                else f"Column '{column}' must say train, validation or test "
                f"(found '{text[index]}')"
            ),
        )
        for index in range(frame.height)
        if normalized[index] is None
    ]
    kept = frame.filter(ok).with_columns(
        pl.Series(column, [label for label in normalized if label is not None], dtype=pl.String)
    )
    kept_rows = [number for number, keep in zip(row_numbers, ok.to_list(), strict=True) if keep]
    return kept, kept_rows, invalid


def prepare_frame(
    frame: pl.DataFrame,
    structure_column: str,
    targets: Sequence[TargetSpec],
    normalizer: StructureNormalizer,
    on_row: RowProgress | None = None,
    *,
    split_column: str | None = None,
    deduplicate: bool = True,
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

    # SMILES first, sequences only as a fallback, and the switch needs *every* row to have
    # failed RDKit. That ordering is the whole safety argument: a column holding even one
    # readable molecule takes exactly the path it took before this existed, so no dataset
    # that works today can be re-read as protein tomorrow. It also resolves the real
    # ambiguity the right way -- `CCN` is both ethylamine and Cys-Cys-Asn, and a chemistry
    # app should answer ethylamine.
    kind = StructureKind.MOLECULE
    if not any(smiles is not None for smiles in canonical):
        sequence_like = sum(looks_like_sequence(value) for value in raw_structures)
        if sequence_like * 2 > total_rows:
            kind = StructureKind.SEQUENCE

    if kind is StructureKind.SEQUENCE:
        # Normalized, not canonicalized: there is no second spelling of a sequence to
        # resolve. Rows that are not sequences stay invalid, exactly as unreadable SMILES
        # does in a molecule dataset.
        canonical = [
            normalize_sequence(value) if looks_like_sequence(value) else None
            for value in raw_structures
        ]

    reason = (
        "Not an amino-acid sequence"
        if kind is StructureKind.SEQUENCE
        else "SMILES could not be parsed"
    )
    invalid = [
        InvalidRow(row_number=index + 1, value=raw_structures[index], reason=reason)
        for index, structure in enumerate(canonical)
        if structure is None
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
    # Each target nulls the cells it cannot use and reports them; a row is removed, and
    # reported as invalid, only when *every* target rejected it. A row measured for some
    # targets and blank for others is the case sparse labels exist for, and the old loop
    # -- which rebound `valid_frame` to each target's survivors in turn -- silently made
    # the dataset the intersection across all of them.
    per_target_failures: list[dict[int, InvalidRow]] = []
    for target in targets:
        valid_frame, row_numbers, bad_targets = _validate_target(valid_frame, target, row_numbers)
        per_target_failures.append({row.row_number: row for row in bad_targets})
    unusable = (
        set.intersection(*(set(failed) for failed in per_target_failures))
        if per_target_failures
        else set()
    )
    # Reported with the *first* target's own reason, not a generic one: "Target 'y' must
    # be 0 or 1 (found 'active')" names the cell to fix, and this file's existing
    # convention is already that a row is reported for the first target it fails. On a
    # single-target dataset that is exactly the behaviour this gate always had.
    invalid.extend(per_target_failures[0][number] for number in sorted(unusable))
    if unusable:
        valid_frame = valid_frame.filter(
            pl.Series([number not in unusable for number in row_numbers])
        )
        row_numbers = [number for number in row_numbers if number not in unusable]
    # Last, after the targets: a row with an unreadable structure and a bad partition is
    # reported for its structure, which is the thing the scientist fixes first.
    if split_column is not None and split_column in valid_frame.columns:
        valid_frame, row_numbers, bad_split = _validate_split_column(
            valid_frame, split_column, row_numbers
        )
        invalid.extend(bad_split)
    invalid.sort(key=lambda row: row.row_number)
    valid_rows = valid_frame.height

    # A salt or mixture is a molecule idea, and `has_multiple_components` parses its
    # argument as SMILES -- on a sequence column it would ask RDKit to read protein and
    # count every row as clean, which is a true number arrived at for a false reason.
    salts_flagged = (
        sum(
            normalizer.has_multiple_components(smiles)
            for smiles in valid_frame[structure_column].to_list()
        )
        if kind is StructureKind.MOLECULE
        else 0
    )

    if valid_rows == 0:
        # Nothing to group. The caller decides what to do with valid_rows == 0; this
        # function's job is to hand back a well-formed report, not to guess or crash.
        return valid_frame, ValidationReport(
            total_rows=total_rows,
            valid_rows=0,
            invalid=invalid,
            salts_flagged=salts_flagged,
            structure_kind=kind,
            labelled_rows=_labelled_rows(valid_frame, targets),
        )

    if not deduplicate:
        # Everything above still ran: the structure gate, the target gates and the split
        # gate all reject rows the same way. Only the grouping is skipped, so the frame
        # keeps the rows the file declared -- which is the point, and also the cost.
        # Without groups there is no replicate spread (the noise floor's only source)
        # and no conflict detection, so one structure labelled two ways now reaches
        # training and can land on both sides of the split.
        return valid_frame, ValidationReport(
            total_rows=total_rows,
            valid_rows=valid_rows,
            invalid=invalid,
            conflicting=[],
            duplicates_collapsed=0,
            salts_flagged=salts_flagged,
            duplicate_spread={},
            structure_kind=kind,
            deduplicated=False,
            labelled_rows=_labelled_rows(valid_frame, targets),
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
        # `drop_nulls()` throughout: a sparse target is null where it was not measured,
        # and a blank is not an answer. Without it polars counts null as a distinct
        # value, so a replicate pair measured once reads as two conflicting answers and
        # the compound is deleted; `.first()` returns null when the first of two rows
        # is the blank one, discarding the measurement on the second; and a group's
        # size is not the number of measurements it actually holds.
        measured = values.drop_nulls()
        if target.kind is TargetKind.NUMERIC:
            aggregations += [
                (measured.max() - measured.min()).alias(f"_spread_{index}"),
                measured.len().alias(f"_measured_{index}"),
                measured.mean().alias(target.column),
            ]
            helpers += [f"_spread_{index}", f"_measured_{index}"]
        else:
            aggregations += [
                measured.n_unique().alias(f"_n_unique_{index}"),
                measured.alias(f"_values_{index}"),
                measured.first().alias(target.column),
            ]
            helpers += [f"_n_unique_{index}", f"_values_{index}"]
    # A designated split column must not be narrowed by `.first()` like any other extra
    # column: replicate rows that disagree about their partition would put the compound
    # on an arbitrary side of the split, which is the one error a split can make that
    # nothing downstream can detect. Count the distinct values so the group can be
    # rejected below, exactly as a disagreeing BINARY target already is.
    split_aggregations: list[pl.Expr] = []
    if split_column is not None and split_column in valid_frame.columns:
        split_aggregations = [
            pl.col(split_column).n_unique().alias("_split_n_unique"),
            pl.col(split_column).alias("_split_values"),
        ]
        helpers += ["_split_n_unique", "_split_values"]
    grouped = valid_frame.group_by(structure_column, maintain_order=True).agg(
        *aggregations, *split_aggregations, *keep_others
    )

    conflicting: list[ConflictRow] = []
    is_conflict = pl.Series([False] * grouped.height)
    if "_split_n_unique" in grouped.columns:
        assert split_column is not None  # set together with the aggregation above
        split_clash = grouped["_split_n_unique"] > 1
        conflicting += [
            ConflictRow(
                structure=str(row[structure_column]),
                column=split_column,
                values=sorted({str(value) for value in row["_split_values"]}),
                row_numbers=sorted(row_numbers_by_structure[str(row[structure_column])]),
            )
            for row in grouped.filter(split_clash).iter_rows(named=True)
        ]
        is_conflict = is_conflict | split_clash
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
        # Gated on how many rows of the group were *measured*, not on how many rows it
        # had. A pair holding one measurement and one blank has a spread of 0.0 -- max
        # and min of a single value -- which would report perfect assay reproducibility
        # for a compound measured once, and that number becomes the Scorecard's noise
        # floor. A group measured on no row at all has a null spread, and `float(None)`
        # raised a TypeError that escaped the PolarsError net as "Could not prepare the
        # dataset. Try again."
        spreads = [
            float(spread)
            for measured_rows, spread in zip(
                agreeing[f"_measured_{index}"].to_list(),
                agreeing[f"_spread_{index}"].to_list(),
                strict=True,
            )
            if measured_rows > 1 and spread is not None
        ]
        if spreads:
            duplicate_spread[target.column] = sum(spreads) / len(spreads)

    deduplicated_frame = agreeing.drop(helpers)
    return deduplicated_frame, ValidationReport(
        total_rows=total_rows,
        valid_rows=valid_rows,
        invalid=invalid,
        conflicting=conflicting,
        duplicates_collapsed=sum(n - 1 for n in group_sizes),
        salts_flagged=salts_flagged,
        duplicate_spread=duplicate_spread,
        structure_kind=kind,
        deduplicated=True,
        labelled_rows=_labelled_rows(deduplicated_frame, targets),
    )
