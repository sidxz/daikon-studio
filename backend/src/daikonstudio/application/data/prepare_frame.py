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

import polars as pl

from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.data.target import TargetKind, TargetSpec
from daikonstudio.domain.data.validation import ConflictRow, InvalidRow, ValidationReport


def prepare_frame(
    frame: pl.DataFrame,
    structure_column: str,
    target: TargetSpec,
    normalizer: StructureNormalizer,
) -> tuple[pl.DataFrame, ValidationReport]:
    total_rows = frame.height
    raw_structures = [str(value) for value in frame[structure_column].to_list()]
    canonical = [normalizer.canonicalize(smiles) for smiles in raw_structures]

    invalid = [
        InvalidRow(row_number=index + 1, value=raw_structures[index], reason="invalid structure")
        for index, smiles in enumerate(canonical)
        if smiles is None
    ]

    is_valid = pl.Series([smiles is not None for smiles in canonical])
    valid_frame = frame.filter(is_valid).with_columns(
        pl.Series(structure_column, [smiles for smiles in canonical if smiles is not None])
    )
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

    other_columns = [c for c in frame.columns if c not in (structure_column, target.column)]
    keep_others = [pl.col(c).first() for c in other_columns]

    if target.kind is TargetKind.NUMERIC:
        grouped = valid_frame.group_by(structure_column, maintain_order=True).agg(
            pl.col(target.column).len().alias("_n"),
            (pl.col(target.column).max() - pl.col(target.column).min()).alias("_spread"),
            pl.col(target.column).mean().alias(target.column),
            *keep_others,
        )
        group_sizes = grouped["_n"].to_list()
        group_spreads = grouped["_spread"].to_list()
        duplicates_collapsed = sum(n - 1 for n in group_sizes if n > 1)
        # Groups of size one carry no spread information and are excluded from the
        # mean entirely -- they don't count as "zero spread", they count as nothing.
        spreads = [
            float(spread) for n, spread in zip(group_sizes, group_spreads, strict=True) if n > 1
        ]
        duplicate_spread = sum(spreads) / len(spreads) if spreads else None
        prepared = grouped.drop("_n", "_spread")
        return prepared, ValidationReport(
            total_rows=total_rows,
            valid_rows=valid_rows,
            invalid=invalid,
            duplicates_collapsed=duplicates_collapsed,
            salts_flagged=salts_flagged,
            duplicate_spread=duplicate_spread,
        )

    # BINARY: agreeing duplicates collapse silently; disagreeing duplicates are a data
    # problem for the scientist to resolve, not one a majority vote papers over.
    grouped = valid_frame.group_by(structure_column, maintain_order=True).agg(
        pl.col(target.column).len().alias("_n"),
        pl.col(target.column).n_unique().alias("_n_unique"),
        pl.col(target.column).alias("_values"),
        pl.col(target.column).first().alias(target.column),
        *keep_others,
    )
    is_conflict = grouped["_n_unique"] > 1
    conflicting = [
        ConflictRow(structure=str(row[structure_column]), values=list(row["_values"]))
        for row in grouped.filter(is_conflict).iter_rows(named=True)
    ]
    agreeing = grouped.filter(~is_conflict)
    duplicates_collapsed = sum(n - 1 for n in agreeing["_n"].to_list())
    prepared = agreeing.drop("_n", "_n_unique", "_values")
    return prepared, ValidationReport(
        total_rows=total_rows,
        valid_rows=valid_rows,
        invalid=invalid,
        conflicting=conflicting,
        duplicates_collapsed=duplicates_collapsed,
        salts_flagged=salts_flagged,
    )
