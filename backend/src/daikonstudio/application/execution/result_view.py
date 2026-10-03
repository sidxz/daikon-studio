"""Sorting and range-filtering a prediction Run's results.

Pure, and separate from `GetPredictionResults`, for two reasons. The row
identity decision here is the one that can silently corrupt a Collection, so
it is worth testing without a repository harness in the way. And a results
file is immutable, so a "view" over it is a value, not a query against
changing state -- it belongs to no aggregate and touches no I/O.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import polars as pl
from returns.result import Failure, Result, Success

from daikonstudio.domain.shared.errors import DomainError, ValidationError

__all__ = ["ROW_ID", "RangeFilter", "SortSpec", "apply_result_view"]

#: The row's position in the original results file, which is what
#: `POST /collections` takes as `row_ids`.
ROW_ID = "row_id"


@dataclass(frozen=True, kw_only=True)
class SortSpec:
    column: str
    descending: bool = False


@dataclass(frozen=True, kw_only=True)
class RangeFilter:
    """One column's bounds. Both ends are optional; an entry with neither is
    rejected at the edge, because a filter that filters nothing is a control
    that silently does nothing."""

    column: str
    minimum: float | None = None
    maximum: float | None = None


def apply_result_view(
    frame: pl.DataFrame,
    *,
    columns: Collection[str],
    sort: SortSpec | None,
    filters: Collection[RangeFilter],
) -> Result[pl.DataFrame, DomainError]:
    """`frame` filtered and sorted, with every row stamped with `ROW_ID`.

    `columns` is what may be sorted or filtered on -- the Protocol's readout
    names plus `uncertainty` and `applicability`. `structure` is deliberately
    not among them: ordering compounds by their SMILES string is alphabetical
    nonsense dressed up as chemistry.
    """
    # Minted first, so it always means "position in the original file". Derive
    # it from the page offset instead and a Collection saved under a sort holds
    # different compounds than the ones that were selected, with no symptom.
    view = frame.with_row_index(name=ROW_ID)

    for range_filter in filters:
        if range_filter.column not in columns:
            return Failure(_unknown(range_filter.column, columns, verb="filter on"))
        column = pl.col(range_filter.column)
        # A comparison against null is null, which `filter` drops -- so a
        # bounded column excludes its own nulls without a separate guard. That
        # is intended: "applicability at least 0.5" cannot honestly include a
        # compound whose applicability was never measurable.
        if range_filter.minimum is not None:
            view = view.filter(column >= range_filter.minimum)
        if range_filter.maximum is not None:
            view = view.filter(column <= range_filter.maximum)

    if sort is not None:
        if sort.column not in columns:
            return Failure(_unknown(sort.column, columns, verb="sort by"))
        # Nulls last either way: `uncertainty` is null for every XGBoost row
        # and `applicability` is null when the training set could not be read.
        # Neither is "the smallest value", so neither may lead a page.
        #
        # `ROW_ID` as a second sort key breaks every tie on `sort.column` in
        # favour of file order. Polars' `sort` is unstable by default (ties may
        # land in either relative order), and `GetPredictionResults` re-reads
        # the Parquet and re-sorts on every page request -- an unstable sort on
        # a column with ties gives two independent, possibly-different
        # permutations of the tied rows for page 1 and page 2, so a row can be
        # emitted on both pages or on neither. This is not a rare edge case:
        # `uncertainty` is null for every row of an XGBoost run (one tie group,
        # the whole frame), and a classification protocol's class column is
        # only ever 0.0 or 1.0 (two tie groups, half the rows each). Sorting on
        # `[sort.column, ROW_ID]` makes the ordering a total order, so it is
        # identical across requests by construction.
        view = view.sort(
            [sort.column, ROW_ID],
            descending=[sort.descending, False],
            nulls_last=[True, False],
        )

    return Success(view)


def _unknown(column: str, columns: Collection[str], *, verb: str) -> DomainError:
    return ValidationError(
        f"Cannot {verb} '{column}': this column is not in the results.",
        detail=f"Available columns: {', '.join(sorted(columns))}.",
    )
