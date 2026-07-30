"""Filtering, sorting and stable row identity over a prediction's results.

The frame here deliberately carries nulls in both nullable columns:
`uncertainty` is null for every XGBoost row, and `applicability` is null when
a Protocol's training structures could not be read. Both are real states, and
where they land in a sorted or filtered view is a decision, not an accident.
"""

import polars as pl
import pytest

from daikonstudio.application.execution.result_view import (
    RangeFilter,
    SortSpec,
    apply_result_view,
)

COLUMNS = frozenset({"solubility", "uncertainty", "applicability"})


@pytest.fixture
def frame() -> pl.DataFrame:
    # Row 0 is the least soluble and the least applicable; row 3 carries the
    # nulls. Ordering by any column gives a different permutation, which is
    # what makes the row_id assertions mean something.
    return pl.DataFrame(
        {
            "structure": ["CCO", "CCC", "CCN", "CCF"],
            "solubility": [1.0, 2.0, 3.0, 4.0],
            "uncertainty": [0.5, 0.1, None, 0.3],
            "applicability": [0.2, 0.9, 0.6, None],
        }
    )


def rows(frame, sort=None, filters=()):
    """The row_ids the view yields, in order -- the only assertion that
    survives every reordering, which is the point of having row_id at all."""
    result = apply_result_view(frame, columns=COLUMNS, sort=sort, filters=filters)
    return result.unwrap()["row_id"].to_list()


def test_every_row_carries_its_position_in_the_original_file(frame):
    assert rows(frame) == [0, 1, 2, 3]


def test_sorting_reorders_rows_without_renumbering_them(frame):
    assert rows(frame, sort=SortSpec(column="solubility", descending=True)) == [3, 2, 1, 0]


def test_nulls_sort_last_in_both_directions(frame):
    # A missing measurement is not "the smallest value", and burying it at the
    # top of a descending sort would be the same lie in the other direction.
    assert rows(frame, sort=SortSpec(column="applicability"))[-1] == 3
    assert rows(frame, sort=SortSpec(column="applicability", descending=True))[-1] == 3


def test_a_fully_tied_sort_column_still_yields_stable_row_order(frame):
    # `GetPredictionResults` re-reads the Parquet and re-runs `apply_result_view`
    # on every page request, so page 1 and page 2 are two *independent* sorts.
    # Polars' `sort` is unstable by default (`maintain_order=False`), so a sort
    # key with ties can permute the tied rows differently each call -- a row
    # could land on both pages, or on neither, with no error to notice it by.
    # This is not a rare edge case to hedge against: `uncertainty` is null for
    # every row of an XGBoost run, which makes the entire frame one tie group,
    # and a classification protocol's class column is only ever 0.0 or 1.0.
    # Modelling the XGBoost case -- every value in the sort column null --
    # is what would have caught the missing `ROW_ID` tiebreak.
    tied = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("uncertainty"))
    assert rows(tied, sort=SortSpec(column="uncertainty")) == [0, 1, 2, 3]


def test_a_minimum_excludes_smaller_values_and_nulls(frame):
    assert rows(frame, filters=(RangeFilter(column="applicability", minimum=0.5),)) == [1, 2]


def test_a_maximum_excludes_larger_values_and_nulls(frame):
    assert rows(frame, filters=(RangeFilter(column="uncertainty", maximum=0.4),)) == [1, 3]


def test_both_bounds_keep_only_what_is_between_them(frame):
    assert rows(frame, filters=(RangeFilter(column="solubility", minimum=2.0, maximum=3.0),)) == [
        1,
        2,
    ]


def test_filter_and_sort_compose_and_preserve_row_id(frame):
    assert rows(
        frame,
        filters=(RangeFilter(column="solubility", minimum=2.0),),
        sort=SortSpec(column="solubility", descending=True),
    ) == [3, 2, 1]


def test_a_filter_that_matches_nothing_is_an_empty_view_not_an_error(frame):
    assert rows(frame, filters=(RangeFilter(column="applicability", minimum=1.1),)) == []


def test_an_unknown_sort_column_is_rejected(frame):
    result = apply_result_view(frame, columns=COLUMNS, sort=SortSpec(column="nope"), filters=())
    assert result.failure() is not None


def test_an_unknown_filter_column_is_rejected(frame):
    result = apply_result_view(
        frame, columns=COLUMNS, sort=None, filters=(RangeFilter(column="nope", minimum=1.0),)
    )
    assert result.failure() is not None


def test_the_structure_column_is_not_sortable(frame):
    # Sorting by SMILES string is alphabetical nonsense dressed as chemistry.
    result = apply_result_view(
        frame, columns=COLUMNS, sort=SortSpec(column="structure"), filters=()
    )
    assert result.failure() is not None
