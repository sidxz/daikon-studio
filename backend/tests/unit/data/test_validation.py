import polars as pl
import pytest

from daikonstudio.application.data.prepare_frame import prepare_frame
from daikonstudio.domain.data.target import Direction, TargetKind, TargetSpec
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer

NUMERIC = TargetSpec(column="y", kind=TargetKind.NUMERIC, unit="nM", direction=Direction.LOW)
BINARY = TargetSpec(column="y", kind=TargetKind.BINARY)
NORMALIZER = RdkitStructureNormalizer()


def test_invalid_structures_are_reported_with_row_numbers():
    frame = pl.DataFrame({"smiles": ["CCO", "not-a-molecule"], "y": [1.0, 2.0]})
    prepared, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER)
    assert prepared.height == 1
    assert report.invalid[0].row_number == 2
    assert "SMILES could not be parsed" in report.invalid[0].reason


def test_numeric_duplicates_are_averaged_and_spread_retained():
    frame = pl.DataFrame({"smiles": ["CCO", "OCC", "c1ccccc1"], "y": [1.0, 3.0, 9.0]})
    prepared, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER)
    assert prepared.height == 2
    assert prepared.filter(pl.col("smiles") == "CCO")["y"].item() == 2.0
    assert report.duplicates_collapsed == 1
    assert report.duplicate_spread == {"y": pytest.approx(2.0)}  # |1.0 - 3.0|


def test_conflicting_binary_duplicates_are_rejected_not_voted():
    """A compound labelled both active and inactive is a data problem to decide about."""
    frame = pl.DataFrame({"smiles": ["CCO", "OCC"], "y": [0, 1]})
    prepared, report = prepare_frame(frame, "smiles", (BINARY,), NORMALIZER)
    assert prepared.height == 0
    assert report.conflicting[0].values == [0, 1]
    assert report.conflicting[0].row_numbers == [1, 2]


def test_a_conflict_names_the_target_column_its_labels_disagree_in():
    frame = pl.DataFrame({"smiles": ["CCO", "OCC"], "y": [0, 1]})
    _, report = prepare_frame(frame, "smiles", (BINARY,), NORMALIZER)
    assert report.conflicting[0].column == "y"


def test_conflicting_row_numbers_are_positions_in_the_uploaded_file_not_in_a_filtered_frame():
    """I4 (whole-branch review, Important): the spec calls for conflicting
    replicates to be "rejected with their row numbers" -- these must be
    1-indexed positions in the file the scientist actually uploaded (matching
    `InvalidRow.row_number`'s own convention), the same numbers a spreadsheet
    would show, not positions re-counted after invalid rows are dropped.

    Row 1 is invalid and filtered out before grouping; the conflicting pair
    that follows it must still report row numbers 2 and 3 -- not 1 and 2, the
    positions they would fall on if row numbers were assigned after the
    invalid row's removal.
    """
    frame = pl.DataFrame(
        {"smiles": ["not-a-molecule", "CCO", "OCC", "c1ccccc1"], "y": [9, 0, 1, 1]}
    )
    prepared, report = prepare_frame(frame, "smiles", (BINARY,), NORMALIZER)
    assert prepared.height == 1  # only the unconflicted benzene row survives
    assert report.invalid[0].row_number == 1
    assert len(report.conflicting) == 1
    assert report.conflicting[0].structure == "CCO"
    assert report.conflicting[0].values == [0, 1]
    assert report.conflicting[0].row_numbers == [2, 3]


def test_agreeing_binary_duplicates_collapse_silently():
    frame = pl.DataFrame({"smiles": ["CCO", "OCC"], "y": [1, 1]})
    prepared, report = prepare_frame(frame, "smiles", (BINARY,), NORMALIZER)
    assert prepared.height == 1
    assert report.conflicting == []
    assert report.duplicates_collapsed == 1


def test_salts_are_flagged_but_kept():
    frame = pl.DataFrame({"smiles": ["CC(=O)O.[Na+]", "CCO"], "y": [1.0, 2.0]})
    prepared, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER)
    assert prepared.height == 2
    assert report.salts_flagged == 1


def test_structures_are_canonicalised_so_equivalent_smiles_deduplicate():
    frame = pl.DataFrame({"smiles": ["C1=CC=CC=C1", "c1ccccc1"], "y": [1.0, 1.0]})
    prepared, _ = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER)
    assert prepared.height == 1


def test_all_rows_invalid_yields_an_empty_but_well_formed_report():
    """valid_rows == 0 must not crash the grouping step -- a later task turns this
    count into an upload rejection, so the report has to exist to be inspected."""
    frame = pl.DataFrame({"smiles": ["not-a-molecule", "also-garbage"], "y": [1.0, 2.0]})
    prepared, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER)
    assert prepared.height == 0
    assert report.valid_rows == 0
    assert len(report.invalid) == 2
    assert report.duplicate_spread == {}


def test_numeric_duplicates_with_identical_values_have_zero_spread_and_still_count():
    """A duplicate group that agrees perfectly is a real zero-noise data point, not an
    absent one, so it counts towards duplicates_collapsed and pulls the spread to 0."""
    frame = pl.DataFrame({"smiles": ["CCO", "OCC"], "y": [5.0, 5.0]})
    prepared, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER)
    assert prepared.height == 1
    assert report.duplicates_collapsed == 1
    assert report.duplicate_spread == {"y": pytest.approx(0.0)}


def test_empty_frame_yields_a_well_formed_report_without_crashing():
    """A CSV with a header row and no data rows reaches prepare_frame at height 0 --
    distinct from the all-invalid case, which has rows that just fail to parse."""
    frame = pl.DataFrame({"smiles": [], "y": []})
    prepared, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER)
    assert prepared.height == 0
    assert report.total_rows == 0
    assert report.valid_rows == 0


def test_a_bom_on_the_first_header_is_stripped():
    from daikonstudio.application.data.prepare_frame import read_csv_upload

    frame = read_csv_upload("﻿smiles,y\nCCO,1.0\n".encode())
    assert frame.columns == ["smiles", "y"]


def test_null_numeric_targets_are_invalid_rows_not_training_failures():
    from daikonstudio.domain.data.validation import InvalidRow

    frame = pl.DataFrame({"smiles": ["CCO", "CCC"], "y": [1.0, None]})
    prepared, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER)
    assert prepared.height == 1
    assert report.valid_rows == 1
    assert report.invalid == [
        InvalidRow(row_number=2, value="", reason="Missing value for target 'y'")
    ]


def test_non_numeric_target_values_are_invalid_rows_not_a_crash():
    """A numeric column holding `NA` or `<10` arrives as text. Before this gate
    the mean/max/min aggregation raised deep inside polars and the request was
    an unhandled 500; now the rows are named and the rest of the file is kept."""
    frame = pl.DataFrame({"smiles": ["CCO", "CCC", "CCN"], "y": ["1.5", "NA", "<10"]})
    prepared, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER)
    assert prepared["y"].dtype == pl.Float64
    assert prepared["y"].to_list() == [1.5]
    assert [row.row_number for row in report.invalid] == [2, 3]
    assert report.invalid[0].reason == "Target 'y' is not numeric: 'NA'"


def test_binary_targets_must_be_zero_or_one():
    frame = pl.DataFrame({"smiles": ["CCO", "CCC", "CCN"], "y": ["1", "active", "2"]})
    prepared, report = prepare_frame(frame, "smiles", (BINARY,), NORMALIZER)
    assert prepared["y"].to_list() == [1]
    assert [row.reason for row in report.invalid] == [
        "Target 'y' must be 0 or 1 (found 'active')",
        "Target 'y' must be 0 or 1 (found '2')",
    ]


def test_a_row_failing_both_gates_is_reported_once_for_its_structure():
    frame = pl.DataFrame({"smiles": ["not-a-molecule", "CCO"], "y": [None, 1.0]})
    _, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER)
    assert [row.reason for row in report.invalid] == ["SMILES could not be parsed"]


def test_nan_and_inf_targets_are_invalid_rows_too():
    """polars parses "nan" and "inf" into floats that are not null, and either
    one is the `Input y contains NaN` failure the gate exists to stop."""
    frame = pl.DataFrame({"smiles": ["CCO", "CCC", "CCN"], "y": ["1.0", "nan", "inf"]})
    prepared, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER)
    assert prepared["y"].to_list() == [1.0]
    assert [row.row_number for row in report.invalid] == [2, 3]


def test_read_csv_upload_keeps_text_verbatim_and_tolerates_a_late_bad_value():
    """Two reasons every column is read as text: an identifier like 00123 must
    not come back as 123, and a numeric column whose first hundred rows parse
    must not make polars raise on row 150's NA -- that row is a reported
    InvalidRow, and the rest of the file is kept."""
    from daikonstudio.application.data.prepare_frame import read_csv_upload

    rows = "\n".join(f"{i:05d},CCO,{i}.5" for i in range(1, 150)) + "\n00150,CCC,NA\n"
    frame = read_csv_upload(f"id,smiles,y\n{rows}".encode())
    assert frame["id"][0] == "00001"

    prepared, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER)
    assert [row.row_number for row in report.invalid] == [150]
    assert report.invalid[0].reason == "Target 'y' is not numeric: 'NA'"
    assert prepared["y"].dtype == pl.Float64


SOLUBILITY = TargetSpec(column="solubility", kind=TargetKind.NUMERIC)
REACTIVE = TargetSpec(column="reactive", kind=TargetKind.BINARY)


def test_every_target_is_gated_and_the_reason_names_its_column():
    frame = pl.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC"],
            "solubility": ["1.0", "2.0", "3.0"],
            "reactive": ["0", "", "1"],
        }
    )
    prepared, report = prepare_frame(frame, "smiles", (SOLUBILITY, REACTIVE), NORMALIZER)
    assert prepared.height == 2
    assert [row.reason for row in report.invalid] == ["Missing value for target 'reactive'"]


def test_duplicates_collapse_per_target_kind_and_keep_column_order():
    frame = pl.DataFrame(
        {
            "smiles": ["CCO", "OCC", "c1ccccc1"],
            "solubility": [1.0, 3.0, 9.0],
            "reactive": [1, 1, 0],
        }
    )
    prepared, report = prepare_frame(frame, "smiles", (SOLUBILITY, REACTIVE), NORMALIZER)
    assert prepared.columns == ["smiles", "solubility", "reactive"]
    row = prepared.filter(pl.col("smiles") == "CCO")
    assert row["solubility"].item() == 2.0
    assert row["reactive"].item() == 1
    assert report.duplicates_collapsed == 1
    assert report.duplicate_spread == {"solubility": pytest.approx(2.0)}


def test_a_conflict_in_one_binary_target_drops_the_compound_and_names_the_column():
    frame = pl.DataFrame(
        {
            "smiles": ["CCO", "OCC", "CCN"],
            "solubility": [1.0, 1.0, 2.0],
            "reactive": [0, 1, 0],
        }
    )
    prepared, report = prepare_frame(frame, "smiles", (SOLUBILITY, REACTIVE), NORMALIZER)
    assert prepared["smiles"].to_list() == ["CCN"]
    assert [(c.column, c.values) for c in report.conflicting] == [("reactive", [0, 1])]


def test_a_column_name_with_braces_is_reported_verbatim():
    target = TargetSpec(column="IC50 {nM}", kind=TargetKind.NUMERIC)
    frame = pl.DataFrame({"smiles": ["CCO", "CCN"], "IC50 {nM}": ["1.0", "NA"]})
    _, report = prepare_frame(frame, "smiles", (target,), NORMALIZER)
    assert report.invalid[0].reason == "Target 'IC50 {nM}' is not numeric: 'NA'"


def test_prepare_frame_reports_progress_every_thousand_rows_and_at_the_end():
    """What the wizard's progress bar is fed during a background build."""
    frame = pl.DataFrame({"smiles": ["CCO", "CCN"] * 1250, "y": [1.0, 2.0] * 1250})
    seen: list[int] = []

    prepare_frame(
        frame,
        "smiles",
        (TargetSpec(column="y", kind=TargetKind.NUMERIC),),
        RdkitStructureNormalizer(),
        on_row=seen.append,
    )

    assert seen == [1000, 2000, 2500]


# --- The designated split column -----------------------------------------------------
#
# A predefined split reads each row's partition out of an uploaded column. Two things
# have to happen here rather than in `assign_split`: a bad cell must be reported against
# its position in the *uploaded* file, which only this layer still knows, and replicate
# rows that disagree about their partition must be rejected rather than collapsed to
# whichever row came first -- `keep_others` narrows every other extra column with
# `.first()`, and for a split assignment that would silently put a compound on the wrong
# side of the split.


def test_a_split_column_value_that_is_not_a_partition_is_an_invalid_row():
    from daikonstudio.domain.data.validation import InvalidRow

    frame = pl.DataFrame(
        {"smiles": ["CCO", "CCC"], "y": [1.0, 2.0], "split": ["train", "holdout"]}
    )
    prepared, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER, split_column="split")
    assert prepared.height == 1
    assert report.invalid == [
        InvalidRow(
            row_number=2,
            value="holdout",
            reason="Column 'split' must say train, validation or test (found 'holdout')",
        )
    ]


def test_a_bad_split_cell_is_reported_against_its_uploaded_row_not_a_filtered_position():
    # Row 2 is dropped for an unreadable structure, so the bad split cell on uploaded
    # row 3 sits at position 2 of the filtered frame. The report must say 3.
    frame = pl.DataFrame(
        {
            "smiles": ["CCO", "not-a-molecule", "CCC"],
            "y": [1.0, 2.0, 3.0],
            "split": ["train", "train", "holdout"],
        }
    )
    _, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER, split_column="split")
    reasons = {row.row_number: row.reason for row in report.invalid}
    assert 3 in reasons
    assert "split" in reasons[3]


def test_replicates_that_agree_about_their_partition_collapse_normally():
    frame = pl.DataFrame(
        {"smiles": ["CCO", "OCC", "CCC"], "y": [1.0, 2.0, 3.0], "split": ["train"] * 3}
    )
    prepared, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER, split_column="split")
    assert prepared.height == 2
    assert report.conflicting == []


def test_replicates_that_disagree_about_their_partition_are_rejected():
    """Taking the first row's label would put the compound on an arbitrary side."""
    frame = pl.DataFrame(
        {"smiles": ["CCO", "OCC", "CCC"], "y": [1.0, 2.0, 3.0], "split": ["train", "test", "test"]}
    )
    prepared, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER, split_column="split")
    assert prepared.height == 1
    assert len(report.conflicting) == 1
    conflict = report.conflicting[0]
    assert conflict.column == "split"
    assert sorted(conflict.values) == ["test", "train"]
    assert conflict.row_numbers == [1, 2]


def test_a_partition_spelling_is_normalised_before_replicates_are_compared():
    # "Train " and "train" are the same partition, so these replicates agree.
    frame = pl.DataFrame(
        {
            "smiles": ["CCO", "OCC", "CCC"],
            "y": [1.0, 2.0, 3.0],
            "split": ["Train ", "train", "test"],
        }
    )
    prepared, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER, split_column="split")
    assert report.conflicting == []
    assert prepared.height == 2


def test_no_split_column_leaves_preparation_exactly_as_it_was():
    frame = pl.DataFrame({"smiles": ["CCO", "OCC"], "y": [1.0, 2.0], "split": ["train", "test"]})
    prepared, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER)
    assert prepared.height == 1
    assert report.conflicting == []
    assert report.invalid == []


def test_an_empty_split_cell_is_a_missing_value_not_a_partition():
    # A partially-labelled file is a real upload: the scientist filtered in Excel and
    # left blanks. Reading a blank as some partition would be worse than saying so.
    from daikonstudio.domain.data.validation import InvalidRow

    frame = pl.DataFrame({"smiles": ["CCO", "CCC"], "y": [1.0, 2.0], "split": ["train", ""]})
    prepared, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER, split_column="split")
    assert prepared.height == 1
    assert report.invalid == [
        InvalidRow(row_number=2, value="", reason="Missing value for column 'split'")
    ]


def test_a_split_column_survives_every_row_being_rejected_first():
    """Picking the wrong structure column is ordinary user error, and it empties the
    frame before the split gate sees it. An empty boolean predicate infers polars' Null
    dtype, which `filter` refuses -- and the TypeError is outside the PolarsError net
    `create_dataset` wraps this in, so it would surface as a crash rather than the
    designed 422 with the per-row report."""
    frame = pl.DataFrame(
        {"smiles": ["nope", "also-nope"], "y": [1.0, 2.0], "split": ["train", "test"]}
    )
    prepared, report = prepare_frame(frame, "smiles", (NUMERIC,), NORMALIZER, split_column="split")
    assert prepared.height == 0
    assert report.valid_rows == 0
    assert len(report.invalid) == 2
