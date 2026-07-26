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
    prepared, report = prepare_frame(frame, "smiles", NUMERIC, NORMALIZER)
    assert prepared.height == 1
    assert report.invalid[0].row_number == 2
    assert "invalid structure" in report.invalid[0].reason


def test_numeric_duplicates_are_averaged_and_spread_retained():
    frame = pl.DataFrame({"smiles": ["CCO", "OCC", "c1ccccc1"], "y": [1.0, 3.0, 9.0]})
    prepared, report = prepare_frame(frame, "smiles", NUMERIC, NORMALIZER)
    assert prepared.height == 2
    assert prepared.filter(pl.col("smiles") == "CCO")["y"].item() == 2.0
    assert report.duplicates_collapsed == 1
    assert report.duplicate_spread == pytest.approx(2.0)  # |1.0 - 3.0|


def test_conflicting_binary_duplicates_are_rejected_not_voted():
    """A compound labelled both active and inactive is a data problem to decide about."""
    frame = pl.DataFrame({"smiles": ["CCO", "OCC"], "y": [0, 1]})
    prepared, report = prepare_frame(frame, "smiles", BINARY, NORMALIZER)
    assert prepared.height == 0
    assert report.conflicting[0].values == [0, 1]


def test_agreeing_binary_duplicates_collapse_silently():
    frame = pl.DataFrame({"smiles": ["CCO", "OCC"], "y": [1, 1]})
    prepared, report = prepare_frame(frame, "smiles", BINARY, NORMALIZER)
    assert prepared.height == 1
    assert report.conflicting == []
    assert report.duplicates_collapsed == 1


def test_salts_are_flagged_but_kept():
    frame = pl.DataFrame({"smiles": ["CC(=O)O.[Na+]", "CCO"], "y": [1.0, 2.0]})
    prepared, report = prepare_frame(frame, "smiles", NUMERIC, NORMALIZER)
    assert prepared.height == 2
    assert report.salts_flagged == 1


def test_structures_are_canonicalised_so_equivalent_smiles_deduplicate():
    frame = pl.DataFrame({"smiles": ["C1=CC=CC=C1", "c1ccccc1"], "y": [1.0, 1.0]})
    prepared, _ = prepare_frame(frame, "smiles", NUMERIC, NORMALIZER)
    assert prepared.height == 1


def test_all_rows_invalid_yields_an_empty_but_well_formed_report():
    """valid_rows == 0 must not crash the grouping step -- a later task turns this
    count into an upload rejection, so the report has to exist to be inspected."""
    frame = pl.DataFrame({"smiles": ["not-a-molecule", "also-garbage"], "y": [1.0, 2.0]})
    prepared, report = prepare_frame(frame, "smiles", NUMERIC, NORMALIZER)
    assert prepared.height == 0
    assert report.valid_rows == 0
    assert len(report.invalid) == 2
    assert report.duplicate_spread is None


def test_numeric_duplicates_with_identical_values_have_zero_spread_and_still_count():
    """A duplicate group that agrees perfectly is a real zero-noise data point, not an
    absent one, so it counts towards duplicates_collapsed and pulls the spread to 0."""
    frame = pl.DataFrame({"smiles": ["CCO", "OCC"], "y": [5.0, 5.0]})
    prepared, report = prepare_frame(frame, "smiles", NUMERIC, NORMALIZER)
    assert prepared.height == 1
    assert report.duplicates_collapsed == 1
    assert report.duplicate_spread == pytest.approx(0.0)


def test_empty_frame_yields_a_well_formed_report_without_crashing():
    """A CSV with a header row and no data rows reaches prepare_frame at height 0 --
    distinct from the all-invalid case, which has rows that just fail to parse."""
    frame = pl.DataFrame({"smiles": [], "y": []})
    prepared, report = prepare_frame(frame, "smiles", NUMERIC, NORMALIZER)
    assert prepared.height == 0
    assert report.total_rows == 0
    assert report.valid_rows == 0
