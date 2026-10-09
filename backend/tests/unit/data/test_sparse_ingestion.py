"""A row measured for some targets and blank for others is data, not an error.

The gate used to rebind `valid_frame` once per target, so the survivors were the
intersection across all targets: a fifth target that is 30% blank discarded 30% of
every other target's rows, and the dataset that trained was not the one the published
numbers described.
"""

from __future__ import annotations

import polars as pl

from daikonstudio.application.data.prepare_frame import prepare_frame
from daikonstudio.domain.data.target import TargetKind, TargetSpec
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer

NORMALIZER = RdkitStructureNormalizer()
TARGETS = (
    TargetSpec(column="a", kind=TargetKind.BINARY),
    TargetSpec(column="b", kind=TargetKind.BINARY),
    TargetSpec(column="c", kind=TargetKind.BINARY),
)


def _staggered() -> pl.DataFrame:
    """Three compounds, each blank for a different target. Under the old gate every
    row failed some target, so nothing survived at all."""
    return pl.DataFrame(
        {
            "smiles": ["CCO", "c1ccccc1", "CCN"],
            "a": [None, 1, 0],
            "b": [1, None, 0],
            "c": [1, 0, None],
        }
    )


def test_a_row_blank_in_one_target_survives_on_the_others() -> None:
    frame, report = prepare_frame(
        _staggered(), structure_column="smiles", targets=TARGETS, normalizer=NORMALIZER
    )

    assert frame.height == 3
    assert report.valid_rows == 3
    assert report.invalid == []
    assert report.labelled_rows == {"a": 2, "b": 2, "c": 2}


def test_a_row_blank_in_every_target_is_still_rejected() -> None:
    """Nothing to learn from and nothing to score: this row is an error, and it is
    reported with the first target's own reason rather than a generic one, because
    "Missing value for target 'a'" names the cell to go and fill. That is also this
    gate's existing convention -- a row is reported for the first target it fails -- and
    on a single-target dataset it is unchanged behaviour."""
    frame = pl.DataFrame(
        {"smiles": ["CCO", "CCN"], "a": [1, None], "b": [0, None], "c": [1, None]}
    )

    kept, report = prepare_frame(
        frame, structure_column="smiles", targets=TARGETS, normalizer=NORMALIZER
    )

    assert kept.height == 1
    assert [row.row_number for row in report.invalid] == [2]
    assert report.invalid[0].reason == "Missing value for target 'a'"


def test_the_labelled_count_survives_a_round_trip_through_the_manifest() -> None:
    """`report_to_dict`/`report_from_dict` is how a preview becomes a frozen Dataset.
    A field written but not read is lost at the moment the Dataset is created, which
    is before it is ever persisted -- so the count would exist only in a preview that
    nobody keeps, and the runner envelope would carry an empty dict."""
    from daikonstudio.domain.data.validation import report_from_dict, report_to_dict

    _, report = prepare_frame(
        _staggered(), structure_column="smiles", targets=TARGETS, normalizer=NORMALIZER
    )

    restored = report_from_dict(report_to_dict(report))

    assert restored.labelled_rows == {"a": 2, "b": 2, "c": 2}


def test_a_replicate_measured_once_is_not_a_conflict() -> None:
    """Deduplication groups by structure and flags a group whose target holds more
    than one distinct value. polars counts null as a value, so `[1, None]` -- one
    measurement and one blank, which is the normal shape of a pooled sparse table --
    read as two conflicting answers. The compound was then deleted entirely, taking
    its other targets' measurements with it, and reported to the scientist as data
    that disagrees with itself."""
    frame = pl.DataFrame(
        {
            "smiles": ["CCO", "OCC", "CCN"],  # first two are the same compound
            "a": [1, None, 0],
            "b": [1, 1, 0],
        }
    )
    targets = (
        TargetSpec(column="a", kind=TargetKind.BINARY),
        TargetSpec(column="b", kind=TargetKind.BINARY),
    )

    kept, report = prepare_frame(
        frame, structure_column="smiles", targets=targets, normalizer=NORMALIZER
    )

    assert report.conflicting == []
    assert kept.height == 2
    assert sorted(kept["a"].drop_nulls().to_list()) == [0, 1]


def test_an_unmeasured_replicate_contributes_no_assay_noise() -> None:
    """A numeric target blank on one row of a replicate pair has one measurement, not
    two agreeing ones. Treating `[5.0, None]` as a spread of 0.0 reports perfect assay
    reproducibility for a compound that was measured once, and that number becomes the
    Scorecard's noise floor."""
    frame = pl.DataFrame(
        {
            "smiles": ["CCO", "OCC", "CCN"],
            "sol": [5.0, None, 2.0],
            "tox": [1, 1, 0],
        }
    )
    targets = (
        TargetSpec(column="sol", kind=TargetKind.NUMERIC),
        TargetSpec(column="tox", kind=TargetKind.BINARY),
    )

    _, report = prepare_frame(
        frame, structure_column="smiles", targets=targets, normalizer=NORMALIZER
    )

    assert "sol" not in report.duplicate_spread


def test_a_replicate_group_with_no_numeric_measurement_does_not_crash() -> None:
    """Both rows blank for the numeric target, kept alive by the binary one. The
    spread is null, and `float(None)` raised a TypeError that escaped the PolarsError
    net and surfaced as "Could not prepare the dataset. Try again." -- no cause, no
    row number, and retrying cannot help."""
    frame = pl.DataFrame(
        {
            "smiles": ["CCO", "OCC", "CCN"],
            "sol": [None, None, 2.0],
            "tox": [1, 1, 0],
        }
    )
    targets = (
        TargetSpec(column="sol", kind=TargetKind.NUMERIC),
        TargetSpec(column="tox", kind=TargetKind.BINARY),
    )

    kept, report = prepare_frame(
        frame, structure_column="smiles", targets=targets, normalizer=NORMALIZER
    )

    assert kept.height == 2
    assert "sol" not in report.duplicate_spread


def test_a_measurement_on_the_second_replicate_row_is_not_lost() -> None:
    """Deduplication collapsed a binary group with `.first()`. When the first of two
    replicate rows is the blank one -- which in a pooled table depends only on upload
    order -- the surviving row took the null and the real measurement was discarded,
    silently, with the row count looking correct."""
    frame = pl.DataFrame(
        {
            "smiles": ["CCO", "OCC", "CCN"],  # first two are the same compound
            "a": [None, 1, 0],  # measured only on the SECOND of the pair
            "b": [1, 1, 0],
        }
    )
    targets = (
        TargetSpec(column="a", kind=TargetKind.BINARY),
        TargetSpec(column="b", kind=TargetKind.BINARY),
    )

    kept, report = prepare_frame(
        frame, structure_column="smiles", targets=targets, normalizer=NORMALIZER
    )

    assert report.conflicting == []
    assert kept.height == 2
    assert report.labelled_rows["a"] == 2
