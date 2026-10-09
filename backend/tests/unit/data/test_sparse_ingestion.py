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
