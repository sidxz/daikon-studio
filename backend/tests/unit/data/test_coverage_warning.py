"""Sparse ingestion turns a loud failure quiet.

A mistyped target column used to reject every row and fail the upload visibly. Now
those rows survive on their other targets and the dataset freezes happily with one
target measured on almost nothing. The review screen has to say so before the freeze,
in the same "Things to consider" list that already carries class imbalance and thin
validation -- stated plainly, not blocking, because Tox21 is a third blank by nature.
"""

from __future__ import annotations

import polars as pl

from daikonstudio.application.data.preview_dataset import dataset_readiness
from daikonstudio.domain.data.target import TargetKind, TargetSpec

TOX = (TargetSpec(column="tox", kind=TargetKind.BINARY),)


def test_a_barely_measured_target_is_flagged() -> None:
    frame = pl.DataFrame({"tox": [1.0, 0.0] + [None] * 98, "split": ["train"] * 100})

    readiness = dataset_readiness(frame, TOX)

    assert any("2 of 100" in warning for warning in readiness.warnings)
    assert any("tox" in warning for warning in readiness.warnings)


def test_a_fully_measured_target_is_not_flagged() -> None:
    frame = pl.DataFrame({"tox": [1.0, 0.0] * 50, "split": ["train"] * 100})

    readiness = dataset_readiness(frame, TOX)

    assert not any("measured for only" in warning for warning in readiness.warnings)


def test_the_labelled_count_reaches_the_balance_rows() -> None:
    """The count the Scorecard and the review screen both need: without it a target
    with 2 labels and one with 100 render the same."""
    frame = pl.DataFrame({"tox": [1.0, 0.0, None, None], "split": ["train"] * 4})

    readiness = dataset_readiness(frame, TOX)

    train = next(b for b in readiness.class_balance if b.split == "train")
    assert train.labelled == 2
