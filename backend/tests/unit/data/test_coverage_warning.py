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


def test_the_profile_page_does_not_count_unmeasured_rows_as_inactive() -> None:
    """The sixth instance of the counting family, on a page the spec's audit missed.

    A Tox21 assay with 400 actives, 4,300 inactives and 3,100 unmeasured rows would
    render as 400 active / 7,400 inactive -- a falsely imbalanced column, stated as a
    measurement, with no caveat.
    """
    import numpy as np

    from daikonstudio.application.data.build_profile import _class_balance

    targets = np.array([1.0, 0.0, float("nan"), float("nan"), 1.0])
    splits = ["train"] * 5

    [balance] = _class_balance(targets, splits)

    assert (balance.positive, balance.negative) == (2, 1)


def test_a_barely_measured_numeric_target_is_flagged_too() -> None:
    """Coverage is per target, not per binary target. A ChEMBL-style export whose IC50
    column reads `NA` on most rows used to fail the upload loudly; now those rows null
    and survive, so the review screen is the only place that can say so."""
    frame = pl.DataFrame({"sol": [1.0, 2.0, 3.0] + [None] * 97, "split": ["train"] * 100})

    readiness = dataset_readiness(frame, (TargetSpec(column="sol", kind=TargetKind.NUMERIC),))

    assert any("3 of 100" in warning for warning in readiness.warnings)


def test_an_all_blank_test_partition_does_not_crash_the_numeric_check() -> None:
    """`len(test)` counts unmeasured rows, so an all-blank test partition is truthy
    and `test.mean()` is None -- `abs(train_mean - None)` raises a TypeError that
    surfaces as "Could not prepare the dataset. Try again." The distribution-shift
    check has to read measured rows, like every other count on this screen."""
    frame = pl.DataFrame(
        {
            "sol": [1.0, 2.0, 3.0, 4.0, None],
            "split": ["train"] * 4 + ["test"],
        }
    )

    readiness = dataset_readiness(frame, (TargetSpec(column="sol", kind=TargetKind.NUMERIC),))

    assert not any("distribution shift" in warning for warning in readiness.warnings)
