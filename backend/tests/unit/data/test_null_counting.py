"""Null is not a value. Four places counted it as one, and sparse labels make all
four reachable: polars' `n_unique()` counts null as a distinct value, and
`count - positive` counts an unmeasured row as a negative result."""

from __future__ import annotations

import polars as pl

from daikonstudio.application.data.preview_dataset import dataset_readiness
from daikonstudio.application.execution.train_protocol import _undefined_reasons
from daikonstudio.domain.data.target import TargetKind, TargetSpec


def test_one_real_class_plus_nulls_is_not_two_classes() -> None:
    """`[1.0, 1.0, None]` is one class with a gap, not two classes. Reading it as two
    makes `_undefined_reasons` pick the wrong sentence and tell the scientist that
    nothing is wrong with a test set that holds a single class."""
    train = pl.DataFrame({"tox": [1.0, 0.0, 1.0, 0.0]})
    test = pl.DataFrame({"tox": [1.0, 1.0, None]})

    reason = _undefined_reasons({"mcc"}, "tox", train_rows=train, test_rows=test)

    assert reason is not None
    assert "all test-set compounds have the same" in reason["mcc"]


def _sparse_frame() -> pl.DataFrame:
    """Twenty training rows, four measured for `tox` and balanced two/two. The other
    sixteen were never run, which is not the same as sixteen negative results."""
    return pl.DataFrame(
        {
            "tox": [1.0, 1.0, 0.0, 0.0] + [None] * 16,
            "split": ["train"] * 20,
        }
    )


def test_unmeasured_rows_are_not_counted_as_inactive() -> None:
    readiness = dataset_readiness(
        _sparse_frame(), (TargetSpec(column="tox", kind=TargetKind.BINARY),)
    )

    train = next(b for b in readiness.class_balance if b.split == "train")
    assert (train.positive, train.negative) == (2, 2)
    assert not any("imbalanced" in warning for warning in readiness.warnings)


def test_an_unmeasured_validation_partition_still_warns_about_the_cutoff() -> None:
    """The counting fix must not silence a true warning. A target with no labelled
    validation row keeps the default cutoff of 0.5 exactly as one with eight labelled
    rows does, and the scientist is told so in both cases -- this is the shape every
    predefined-split reproduction has, where the benchmark ships train and test only."""
    frame = pl.DataFrame(
        {
            "tox": [1.0, 0.0] * 10 + [None] * 4,
            "split": ["train"] * 20 + ["validation"] * 4,
        }
    )

    readiness = dataset_readiness(frame, (TargetSpec(column="tox", kind=TargetKind.BINARY),))

    assert any("default cutoff of 0.5" in warning for warning in readiness.warnings)
