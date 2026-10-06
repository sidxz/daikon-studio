import polars as pl

from daikonstudio.application.data.preview_dataset import dataset_readiness
from daikonstudio.domain.data.target import TargetKind, TargetSpec


def test_class_balance_is_computed_for_each_target_and_partition():
    frame = pl.DataFrame(
        {
            "split": ["train", "train", "validation", "test"],
            "active": [0, 1, 1, 0],
            "toxic": [1, 1, 0, 1],
        }
    )
    readiness = dataset_readiness(
        frame,
        (
            TargetSpec(column="active", kind=TargetKind.BINARY),
            TargetSpec(column="toxic", kind=TargetKind.BINARY),
        ),
    )
    assert readiness.row_count == 4
    assert readiness.partition_counts == {"train": 2, "validation": 1, "test": 1}
    counts = {
        (entry.column, entry.split): (entry.positive, entry.negative)
        for entry in readiness.class_balance
    }
    assert counts[("active", "train")] == (1, 1)
    assert counts[("active", "validation")] == (1, 0)
    assert counts[("toxic", "train")] == (2, 0)
    assert any(
        "'active'" in warning and "default cutoff" in warning for warning in readiness.warnings
    )


def test_empty_partitions_have_exact_zero_counts_and_no_fabricated_class_metrics():
    frame = pl.DataFrame({"split": ["train", "train"], "active": [0, 1]})
    readiness = dataset_readiness(frame, (TargetSpec(column="active", kind=TargetKind.BINARY),))
    assert readiness.partition_counts["test"] == 0
    test = next(entry for entry in readiness.class_balance if entry.split == "test")
    assert (test.positive, test.negative) == (0, 0)
    assert not any("only one class in the test" in warning for warning in readiness.warnings)


def test_numeric_distribution_shift_is_described_as_a_heuristic_warning():
    frame = pl.DataFrame(
        {"split": ["train", "train", "train", "test", "test"], "y": [0.0, 1.0, 2.0, 10.0, 12.0]}
    )
    readiness = dataset_readiness(frame, (TargetSpec(column="y", kind=TargetKind.NUMERIC),))
    assert readiness.class_balance == []
    assert any("may indicate a distribution shift" in warning for warning in readiness.warnings)
