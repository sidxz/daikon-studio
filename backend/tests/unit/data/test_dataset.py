import uuid

import pytest

from daikonstudio.domain.data.dataset import Dataset, check_id_column
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.data.target import TargetKind, TargetSpec
from daikonstudio.domain.data.validation import ValidationReport
from daikonstudio.domain.shared.errors import ValidationError


def _dataset(*columns: str) -> Dataset:
    return Dataset(
        workspace_id=uuid.uuid4(),
        name="d",
        structure_column="smiles",
        targets=tuple(TargetSpec(column=c, kind=TargetKind.BINARY) for c in columns),
        split=SplitSpec(strategy=SplitStrategy.RANDOM, seed=1),
        content_hash="h",
        snapshot_uri="x",
        row_count=1,
        validation_report=ValidationReport(total_rows=1, valid_rows=1),
    )


def test_target_columns_keep_the_order_chosen():
    assert _dataset("b", "a").target_columns == ("b", "a")


def test_single_target_refuses_a_dataset_with_several():
    assert _dataset("y").single_target().column == "y"
    with pytest.raises(ValidationError):
        _dataset("a", "b").single_target()


def test_an_identifier_may_not_be_any_of_the_targets():
    with pytest.raises(ValidationError):
        check_id_column(
            ["smiles", "a", "b", "id"],
            id_column="b",
            structure_column="smiles",
            target_columns=("a", "b"),
        )
