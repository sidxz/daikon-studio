import pytest

from daikonstudio.domain.data.target import (
    TargetKind,
    TargetSpec,
    check_targets,
    prediction_columns,
)
from daikonstudio.domain.shared.errors import ValidationError


def numeric(column: str) -> TargetSpec:
    return TargetSpec(column=column, kind=TargetKind.NUMERIC)


def binary(column: str) -> TargetSpec:
    return TargetSpec(column=column, kind=TargetKind.BINARY)


def test_several_distinct_targets_of_mixed_kinds_are_accepted():
    check_targets([numeric("solubility"), binary("reactive")])


def test_no_target_is_refused():
    with pytest.raises(ValidationError, match="at least one"):
        check_targets([])


def test_a_target_chosen_twice_is_refused():
    with pytest.raises(ValidationError, match="'y' is chosen more than once"):
        check_targets([numeric("y"), numeric("y")])


@pytest.mark.parametrize("column", ["target", "uncertainty", "row_id", "structure"])
def test_a_reserved_column_name_is_refused(column):
    with pytest.raises(ValidationError, match="cannot be used as a target column"):
        check_targets([numeric(column)])


def test_a_binary_target_beside_its_own_probability_column_is_refused():
    with pytest.raises(ValidationError, match="foo_probability"):
        check_targets([binary("foo"), numeric("foo_probability")])


def test_with_several_targets_an_uncertainty_column_can_clash_too():
    with pytest.raises(ValidationError, match="foo_uncertainty"):
        check_targets([numeric("foo"), numeric("foo_uncertainty")])


def test_one_target_writes_no_per_target_uncertainty_column():
    assert prediction_columns([binary("y")]) == ["y_probability", "y"]
    assert prediction_columns([binary("a"), numeric("b")]) == [
        "a_probability",
        "a",
        "a_uncertainty",
        "b",
        "b_uncertainty",
    ]
