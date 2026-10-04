from daikonstudio.application.catalog.derive_readouts import derive_readouts, target_columns_of
from daikonstudio.domain.data.target import Direction, TargetKind, TargetSpec

SOLUBILITY = TargetSpec(
    column="solubility", kind=TargetKind.NUMERIC, unit="logS", direction=Direction.HIGH
)
REACTIVE = TargetSpec(column="reactive", kind=TargetKind.BINARY)


def test_a_mixed_dataset_gets_a_value_beside_a_probability_and_class_pair():
    readouts = derive_readouts((SOLUBILITY, REACTIVE))
    assert [(r.name, r.type.value, r.unit) for r in readouts] == [
        ("solubility", "numeric", "logS"),
        ("reactive_probability", "probability", None),
        ("reactive", "class", None),
    ]


def test_four_binary_targets_give_eight_readouts():
    targets = tuple(TargetSpec(column=c, kind=TargetKind.BINARY) for c in "abcd")
    assert len(derive_readouts(targets)) == 8


def test_the_targets_are_recovered_from_the_readouts_in_order():
    assert target_columns_of(derive_readouts((REACTIVE, SOLUBILITY))) == ("reactive", "solubility")
