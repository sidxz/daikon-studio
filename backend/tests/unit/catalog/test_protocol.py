import uuid
from typing import Any

import pytest

from daikonstudio.application.catalog.derive_readouts import derive_readouts
from daikonstudio.domain.catalog.protocol import InSilicoProtocol, ProtocolStatus
from daikonstudio.domain.catalog.readout import ReadoutType
from daikonstudio.domain.data.target import Direction, TargetKind, TargetSpec
from daikonstudio.domain.shared.errors import ConflictError, DataLockedError


def _draft(conditions: dict[str, Any] | None = None) -> InSilicoProtocol:
    return InSilicoProtocol(
        workspace_id=uuid.uuid4(),
        name="solubility rf",
        dataset_id=uuid.uuid4(),
        engine_id="ecfp4-randomforest",
        artifact_uri="s3://x/1",
        readouts=derive_readouts(
            (
                TargetSpec(
                    column="ic50", kind=TargetKind.NUMERIC, unit="nM", direction=Direction.LOW
                ),
            )
        ),
        conditions={} if conditions is None else conditions,
    )


def test_regression_readout_inherits_unit_and_direction_from_the_target():
    target = TargetSpec(column="ic50", kind=TargetKind.NUMERIC, unit="nM", direction=Direction.LOW)
    readouts = derive_readouts((target,))
    assert len(readouts) == 1
    assert readouts[0].type == ReadoutType.NUMERIC
    assert readouts[0].unit == "nM"
    assert readouts[0].direction == Direction.LOW


def test_classification_produces_a_probability_and_a_class_readout():
    target = TargetSpec(column="active", kind=TargetKind.BINARY)
    readouts = derive_readouts((target,))
    assert [r.type for r in readouts] == [ReadoutType.PROBABILITY, ReadoutType.CLASS]
    assert readouts[0].unit is None
    assert readouts[0].direction == Direction.HIGH


def test_publishing_locks_the_protocol():
    protocol = _draft()
    assert protocol.status == ProtocolStatus.DRAFT
    assert protocol.is_locked is False
    protocol.publish()
    assert protocol.status == ProtocolStatus.PUBLISHED
    assert protocol.is_locked is True
    assert protocol.published_at is not None


def test_a_published_protocol_cannot_be_republished():
    protocol = _draft()
    protocol.publish()
    with pytest.raises(DataLockedError):
        protocol.publish()
    # ...and the failed attempt left the original publish timestamp untouched.
    published_at = protocol.published_at
    with pytest.raises(DataLockedError):
        protocol.publish()
    assert protocol.published_at == published_at


def test_a_new_version_chains_to_its_parent():
    parent = _draft()
    parent.publish()
    child = parent.new_version(artifact_uri="s3://x/2")
    assert child.parent_protocol_id == parent.id
    assert child.protocol_version == parent.protocol_version + 1
    assert child.status == ProtocolStatus.DRAFT
    assert child.is_locked is False
    assert child.id != parent.id
    # The version carries forward what training produced, not just the pointer.
    assert child.readouts == parent.readouts
    assert child.dataset_id == parent.dataset_id
    assert child.workspace_id == parent.workspace_id


def test_repeated_new_version_calls_each_chain_to_the_same_parent_independently():
    parent = _draft()
    parent.publish()
    first_child = parent.new_version(artifact_uri="s3://x/2")
    second_child = parent.new_version(artifact_uri="s3://x/3")
    assert first_child.id != second_child.id
    assert first_child.parent_protocol_id == parent.id
    assert second_child.parent_protocol_id == parent.id
    expected_version = parent.protocol_version + 1
    assert first_child.protocol_version == expected_version
    assert second_child.protocol_version == expected_version


def test_a_new_version_can_itself_be_published_and_chained_again():
    parent = _draft()
    parent.publish()
    child = parent.new_version(artifact_uri="s3://x/2")
    child.publish()
    grandchild = child.new_version(artifact_uri="s3://x/3")
    assert grandchild.parent_protocol_id == child.id
    assert grandchild.protocol_version == child.protocol_version + 1 == parent.protocol_version + 2


def test_mutating_a_childs_conditions_leaves_the_published_parent_unchanged():
    """`new_version()` hands the parent's conditions to the child; before the
    read-only view, that meant a live shared dict, and mutating the child's copy
    silently rewrote the already-published parent's. Now the write itself is
    rejected -- and either way, the parent must come out untouched."""
    parent = _draft(conditions={"n_estimators": 200})
    parent.publish()
    child = parent.new_version(artifact_uri="s3://x/2")

    with pytest.raises(TypeError):
        child.conditions["n_estimators"] = 999

    assert parent.conditions == {"n_estimators": 200}
    assert child.conditions == {"n_estimators": 200}


def test_mutating_the_dict_passed_into_the_constructor_does_not_reach_the_aggregate():
    original = {"n_estimators": 200}
    protocol = _draft(conditions=original)

    original["n_estimators"] = 999

    assert protocol.conditions == {"n_estimators": 200}


def test_conditions_cannot_be_mutated_in_place_once_published():
    """The direct attack: reach into a published, supposedly-immutable Protocol's
    `.conditions` and write through it. Must raise, not silently succeed and not
    silently vanish -- a `MappingProxyType` view over a copy-on-read property,
    because the latter would let the write appear to work."""
    protocol = _draft(conditions={"n_estimators": 200})
    protocol.publish()

    with pytest.raises(TypeError):
        protocol.conditions["n_estimators"] = 12345

    assert protocol.conditions == {"n_estimators": 200}
    assert protocol.is_locked is True


def test_new_version_on_an_unpublished_draft_is_rejected():
    draft = _draft()
    with pytest.raises(ConflictError):
        draft.new_version(artifact_uri="s3://x/2")
