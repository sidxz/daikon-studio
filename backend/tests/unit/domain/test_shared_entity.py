import uuid

from daikonstudio.domain.shared.entity import AggregateRoot, Entity
from daikonstudio.domain.shared.errors import ConcurrencyConflictError, NotFoundError
from daikonstudio.domain.shared.global_workspace import GLOBAL_WORKSPACE_ID


def test_entities_are_equal_by_id():
    shared_id = uuid.uuid4()
    assert Entity(id=shared_id) == Entity(id=shared_id)
    assert Entity() != Entity()


def test_aggregate_collects_and_clears_events():
    aggregate = AggregateRoot()
    assert aggregate.version == 1
    aggregate.register_event(object())
    assert len(aggregate.collect_events()) == 1
    aggregate.clear_events()
    assert aggregate.collect_events() == []


def test_not_found_error_names_the_entity():
    error = NotFoundError("Dataset", "abc")
    assert error.message == "Dataset 'abc' not found"


def test_concurrency_error_is_a_domain_error():
    error = ConcurrencyConflictError("Protocol", "abc")
    assert "modified by another transaction" in error.message


def test_global_workspace_is_the_nil_uuid():
    assert uuid.UUID(int=0) == GLOBAL_WORKSPACE_ID
