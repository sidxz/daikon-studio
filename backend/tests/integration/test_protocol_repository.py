"""Round-trip check for SqlAlchemyProtocolRepository against real Postgres.

Not part of the aggregate's behavioural contract (that's tests/unit/catalog) --
this exists because the repository's JSONB (de)serialisation, the status enum
round trip and the self-referencing `parent_protocol_id` FK are exactly the
kind of thing that looks right in a code review and breaks the first time it
touches a real database.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from daikonstudio.application.catalog.derive_readouts import derive_readouts
from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.domain.catalog.protocol import InSilicoProtocol, ProtocolStatus
from daikonstudio.domain.data.target import Direction, TargetKind, TargetSpec
from daikonstudio.domain.shared.errors import ConcurrencyConflictError
from daikonstudio.infrastructure.persistence.sqlalchemy.catalog.models import (
    InSilicoProtocolModel,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.catalog.repository import (
    SqlAlchemyProtocolRepository,
)


@pytest_asyncio.fixture
async def session_factory(_migrated_engine: AsyncEngine) -> AsyncIterator[async_sessionmaker]:
    """Same recipe as tests/api/conftest.py's fixture of the same name: one
    connection + one outer transaction per test, rolled back at teardown."""
    async with _migrated_engine.connect() as connection:
        await connection.begin()
        yield async_sessionmaker(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        await connection.rollback()


def _draft(**overrides: object) -> InSilicoProtocol:
    defaults: dict[str, object] = dict(
        workspace_id=uuid.uuid4(),
        name="solubility rf",
        dataset_id=uuid.uuid4(),
        engine_id="ecfp4-randomforest",
        artifact_uri="s3://bucket/artifact.joblib",
        readouts=derive_readouts(
            TargetSpec(column="ic50", kind=TargetKind.NUMERIC, unit="nM", direction=Direction.LOW),
            TaskType.REGRESSION,
        ),
        conditions={"n_estimators": 200},
    )
    defaults.update(overrides)
    return InSilicoProtocol(**defaults)  # type: ignore[arg-type]


async def test_add_then_get_round_trips_readouts_conditions_and_status(session_factory):
    """`protocol.conditions` is a read-only `MappingProxyType`, not a plain dict --
    this also checks that the repository unwraps it to something JSONB can actually
    store, and that what comes back out of the database is a plain `dict` again, not
    a proxy, a JSON string, or anything else the domain layer wouldn't accept."""
    repository = SqlAlchemyProtocolRepository(session_factory)
    protocol = _draft()

    await repository.add(protocol)
    fetched = await repository.get(protocol.workspace_id, protocol.id)

    assert fetched is not None
    assert fetched.id == protocol.id
    assert fetched.status == ProtocolStatus.DRAFT
    assert fetched.readouts == protocol.readouts
    assert fetched.conditions == protocol.conditions
    assert fetched.artifact_uri == protocol.artifact_uri

    async with session_factory() as session:
        row = (
            await session.execute(
                select(InSilicoProtocolModel).where(InSilicoProtocolModel.id == protocol.id)
            )
        ).scalar_one()
    assert type(row.conditions) is dict
    assert row.conditions == {"n_estimators": 200}


async def test_publish_persists_through_update(session_factory):
    repository = SqlAlchemyProtocolRepository(session_factory)
    protocol = _draft()
    await repository.add(protocol)

    protocol.publish()
    await repository.update(protocol)

    fetched = await repository.get(protocol.workspace_id, protocol.id)
    assert fetched is not None
    assert fetched.status == ProtocolStatus.PUBLISHED
    assert fetched.is_locked is True
    assert fetched.published_at is not None


async def test_a_new_versions_parent_id_round_trips(session_factory):
    repository = SqlAlchemyProtocolRepository(session_factory)
    parent = _draft()
    parent.publish()
    await repository.add(parent)

    child = parent.new_version(artifact_uri="s3://bucket/artifact-v2.joblib")
    await repository.add(child)

    fetched_child = await repository.get(child.workspace_id, child.id)
    assert fetched_child is not None
    assert fetched_child.parent_protocol_id == parent.id
    assert fetched_child.protocol_version == parent.protocol_version + 1


async def test_two_stale_writers_produce_one_success_and_one_conflict(session_factory):
    """Two in-memory copies loaded before either was written back -- the second
    writer's `version` is stale the moment the first one's `update()` commits, and
    the repository must catch that rather than silently overwrite the first write."""
    repository = SqlAlchemyProtocolRepository(session_factory)
    protocol = _draft()
    await repository.add(protocol)

    first = await repository.get(protocol.workspace_id, protocol.id)
    second = await repository.get(protocol.workspace_id, protocol.id)
    assert first is not None and second is not None

    first.publish()
    await repository.update(first)  # succeeds: version 1 -> 2

    second.publish()
    with pytest.raises(ConcurrencyConflictError):
        await repository.update(second)  # still holds the stale version 1

    fetched = await repository.get(protocol.workspace_id, protocol.id)
    assert fetched is not None
    assert fetched.version == 2
    assert fetched.status == ProtocolStatus.PUBLISHED


async def test_update_cannot_cross_a_workspace_boundary(session_factory):
    """I5 (whole-branch review, Important): `update()`'s WHERE clause used to
    match only `(id, version)`, not `workspace_id`. No live caller reaches
    this -- every one loads its `protocol` through `get()`, itself
    workspace-scoped -- but it was the last predicate in this repository
    that was conventional rather than enforced. Reproduced directly: a
    Protocol object whose in-memory `workspace_id` has been tampered with
    after loading must not be able to mutate a row it no longer claims to
    own.
    """
    repository = SqlAlchemyProtocolRepository(session_factory)
    protocol = _draft()
    await repository.add(protocol)

    stolen = await repository.get(protocol.workspace_id, protocol.id)
    assert stolen is not None
    stolen.workspace_id = uuid.uuid4()
    stolen.publish()
    with pytest.raises(ConcurrencyConflictError):
        await repository.update(stolen)

    fetched = await repository.get(protocol.workspace_id, protocol.id)
    assert fetched is not None
    assert fetched.status == ProtocolStatus.DRAFT  # untouched
    assert fetched.version == 1


async def test_list_excludes_other_workspaces(session_factory):
    repository = SqlAlchemyProtocolRepository(session_factory)
    mine = _draft()
    theirs = _draft(workspace_id=uuid.uuid4())
    await repository.add(mine)
    await repository.add(theirs)

    listed = await repository.list(mine.workspace_id)

    assert [p.id for p in listed] == [mine.id]
