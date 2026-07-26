"""Round-trip check for SqlAlchemyRunRepository against real Postgres.

Not part of the aggregate's behavioural contract (that's tests/unit/execution
and tests/integration/test_run_lifecycle.py) -- this exists because the
CheckConstraint, the enum round trip and the optimistic-concurrency `update()`
are exactly the kind of thing that looks right in review and breaks the first
time it touches a real database.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from daikonstudio.domain.execution.run import Run, RunKind, RunStatus
from daikonstudio.domain.shared.errors import ConcurrencyConflictError
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.models import RunModel
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.repository import (
    SqlAlchemyRunRepository,
)


@pytest_asyncio.fixture
async def session_factory(_migrated_engine: AsyncEngine) -> AsyncIterator[async_sessionmaker]:
    """Same recipe as tests/integration/test_protocol_repository.py's fixture
    of the same name: one connection + one outer transaction per test, rolled
    back at teardown."""
    async with _migrated_engine.connect() as connection:
        await connection.begin()
        yield async_sessionmaker(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        await connection.rollback()


def _pending(**overrides: object) -> Run:
    defaults: dict[str, object] = dict(
        kind=RunKind.TRAINING,
        workspace_id=uuid.uuid4(),
        requested_by=uuid.uuid4(),
        cache_key="deadbeef",
    )
    defaults.update(overrides)
    return Run(**defaults)  # type: ignore[arg-type]


async def test_add_then_get_round_trips_status_and_progress(session_factory):
    repository = SqlAlchemyRunRepository(session_factory)
    run = _pending()

    await repository.add(run)
    fetched = await repository.get(run.workspace_id, run.id)

    assert fetched is not None
    assert fetched.id == run.id
    assert fetched.kind is RunKind.TRAINING
    assert fetched.status is RunStatus.PENDING
    assert fetched.progress == 0.0
    assert fetched.cache_key == "deadbeef"


async def test_get_by_id_ignores_workspace_for_the_worker(session_factory):
    repository = SqlAlchemyRunRepository(session_factory)
    run = _pending()
    await repository.add(run)

    fetched = await repository.get_by_id(run.id)

    assert fetched is not None
    assert fetched.id == run.id


async def test_update_persists_lifecycle_transitions(session_factory):
    repository = SqlAlchemyRunRepository(session_factory)
    run = _pending()
    await repository.add(run)

    run.start()
    run.report_progress(0.5, phase="training baseline")
    await repository.update(run)

    fetched = await repository.get(run.workspace_id, run.id)
    assert fetched is not None
    assert fetched.status is RunStatus.RUNNING
    assert fetched.progress == 0.5
    assert fetched.phase == "training baseline"


async def test_two_stale_writers_produce_one_success_and_one_conflict(session_factory):
    """Two in-memory copies loaded before either was written back -- the second
    writer's `version` is stale the moment the first one's `update()` commits,
    and the repository must catch that rather than silently overwrite it."""
    repository = SqlAlchemyRunRepository(session_factory)
    run = _pending()
    await repository.add(run)

    first = await repository.get(run.workspace_id, run.id)
    second = await repository.get(run.workspace_id, run.id)
    assert first is not None and second is not None

    first.start()
    await repository.update(first)  # succeeds: version 1 -> 2

    second.start()
    with pytest.raises(ConcurrencyConflictError):
        await repository.update(second)  # still holds the stale version 1

    fetched = await repository.get(run.workspace_id, run.id)
    assert fetched is not None
    assert fetched.version == 2
    assert fetched.status is RunStatus.RUNNING


async def test_update_cannot_cross_a_workspace_boundary(session_factory):
    """I5 (whole-branch review, Important): `update()`'s WHERE clause used to
    match only `(id, version)`, not `workspace_id` -- the one predicate in
    this repository that was conventional rather than enforced. No live
    caller reaches this: every one loads its `run` through `get()`, itself
    workspace-scoped, before calling `update()`. Reproduced directly instead:
    a Run object whose in-memory `workspace_id` has been tampered with after
    loading (however that could happen) must not be able to mutate a row it
    no longer claims to own -- if the predicate ignores `workspace_id`, `(id,
    version)` alone still matches the real row and the write goes through.
    """
    repository = SqlAlchemyRunRepository(session_factory)
    run = _pending()
    await repository.add(run)

    stolen = await repository.get(run.workspace_id, run.id)
    assert stolen is not None
    stolen.workspace_id = uuid.uuid4()
    stolen.start()
    with pytest.raises(ConcurrencyConflictError):
        await repository.update(stolen)

    fetched = await repository.get(run.workspace_id, run.id)
    assert fetched is not None
    assert fetched.status is RunStatus.PENDING  # untouched
    assert fetched.version == 1


async def test_find_by_cache_key_scopes_to_workspace(session_factory):
    repository = SqlAlchemyRunRepository(session_factory)
    mine = _pending(cache_key="shared-key")
    theirs = _pending(cache_key="shared-key", workspace_id=uuid.uuid4())
    await repository.add(mine)
    await repository.add(theirs)

    found = await repository.find_by_cache_key(mine.workspace_id, "shared-key")

    assert found is not None
    assert found.id == mine.id


async def test_find_by_cache_key_does_not_raise_when_two_rows_share_a_key(session_factory):
    """CRITICAL fix (Task 17 review): the `(workspace_id, cache_key)` index is
    not unique, and Task 17's own caching deliberately creates a second Run
    under the same cache_key whenever the first one is FAILED or CANCELLED
    (see `PredictWithProtocol`) -- so two rows sharing a key is an expected,
    recurring state, not a corrupted one. Before `.limit(1)` was added,
    `scalar_one_or_none()` raised `MultipleResultsFound` the moment this
    happened, which escaped every caller as a raw 500 (not a `DomainError`).
    """
    repository = SqlAlchemyRunRepository(session_factory)
    workspace_id = uuid.uuid4()
    older = _pending(
        cache_key="dup",
        workspace_id=workspace_id,
        created_at=datetime.now(UTC) - timedelta(seconds=5),
    )
    newer = _pending(cache_key="dup", workspace_id=workspace_id)
    await repository.add(older)
    await repository.add(newer)

    found = await repository.find_by_cache_key(workspace_id, "dup")

    assert found is not None
    assert found.id == newer.id


async def test_list_excludes_other_workspaces(session_factory):
    repository = SqlAlchemyRunRepository(session_factory)
    mine = _pending()
    theirs = _pending(workspace_id=uuid.uuid4())
    await repository.add(mine)
    await repository.add(theirs)

    listed = await repository.list(mine.workspace_id)

    assert [r.id for r in listed] == [mine.id]


async def test_status_check_constraint_rejects_an_invalid_status(session_factory):
    """The database enforces the same five statuses the aggregate does -- a
    stray write from outside the aggregate cannot park a row in limbo."""
    async with session_factory() as session:
        session.add(
            RunModel(
                workspace_id=uuid.uuid4(),
                kind="training",
                requested_by=uuid.uuid4(),
                cache_key="x",
                status="bogus",
                progress=0.0,
            )
        )
        with pytest.raises(IntegrityError):
            await session.commit()
