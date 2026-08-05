"""Round-trip check for SqlAlchemyRunnerRepository against real Postgres.

Not part of the aggregate's behavioural contract (that's tests/unit/runners)
-- this exists because the JSONB lanes round trip, the unique constraints and
the targeted `touch_last_seen`/`revoke` updates are exactly the kind of thing
that looks right in review and breaks the first time it touches a real
database.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from daikonstudio.domain.runners.runner import Runner
from daikonstudio.domain.shared.errors import ConflictError
from daikonstudio.infrastructure.persistence.sqlalchemy.runners.repository import (
    SqlAlchemyRunnerRepository,
)


@pytest_asyncio.fixture
async def session_factory(_migrated_engine: AsyncEngine) -> AsyncIterator[async_sessionmaker]:
    """Same recipe as tests/integration/test_run_repository.py's fixture of
    the same name: one connection + one outer transaction per test, rolled
    back at teardown."""
    async with _migrated_engine.connect() as connection:
        await connection.begin()
        yield async_sessionmaker(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        await connection.rollback()


def _runner(**overrides: object) -> Runner:
    defaults: dict[str, object] = dict(
        name=f"gpu-{uuid.uuid4()}",
        lanes=("gpu",),
        token_hash=uuid.uuid4().hex + uuid.uuid4().hex,
    )
    defaults.update(overrides)
    return Runner(**defaults)  # type: ignore[arg-type]


async def test_add_then_get_round_trips_all_fields(session_factory):
    repository = SqlAlchemyRunnerRepository(session_factory)
    runner = _runner(lanes=("default", "gpu"))

    await repository.add(runner)
    fetched = await repository.get(runner.id)

    assert fetched is not None
    assert fetched.id == runner.id
    assert fetched.name == runner.name
    assert fetched.lanes == ("default", "gpu")
    assert fetched.token_hash == runner.token_hash
    assert fetched.is_revoked is False


async def test_get_returns_none_for_unknown_id(session_factory):
    repository = SqlAlchemyRunnerRepository(session_factory)

    assert await repository.get(uuid.uuid4()) is None


async def test_get_by_token_hash_finds_it_and_none_for_unknown(session_factory):
    repository = SqlAlchemyRunnerRepository(session_factory)
    runner = _runner()
    await repository.add(runner)

    found = await repository.get_by_token_hash(runner.token_hash)
    assert found is not None
    assert found.id == runner.id

    assert await repository.get_by_token_hash("b" * 64) is None


async def test_list_returns_all(session_factory):
    repository = SqlAlchemyRunnerRepository(session_factory)
    first = _runner()
    second = _runner()
    await repository.add(first)
    await repository.add(second)

    listed = await repository.list()

    assert {r.id for r in listed} == {first.id, second.id}


async def test_touch_last_seen_sets_a_timestamp_visible_on_reget(session_factory):
    repository = SqlAlchemyRunnerRepository(session_factory)
    runner = _runner()
    await repository.add(runner)
    assert runner.last_seen_at is None

    await repository.touch_last_seen(runner.id)

    fetched = await repository.get(runner.id)
    assert fetched is not None
    assert fetched.last_seen_at is not None


async def test_revoke_sets_revoked_at_and_is_idempotent(session_factory):
    repository = SqlAlchemyRunnerRepository(session_factory)
    runner = _runner()
    await repository.add(runner)

    await repository.revoke(runner.id)
    first = await repository.get(runner.id)
    assert first is not None and first.revoked_at is not None

    await repository.revoke(runner.id)
    second = await repository.get(runner.id)
    assert second is not None
    assert second.revoked_at == first.revoked_at


async def test_add_duplicate_name_raises_conflict(session_factory):
    repository = SqlAlchemyRunnerRepository(session_factory)
    await repository.add(_runner(name="dup"))

    with pytest.raises(ConflictError):
        await repository.add(_runner(name="dup"))
