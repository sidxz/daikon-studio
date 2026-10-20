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


async def test_record_build_round_trips_the_three_values(session_factory):
    repository = SqlAlchemyRunnerRepository(session_factory)
    runner = _runner()
    await repository.add(runner)
    assert runner.build_version is None

    await repository.record_build(
        runner.id, version="0.8.0", git_sha="deadbee", build_date="2026-10-20T09:00:00Z"
    )

    fetched = await repository.get(runner.id)
    assert fetched is not None
    assert fetched.build_version == "0.8.0"
    assert fetched.build_git_sha == "deadbee"
    assert fetched.build_date == "2026-10-20T09:00:00Z"


async def test_record_build_replaces_an_earlier_report(session_factory):
    repository = SqlAlchemyRunnerRepository(session_factory)
    runner = _runner()
    await repository.add(runner)

    await repository.record_build(runner.id, version="0.7.0", git_sha="old", build_date="then")
    await repository.record_build(runner.id, version="0.8.0", git_sha="new", build_date="now")

    fetched = await repository.get(runner.id)
    assert fetched is not None
    assert (fetched.build_version, fetched.build_git_sha) == ("0.8.0", "new")


async def test_a_runner_that_never_reported_a_build_reads_back_as_none(session_factory):
    """The nullable columns are the whole point: an older agent reports
    nothing, and the row must still load and still be usable."""
    repository = SqlAlchemyRunnerRepository(session_factory)
    runner = _runner()
    await repository.add(runner)

    fetched = await repository.get(runner.id)
    assert fetched is not None
    assert (fetched.build_version, fetched.build_git_sha, fetched.build_date) == (
        None,
        None,
        None,
    )
    assert fetched.is_revoked is False


async def test_delete_removes_the_row(session_factory):
    repository = SqlAlchemyRunnerRepository(session_factory)
    runner = _runner()
    await repository.add(runner)

    await repository.delete(runner.id)

    assert await repository.get(runner.id) is None
    assert await repository.get_by_token_hash(runner.token_hash) is None


async def test_delete_of_an_unknown_id_is_a_no_op(session_factory):
    repository = SqlAlchemyRunnerRepository(session_factory)
    runner = _runner()
    await repository.add(runner)

    await repository.delete(uuid.uuid4())

    assert await repository.get(runner.id) is not None


async def test_the_name_of_a_deleted_runner_can_be_registered_again(session_factory):
    """`uq_runners_name` is why delete exists at all: a runner registered under
    a typo'd name holds that name until the row goes."""
    repository = SqlAlchemyRunnerRepository(session_factory)
    first = _runner(name="gpu-typo")
    await repository.add(first)

    await repository.delete(first.id)
    await repository.add(_runner(name="gpu-typo"))
