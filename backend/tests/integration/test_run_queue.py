"""Integration tests for SqlAlchemyRunQueue against real Postgres.

The runs table doubles as a job queue -- SKIP LOCKED claiming, lease expiry,
and the per-workspace fairness cap are exactly the kind of concurrency logic
that looks right on inspection and breaks the first time two runners race for
the same row. Every test hits a real Postgres, never a mock: SKIP LOCKED,
make_interval, and the partial index are Postgres idioms with no in-memory
equivalent worth trusting.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from daikonstudio.infrastructure.persistence.sqlalchemy.execution.models import RunModel
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.queue import SqlAlchemyRunQueue


@pytest_asyncio.fixture
async def session_factory(_migrated_engine: AsyncEngine) -> AsyncIterator[async_sessionmaker]:
    """Same recipe as tests/integration/test_run_repository.py's fixture of the
    same name: one connection + one outer transaction per test, rolled back at
    teardown."""
    async with _migrated_engine.connect() as connection:
        await connection.begin()
        yield async_sessionmaker(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        await connection.rollback()


@pytest.fixture
def queue(session_factory: async_sessionmaker) -> SqlAlchemyRunQueue:
    return SqlAlchemyRunQueue(session_factory)


async def _pending_run(
    session_factory: async_sessionmaker,
    *,
    lane: str | None = "default",
    workspace_id: uuid.UUID | None = None,
    attempts: int = 0,
    status: str = "pending",
    claimed_by: uuid.UUID | None = None,
    lease_expires_at: datetime | None = None,
    created_at: datetime | None = None,
    progress: float = 0.0,
    phase: str | None = None,
) -> uuid.UUID:
    """Inserts a minimal valid RunModel directly -- the queue operates on raw
    rows, not the Run aggregate -- and returns its id."""
    model = RunModel(
        kind="training",
        requested_by=uuid.uuid4(),
        cache_key=uuid.uuid4().hex,
        params={},
        status=status,
        progress=progress,
        phase=phase,
        workspace_id=workspace_id if workspace_id is not None else uuid.uuid4(),
        version=1,
        lane=lane,
        attempts=attempts,
        claimed_by=claimed_by,
        lease_expires_at=lease_expires_at,
    )
    if created_at is not None:
        model.created_at = created_at
    async with session_factory() as session:
        session.add(model)
        await session.commit()
    return model.id


async def _fetch(session_factory: async_sessionmaker, run_id: uuid.UUID) -> RunModel:
    async with session_factory() as session:
        model = (await session.execute(select(RunModel).where(RunModel.id == run_id))).scalar_one()
        session.expunge(model)
        return model


_CLAIM_DEFAULTS: dict[str, object] = dict(
    lanes=["default"], lease_seconds=60, max_active_per_workspace=10, max_attempts=3
)


async def test_claim_returns_oldest_matching_lane(queue, session_factory):
    older = await _pending_run(
        session_factory, created_at=datetime.now(UTC) - timedelta(seconds=5)
    )
    await _pending_run(session_factory)
    await _pending_run(session_factory, lane="gpu")

    claimed = await queue.claim_next(runner_id=uuid.uuid4(), **_CLAIM_DEFAULTS)

    assert claimed == older


async def test_claim_skips_other_lanes(queue, session_factory):
    await _pending_run(session_factory, lane="gpu")

    claimed = await queue.claim_next(runner_id=uuid.uuid4(), **_CLAIM_DEFAULTS)

    assert claimed is None


async def test_claim_with_empty_lanes_returns_none(queue, session_factory):
    """A runner registered with no lanes must get a clean `None`, not a
    Postgres type error -- `lane IN :lanes` against an empty expanding
    bindparam only degrades gracefully with an explicit `type_` (asyncpg
    otherwise guesses `integer` for the empty-list placeholder and
    `character varying = integer` is a hard DB error, not an empty match)."""
    await _pending_run(session_factory)

    claimed = await queue.claim_next(runner_id=uuid.uuid4(), **{**_CLAIM_DEFAULTS, "lanes": []})

    assert claimed is None


async def test_claim_sets_claimant_lease_and_attempts(queue, session_factory):
    run_id = await _pending_run(session_factory)
    runner_id = uuid.uuid4()

    claimed = await queue.claim_next(
        runner_id=runner_id, **{**_CLAIM_DEFAULTS, "lease_seconds": 120}
    )
    assert claimed == run_id

    fetched = await _fetch(session_factory, run_id)
    assert fetched.claimed_by == runner_id
    assert fetched.attempts == 1
    assert fetched.lease_expires_at is not None
    expected = datetime.now(UTC) + timedelta(seconds=120)
    assert abs((fetched.lease_expires_at - expected).total_seconds()) < 5


async def test_claimed_run_is_not_claimable_again(queue, session_factory):
    await _pending_run(session_factory)

    first = await queue.claim_next(runner_id=uuid.uuid4(), **_CLAIM_DEFAULTS)
    assert first is not None

    second = await queue.claim_next(runner_id=uuid.uuid4(), **_CLAIM_DEFAULTS)
    assert second is None


async def test_two_concurrent_claims_get_different_runs(_migrated_engine: AsyncEngine):
    """SKIP LOCKED only shows its effect across two genuinely separate DB
    transactions racing each other; the shared savepoint-bound
    `session_factory` fixture funnels every test through one connection,
    which would serialize the two claims and hide the very race this test
    exists to catch. So this test opens two independent sessionmakers
    straight on the engine (real, committed rows, real parallel
    transactions) and cleans up after itself in a `finally` instead of
    relying on the fixture's rollback.
    """
    sessions_a = async_sessionmaker(bind=_migrated_engine, expire_on_commit=False)
    sessions_b = async_sessionmaker(bind=_migrated_engine, expire_on_commit=False)
    queue_a = SqlAlchemyRunQueue(sessions_a)
    queue_b = SqlAlchemyRunQueue(sessions_b)

    workspace_id = uuid.uuid4()
    run_1 = RunModel(
        kind="training",
        requested_by=uuid.uuid4(),
        cache_key=uuid.uuid4().hex,
        params={},
        status="pending",
        progress=0.0,
        workspace_id=workspace_id,
        version=1,
        lane="default",
    )
    run_2 = RunModel(
        kind="training",
        requested_by=uuid.uuid4(),
        cache_key=uuid.uuid4().hex,
        params={},
        status="pending",
        progress=0.0,
        workspace_id=workspace_id,
        version=1,
        lane="default",
    )
    async with sessions_a() as session:
        session.add_all([run_1, run_2])
        await session.commit()
    run_ids = {run_1.id, run_2.id}

    try:
        results = await asyncio.gather(
            queue_a.claim_next(runner_id=uuid.uuid4(), **_CLAIM_DEFAULTS),
            queue_b.claim_next(runner_id=uuid.uuid4(), **_CLAIM_DEFAULTS),
        )
        assert None not in results
        assert results[0] != results[1]
        assert set(results) == run_ids
    finally:
        async with sessions_a() as session:
            await session.execute(sa_delete(RunModel).where(RunModel.id.in_(run_ids)))
            await session.commit()


async def test_sweep_requeues_expired_lease(queue, session_factory):
    run_id = await _pending_run(
        session_factory,
        status="running",
        claimed_by=uuid.uuid4(),
        lease_expires_at=datetime.now(UTC) - timedelta(minutes=5),
        progress=0.7,
        phase="training baseline",
    )

    await queue.sweep(max_attempts=3)

    fetched = await _fetch(session_factory, run_id)
    assert fetched.status == "pending"
    assert fetched.claimed_by is None
    assert fetched.lease_expires_at is None
    assert fetched.progress == 0.0
    assert fetched.phase is None


async def test_sweep_fails_run_after_max_attempts(queue, session_factory):
    run_id = await _pending_run(session_factory, attempts=3)

    await queue.sweep(max_attempts=3)

    fetched = await _fetch(session_factory, run_id)
    assert fetched.status == "failed"
    assert fetched.error_message is not None
    assert "lease" in fetched.error_message


async def test_workspace_cap_blocks_claim(queue, session_factory):
    workspace_id = uuid.uuid4()
    await _pending_run(session_factory, workspace_id=workspace_id, status="running")
    await _pending_run(session_factory, workspace_id=workspace_id)
    other_workspace_run = await _pending_run(session_factory)

    claimed = await queue.claim_next(
        runner_id=uuid.uuid4(), **{**_CLAIM_DEFAULTS, "max_active_per_workspace": 1}
    )

    assert claimed == other_workspace_run


async def test_workspace_cap_counts_claimed_but_still_pending_runs(queue, session_factory):
    """A claim leaves its run `pending` with `claimed_by` set until the
    runner reports back `running` -- during that window it must still
    consume the workspace's fairness slot, or two concurrent pollers could
    both slip a cap of 1."""
    workspace_id = uuid.uuid4()
    await _pending_run(session_factory, workspace_id=workspace_id, claimed_by=uuid.uuid4())
    capped_pending = await _pending_run(session_factory, workspace_id=workspace_id)
    other_workspace_run = await _pending_run(session_factory)

    claimed = await queue.claim_next(
        runner_id=uuid.uuid4(), **{**_CLAIM_DEFAULTS, "max_active_per_workspace": 1}
    )

    assert claimed == other_workspace_run
    assert claimed != capped_pending


async def test_lane_null_is_never_claimable(queue, session_factory):
    await _pending_run(session_factory, lane=None)

    claimed = await queue.claim_next(runner_id=uuid.uuid4(), **_CLAIM_DEFAULTS)

    assert claimed is None


async def test_verify_claim_true_for_claimant_and_extends_lease(queue, session_factory):
    runner_id = uuid.uuid4()
    run_id = await _pending_run(
        session_factory,
        claimed_by=runner_id,
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=10),
    )

    ok = await queue.verify_claim(
        run_id, runner_id=runner_id, lease_seconds=120, require_active=True
    )
    assert ok is True

    fetched = await _fetch(session_factory, run_id)
    assert fetched.lease_expires_at is not None
    expected = datetime.now(UTC) + timedelta(seconds=120)
    assert abs((fetched.lease_expires_at - expected).total_seconds()) < 5


async def test_verify_claim_false_for_other_runner(queue, session_factory):
    run_id = await _pending_run(session_factory, claimed_by=uuid.uuid4())

    ok = await queue.verify_claim(
        run_id, runner_id=uuid.uuid4(), lease_seconds=60, require_active=True
    )

    assert ok is False


async def test_verify_claim_require_active_false_on_terminal(queue, session_factory):
    runner_id = uuid.uuid4()
    run_id = await _pending_run(session_factory, status="ready", claimed_by=runner_id)

    ok = await queue.verify_claim(
        run_id, runner_id=runner_id, lease_seconds=60, require_active=False
    )

    assert ok is True


async def test_verify_claim_require_active_true_allows_running_and_rejects_terminal(
    queue, session_factory
):
    runner_id = uuid.uuid4()
    running_id = await _pending_run(session_factory, status="running", claimed_by=runner_id)
    terminal_id = await _pending_run(session_factory, status="ready", claimed_by=runner_id)

    assert (
        await queue.verify_claim(
            running_id, runner_id=runner_id, lease_seconds=60, require_active=True
        )
        is True
    )
    assert (
        await queue.verify_claim(
            terminal_id, runner_id=runner_id, lease_seconds=60, require_active=True
        )
        is False
    )


async def test_set_lane_makes_run_claimable(queue, session_factory):
    run_id = await _pending_run(session_factory, lane=None)
    assert await queue.claim_next(runner_id=uuid.uuid4(), **_CLAIM_DEFAULTS) is None

    await queue.set_lane(run_id, "default")

    claimed = await queue.claim_next(runner_id=uuid.uuid4(), **_CLAIM_DEFAULTS)
    assert claimed == run_id


async def test_cancelled_pending_run_is_not_claimed(queue, session_factory):
    await _pending_run(session_factory, status="cancelled")

    claimed = await queue.claim_next(runner_id=uuid.uuid4(), **_CLAIM_DEFAULTS)

    assert claimed is None


async def test_set_lane_hands_a_retried_run_back_to_the_queue_clean(queue, session_factory):
    """A run that failed on a runner keeps that runner's claim, a live lease and
    its used-up attempts on the row. Retry re-enqueues through `set_lane`, which
    must clear all three -- otherwise the sweep fails it again on the next poll
    ("lease expired after 3 attempts") and nothing ever claims it."""
    run_id = await _pending_run(
        session_factory,
        attempts=3,
        claimed_by=uuid.uuid4(),
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )

    await queue.set_lane(run_id, "default")

    fetched = await _fetch(session_factory, run_id)
    assert (fetched.claimed_by, fetched.lease_expires_at, fetched.attempts) == (None, None, 0)
    await queue.sweep(max_attempts=3)
    assert (await _fetch(session_factory, run_id)).status == "pending"
    assert await queue.claim_next(runner_id=uuid.uuid4(), **_CLAIM_DEFAULTS) == run_id
