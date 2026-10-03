"""The runs table as a job queue. Raw SQL by design: SKIP LOCKED, a partial
index, and make_interval are Postgres idioms, and hiding them behind the ORM
would only obscure the one query whose exact shape is the point.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import String, bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_SWEEP_REQUEUE = text("""
    UPDATE runs
    SET status = 'pending', claimed_by = NULL, lease_expires_at = NULL,
        progress = 0.0, phase = NULL, updated_at = now()
    WHERE lease_expires_at < now() AND status IN ('pending', 'running')
""")

_SWEEP_FAIL = text("""
    UPDATE runs
    SET status = 'failed', updated_at = now(),
        error_message = 'No runner completed this run after ' || attempts
            || ' attempts; the runner stopped responding.'
    WHERE status = 'pending' AND claimed_by IS NULL AND lane IS NOT NULL
      AND attempts >= :max_attempts
""")

# ponytail: the fairness cap is a correlated count per candidate row -- fine
# while the pending queue is small; switch to a windowed CTE if it ever isn't.
#
# `lane IN :lanes` (expanding bindparam) instead of `lane = ANY(:lanes)`:
# asyncpg cannot infer the array element type of a plain `:lanes` bind
# through text() reliably, and this is the brief's noted equivalent.
#
# The cap counts 'running' rows *and* claimed-but-still-pending ones: a claim
# leaves a run 'pending' with claimed_by set until the runner reports back
# 'running', so counting only 'running' would let two concurrent pollers both
# slip under a cap of 1 during that window -- the candidate row itself can
# never self-count since candidates require claimed_by IS NULL.
_CLAIM = text("""
    WITH candidate AS (
        SELECT r.id FROM runs r
        WHERE r.status = 'pending' AND r.claimed_by IS NULL
          AND r.lane IN :lanes AND r.attempts < :max_attempts
          AND (SELECT count(*) FROM runs a
               WHERE a.workspace_id = r.workspace_id
                 AND (a.status = 'running'
                      OR (a.status = 'pending' AND a.claimed_by IS NOT NULL)))
              < :cap
        ORDER BY r.created_at
        LIMIT 1
        FOR UPDATE SKIP LOCKED
    )
    UPDATE runs
    SET claimed_by = :runner_id, attempts = attempts + 1, updated_at = now(),
        lease_expires_at = now() + make_interval(secs => :lease_seconds)
    FROM candidate WHERE runs.id = candidate.id
    RETURNING runs.id
""").bindparams(bindparam("lanes", expanding=True, type_=String))

_VERIFY = text("""
    UPDATE runs
    SET lease_expires_at = now() + make_interval(secs => :lease_seconds)
    WHERE id = :run_id AND claimed_by = :runner_id
      AND (NOT :require_active OR status IN ('pending', 'running'))
    RETURNING id
""")

# Enqueue hands the row to the queue *clean*. For a fresh run these three are
# already NULL/0; for a retried one they are the stale claim, live lease and
# used-up attempts of the failed execution -- left in place, the sweep would
# fail the run again on the next poll ("lease expired after 3 attempts") or no
# runner could claim it until the old lease ran out. Clearing claimed_by also
# fences a runner still computing the abandoned attempt: its next write 403s.
_SET_LANE = text("""
    UPDATE runs
    SET lane = :lane, claimed_by = NULL, lease_expires_at = NULL, attempts = 0,
        updated_at = now()
    WHERE id = :run_id
""")

_ACTIVE_RUN_BY_RUNNER = text("""
    SELECT DISTINCT ON (claimed_by) claimed_by, id FROM runs
    WHERE claimed_by IS NOT NULL AND status IN ('pending', 'running')
    ORDER BY claimed_by, created_at DESC
""")


class SqlAlchemyRunQueue:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def sweep(self, *, max_attempts: int) -> None:
        async with self._sessions() as session:
            await session.execute(_SWEEP_REQUEUE)
            await session.execute(_SWEEP_FAIL, {"max_attempts": max_attempts})
            await session.commit()

    async def claim_next(
        self,
        *,
        runner_id: uuid.UUID,
        lanes: Sequence[str],
        lease_seconds: int,
        max_active_per_workspace: int,
        max_attempts: int,
    ) -> uuid.UUID | None:
        """Sweeps first, then claims -- in the same transaction, so a runner
        that just missed its lease is eligible again before anyone else tries
        to claim past it."""
        async with self._sessions() as session:
            await session.execute(_SWEEP_REQUEUE)
            await session.execute(_SWEEP_FAIL, {"max_attempts": max_attempts})
            result = await session.execute(
                _CLAIM,
                {
                    "lanes": list(lanes),
                    "max_attempts": max_attempts,
                    "cap": max_active_per_workspace,
                    "runner_id": runner_id,
                    "lease_seconds": lease_seconds,
                },
            )
            row = result.first()
            await session.commit()
            return row.id if row is not None else None

    async def verify_claim(
        self,
        run_id: uuid.UUID,
        *,
        runner_id: uuid.UUID,
        lease_seconds: int,
        require_active: bool,
    ) -> bool:
        async with self._sessions() as session:
            result = await session.execute(
                _VERIFY,
                {
                    "run_id": run_id,
                    "runner_id": runner_id,
                    "lease_seconds": lease_seconds,
                    "require_active": require_active,
                },
            )
            row = result.first()
            await session.commit()
            return row is not None

    async def set_lane(self, run_id: uuid.UUID, lane: str) -> None:
        async with self._sessions() as session:
            await session.execute(_SET_LANE, {"run_id": run_id, "lane": lane})
            await session.commit()

    async def active_run_by_runner(self) -> dict[uuid.UUID, uuid.UUID]:
        async with self._sessions() as session:
            result = await session.execute(_ACTIVE_RUN_BY_RUNNER)
            return {row.claimed_by: row.id for row in result}
