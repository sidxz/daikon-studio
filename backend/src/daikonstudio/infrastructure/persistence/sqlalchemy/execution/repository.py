"""SQLAlchemy Run repository.

Every workspace-scoped read filters on `workspace_id` inside the SQL, never
after it -- a cross-tenant row is not fetched and then discarded, it is never
selected. `get_by_id` is the one deliberate exception: a claiming runner is
handed a bare `run_id` by the claim endpoint with no tenant context of its own,
and the Run was already scoped to its workspace when the use case that enqueued it
created the row. The runner's job is to execute it, not re-authorize it.

Like `SqlAlchemyProtocolRepository`, `update()` does optimistic concurrency
with `UPDATE ... WHERE version = :expected` and raises `ConcurrencyConflictError`
on zero rows affected -- never `session.merge()`, which would silently let a
stale in-memory copy overwrite a concurrent write instead of raising.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CursorResult, Select, select, tuple_
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from daikonstudio.domain.execution.run import Run, RunKind, RunStatus
from daikonstudio.domain.shared.errors import ConcurrencyConflictError
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.models import RunModel


def _to_domain(model: RunModel) -> Run:
    return Run(
        id=model.id,
        workspace_id=model.workspace_id,
        kind=RunKind(model.kind),
        requested_by=model.requested_by,
        cache_key=model.cache_key,
        params=model.params,
        protocol_id=model.protocol_id,
        status=RunStatus(model.status),
        progress=model.progress,
        phase=model.phase,
        result_uri=model.result_uri,
        error_message=model.error_message,
        created_at=model.created_at,
        updated_at=model.updated_at,
        version=model.version,
    )


def _to_model(run: Run) -> RunModel:
    return RunModel(
        id=run.id,
        workspace_id=run.workspace_id,
        kind=run.kind.value,
        requested_by=run.requested_by,
        cache_key=run.cache_key,
        params=run.params,
        protocol_id=run.protocol_id,
        status=run.status.value,
        progress=run.progress,
        phase=run.phase,
        result_uri=run.result_uri,
        error_message=run.error_message,
        version=run.version,
        # Explicit, rather than the column's server_default: otherwise the
        # created_at in a 201 response body (the aggregate's own clock) and the
        # one a later GET returns (the database's) would differ by however long
        # the insert took.
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


class SqlAlchemyRunRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(self, run: Run) -> None:
        async with self._sessions() as session:
            session.add(_to_model(run))
            await session.commit()

    async def update(self, run: Run) -> None:
        """Optimistic concurrency: the UPDATE only lands if the row's `version`
        still matches what this in-memory copy was loaded with, and bumps it by
        one when it does. Zero rows affected means someone else's write (e.g. a
        worker's progress update racing a cancel request) already moved the row
        on -- raise rather than silently last-writer-wins overwriting it.

        `workspace_id` is also part of the predicate (whole-branch review,
        Important 5) -- every real caller already loads its `run` through
        `get()` or `get_by_id()` (the worker's own, deliberately unscoped
        read of a row already scoped when its owning use case created it --
        see this module's docstring), so this cannot fire today. It stays
        because it is the one invariant in this module that was conventional
        rather than enforced, at the cost of one clause.
        """
        model = _to_model(run)
        expected_version = run.version
        async with self._sessions() as session:
            result = await session.execute(
                sa_update(RunModel)
                .where(
                    RunModel.id == run.id,
                    RunModel.workspace_id == run.workspace_id,
                    RunModel.version == expected_version,
                )
                .values(
                    # Persisted by `update` because it is an *outcome*, unlike
                    # `params` which stays write-once -- see `Run.link_protocol`.
                    protocol_id=model.protocol_id,
                    status=model.status,
                    progress=model.progress,
                    phase=model.phase,
                    result_uri=model.result_uri,
                    error_message=model.error_message,
                    updated_at=model.updated_at,
                    version=expected_version + 1,
                )
            )
            # `Session.execute()` is typed to return the generic `Result[Any]`, but an
            # UPDATE statement always yields the richer `CursorResult` that actually
            # carries `rowcount` -- this assertion is the real runtime shape, not a
            # type-checker workaround.
            assert isinstance(result, CursorResult)
            if result.rowcount == 0:
                await session.rollback()
                raise ConcurrencyConflictError("Run", str(run.id))
            await session.commit()
        # The caller's in-memory copy now matches what was just persisted -- without
        # this, a second `update()` on the same object would immediately self-conflict.
        run.version = expected_version + 1

    async def get(self, workspace_id: uuid.UUID, run_id: uuid.UUID) -> Run | None:
        return await self._one(
            select(RunModel).where(RunModel.id == run_id, RunModel.workspace_id == workspace_id)
        )

    async def get_by_id(self, run_id: uuid.UUID) -> Run | None:
        """No workspace filter -- see module docstring. For the worker only."""
        return await self._one(select(RunModel).where(RunModel.id == run_id))

    async def find_by_cache_key(self, workspace_id: uuid.UUID, cache_key: str) -> Run | None:
        """The newest Run for this `(workspace_id, cache_key)` pair.

        `.limit(1)` is load-bearing, not cosmetic: the index on this pair is
        not unique (Task 17's caching deliberately falls through past a
        FAILED or CANCELLED hit and creates a fresh Run under the *same*
        cache_key, so two -- or more -- rows sharing a key is an expected,
        recurring state, not an edge case). Without it, `_one`'s
        `scalar_one_or_none()` raises `MultipleResultsFound` the moment a
        second row exists, which is not a `DomainError` and so escapes every
        caller as a raw 500 (CRITICAL, Task 17 review) instead of the 202 the
        caller's next identical request is entitled to.
        """
        return await self._one(
            select(RunModel)
            .where(RunModel.workspace_id == workspace_id, RunModel.cache_key == cache_key)
            .order_by(RunModel.created_at.desc())
            .limit(1)
        )

    async def list(
        self,
        workspace_id: uuid.UUID,
        *,
        kind: RunKind | None = None,
        protocol_id: uuid.UUID | None = None,
        cursor: tuple[datetime, uuid.UUID] | None = None,
        limit: int = 50,
    ) -> list[Run]:
        statement = (
            select(RunModel)
            .where(RunModel.workspace_id == workspace_id)
            .order_by(RunModel.created_at.desc(), RunModel.id.desc())
        )
        if kind is not None:
            statement = statement.where(RunModel.kind == kind.value)
        if protocol_id is not None:
            # Served by `ix_runs_workspace_protocol_id`, created by migration 007
            # for exactly this query and unused until now.
            statement = statement.where(RunModel.protocol_id == protocol_id)
        if cursor is not None:
            statement = statement.where(tuple_(RunModel.created_at, RunModel.id) < cursor)
        async with self._sessions() as session:
            result = await session.execute(statement.limit(limit))
            return [_to_domain(model) for model in result.scalars()]

    async def _one(self, statement: Select[tuple[RunModel]]) -> Run | None:
        async with self._sessions() as session:
            model = (await session.execute(statement)).scalar_one_or_none()
            return _to_domain(model) if model is not None else None
