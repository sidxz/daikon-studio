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

import builtins
import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import CursorResult, Select, false, func, or_, select, tuple_
from sqlalchemy import delete as sa_delete
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from daikonstudio.application.engines.context import EpochPoint
from daikonstudio.application.ports.run_repository import SweepSummary, TrainingVisibility
from daikonstudio.domain.execution.run import Run, RunKind, RunStatus
from daikonstudio.domain.shared.errors import ConcurrencyConflictError
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.models import (
    RunEpochModel,
    RunModel,
)


def _to_domain(model: RunModel) -> Run:
    return Run(
        id=model.id,
        workspace_id=model.workspace_id,
        kind=RunKind(model.kind),
        requested_by=model.requested_by,
        cache_key=model.cache_key,
        params=model.params,
        protocol_id=model.protocol_id,
        sweep_id=model.sweep_id,
        metrics=model.metrics,
        lane=model.lane,
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
        sweep_id=run.sweep_id,
        metrics=run.metrics,
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

    async def append_epochs(self, run_id: uuid.UUID, points: Sequence[EpochPoint]) -> None:
        if not points:
            return
        async with self._sessions() as session:
            attempt = await session.scalar(select(RunModel.attempts).where(RunModel.id == run_id))
            if attempt is None:  # the run is gone; so is anything to chart
                return
            session.add_all(
                RunEpochModel(
                    run_id=run_id,
                    attempt=attempt,
                    fit=point.fit,
                    target=point.target,
                    member=point.member,
                    members=point.members,
                    epoch=point.epoch,
                    epochs=point.epochs,
                    train_loss=point.train_loss,
                    val_loss=point.val_loss,
                    scores=point.scores,
                    device=point.device,
                    kept_epoch=point.kept_epoch,
                    kept_by=point.kept_by,
                    recorded_at=point.at,
                )
                for point in points
            )
            await session.commit()

    async def list_epochs(self, run_id: uuid.UUID) -> builtins.list[EpochPoint]:
        latest = (
            select(func.max(RunEpochModel.attempt))
            .where(RunEpochModel.run_id == run_id)
            .scalar_subquery()
        )
        async with self._sessions() as session:
            rows = await session.scalars(
                select(RunEpochModel)
                .where(RunEpochModel.run_id == run_id, RunEpochModel.attempt == latest)
                .order_by(RunEpochModel.id)
            )
            return [
                EpochPoint(
                    epoch=row.epoch,
                    epochs=row.epochs,
                    train_loss=row.train_loss,
                    val_loss=row.val_loss,
                    scores=row.scores,
                    device=row.device,
                    member=row.member,
                    members=row.members,
                    target=row.target,
                    fit=row.fit,
                    kept_epoch=row.kept_epoch,
                    kept_by=row.kept_by,
                    at=row.recorded_at,
                )
                for row in rows
            ]

    async def clear_epochs(self, run_id: uuid.UUID) -> None:
        async with self._sessions() as session:
            await session.execute(sa_delete(RunEpochModel).where(RunEpochModel.run_id == run_id))
            await session.commit()

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
                    # An outcome, like protocol_id above -- see
                    # `Run.record_metrics`. `sweep_id` is deliberately absent:
                    # it is an instruction, and stays write-once like `params`.
                    metrics=model.metrics,
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

    async def delete_many(self, workspace_id: uuid.UUID, run_ids: Sequence[uuid.UUID]) -> None:
        if not run_ids:
            return
        async with self._sessions() as session:
            await session.execute(
                sa_delete(RunModel).where(
                    RunModel.workspace_id == workspace_id, RunModel.id.in_(run_ids)
                )
            )
            await session.commit()

    async def list_training_for_dataset(
        self, workspace_id: uuid.UUID, dataset_id: uuid.UUID
    ) -> builtins.list[Run]:
        # ponytail: a JSONB scan within the workspace, no index. Fine while a
        # workspace has thousands of runs; add an expression index on
        # (workspace_id, params->>'dataset_id') if deletes get slow.
        statement = select(RunModel).where(
            RunModel.workspace_id == workspace_id,
            RunModel.kind == RunKind.TRAINING.value,
            RunModel.params["dataset_id"].astext == str(dataset_id),
        )
        async with self._sessions() as session:
            result = await session.execute(statement)
            return [_to_domain(model) for model in result.scalars()]

    async def list_stopped_training(self, stopped_before: datetime) -> builtins.list[Run]:
        # ponytail: every old cancelled or failed training run, every day, including
        # ones whose progress is already gone. Fine at hundreds; add a column marking
        # the progress discarded if it reaches thousands.
        statement = select(RunModel).where(
            RunModel.kind == RunKind.TRAINING.value,
            RunModel.status.in_((RunStatus.CANCELLED.value, RunStatus.FAILED.value)),
            RunModel.updated_at < stopped_before,
        )
        async with self._sessions() as session:
            result = await session.execute(statement)
            return [_to_domain(model) for model in result.scalars()]

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
        training_visible_to: TrainingVisibility | None = None,
        requested_by: uuid.UUID | None = None,
        statuses: Sequence[RunStatus] | None = None,
        protocol_ids: frozenset[uuid.UUID] | None = None,
        name_contains: str | None = None,
        name_or_protocol_ids: frozenset[uuid.UUID] = frozenset(),
        created_from: datetime | None = None,
        created_before: datetime | None = None,
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
        if requested_by is not None:
            statement = statement.where(RunModel.requested_by == requested_by)
        if created_from is not None:
            statement = statement.where(RunModel.created_at >= created_from)
        if created_before is not None:
            statement = statement.where(RunModel.created_at < created_before)
        if statuses:
            statement = statement.where(RunModel.status.in_([s.value for s in statuses]))
        if protocol_ids is not None:
            statement = statement.where(
                RunModel.protocol_id.in_(protocol_ids) if protocol_ids else false()
            )
        if name_contains:
            # The typed text is matched literally: `%` and `_` are escaped, not wildcards.
            escaped = name_contains.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            by_name = RunModel.params["name"].astext.ilike(f"%{escaped}%", escape="\\")
            # Runs from before run names existed are labelled with their protocol's name.
            statement = statement.where(
                or_(by_name, RunModel.protocol_id.in_(name_or_protocol_ids))
                if name_or_protocol_ids
                else by_name
            )
        if training_visible_to is not None:
            visible = training_visible_to
            statement = statement.where(
                or_(
                    RunModel.kind == RunKind.PREDICTION.value,
                    RunModel.requested_by == visible.user_id,
                    RunModel.protocol_id.in_(visible.protocol_ids)
                    if visible.protocol_ids
                    else false(),
                )
            )
        if cursor is not None:
            statement = statement.where(tuple_(RunModel.created_at, RunModel.id) < cursor)
        async with self._sessions() as session:
            result = await session.execute(statement.limit(limit))
            return [_to_domain(model) for model in result.scalars()]

    # `builtins.list[...]`, not the bare generic -- see the matching comment
    # on the port's `list_by_sweep`: this class already has a method named
    # `list`, and an unqualified `list[...]` used after it resolves to that
    # method rather than the builtin under Python 3.14's lazy annotation
    # evaluation (PEP 649), a landmine mypy also flags statically.
    async def list_by_sweep(
        self, workspace_id: uuid.UUID, sweep_id: uuid.UUID
    ) -> builtins.list[Run]:
        statement = (
            select(RunModel)
            .where(RunModel.workspace_id == workspace_id, RunModel.sweep_id == sweep_id)
            .order_by(RunModel.created_at, RunModel.id)
        )
        async with self._sessions() as session:
            result = await session.execute(statement)
            return [_to_domain(model) for model in result.scalars()]

    async def sweep_summaries(
        self,
        workspace_id: uuid.UUID,
        *,
        limit: int = 50,
        requested_by: uuid.UUID | None = None,
    ) -> builtins.list[SweepSummary]:
        """One row per sweep, via FILTER-ed aggregates so the LIMIT applies to
        sweeps rather than to (sweep, status) pairs.

        ponytail: no cursor. Sweeps are created by hand, a handful at a time --
        add keyset pagination here the day a workspace has more than `limit`,
        the same shape `list()` already uses.
        """
        counts = {
            status: func.count().filter(RunModel.status == status).label(f"n_{status}")
            for status in ("pending", "running", "ready", "failed", "cancelled")
        }
        statement = (
            select(
                RunModel.sweep_id,
                func.min(RunModel.created_at).label("created_at"),
                func.min(RunModel.params["sweep_name"].astext).label("name"),
                func.min(RunModel.params["dataset_id"].astext).label("dataset_id"),
                func.count().label("total"),
                *counts.values(),
            )
            .where(RunModel.workspace_id == workspace_id, RunModel.sweep_id.is_not(None))
            .group_by(RunModel.sweep_id)
            .order_by(func.min(RunModel.created_at).desc())
            .limit(limit)
        )
        if requested_by is not None:
            statement = statement.where(RunModel.requested_by == requested_by)
        async with self._sessions() as session:
            rows = (await session.execute(statement)).all()
        return [
            SweepSummary(
                sweep_id=row.sweep_id,
                name=row.name,
                dataset_id=uuid.UUID(row.dataset_id) if row.dataset_id else None,
                created_at=row.created_at,
                total=row.total,
                by_status={
                    status: getattr(row, f"n_{status}")
                    for status in counts
                    if getattr(row, f"n_{status}")
                },
            )
            for row in rows
        ]

    async def _one(self, statement: Select[tuple[RunModel]]) -> Run | None:
        async with self._sessions() as session:
            model = (await session.execute(statement)).scalar_one_or_none()
            return _to_domain(model) if model is not None else None
