"""SQLAlchemy Runner repository.

`touch_last_seen` and `revoke` are targeted `UPDATE` statements rather than a
`get` + mutate + `add`-style round trip, and neither bumps `version` -- a
heartbeat (or a revoke, which is idempotent by construction) is not an edit
in the optimistic-concurrency sense `SqlAlchemyRunRepository.update` protects.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Select, select
from sqlalchemy import delete as sa_delete
from sqlalchemy import update as sa_update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from daikonstudio.domain.runners.runner import Runner
from daikonstudio.domain.shared.errors import ConflictError
from daikonstudio.infrastructure.persistence.sqlalchemy.runners.models import RunnerModel


def _to_domain(model: RunnerModel) -> Runner:
    return Runner(
        id=model.id,
        name=model.name,
        lanes=tuple(model.lanes),
        token_hash=model.token_hash,
        last_seen_at=model.last_seen_at,
        revoked_at=model.revoked_at,
        build_version=model.build_version,
        build_git_sha=model.build_git_sha,
        build_date=model.build_date,
        created_at=model.created_at,
        updated_at=model.updated_at,
        version=model.version,
    )


def _to_model(runner: Runner) -> RunnerModel:
    return RunnerModel(
        id=runner.id,
        name=runner.name,
        lanes=list(runner.lanes),
        token_hash=runner.token_hash,
        last_seen_at=runner.last_seen_at,
        revoked_at=runner.revoked_at,
        build_version=runner.build_version,
        build_git_sha=runner.build_git_sha,
        build_date=runner.build_date,
        version=runner.version,
        # Explicit, rather than the column's server_default: otherwise the
        # created_at in a response body (the aggregate's own clock) and the
        # one a later GET returns (the database's) would differ by however
        # long the insert took.
        created_at=runner.created_at,
        updated_at=runner.updated_at,
    )


class SqlAlchemyRunnerRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(self, runner: Runner) -> None:
        async with self._sessions() as session:
            session.add(_to_model(runner))
            try:
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                raise ConflictError(f"A runner named '{runner.name}' already exists") from error

    async def get(self, runner_id: uuid.UUID) -> Runner | None:
        return await self._one(select(RunnerModel).where(RunnerModel.id == runner_id))

    async def get_by_token_hash(self, token_hash: str) -> Runner | None:
        return await self._one(select(RunnerModel).where(RunnerModel.token_hash == token_hash))

    async def list(self) -> list[Runner]:
        async with self._sessions() as session:
            result = await session.execute(
                select(RunnerModel).order_by(RunnerModel.created_at.desc())
            )
            return [_to_domain(model) for model in result.scalars()]

    async def touch_last_seen(self, runner_id: uuid.UUID) -> None:
        """A heartbeat, not an edit -- does not bump `version`."""
        async with self._sessions() as session:
            await session.execute(
                sa_update(RunnerModel)
                .where(RunnerModel.id == runner_id)
                .values(last_seen_at=datetime.now(UTC))
            )
            await session.commit()

    async def record_build(
        self, runner_id: uuid.UUID, *, version: str, git_sha: str, build_date: str
    ) -> None:
        """What the agent reported at start. Like `touch_last_seen`, a targeted
        update that does not bump `version`: an agent restarting on a new image
        is not an edit anyone holds a stale copy of."""
        async with self._sessions() as session:
            await session.execute(
                sa_update(RunnerModel)
                .where(RunnerModel.id == runner_id)
                .values(build_version=version, build_git_sha=git_sha, build_date=build_date)
            )
            await session.commit()

    async def delete(self, runner_id: uuid.UUID) -> None:
        """Removes the row. `DeleteRunner` is what decides a runner may go --
        `runs.claimed_by` has no foreign key behind it, so nothing here would
        stop a delete that leaves run history naming a runner that is gone."""
        async with self._sessions() as session:
            await session.execute(sa_delete(RunnerModel).where(RunnerModel.id == runner_id))
            await session.commit()

    async def revoke(self, runner_id: uuid.UUID) -> None:
        """Idempotent: only sets `revoked_at` when it is still NULL, so a
        retry cannot slide the recorded instant forward. Not an edit either
        -- does not bump `version`."""
        async with self._sessions() as session:
            await session.execute(
                sa_update(RunnerModel)
                .where(RunnerModel.id == runner_id, RunnerModel.revoked_at.is_(None))
                .values(revoked_at=datetime.now(UTC))
            )
            await session.commit()

    async def _one(self, statement: Select[tuple[RunnerModel]]) -> Runner | None:
        async with self._sessions() as session:
            model = (await session.execute(statement)).scalar_one_or_none()
            return _to_domain(model) if model is not None else None
