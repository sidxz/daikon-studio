"""SQLAlchemy InSilicoProtocol repository.

Every read filters on `workspace_id` inside the SQL, never after it: a
cross-tenant row is not fetched and then discarded, it is never selected.

Unlike Dataset, a Protocol is mutable up to the moment it is published
(`publish()` flips `status` and stamps `published_at` in place on the same
aggregate) -- so, unlike `SqlAlchemyDatasetRepository`, this repository needs
an `update`, not just `add`/`get`/`list`.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CursorResult, Select, select, tuple_
from sqlalchemy import delete as sa_delete
from sqlalchemy import update as sa_update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from daikonstudio.domain.catalog.protocol import InSilicoProtocol, ProtocolStatus
from daikonstudio.domain.catalog.readout import Readout, ReadoutType
from daikonstudio.domain.shared.errors import ConcurrencyConflictError, ConflictError
from daikonstudio.infrastructure.persistence.sqlalchemy.catalog.models import (
    InSilicoProtocolModel,
)


def _readout_to_dict(readout: Readout) -> dict[str, Any]:
    return {
        "name": readout.name,
        "type": readout.type.value,
        "unit": readout.unit,
        "direction": readout.direction,
        "description": readout.description,
    }


def _readout_from_dict(data: dict[str, Any]) -> Readout:
    return Readout(
        name=data["name"],
        type=ReadoutType(data["type"]),
        unit=data.get("unit"),
        direction=data.get("direction"),
        description=data["description"],
    )


def _to_domain(model: InSilicoProtocolModel) -> InSilicoProtocol:
    return InSilicoProtocol(
        id=model.id,
        workspace_id=model.workspace_id,
        name=model.name,
        dataset_id=model.dataset_id,
        engine_id=model.engine_id,
        artifact_uri=model.artifact_uri,
        readouts=tuple(_readout_from_dict(r) for r in model.readouts),
        conditions=model.conditions,
        status=ProtocolStatus(model.status),
        published_at=model.published_at,
        parent_protocol_id=model.parent_protocol_id,
        protocol_version=model.protocol_version,
        created_by=model.created_by,
        created_at=model.created_at,
        updated_at=model.updated_at,
        version=model.version,
    )


def _to_model(protocol: InSilicoProtocol) -> InSilicoProtocolModel:
    return InSilicoProtocolModel(
        id=protocol.id,
        workspace_id=protocol.workspace_id,
        name=protocol.name,
        dataset_id=protocol.dataset_id,
        engine_id=protocol.engine_id,
        artifact_uri=protocol.artifact_uri,
        readouts=[_readout_to_dict(r) for r in protocol.readouts],
        # `protocol.conditions` is a read-only `MappingProxyType`, not a plain dict --
        # JSONB serialisation (and the psycopg/asyncpg JSON codec under it) only knows
        # how to encode an actual `dict`, so unwrap it here rather than at every caller.
        conditions=dict(protocol.conditions),
        status=protocol.status.value,
        published_at=protocol.published_at,
        parent_protocol_id=protocol.parent_protocol_id,
        protocol_version=protocol.protocol_version,
        created_by=protocol.created_by,
        version=protocol.version,
        # Explicit, rather than the column's server_default: otherwise the
        # created_at in a 201 response body (the aggregate's own clock) and the
        # one a later GET returns (the database's) would differ by however long
        # the insert took.
        created_at=protocol.created_at,
        updated_at=protocol.updated_at,
    )


class SqlAlchemyProtocolRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(self, protocol: InSilicoProtocol) -> None:
        async with self._sessions() as session:
            session.add(_to_model(protocol))
            try:
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                raise ConflictError(f"Protocol '{protocol.id}' already exists") from error

    async def update(self, protocol: InSilicoProtocol) -> None:
        """Optimistic concurrency: the UPDATE only lands if the row's `version`
        still matches what this in-memory copy was loaded with, and bumps it by
        one when it does. Zero rows affected means someone else's write (e.g. a
        concurrent `publish()`) already moved the row on -- raise rather than
        silently last-writer-wins overwriting it.

        `workspace_id` is also part of the predicate (whole-branch review,
        Important 5) -- every real caller already loads its `protocol` through
        `get()`, which is itself workspace-scoped, so this cannot fire today.
        It stays because it is the one invariant in this module that was
        conventional rather than enforced, at the cost of one clause.
        """
        model = _to_model(protocol)
        expected_version = protocol.version
        async with self._sessions() as session:
            result = await session.execute(
                sa_update(InSilicoProtocolModel)
                .where(
                    InSilicoProtocolModel.id == protocol.id,
                    InSilicoProtocolModel.workspace_id == protocol.workspace_id,
                    InSilicoProtocolModel.version == expected_version,
                )
                .values(
                    name=model.name,
                    dataset_id=model.dataset_id,
                    engine_id=model.engine_id,
                    artifact_uri=model.artifact_uri,
                    readouts=model.readouts,
                    conditions=model.conditions,
                    status=model.status,
                    published_at=model.published_at,
                    parent_protocol_id=model.parent_protocol_id,
                    protocol_version=model.protocol_version,
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
                raise ConcurrencyConflictError("Protocol", str(protocol.id))
            await session.commit()
        # The caller's in-memory copy now matches what was just persisted -- without
        # this, a second `update()` on the same object would immediately self-conflict.
        protocol.version = expected_version + 1

    async def delete(self, workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> None:
        async with self._sessions() as session:
            await session.execute(
                sa_delete(InSilicoProtocolModel).where(
                    InSilicoProtocolModel.id == protocol_id,
                    InSilicoProtocolModel.workspace_id == workspace_id,
                )
            )
            await session.commit()

    async def get(
        self, workspace_id: uuid.UUID, protocol_id: uuid.UUID
    ) -> InSilicoProtocol | None:
        return await self._one(
            select(InSilicoProtocolModel).where(
                InSilicoProtocolModel.id == protocol_id,
                InSilicoProtocolModel.workspace_id == workspace_id,
            )
        )

    async def list(
        self,
        workspace_id: uuid.UUID,
        *,
        cursor: tuple[datetime, uuid.UUID] | None = None,
        limit: int = 50,
    ) -> list[InSilicoProtocol]:
        statement = (
            select(InSilicoProtocolModel)
            .where(InSilicoProtocolModel.workspace_id == workspace_id)
            .order_by(InSilicoProtocolModel.created_at.desc(), InSilicoProtocolModel.id.desc())
        )
        if cursor is not None:
            statement = statement.where(
                tuple_(InSilicoProtocolModel.created_at, InSilicoProtocolModel.id) < cursor
            )
        async with self._sessions() as session:
            result = await session.execute(statement.limit(limit))
            return [_to_domain(model) for model in result.scalars()]

    async def _one(
        self, statement: Select[tuple[InSilicoProtocolModel]]
    ) -> InSilicoProtocol | None:
        async with self._sessions() as session:
            model = (await session.execute(statement)).scalar_one_or_none()
            return _to_domain(model) if model is not None else None
