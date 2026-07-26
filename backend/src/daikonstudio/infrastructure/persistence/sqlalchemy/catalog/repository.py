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

from sqlalchemy import Select, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from daikonstudio.domain.catalog.protocol import InSilicoProtocol, ProtocolStatus
from daikonstudio.domain.catalog.readout import Readout, ReadoutType
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
        conditions=protocol.conditions,
        status=protocol.status.value,
        published_at=protocol.published_at,
        parent_protocol_id=protocol.parent_protocol_id,
        protocol_version=protocol.protocol_version,
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
            await session.commit()

    async def update(self, protocol: InSilicoProtocol) -> None:
        async with self._sessions() as session:
            await session.merge(_to_model(protocol))
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
