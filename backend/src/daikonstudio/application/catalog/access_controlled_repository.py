"""The server-side ProtocolRepository. Every protocol it creates is registered with
Duar before its row exists, so its creator never sees a READY run whose protocol 404s.
Every one it deletes is forgotten. Both training paths create protocols through this:
a runner's report (`runner_api.create_protocol`) and the inline job (`jobs.py`). So
neither can skip registration. Reads pass straight through.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from daikonstudio.application.ports.protocol_access import ProtocolAccess
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.domain.catalog.protocol import InSilicoProtocol


class AccessControlledProtocolRepository:
    def __init__(self, inner: ProtocolRepository, access: ProtocolAccess) -> None:
        self._inner = inner
        self._access = access

    async def add(self, protocol: InSilicoProtocol) -> None:
        await self._access.register(protocol)  # never raises; see ProtocolAccess
        await self._inner.add(protocol)

    async def delete(self, workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> None:
        await self._inner.delete(workspace_id, protocol_id)
        await self._access.deregister(protocol_id)

    async def update(self, protocol: InSilicoProtocol) -> None:
        await self._inner.update(protocol)

    async def get(
        self, workspace_id: uuid.UUID, protocol_id: uuid.UUID
    ) -> InSilicoProtocol | None:
        return await self._inner.get(workspace_id, protocol_id)

    async def list(
        self,
        workspace_id: uuid.UUID,
        *,
        cursor: tuple[datetime, uuid.UUID] | None = None,
        limit: int = 50,
        dataset_id: uuid.UUID | None = None,
    ) -> list[InSilicoProtocol]:
        return await self._inner.list(
            workspace_id, cursor=cursor, limit=limit, dataset_id=dataset_id
        )
