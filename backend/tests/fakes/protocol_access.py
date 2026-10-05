"""In-memory ProtocolAccess with Duar's rules (identity-service, 2026-10-05)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from daikonstudio.domain.shared.errors import ServiceUnavailableError

_FULL = ("admin", "owner")


@dataclass
class FakeProtocolAccess:
    # protocol id -> (workspace id, owner id, "private" | "workspace")
    acl: dict[uuid.UUID, tuple[uuid.UUID, uuid.UUID, str]] = field(default_factory=dict)
    down: bool = False  # simulate Duar being unreachable for reads

    async def register(self, protocol: InSilicoProtocol) -> None:
        if protocol.created_by is None:
            return
        visibility = "workspace" if protocol.is_locked else "private"
        self.acl.setdefault(protocol.id, (protocol.workspace_id, protocol.created_by, visibility))

    async def deregister(self, protocol_id: uuid.UUID) -> None:
        self.acl.pop(protocol_id, None)

    async def make_workspace_visible(self, auth: Any, protocol: InSilicoProtocol) -> None:
        self._check()
        workspace, owner, _ = self.acl.get(
            protocol.id, (protocol.workspace_id, protocol.created_by or auth.user_id, "private")
        )
        self.acl[protocol.id] = (workspace, owner, "workspace")

    async def visible_ids(self, auth: Any) -> frozenset[uuid.UUID] | None:
        self._check()
        if auth.workspace_role in _FULL:
            return None
        return frozenset(pid for pid in self.acl if self._allowed(auth, pid))

    async def can_view(self, auth: Any, protocol_id: uuid.UUID) -> bool:
        self._check()
        return auth.workspace_role in _FULL or self._allowed(auth, protocol_id)

    def _allowed(self, auth: Any, protocol_id: uuid.UUID) -> bool:
        entry = self.acl.get(protocol_id)
        if entry is None:
            return False
        workspace, owner, visibility = entry
        return workspace == auth.workspace_id and (
            owner == auth.user_id or visibility == "workspace"
        )

    def _check(self) -> None:
        if self.down:
            raise ServiceUnavailableError("Protocol permissions could not be checked.")
