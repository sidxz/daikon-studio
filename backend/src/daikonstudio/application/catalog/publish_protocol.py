"""Publish a Protocol -- the one irreversible step in its lifecycle.

`InSilicoProtocol.publish()` raises `DataLockedError` on a second call
(mapped to 423 by the shared error handler in `interface/error_handlers.py`),
so the aggregate's own invariant is the only place that rule lives. This use
case checks it up front, only so that Duar is not touched for a protocol that is
already published; the Failure is the same `DataLockedError`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.catalog.visibility import visible_protocol
from daikonstudio.application.ports.protocol_access import ProtocolAccess
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from daikonstudio.domain.shared.errors import DataLockedError, DomainError, NotFoundError


@dataclass(frozen=True, kw_only=True)
class PublishProtocolCommand:
    protocol_id: uuid.UUID


class PublishProtocol:
    def __init__(self, repository: ProtocolRepository, access: ProtocolAccess) -> None:
        self._repository = repository
        self._access = access

    async def __call__(
        self, command: PublishProtocolCommand, auth: AuthContext | None = None
    ) -> Result[InSilicoProtocol, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None

        protocol = await visible_protocol(
            self._repository, self._access, auth, auth.workspace_id, command.protocol_id
        )
        if protocol is None:
            return Failure(NotFoundError("Protocol", str(command.protocol_id)))
        if protocol.is_locked:
            return Failure(DataLockedError("This protocol is already published."))

        # Duar first: if it fails, the protocol stays a draft and Publish can be retried.
        await self._access.make_workspace_visible(auth, protocol)
        try:
            protocol.publish()
        except DomainError as error:
            return Failure(error)
        await self._repository.update(protocol)
        return Success(protocol)
