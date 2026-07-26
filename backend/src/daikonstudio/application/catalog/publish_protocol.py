"""Publish a Protocol -- the one irreversible step in its lifecycle.

`InSilicoProtocol.publish()` raises `DataLockedError` on a second call
(mapped to 423 by the shared error handler in `interface/error_handlers.py`),
so this use case does not special-case "already published" at all: the
aggregate's own invariant is the only place that rule lives, and a Failure
here is that same error propagating, not a fresh one.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from daikonstudio.domain.shared.errors import DomainError, NotFoundError


@dataclass(frozen=True, kw_only=True)
class PublishProtocolCommand:
    protocol_id: uuid.UUID


class PublishProtocol:
    def __init__(self, repository: ProtocolRepository) -> None:
        self._repository = repository

    async def __call__(
        self, command: PublishProtocolCommand, auth: AuthContext | None = None
    ) -> Result[InSilicoProtocol, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None

        protocol = await self._repository.get(auth.workspace_id, command.protocol_id)
        if protocol is None:
            return Failure(NotFoundError("Protocol", str(command.protocol_id)))

        try:
            protocol.publish()
        except DomainError as error:
            return Failure(error)
        await self._repository.update(protocol)
        return Success(protocol)
