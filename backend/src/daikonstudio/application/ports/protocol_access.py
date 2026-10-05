"""Who may see a Protocol. Duar owns the answer: each protocol is a `studio_protocol`
resource, private to its creator while a draft and workspace-wide once published.
Workspace admins and owners see every one; that is Duar's rule, not Studio's.

The port keeps SDK types out of the application layer. `register` and `deregister`
never raise: they are called from the paths that create and delete protocols, where
losing a trained model to a permissions outage would be worse than a protocol that
stays hidden until `register_protocols` runs again. The read methods raise
`ServiceUnavailableError` when Duar cannot answer; Studio fails closed.
"""

from __future__ import annotations

import uuid
from typing import Protocol

from daikonstudio.application.auth import AuthContext
from daikonstudio.domain.catalog.protocol import InSilicoProtocol


class ProtocolAccess(Protocol):
    async def register(self, protocol: InSilicoProtocol) -> None: ...

    async def deregister(self, protocol_id: uuid.UUID) -> None: ...

    async def make_workspace_visible(
        self, auth: AuthContext, protocol: InSilicoProtocol
    ) -> None: ...

    async def visible_ids(self, auth: AuthContext) -> frozenset[uuid.UUID] | None:
        """Every protocol id the caller may view. None means all of them."""
        ...

    async def can_view(self, auth: AuthContext, protocol_id: uuid.UUID) -> bool: ...
