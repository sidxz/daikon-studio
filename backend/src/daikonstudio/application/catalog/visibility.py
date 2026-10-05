"""The one read path for a user looking at a Protocol: workspace-scoped, then Duar's
`view` check. Hidden reads return None, which callers turn into NotFound: a 403 would
confirm a colleague's draft exists. `auth=None` is a system/worker call and skips the
check."""

from __future__ import annotations

import uuid

from daikonstudio.application.auth import AuthContext
from daikonstudio.application.ports.protocol_access import ProtocolAccess
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.domain.catalog.protocol import InSilicoProtocol


async def visible_protocol(
    protocols: ProtocolRepository,
    access: ProtocolAccess,
    auth: AuthContext | None,
    workspace_id: uuid.UUID,
    protocol_id: uuid.UUID,
) -> InSilicoProtocol | None:
    protocol = await protocols.get(workspace_id, protocol_id)
    if protocol is None or auth is None:
        return protocol
    return protocol if await access.can_view(auth, protocol.id) else None
