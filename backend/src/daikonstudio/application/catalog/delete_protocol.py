"""Delete a draft Protocol, with the training run that produced it and its files.

Only a draft. A published Protocol is citable, and predictions and collections
may depend on it. A draft has neither (a prediction needs a published Protocol,
and `new_version` needs a published parent), so its training run and its folder
of files are everything that depends on it.

Rows go first, then files: a failed file delete leaves an orphan folder, which is
harmless, where the reverse order could leave a row pointing at files that are
gone. Runs go before the Protocol, so a failure between the two leaves a draft
that can simply be deleted again.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import (
    AuthContext,
    require_authenticated,
    require_may_delete,
)
from daikonstudio.application.catalog.visibility import visible_protocol
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.protocol_access import ProtocolAccess
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.domain.execution.run import RunStatus
from daikonstudio.domain.shared.errors import ConflictError, DomainError, NotFoundError

logger = logging.getLogger(__name__)

_ACTIVE = {RunStatus.PENDING, RunStatus.RUNNING}


def protocol_folder(workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> str:
    """Every file a Protocol owns lives under this folder: the artifact, the scorecard
    inputs and the chemical-space map. Never delete `artifact_uri` on its own: a
    versioned Protocol's can point into another Protocol's folder."""
    return f"{workspace_id}/protocols/{protocol_id}/"


@dataclass(frozen=True, kw_only=True)
class DeleteProtocolCommand:
    protocol_id: uuid.UUID


class DeleteProtocol:
    def __init__(
        self,
        protocols: ProtocolRepository,
        runs: RunRepository,
        store: BlobStore,
        access: ProtocolAccess,
    ) -> None:
        self._access = access
        self._protocols = protocols
        self._runs = runs
        self._store = store

    async def __call__(
        self, command: DeleteProtocolCommand, auth: AuthContext | None = None
    ) -> Result[None, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None

        protocol = await visible_protocol(
            self._protocols, self._access, auth, auth.workspace_id, command.protocol_id
        )
        if protocol is None:
            return Failure(NotFoundError("Protocol", str(command.protocol_id)))
        require_may_delete(auth, protocol.created_by)
        if protocol.is_locked:
            return Failure(ConflictError("Published protocols cannot be deleted."))

        # A draft has one training run; the limit is a ceiling, not a page size.
        runs = await self._runs.list(auth.workspace_id, protocol_id=protocol.id, limit=100)
        if any(run.status in _ACTIVE for run in runs):
            return Failure(
                ConflictError(
                    "This protocol's training run is still finishing. Try again when it completes."
                )
            )

        await self._runs.delete_many(auth.workspace_id, [run.id for run in runs])
        await self._protocols.delete(auth.workspace_id, protocol.id)
        folder = protocol_folder(auth.workspace_id, protocol.id)
        try:
            self._store.delete_prefix(folder)
        except Exception:
            logger.exception("Deleting %s failed; the folder is orphaned", folder)
        logger.info("Protocol %s deleted by %s", protocol.id, auth.user_id)
        return Success(None)
