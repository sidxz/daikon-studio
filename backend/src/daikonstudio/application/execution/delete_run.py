"""Delete a stopped training run that never produced a Protocol.

Such a run is listed nowhere else and owns nothing but its row, its epochs (which
cascade), its pages and its saved progress. A run that produced a Protocol is deleted with that
Protocol (`DeleteProtocol`); a prediction is not deletable at all.

Files first, then the row: a failed file delete leaves the run listed and deletable
again, where the reverse order would orphan saved progress that nothing lists or
expires any more.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_may_delete
from daikonstudio.application.engines.checkpoints import checkpoint_root
from daikonstudio.application.execution.visibility import run_visible
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.page_repository import PageRepository
from daikonstudio.application.ports.protocol_access import ProtocolAccess
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.domain.execution.run import RunKind, RunStatus
from daikonstudio.domain.shared.errors import ConflictError, DomainError, NotFoundError
from daikonstudio.domain.shared.page import PageOwnerKind

logger = logging.getLogger(__name__)

_STOPPED = {RunStatus.FAILED, RunStatus.CANCELLED}


@dataclass(frozen=True, kw_only=True)
class DeleteRunCommand:
    run_id: uuid.UUID


class DeleteRun:
    def __init__(
        self, runs: RunRepository, store: BlobStore, access: ProtocolAccess, pages: PageRepository
    ) -> None:
        self._runs = runs
        self._store = store
        self._access = access
        self._pages = pages

    async def __call__(
        self, command: DeleteRunCommand, auth: AuthContext | None = None
    ) -> Result[None, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None

        run = await self._runs.get(auth.workspace_id, command.run_id)
        if run is None or not await run_visible(run, auth, self._access):
            return Failure(NotFoundError("Run", str(command.run_id)))
        if run.kind is not RunKind.TRAINING or run.protocol_id is not None:
            return Failure(
                ConflictError("Only a training run that produced no protocol can be deleted.")
            )
        if run.status not in _STOPPED:
            return Failure(ConflictError("Only a failed or cancelled run can be deleted."))
        require_may_delete(auth, run.requested_by)

        dataset_id = run.params.get("dataset_id")
        if dataset_id:
            await asyncio.to_thread(
                self._store.delete_prefix,
                checkpoint_root(run.workspace_id, uuid.UUID(str(dataset_id)), run.id),
            )
        await self._pages.delete_for_owners(auth.workspace_id, PageOwnerKind.RUN, [run.id])
        await self._runs.delete_many(auth.workspace_id, [run.id])
        logger.info("Run %s deleted by %s", run.id, auth.user_id)
        return Success(None)
