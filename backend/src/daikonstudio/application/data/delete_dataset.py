"""Delete a Dataset that nothing depends on.

A Dataset is immutable and cited by id, which is why one that anything still
depends on cannot be deleted: any Protocol trained on it (a published one reads
its training compounds on every prediction run, to measure applicability
domain), and any training run on it still pending or running. Training runs that
failed or were cancelled have no Protocol, and are deleted with it, as are the
pages on it and on them.

Rows go first, then files, for the reason given in `delete_protocol.py`.
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
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.page_repository import PageRepository
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.domain.execution.run import RunStatus
from daikonstudio.domain.shared.errors import ConflictError, DomainError, NotFoundError
from daikonstudio.domain.shared.page import PageOwnerKind

logger = logging.getLogger(__name__)

_ACTIVE = {RunStatus.PENDING, RunStatus.RUNNING}

PROTOCOLS_FIRST = (
    "Protocols trained on this dataset must be deleted first, "
    "including drafts you may not be able to see. "
    "A dataset used by a published protocol cannot be deleted."
)


def dataset_folder(workspace_id: uuid.UUID, dataset_id: uuid.UUID) -> str:
    """The snapshot and the profile both live under this folder (`snapshot.py`,
    `get_dataset_profile.py`)."""
    return f"{workspace_id}/datasets/{dataset_id}/"


@dataclass(frozen=True, kw_only=True)
class DeleteDatasetCommand:
    dataset_id: uuid.UUID


class DeleteDataset:
    def __init__(
        self,
        datasets: DatasetRepository,
        protocols: ProtocolRepository,
        runs: RunRepository,
        store: BlobStore,
        pages: PageRepository,
    ) -> None:
        self._datasets = datasets
        self._protocols = protocols
        self._runs = runs
        self._store = store
        self._pages = pages

    async def __call__(
        self, command: DeleteDatasetCommand, auth: AuthContext | None = None
    ) -> Result[None, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None

        dataset = await self._datasets.get(auth.workspace_id, command.dataset_id)
        if dataset is None:
            return Failure(NotFoundError("Dataset", str(command.dataset_id)))
        require_may_delete(auth, dataset.created_by)
        if await self._protocols.list(auth.workspace_id, dataset_id=dataset.id, limit=1):
            return Failure(ConflictError(PROTOCOLS_FIRST))
        runs = await self._runs.list_training_for_dataset(auth.workspace_id, dataset.id)
        if any(run.status in _ACTIVE for run in runs):
            return Failure(ConflictError("A training run on this dataset is still in progress."))

        # ponytail: a training run submitted between the check above and the
        # delete below fails when it reads the missing snapshot. Rare (it needs
        # two people acting on one dataset in the same second), and loud.
        run_ids = [run.id for run in runs]
        await self._pages.delete_for_owners(auth.workspace_id, PageOwnerKind.RUN, run_ids)
        await self._pages.delete_for_owners(auth.workspace_id, PageOwnerKind.DATASET, [dataset.id])
        await self._runs.delete_many(auth.workspace_id, run_ids)
        await self._datasets.delete(auth.workspace_id, dataset.id)
        folder = dataset_folder(auth.workspace_id, dataset.id)
        try:
            self._store.delete_prefix(folder)
        except Exception:
            logger.exception("Deleting %s failed; the folder is orphaned", folder)
        logger.info("Dataset %s deleted by %s", dataset.id, auth.user_id)
        return Success(None)
