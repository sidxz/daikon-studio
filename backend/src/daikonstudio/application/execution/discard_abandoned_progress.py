"""Saved training progress that nobody came back for.

A training run that succeeds deletes its saved progress (`TrainProtocol`). One that is
cancelled or fails keeps it, so Resume can continue from the last save -- and before
this, nothing else removed it: every run cancelled and never resumed kept its progress
for as long as its Dataset existed, several hundred MB for a MoLFormer fit. This deletes
it once the run has been stopped for `SAVED_PROGRESS_DAYS`. The run itself, its error
and its epochs stay; a Resume after that trains from the start.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta

from daikonstudio.application.engines.checkpoints import checkpoint_root
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.run_repository import RunRepository

logger = logging.getLogger(__name__)

#: How long a cancelled or failed training run keeps its saved progress. The run page
#: says so beside Resume (`run-detail.tsx`); change both together.
SAVED_PROGRESS_DAYS = 7


class DiscardAbandonedProgress:
    def __init__(self, runs: RunRepository, store: BlobStore) -> None:
        self._runs = runs
        self._store = store

    async def __call__(self, now: datetime | None = None) -> int:
        """Delete the saved progress of every training run stopped more than
        `SAVED_PROGRESS_DAYS` ago, in every workspace; returns how many runs were
        checked. Idempotent: progress already gone is skipped."""
        cutoff = (now or datetime.now(UTC)) - timedelta(days=SAVED_PROGRESS_DAYS)
        runs = await self._runs.list_stopped_training(stopped_before=cutoff)
        for run in runs:
            dataset_id = run.params.get("dataset_id")
            if not dataset_id:
                continue
            root = checkpoint_root(run.workspace_id, uuid.UUID(str(dataset_id)), run.id)
            try:
                await asyncio.to_thread(self._store.delete_prefix, root)
            except Exception:
                logger.warning("Could not delete saved progress under %s", root, exc_info=True)
        return len(runs)
