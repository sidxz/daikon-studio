"""Re-execute a failed or cancelled Run in place.

Design: `2026-08-04-retry-a-failed-run-design.md` (git history, `50860b2^`).
The same Run, not a new one: `params` is write-once, so the re-enqueued job
re-reads its original instructions by construction, and the cache key stays
the same because the inputs are. `Run.retry()` owns the status rule; this use
case loads the right row, resolves the lane the way the original enqueue did,
and orders the writes so a worker can never claim a row still reading FAILED.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.engines.checkpoints import Checkpoints, checkpoint_root
from daikonstudio.application.engines.registry import EngineRegistry, UnknownEngineError
from daikonstudio.application.execution.enqueue import JobEnqueuer
from daikonstudio.application.execution.train_protocol import training_lane
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.domain.shared.errors import DomainError, NotFoundError


@dataclass(frozen=True, kw_only=True)
class RetryRunCommand:
    run_id: uuid.UUID
    # Start over: discard the run's saved training progress before requeueing it, so
    # the next attempt fits everything again. A prediction run keeps none; ignored there.
    fresh: bool = False


class RetryRun:
    def __init__(
        self,
        runs: RunRepository,
        protocols: ProtocolRepository,
        enqueuer: JobEnqueuer,
        engines: EngineRegistry,
        store: BlobStore,
    ) -> None:
        self._runs = runs
        self._protocols = protocols
        self._enqueuer = enqueuer
        self._engines = engines
        self._store = store

    async def __call__(
        self, command: RetryRunCommand, auth: AuthContext | None = None
    ) -> Result[Run, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None

        run = await self._runs.get(auth.workspace_id, command.run_id)
        if run is None:
            return Failure(NotFoundError("Run", str(command.run_id)))
        try:
            # Resolve the lane before touching the row, so an engine this deployment
            # no longer ships leaves the run exactly as it was.
            lane = await self._lane(run, auth)
            run.retry()
        except DomainError as error:
            return Failure(error)
        if command.fresh and run.kind is RunKind.TRAINING:
            # Before the update and the enqueue, so the requeued attempt can never load
            # what it was asked to forget.
            Checkpoints(
                self._store,
                checkpoint_root(
                    run.workspace_id, uuid.UUID(str(run.params["dataset_id"])), run.id
                ),
            ).clear()
        # Update, then enqueue. Enqueueing first would let a worker pick up a run
        # whose row still reads FAILED, and `run_job` drops redeliveries for
        # terminal runs -- the retry would vanish silently.
        await self._runs.update(run)
        await self._enqueuer.enqueue(run.id, lane=lane)
        return Success(run)

    async def _lane(self, run: Run, auth: AuthContext) -> str:
        try:
            if run.kind is RunKind.TRAINING:
                return training_lane(
                    self._engines, run.params["engine_id"], run.params.get("baseline_engine_id")
                )
            protocol_id = uuid.UUID(run.params["protocol_id"])
            protocol = await self._protocols.get(auth.workspace_id, protocol_id)
            if protocol is None:
                raise NotFoundError("Protocol", str(protocol_id))
            return self._engines.get(protocol.engine_id).manifest().lane
        except UnknownEngineError as error:
            raise NotFoundError("Engine") from error
