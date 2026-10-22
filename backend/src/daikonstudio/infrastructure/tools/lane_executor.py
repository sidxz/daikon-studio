"""Runs a tool where its lane says: here for the default lane, otherwise as a queued
TOOL Run that a runner serving that lane claims, while this waits on the row.

ponytail: the wait polls the Run row; a notification channel if the poll ever shows up
in the database's load.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from daikonstudio.application.execution.enqueue import JobEnqueuer
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.application.ports.runner_repository import RunnerRepository
from daikonstudio.application.ports.tool_executor import ToolInput, ToolOutput, ToolRunContext
from daikonstudio.application.tools.manifest import DEFAULT_LANE
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.application.tools.tool_run import ToolRunParams, read_outputs, tool_run_prefix
from daikonstudio.domain.execution.run import Run, RunKind, RunStatus, compute_cache_key
from daikonstudio.domain.shared.errors import ConflictError
from daikonstudio.infrastructure.tools.runtime import ChildProcessToolExecutor

logger = logging.getLogger(__name__)

# A runner touches `last_seen_at` on every poll, and polls every few seconds. One
# not seen for this long is registered but not serving, and a step sent to it
# would wait out the whole step timeout rather than be skipped with a reason.
RUNNER_STALE_SECONDS = 120


class LaneToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        local: ChildProcessToolExecutor,
        runs: RunRepository,
        enqueue: JobEnqueuer,
        runners: RunnerRepository,
        store: BlobStore,
        *,
        poll_seconds: float = 2.0,
    ) -> None:
        self._registry = registry
        self._local = local
        self._runs = runs
        self._enqueue = enqueue
        self._runners = runners
        self._store = store
        self._poll_seconds = poll_seconds

    async def available(self, tool_id: str) -> bool:
        lane = self._registry.get(tool_id).lane
        if lane == DEFAULT_LANE:
            return await self._local.available(tool_id)
        cutoff = datetime.now(UTC) - timedelta(seconds=RUNNER_STALE_SECONDS)
        return any(
            r.revoked_at is None
            and lane in r.lanes
            and r.last_seen_at is not None
            and r.last_seen_at >= cutoff
            for r in await self._runners.list()
        )

    async def run(
        self,
        *,
        tool_id: str,
        params: Mapping[str, Any],
        inputs: Sequence[ToolInput],
        context: ToolRunContext,
    ) -> list[ToolOutput]:
        lane = self._registry.get(tool_id).lane
        if lane == DEFAULT_LANE:
            return await self._local.run(
                tool_id=tool_id, params=params, inputs=inputs, context=context
            )
        run_id = uuid.uuid4()
        prefix = tool_run_prefix(context.workspace_id, context.study_id, run_id)
        run = Run(
            id=run_id,
            kind=RunKind.TOOL,
            workspace_id=context.workspace_id,
            requested_by=context.requested_by,
            cache_key=compute_cache_key(
                tool=tool_id,
                study=str(context.study_id),
                position=context.position,
                nonce=str(uuid.uuid4()),
            ),
            params=ToolRunParams(
                tool_id=tool_id,
                lane=lane,
                study_id=context.study_id,
                position=context.position,
                params=dict(params),
                inputs=tuple(inputs),
                output_prefix=prefix,
            ).to_params(),
        )
        await self._runs.add(run)
        try:
            await self._enqueue.enqueue(run_id, lane=lane)
            while True:
                current = await self._runs.get_by_id(run_id)
                if current is None:
                    raise RuntimeError("The run was deleted.")
                if current.status is RunStatus.READY:
                    return await asyncio.to_thread(read_outputs, self._store, prefix)
                if current.status is RunStatus.FAILED:
                    raise RuntimeError(current.error_message or "The runner failed the step.")
                if current.status is RunStatus.CANCELLED:
                    raise RuntimeError("The run was cancelled.")
                await asyncio.sleep(self._poll_seconds)
        except asyncio.CancelledError:
            # The step's timeout or a shutdown: stop the run, or a runner picks it up
            # for a step nobody is waiting on.
            await self._end(run_id, None)
            raise
        except Exception as error:
            await self._end(run_id, f"{type(error).__name__}: {error}"[:500])
            raise

    async def _end(self, run_id: uuid.UUID, failure: str | None) -> None:
        """Cancel the run (or fail it, given a message) unless it has ended already."""
        try:
            run = await self._runs.get_by_id(run_id)
            if run is None:
                return
            if failure is None:
                run.cancel()
            else:
                run.fail(failure)
            await self._runs.update(run)
        except ConflictError:
            pass  # already ended: ready, failed or cancelled
        except Exception:
            logger.exception("Tool run %s could not be stopped", run_id)
