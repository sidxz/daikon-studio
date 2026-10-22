"""Advancing a study: run its steps in order until a gate, a failure, or the end;
or run one step alone, like a notebook cell.

`AdvanceStudy` answers at once and runs the steps on a task of its own, writing the
study after every change so a reader sees each step start and finish. One advance
re-runs a failed step, and a step a restart left running.

ponytail: the run lives inside the API process, so a restart drops it mid-step; the
step is recorded as failed ("Interrupted by a restart.") when the shutdown cancels it
cleanly, stays `running` when the process dies outright, and the next advance re-runs
it either way. Move it to a runner job if studies ever need to survive a deploy.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
import weakref
from datetime import UTC, datetime

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.study_repository import StudyRepository
from daikonstudio.application.ports.tool_executor import ToolExecutor, ToolInput, ToolRunContext
from daikonstudio.application.tools.manifest import DEFAULT_LANE
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.shared.errors import DomainError, NotFoundError, ValidationError
from daikonstudio.domain.studies.study import SlotValue, Step, StepStatus, Study

logger = logging.getLogger(__name__)

# asyncio holds only weak references to tasks: without these, a run could be
# garbage-collected mid-step.
_running: set[asyncio.Task[None]] = set()
# ponytail: single API replica assumed; with several, a duplicate advance from another
# replica ends in ConcurrencyConflictError on the step, not a double record -- derive
# `running` from the row (started_at within a bound) when a second replica exists.
_in_flight: dict[uuid.UUID, asyncio.Task[None]] = {}

# A tool holds the one tool lock while it runs, so a hung one would stall every study.
TOOL_TIMEOUT_SECONDS: float = 60 * 60

# Per loop, for the reason `get_dataset_profile._COMPUTE_LOCKS` gives.
_TOOL_LOCKS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
    weakref.WeakKeyDictionary()
)


def _tool_lock() -> asyncio.Lock:
    """ponytail: one child at a time per process; a semaphore if studies queue."""
    loop = asyncio.get_running_loop()
    lock = _TOOL_LOCKS.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _TOOL_LOCKS[loop] = lock
    return lock


def in_flight(study_id: uuid.UUID) -> bool:
    return study_id in _in_flight


class AdvanceStudy:
    def __init__(
        self,
        studies: StudyRepository,
        tools: ToolRegistry,
        executor: ToolExecutor,
        store: BlobStore,
    ) -> None:
        self._studies = studies
        self._tools = tools
        self._executor = executor
        self._store = store

    async def __call__(
        self,
        study_id: uuid.UUID,
        auth: AuthContext | None = None,
        *,
        position: int | None = None,
    ) -> Result[Study, DomainError]:
        """Run the steps from the next one to a gate; or, given a position, that one
        step alone, whatever its gate or status says."""
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        study = await self._studies.get(auth.workspace_id, study_id)
        if study is None or study.created_by != auth.user_id:
            return Failure(NotFoundError("Study", str(study_id)))
        if position is not None and all(s.position != position for s in study.steps):
            return Failure(ValidationError(f"No step {position} in this study."))
        if in_flight(study_id) or (position is None and study.next_step is None):
            return Success(study)
        run = (
            self._run(auth.workspace_id, study_id)
            if position is None
            else self._run_one(auth.workspace_id, study_id, position)
        )
        task = asyncio.create_task(run)
        _running.add(task)
        _in_flight[study_id] = task

        def _forget(done: asyncio.Task[None]) -> None:
            _running.discard(done)
            if _in_flight.get(study_id) is done:
                del _in_flight[study_id]

        task.add_done_callback(_forget)
        return Success(study)

    async def _run(self, workspace_id: uuid.UUID, study_id: uuid.UUID) -> None:
        while True:
            study = await self._studies.get(workspace_id, study_id)
            step = study.next_step if study else None
            if study is None or step is None:
                return
            if not await self._guarded(workspace_id, study, step):
                return

    async def _run_one(self, workspace_id: uuid.UUID, study_id: uuid.UUID, position: int) -> None:
        study = await self._studies.get(workspace_id, study_id)
        step = next((s for s in study.steps if s.position == position), None) if study else None
        if study is not None and step is not None:
            await self._guarded(workspace_id, study, step)

    async def _guarded(self, workspace_id: uuid.UUID, study: Study, step: Step) -> bool:
        """`_step`, with whatever it raises recorded on the step; whether to go on."""
        try:
            return await self._step(workspace_id, study, step)
        except asyncio.CancelledError:
            # The API is shutting down mid-step; say so rather than leave it `running`.
            await self._record_failure(
                workspace_id, study.id, step.position, "Interrupted by a restart."
            )
            raise
        except Exception as error:
            logger.exception("Study %s step %s failed", study.id, step.position)
            message = f"{type(error).__name__}: {error}"[:500]
            await self._record_failure(workspace_id, study.id, step.position, message)
            return False

    async def _record_failure(
        self, workspace_id: uuid.UUID, study_id: uuid.UUID, position: int, message: str
    ) -> None:
        try:
            reloaded = await self._studies.get(workspace_id, study_id)
            if reloaded is not None:
                await self._fail(reloaded, position, message)
        except Exception:
            # The step stays `running`; the next advance re-runs it.
            logger.exception("Study %s could not record its failure", study_id)

    async def _step(self, workspace_id: uuid.UUID, study: Study, step: Step) -> bool:
        """Run one step; whether the loop goes on to the next."""
        position = step.position
        manifest = self._tools.get(step.tool_id)
        if not await self._executor.available(step.tool_id):
            why = (
                "Not available on this server yet."
                if manifest.lane == DEFAULT_LANE
                else f"No runner is serving the {manifest.lane} lane."
            )
            if step.status is StepStatus.SKIPPED:
                return True  # run alone after it was skipped: it already says why
            if step.status is not StepStatus.PENDING:
                await self._fail(study, position, why)
                return False
            study.skip_step(position, why)
            await self._studies.update(study)
            return True

        bound: dict[str, SlotValue] = {}
        for slot in manifest.inputs:
            value = study.latest(slot.kind.value, before=position)
            if value is not None:
                bound[slot.key] = value
            elif not slot.optional:
                kind = slot.kind.value.replace("_", " ")
                await self._fail(study, position, f"Needs {kind}, which nothing produced.")
                return False

        study.start_step(position)
        await self._studies.update(study)
        inputs = [
            ToolInput(key=key, kind=value.kind, blob_key=value.blob_key)
            for key, value in bound.items()
        ]
        context = ToolRunContext(
            workspace_id=workspace_id,
            requested_by=study.created_by,
            study_id=study.id,
            position=position,
            output_prefix=f"{workspace_id}/studies/{study.id}/values/",
        )
        # A lane step waits on a runner, not on this process, so it holds no lock.
        lock = _tool_lock() if manifest.lane == DEFAULT_LANE else contextlib.nullcontext()
        async with lock:
            # Made inside the lock: the deadline is fixed when the timer is created.
            timer = asyncio.timeout(TOOL_TIMEOUT_SECONDS)
            try:
                async with timer:
                    outputs = await self._executor.run(
                        tool_id=step.tool_id, params=step.params, inputs=inputs, context=context
                    )
            except TimeoutError:
                if not timer.expired():
                    raise  # the tool's own timeout, reported as it said it
                minutes = f"{TOOL_TIMEOUT_SECONDS / 60:g}"
                await self._fail(study, position, f"Timed out after {minutes} minutes.")
                return False
        values = [
            SlotValue(
                id=uuid.uuid4(),
                key=output.key,
                kind=output.kind,
                blob_key=output.blob_key,
                summary=output.summary,
                produced_by=position,
                created_at=datetime.now(UTC),
            )
            for output in outputs
        ]
        study.finish_step(position, values)
        await self._studies.update(study)
        return not step.gated

    async def _fail(self, study: Study, position: int, message: str) -> None:
        step = next(s for s in study.steps if s.position == position)
        if step.status is not StepStatus.RUNNING:
            study.start_step(position)
        study.fail_step(position, message)
        await self._studies.update(study)
