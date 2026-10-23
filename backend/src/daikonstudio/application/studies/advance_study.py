"""Advancing a study: run its steps in plan order from where the run starts, a step that has
run already included, until a gate, a failure, a stop the agent asks for, or the end; or run
one step alone, like a notebook cell. A step that ran again changed what every step after it
was made from, so a run never skips one because it is done.

`AdvanceStudy` answers at once and runs the steps on a task of its own, writing the
study after every change so a reader sees each step start and finish. After each step that
finishes or fails the agent looks at it, and the loop waits for that look: the one thing it
can do there is stop the run, and a stop that came after the next step had begun would be no
stop. One advance re-runs a failed step, and a step a restart left running.

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
from typing import TYPE_CHECKING, Literal

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.study_repository import StudyRepository
from daikonstudio.application.ports.tool_executor import ToolExecutor, ToolInput, ToolRunContext
from daikonstudio.application.studies.edits import _label
from daikonstudio.application.tools.manifest import DEFAULT_LANE
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.shared.errors import (
    ConflictError,
    DomainError,
    NotFoundError,
)
from daikonstudio.domain.studies.study import SlotValue, Step, StepStatus, Study, Turn

if TYPE_CHECKING:
    # `agent` imports `in_flight` from here, so this module cannot import it back at run time.
    from daikonstudio.application.studies.agent import ObserveStep

logger = logging.getLogger(__name__)

# Who asked for a run. Only the scientist's is written into the thread as a run.
StartedBy = Literal["scientist", "agent"]

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


# What a request that would change a study says while a run is under way in it.
RUNNING_MESSAGE = "A run is under way in this study. Wait for it to stop, then try again."


def in_flight(study_id: uuid.UUID) -> bool:
    return study_id in _in_flight


def _event(kind: str, say: str, about: uuid.UUID | None) -> Turn:
    return Turn(role="event", kind=kind, say=say, about=about, at=datetime.now(UTC))


def _reach(study: Study, until: uuid.UUID | None) -> int:
    """The last position a run may take: `until`'s, none at all (-1) if it has left the plan,
    or, with no `until`, any."""
    if until is None:
        return len(study.steps)
    return next((s.position for s in study.steps if s.id == until), -1)


def _ran(study: Study, start: uuid.UUID | None, until: uuid.UUID | None) -> str:
    """A run as the thread says it when it starts: from where, and where it is headed. Said as
    intent, because a gate, a failure or a pause can end it before it gets there."""
    if start is not None and start == until:
        return f"Started a run of {_label(study.step(start))}."
    first = study.next_step if start is None else study.step(start)
    begin = "the next step waiting" if first is None else f"step {first.position + 1}"
    if until is None:
        return f"Started a run from {begin} to the next stop."
    return f"Started a run from {begin} toward step {study.step(until).position + 1}."


class AdvanceStudy:
    def __init__(
        self,
        studies: StudyRepository,
        tools: ToolRegistry,
        executor: ToolExecutor,
        store: BlobStore,
        observe: ObserveStep,
    ) -> None:
        self._studies = studies
        self._tools = tools
        self._executor = executor
        self._store = store
        self._observe = observe

    async def __call__(
        self,
        study_id: uuid.UUID,
        auth: AuthContext | None = None,
        *,
        start: uuid.UUID | None = None,
        until: uuid.UUID | None = None,
        started_by: StartedBy = "scientist",
    ) -> Result[Study, DomainError]:
        """Run the steps from `start`, whatever its status, or else from the next one waiting,
        and on in plan order, steps already done included, to `until`, the next gate, a
        failure, a pause the agent asks for, or the end.

        `until` is a place in the plan: nothing beyond it runs, whether or not it has to run
        itself. Only a scientist's run is written into the thread as such; the agent's own
        turn already says what it ran.
        """
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        study = await self._studies.get(auth.workspace_id, study_id)
        if study is None or study.created_by != auth.user_id:
            return Failure(NotFoundError("Study", str(study_id)))
        known = {s.id for s in study.steps}
        unknown = next((s for s in (start, until) if s is not None and s not in known), None)
        if unknown is not None:
            return Failure(NotFoundError("Step", str(unknown)))
        if study.started_at is None:
            return Failure(ConflictError("This study has not started."))
        if in_flight(study_id) or (start is None and study.next_step is None):
            return Success(study)
        ran = (
            _event("ran", _ran(study, start, until), start) if started_by == "scientist" else None
        )
        task = asyncio.create_task(self._run(auth.workspace_id, study_id, start, until, ran))
        _running.add(task)
        _in_flight[study_id] = task

        def _forget(done: asyncio.Task[None]) -> None:
            _running.discard(done)
            if _in_flight.get(study_id) is done:
                del _in_flight[study_id]

        task.add_done_callback(_forget)
        return Success(study)

    async def _run(
        self,
        workspace_id: uuid.UUID,
        study_id: uuid.UUID,
        start: uuid.UUID | None,
        until: uuid.UUID | None,
        ran: Turn | None,
    ) -> None:
        # Written here, not before the task exists: nothing may be awaited between
        # `in_flight` answering no and this run being the one in flight.
        if ran is not None:
            await self._say(workspace_id, study_id, ran)
        step_id = start
        while True:
            # Afresh every time: the last step and its observation both wrote to it.
            study = await self._studies.get(workspace_id, study_id)
            if study is None:
                return
            last = _reach(study, until)
            step = (
                study.next_step
                if step_id is None
                else next((s for s in study.steps if s.id == step_id), None)
            )
            if step is None or step.position > last:
                return
            ended = await self._guarded(workspace_id, study, step)
            # On to the step after this one in the plan, done or not: it was made from what
            # this one has just replaced. Only while `until` is not behind it.
            following = study.step_at(step.position + 1)
            go_on = following is not None and following.position <= last
            if ended is not StepStatus.SKIPPED:  # a skipped step made nothing to look at
                go_on = go_on and ended is StepStatus.DONE and not step.gated
                paused = await self._observe(workspace_id, study_id, step.id, driving=go_on)
                if paused and go_on:  # a pause means something only where the run would go on
                    pause = _event("paused", f"Paused by the agent after {_label(step)}.", step.id)
                    await self._say(workspace_id, study_id, pause)
                    return
            if following is None or not go_on:
                return
            step_id = following.id

    async def _say(self, workspace_id: uuid.UUID, study_id: uuid.UUID, turn: Turn) -> None:
        """Add an event to the study as it stands now, after whatever the agent wrote.

        The event only records the run, so one that cannot be written is logged and the run
        goes on, as an observation that cannot be taken is.
        """
        try:
            study = await self._studies.get(workspace_id, study_id)
            if study is not None:
                study.say(turn)
                await self._studies.update(study)
        except Exception:
            logger.exception("Study %s could not record a %s event", study_id, turn.kind)

    async def _guarded(self, workspace_id: uuid.UUID, study: Study, step: Step) -> StepStatus:
        """`_step`, with whatever it raises recorded on the step; how the step ended."""
        try:
            return await self._step(workspace_id, study, step)
        except asyncio.CancelledError:
            # The API is shutting down mid-step; say so rather than leave it `running`.
            await self._record_failure(
                workspace_id, study.id, step.id, "Interrupted by a restart."
            )
            raise
        except Exception as error:
            logger.exception("Study %s step %s failed", study.id, step.position + 1)
            message = f"{type(error).__name__}: {error}"[:500]
            await self._record_failure(workspace_id, study.id, step.id, message)
            return StepStatus.FAILED

    async def _record_failure(
        self, workspace_id: uuid.UUID, study_id: uuid.UUID, step_id: uuid.UUID, message: str
    ) -> None:
        try:
            reloaded = await self._studies.get(workspace_id, study_id)
            if reloaded is not None:
                await self._fail(reloaded, step_id, message)
        except Exception:
            # The step stays `running`; the next advance re-runs it.
            logger.exception("Study %s could not record its failure", study_id)

    async def _step(self, workspace_id: uuid.UUID, study: Study, step: Step) -> StepStatus:
        """Run one step; how it ended."""
        manifest = self._tools.get(step.tool_id)
        if not await self._executor.available(step.tool_id):
            why = (
                "Not available on this server yet."
                if manifest.lane == DEFAULT_LANE
                else f"No runner is serving the {manifest.lane} lane."
            )
            if step.status is StepStatus.SKIPPED:
                return StepStatus.SKIPPED  # run alone after it was skipped: it already says why
            if step.status is not StepStatus.PENDING:
                await self._fail(study, step.id, why)
                return StepStatus.FAILED
            study.skip_step(step.id, why)
            await self._studies.update(study)
            return StepStatus.SKIPPED

        bound: dict[str, SlotValue] = {}
        for slot in manifest.inputs:
            value = study.latest(slot.kind.value, before=step.id)
            if value is not None:
                bound[slot.key] = value
            elif not slot.optional:
                kind = slot.kind.value.replace("_", " ")
                await self._fail(study, step.id, f"Needs {kind}, which nothing produced.")
                return StepStatus.FAILED

        study.start_step(step.id)
        await self._studies.update(study)
        inputs = [
            ToolInput(key=key, kind=value.kind, blob_key=value.blob_key)
            for key, value in bound.items()
        ]
        context = ToolRunContext(
            workspace_id=workspace_id,
            requested_by=study.created_by,
            study_id=study.id,
            step_id=step.id,
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
                await self._fail(study, step.id, f"Timed out after {minutes} minutes.")
                return StepStatus.FAILED
        values = [
            SlotValue(
                id=uuid.uuid4(),
                key=output.key,
                kind=output.kind,
                blob_key=output.blob_key,
                summary=output.summary,
                produced_by=step.id,
                created_at=datetime.now(UTC),
            )
            for output in outputs
        ]
        study.finish_step(step.id, values)
        await self._studies.update(study)
        return StepStatus.DONE

    async def _fail(self, study: Study, step_id: uuid.UUID, message: str) -> None:
        if study.step(step_id).status is not StepStatus.RUNNING:
            study.start_step(step_id)
        study.fail_step(step_id, message)
        await self._studies.update(study)
