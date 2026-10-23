"""The Study aggregate -- a plan drafted in conversation, then run one step at a time.

A study is a draft from the scientist's first message until `start()` stamps
`started_at`: it holds a plan, files and the one conversation, and nothing runs.
Status is derived from `started_at` and the steps, never stored, so it cannot
disagree with them. A step that is `running` with nothing behind it was
interrupted; `next_step` returns it so it is re-run rather than waited on.

A step is named by its id. `position` is only its place in the order, renumbered
after an insert or a removal, so a value, a turn or a pointer that names a step
keeps meaning that step however the plan around it is edited.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from daikonstudio.domain.shared.entity import AggregateRoot
from daikonstudio.domain.shared.errors import ConflictError


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class StudyStatus(StrEnum):
    DRAFT = "draft"
    OPEN = "open"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True, kw_only=True)
class SlotValue:
    id: uuid.UUID
    key: str
    kind: str
    blob_key: str
    summary: dict[str, Any]
    produced_by: uuid.UUID | None  # a step id; None when the scientist brought it
    asserted_by: uuid.UUID | None = None  # the scientist who approved it, else None
    created_at: datetime


@dataclass(frozen=True, kw_only=True)
class Answered:
    """A question the planner asked before it planned, and what the scientist said."""

    ask: str
    answer: str  # "" when left unanswered, "Not sure" when they said so


@dataclass(frozen=True, kw_only=True)
class Asked:
    """One question the agent put to the scientist before it planned.

    What the scientist said back is an `Answered`, kept on the study.
    """

    ask: str
    why: str
    options: tuple[str, ...] = ()
    default: str | None = None


@dataclass(frozen=True, kw_only=True)
class Pointer:
    """One claim an agent makes about a result, as a reference into it.

    A judgement step may produce prose, choices and proposed steps, never
    numbers. A pointer is how a choice is expressed without breaking that: it
    names a candidate and what the agent believes it is, and `cites` carries the
    fields it read, copied from the result rather than restated. Anything here
    was checked against the step's own output before it was stored, so a reader
    can follow every part of it back. `step` is the step whose result it is
    about; a reader leaves it empty and the agent fills it in.
    """

    step: uuid.UUID | None
    at: str
    label: str
    evidence: str
    cites: tuple[str, ...] = ()


@dataclass(kw_only=True)
class Proposed:
    """Something an agent asked the study to do, waiting on the scientist.

    `action` is the change as plain data, every step in it named by id, so a
    button pressed after the plan was edited still means the same step. It
    carries its own id because it is acted on later, from a button in the
    conversation, long after the turn that made it. `applied_at` is what stops
    the same proposal being applied twice by two presses.
    """

    id: uuid.UUID
    action: dict[str, Any]
    why: str
    applied_at: datetime | None = None


@dataclass(frozen=True, kw_only=True)
class Acted:
    """Something the agent did on the scientist's word, and the words it relied on.

    `quoting` is the part of the scientist's message the action carries out, so a
    reader can see what the agent was told and that it did no more.
    """

    does: str
    step: uuid.UUID | None
    detail: str
    quoting: str


@dataclass(frozen=True, kw_only=True)
class Turn:
    """One entry in a study's conversation.

    `role` is "scientist", "agent" or "event" (something the study itself did,
    written by the services and never by the aggregate). An agent turn's `kind`
    is asking, planning, observe or reply; an event's is started, ran, swapped,
    approved, applied, attached, paused, edited, run_refused or observe_failed.
    `about` is the step a scientist's message is about, if any.

    `dropped` is kept rather than hidden: it counts the claims the agent made
    that its own step's result did not support, and a turn that drops several is
    worth less than one that drops none. `silent` marks an observation with
    nothing to say: kept as a tick on the step, left out of the rail.
    """

    role: str
    kind: str = ""
    say: str
    about: uuid.UUID | None = None
    questions: tuple[Asked, ...] = ()
    points: tuple[Pointer, ...] = ()
    proposals: tuple[Proposed, ...] = ()
    unresolved: tuple[str, ...] = ()
    actions: tuple[Acted, ...] = ()
    looked: tuple[str, ...] = ()
    dropped: int = 0
    silent: bool = False
    written_by: str | None = None
    at: datetime


@dataclass(kw_only=True)
class Thread:
    """A study's one conversation, and the paragraph that stands in for its old turns.

    `summary` is what someone opening the study cold needs to know, rewritten by
    the agent every turn. Context is assembled from it in place of the turns that
    have scrolled out of the window.
    """

    turns: list[Turn] = field(default_factory=list)
    summary: str = ""


@dataclass(kw_only=True)
class Step:
    id: uuid.UUID
    position: int  # 0-based order in the plan, renumbered after an insert or a removal
    tool_id: str
    tool_version: str
    name: str
    purpose: str
    params: dict[str, Any]
    gated: bool
    always: bool
    expanded_from: str | None
    status: StepStatus = StepStatus.PENDING
    error: str | None = None
    problem: str | None = None  # why the plan cannot go ahead as it stands; blocks start()
    started_at: datetime | None = None
    finished_at: datetime | None = None


class Study(AggregateRoot):
    def __init__(
        self,
        *,
        workspace_id: uuid.UUID,
        created_by: uuid.UUID,
        goal: str,
        restated_goal: str,
        checks: tuple[str, ...],
        stop_conditions: tuple[str, ...],
        steps: list[Step],
        values: list[SlotValue] | None = None,
        answers: Sequence[Answered] = (),
        thread: Thread | None = None,
        started_at: datetime | None = None,
        mode: str = "run",
        amended_after_steps: int | None = None,
        id: uuid.UUID | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        version: int = 1,
    ) -> None:
        super().__init__(id=id, created_at=created_at, updated_at=updated_at, version=version)
        self.workspace_id = workspace_id
        self.created_by = created_by
        self.goal = goal
        self.restated_goal = restated_goal
        self.checks = checks
        self.stop_conditions = stop_conditions
        self.steps = steps
        self.values: list[SlotValue] = values if values is not None else []
        self.answers = tuple(answers)
        self.thread = thread if thread is not None else Thread()
        # None while a draft; Start sets it. What the steps may do hangs on this.
        self.started_at = started_at
        # "run" is the only mode: the run loop pauses at a gate or on the agent's word.
        self.mode = mode
        # How many steps had run when the checks or stop conditions were last
        # changed after Start. None if they never were.
        self.amended_after_steps = amended_after_steps

    @property
    def status(self) -> StudyStatus:
        if self.started_at is None:
            return StudyStatus.DRAFT
        if any(s.status is StepStatus.FAILED for s in self.steps):
            return StudyStatus.FAILED
        if all(s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in self.steps):
            return StudyStatus.DONE
        return StudyStatus.OPEN

    @property
    def phase(self) -> str:
        """Which turn the agent owes next, and so which schema it answers in.

        A draft is asking until the agent has said anything, then planning;
        a started study is running.
        """
        if self.started_at is not None:
            return "running"
        return "planning" if any(t.role == "agent" for t in self.thread.turns) else "asking"

    @property
    def done_count(self) -> int:
        """How many steps have run to a result. A skipped step never ran."""
        return sum(s.status is StepStatus.DONE for s in self.steps)

    @property
    def next_step(self) -> Step | None:
        # A failed step is next too: the same button that runs a plan re-runs
        # it from where it broke, and a step left running by a restart is
        # re-run rather than waited on.
        return next(
            (
                s
                for s in self.steps
                if s.status in (StepStatus.PENDING, StepStatus.RUNNING, StepStatus.FAILED)
            ),
            None,
        )

    def step(self, step_id: uuid.UUID) -> Step:
        return self.steps[self._index(step_id)]

    def step_at(self, position: int) -> Step | None:
        """The step at a stored, 0-based position; the model and error labels speak 1-based."""
        return next((s for s in self.steps if s.position == position), None)

    def say(self, turn: Turn) -> Thread:
        """Add a turn to the study's one conversation."""
        self.thread.turns.append(turn)
        self._touch()
        return self.thread

    def proposal(self, proposal_id: uuid.UUID) -> Proposed | None:
        return next(
            (p for turn in self.thread.turns for p in turn.proposals if p.id == proposal_id),
            None,
        )

    def start(self) -> None:
        """End the draft: from here the steps can run.

        A plan that still has a problem is not started on the hope that it will
        work out; the problem is named so it can be fixed first.
        """
        if self.started_at is not None:
            raise ConflictError("This study has already started.")
        if not self.steps:
            raise ConflictError("This study has no steps to run.")
        problem = next((s.problem for s in self.steps if s.problem), None)
        if problem is not None:
            raise ConflictError(f"The plan cannot start yet: {problem}")
        self.started_at = datetime.now(UTC)
        self._touch()

    def replace_steps(self, steps: list[Step]) -> None:
        """Put a whole different plan in a draft's place, as the planner writes one.

        Once started the plan changes one step at a time, so what has run is
        never swept away with it.
        """
        if self.started_at is not None:
            raise ConflictError("A study that has started changes its plan one step at a time.")
        self.steps = steps
        self._renumber()
        self._touch()

    def insert_step(self, step: Step, before: uuid.UUID | None) -> Step:
        """Put a step in the plan: before the step `before` names, or at the end."""
        self.steps.insert(len(self.steps) if before is None else self._index(before), step)
        self._renumber()
        self._touch()
        return step

    def remove_step(self, step_id: uuid.UUID) -> None:
        """Take a step out of the plan, if it has not run.

        A done step is the record of what was found and stays, and a running one
        is in the middle of finding it.
        """
        index = self._index(step_id)
        step = self.steps[index]
        if step.status not in (StepStatus.PENDING, StepStatus.SKIPPED, StepStatus.FAILED):
            raise ConflictError(
                f"{self._label(step)} is {step.status.value} and cannot be removed."
            )
        del self.steps[index]
        self._renumber()
        self._touch()

    def set_tool(
        self,
        step_id: uuid.UUID,
        *,
        tool_id: str,
        tool_version: str,
        name: str,
        params: dict[str, Any],
    ) -> Step:
        """Point a step at another tool, and leave it waiting to run again.

        Its old error and problem were the old tool's, so they go with it.
        """
        step = self.step(step_id)
        step.tool_id = tool_id
        step.tool_version = tool_version
        step.name = name
        step.params = params
        step.status = StepStatus.PENDING
        step.error = None
        step.problem = None
        self._touch()
        return step

    def set_params(self, step_id: uuid.UUID, params: dict[str, Any]) -> Step:
        """Replace a step's settings. Back to pending, so it reads as waiting to run
        rather than as done with a result that no longer matches the settings beside it.
        The last error goes too, as it described the old settings.

        `problem` stays: only the plan check knows whether the new settings fix it.
        """
        step = self.step(step_id)
        step.params = params
        step.status = StepStatus.PENDING
        step.error = None
        self._touch()
        return step

    def set_gate(self, step_id: uuid.UUID, gated: bool) -> Step:
        step = self.step(step_id)
        step.gated = gated
        self._touch()
        return step

    def amend_terms(
        self, checks: tuple[str, ...] | None, stop_conditions: tuple[str, ...] | None
    ) -> None:
        """Change the checks or the stop conditions, whichever is given.

        Once started, the change is marked with how many steps had run, so the
        terms never read as if they were written before the data.
        """
        if checks is None and stop_conditions is None:
            return  # nothing changed, so nothing to mark
        if checks is not None:
            self.checks = checks
        if stop_conditions is not None:
            self.stop_conditions = stop_conditions
        if self.started_at is not None:
            self.amended_after_steps = self.done_count
        self._touch()

    def latest(self, kind: str, before: uuid.UUID | None = None) -> SlotValue | None:
        """The newest value of a kind, optionally only from earlier in the plan.

        A step whose output is the same kind as its input — preparing a
        structure into a structure — would otherwise be handed its own last
        output when it is run a second time, and re-prepare the prepared file
        rather than the deposited one. `before` is a step id, read through that
        step's position because ids carry no order. With it, values the scientist
        brought are always in scope and those of a step since removed from the
        plan never are. A choice made on a result its step has since replaced is
        never in scope (`superseded`).
        """
        bound = None if before is None else self.step(before).position
        positions = {s.id: s.position for s in self.steps}
        of_kind = [
            v
            for v in self.values
            if v.kind == kind
            and (
                bound is None
                or v.produced_by is None
                or (v.produced_by in positions and positions[v.produced_by] < bound)
            )
            and not self.superseded(v)
        ]
        return max(of_kind, key=lambda v: v.created_at, default=None)

    def superseded(self, value: SlotValue) -> bool:
        """Whether this is the scientist's choice made on a result its step has since replaced.

        A choice names its candidate by rank, and a step run again ranks a different list, so
        the same rank may now be another candidate. `from` in its summary is the result it was
        made on; a choice stored without one is taken as standing.
        """
        made_on = value.summary.get("from")
        return (
            value.asserted_by is not None
            and value.produced_by is not None
            and made_on is not None
            and made_on not in {str(v.id) for v in self.current_outputs_of(value.produced_by)}
        )

    def outputs_of(self, step_id: uuid.UUID) -> list[SlotValue]:
        return [v for v in self.values if v.produced_by == step_id]

    def current_outputs_of(self, step_id: uuid.UUID) -> list[SlotValue]:
        """The newest value per output key, as a notebook cell shows only its
        latest output. Every earlier value stays in `values`; this is a view."""
        newest: dict[str, SlotValue] = {}
        for value in sorted(self.outputs_of(step_id), key=lambda v: v.created_at):
            newest[value.key] = value
        return list(newest.values())

    def bring(self, value: SlotValue) -> None:
        if value.produced_by is not None:
            raise ConflictError("A value the scientist brings cannot name a step that made it.")
        self.values.append(value)
        self._touch()

    def approve(self, step_id: uuid.UUID, value: SlotValue) -> None:
        """Record the scientist's choice of a candidate a step reported.

        The one place a computed result becomes an asserted one, and the value
        names who asserted it. Approving again adds a newer value and `latest`
        picks it; steps that already ran against the earlier one are not
        invalidated.
        """
        self.step(step_id)
        if value.asserted_by is None or value.produced_by != step_id:
            raise ConflictError(
                "An approved value must name who chose it and the step it settles."
            )
        self.values.append(value)
        self._touch()

    def start_step(self, step_id: uuid.UUID) -> Step:
        """Any step can be (re)started: a scientist runs one step at a time, like a cell.

        ponytail: re-running a step does not invalidate later steps; forking is the
        feature that decides what happens to them. Its earlier values stay, and
        `latest` picks the newest.
        """
        step = self.step(step_id)
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now(UTC)
        step.error = None
        step.finished_at = None
        self._touch()
        return step

    def finish_step(self, step_id: uuid.UUID, values: Sequence[SlotValue]) -> None:
        step = self.step(step_id)
        self._require_running(step)
        if any(v.produced_by != step_id for v in values):
            raise ConflictError(f"Every value of {self._label(step)} must name that step.")
        self.values.extend(values)
        step.status = StepStatus.DONE
        step.finished_at = datetime.now(UTC)
        self._touch()

    def fail_step(self, step_id: uuid.UUID, message: str) -> None:
        step = self.step(step_id)
        self._require_running(step)
        step.status = StepStatus.FAILED
        step.error = message
        step.finished_at = datetime.now(UTC)
        self._touch()

    def skip_step(self, step_id: uuid.UUID, why: str) -> None:
        step = self.step(step_id)
        if step.status is not StepStatus.PENDING:
            raise ConflictError(
                f"{self._label(step)} is {step.status.value} and cannot be skipped."
            )
        step.status = StepStatus.SKIPPED
        step.error = why
        self._touch()

    def _index(self, step_id: uuid.UUID) -> int:
        for index, step in enumerate(self.steps):
            if step.id == step_id:
                return index
        raise ConflictError("This study has no such step.")

    def _renumber(self) -> None:
        for i, s in enumerate(self.steps):
            s.position = i

    @staticmethod
    def _label(step: Step) -> str:
        # Stored positions are 0-based; people read the plan from 1.
        return f"Step {step.position + 1} ({step.name})"

    @classmethod
    def _require_running(cls, step: Step) -> None:
        if step.status is not StepStatus.RUNNING:
            raise ConflictError(f"{cls._label(step)} is {step.status.value}, not running.")

    def _touch(self) -> None:
        self.updated_at = datetime.now(UTC)
