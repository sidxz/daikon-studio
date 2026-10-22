"""The Study aggregate -- a saved analysis plan whose steps run one at a time.

Status is derived from the steps, never stored, so it cannot disagree with them.
A step that is `running` with nothing behind it was interrupted; `next_step`
returns it so it is re-run rather than waited on.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    produced_by: int | None  # step position; None when the scientist brought it
    asserted_by: uuid.UUID | None = None
    created_at: datetime


@dataclass(frozen=True, kw_only=True)
class Answered:
    """A question the planner asked before it planned, and what the scientist said."""

    ask: str
    answer: str  # "" when left unanswered, "Not sure" when they said so


@dataclass(frozen=True, kw_only=True)
class Pointer:
    """One claim an agent makes about a result, as a reference into it.

    A judgement step may produce prose, choices and proposed steps, never
    numbers. A pointer is how a choice is expressed without breaking that: it
    names a candidate and what the agent believes it is, and `cites` carries the
    fields it read, copied from the result rather than restated. Anything here
    was checked against the step's own output before it was stored, so a reader
    can follow every part of it back.
    """

    at: str
    label: str
    evidence: str
    cites: tuple[str, ...] = ()


@dataclass(kw_only=True)
class Proposed:
    """Something an agent asked the study to do, waiting on the scientist.

    It carries its own id because it is acted on later, from a button in a
    conversation, long after the turn that made it. `applied_at` is what stops
    the same proposal being applied twice by two presses.
    """

    id: uuid.UUID
    does: str  # "rerun" or "add"
    step: int | None
    tool_id: str | None
    settings: dict[str, Any]
    why: str
    applied_at: datetime | None = None


@dataclass(frozen=True, kw_only=True)
class Turn:
    """One exchange in a stage's conversation.

    `dropped` is kept rather than hidden: it counts the claims the agent made
    that its own step's result did not support, and a turn that drops several is
    worth less than one that drops none.
    """

    role: str  # "agent" or "scientist"
    say: str
    points: tuple[Pointer, ...] = ()
    proposes: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    proposals: tuple[Proposed, ...] = ()
    dropped: int = 0
    written_by: str | None = None
    at: datetime


@dataclass(kw_only=True)
class Thread:
    """A stage's conversation, and the one paragraph that leaves it.

    `handoff` is what a study-wide agent reads instead of the whole exchange:
    what this stage settled, on what evidence, and what it could not. Kept on the
    thread so coordination never means replaying every turn of every stage.
    """

    position: int | None
    turns: list[Turn]
    handoff: str = ""


@dataclass(kw_only=True)
class Step:
    position: int
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
        planning: Mapping[str, Any] | None = None,
        conversations: Sequence[Thread] = (),
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
        # The plan it was started from (questions, understanding, what was left
        # out, who wrote it), kept as the planner returned it. Read-only after start.
        self.planning: dict[str, Any] = dict(planning or {})
        # One per stage, created when something is first said about that step.
        self.conversations: list[Thread] = list(conversations)

    def proposal(self, proposal_id: uuid.UUID) -> Proposed | None:
        return next(
            (
                p
                for thread in self.conversations
                for turn in thread.turns
                for p in turn.proposals
                if p.id == proposal_id
            ),
            None,
        )

    def add_step(
        self, *, tool_id: str, tool_version: str, name: str, purpose: str, params: dict[str, Any]
    ) -> Step:
        """Put a step at the end of the plan.

        At the end, never in the middle: a step inserted among the others would
        renumber them, and every conversation, value and record that already
        names a position would quietly come to mean something else.
        """
        step = Step(
            position=len(self.steps),
            tool_id=tool_id,
            tool_version=tool_version,
            name=name,
            purpose=purpose,
            params=params,
            gated=False,
            always=False,
            expanded_from=None,
        )
        self.steps.append(step)
        self._touch()
        return step

    def thread(self, position: int | None) -> Thread | None:
        """A stage's conversation, or the study's own when `position` is None."""
        return next((t for t in self.conversations if t.position == position), None)

    def say(self, position: int | None, turn: Turn) -> Thread:
        """Add a turn to a conversation, starting it if nothing was said yet.

        `position` None is the study's own thread, which is about the plan and
        where the steps have got to rather than about any one result.
        """
        if position is not None and not any(s.position == position for s in self.steps):
            raise ConflictError(f"This study has no step {position}.")
        thread = self.thread(position)
        if thread is None:
            thread = Thread(position=position, turns=[])
            self.conversations.append(thread)
        thread.turns.append(turn)
        self._touch()
        return thread

    @property
    def status(self) -> StudyStatus:
        if any(s.status is StepStatus.FAILED for s in self.steps):
            return StudyStatus.FAILED
        if all(s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in self.steps):
            return StudyStatus.DONE
        return StudyStatus.OPEN

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

    def latest(self, kind: str, before: int | None = None) -> SlotValue | None:
        """The newest value of a kind, optionally only from earlier in the study.

        A step whose output is the same kind as its input — preparing a
        structure into a structure — would otherwise be handed its own last
        output when it is run a second time, and re-prepare the prepared file
        rather than the deposited one. Values the scientist brought have no
        position and are always in scope.
        """
        of_kind = [
            v
            for v in self.values
            if v.kind == kind
            and (before is None or v.produced_by is None or v.produced_by < before)
        ]
        return max(of_kind, key=lambda v: v.created_at, default=None)

    def outputs_of(self, position: int) -> list[SlotValue]:
        return [v for v in self.values if v.produced_by == position]

    def current_outputs_of(self, position: int) -> list[SlotValue]:
        """The newest value per output key, as a notebook cell shows only its
        latest output. Every earlier value stays in `values`; this is a view."""
        newest: dict[str, SlotValue] = {}
        for value in sorted(self.outputs_of(position), key=lambda v: v.created_at):
            newest[value.key] = value
        return list(newest.values())

    def bring(self, value: SlotValue) -> None:
        if value.produced_by is not None:
            raise ConflictError("A value the scientist brings cannot name a step that made it.")
        self.values.append(value)
        self._touch()

    def start_step(self, position: int) -> Step:
        """Any step can be (re)started: a scientist runs one step at a time, like a cell.

        ponytail: re-running a step does not invalidate later steps; forking is the
        feature that decides what happens to them. Its earlier values stay, and
        `latest` picks the newest.
        """
        step = self._step(position)
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now(UTC)
        step.error = None
        step.finished_at = None
        self._touch()
        return step

    def finish_step(self, position: int, values: Sequence[SlotValue]) -> None:
        step = self._step(position)
        self._require_running(step)
        if any(v.produced_by != position for v in values):
            raise ConflictError(f"Every value of step {position} must name step {position}.")
        self.values.extend(values)
        step.status = StepStatus.DONE
        step.finished_at = datetime.now(UTC)
        self._touch()

    def fail_step(self, position: int, message: str) -> None:
        step = self._step(position)
        self._require_running(step)
        step.status = StepStatus.FAILED
        step.error = message
        step.finished_at = datetime.now(UTC)
        self._touch()

    def skip_step(self, position: int, why: str) -> None:
        step = self._step(position)
        if step.status is not StepStatus.PENDING:
            raise ConflictError(f"Step {position} is {step.status.value} and cannot be skipped.")
        step.status = StepStatus.SKIPPED
        step.error = why
        self._touch()

    def _step(self, position: int) -> Step:
        for step in self.steps:
            if step.position == position:
                return step
        raise ConflictError(f"This study has no step {position}.")

    @staticmethod
    def _require_running(step: Step) -> None:
        if step.status is not StepStatus.RUNNING:
            raise ConflictError(f"Step {step.position} is {step.status.value}, not running.")

    def _touch(self) -> None:
        self.updated_at = datetime.now(UTC)
