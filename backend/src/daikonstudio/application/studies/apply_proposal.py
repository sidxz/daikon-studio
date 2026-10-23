"""Carrying out a proposal, once the scientist has pressed its button.

The press is the scientist's word, so a proposal is carried out by the same edits the agent
makes on a quoted instruction, with nothing left to quote. It is rebuilt as the action it
stands for, against the plan as it is now, and the edits check it as they check any other.
That is why a proposal keeps its steps by id: a button pressed after the plan was edited still
means the step it meant, and one whose step has left the plan, or whose change the edits would
no longer make, is refused and left unapplied rather than made to mean something else.

An edit changes the plan and stops there, leaving the step for the play control every other
step has, so a confirmation never turns into a long job the scientist did not know they were
starting. A proposal to run is the exception, and its button says what it runs: it starts the
loop, and the event it leaves stands in for the one a run the scientist starts directly leaves,
so a single press is never recorded twice.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.study_repository import StudyRepository
from daikonstudio.application.studies.advance_study import RUNNING_MESSAGE, _ran, in_flight
from daikonstudio.application.studies.agent import Advance
from daikonstudio.application.studies.approve import approve_candidate
from daikonstudio.application.studies.edits import _label, apply_actions
from daikonstudio.application.studies.get_study import GetStudy
from daikonstudio.application.studies.schemas import Action, Setting
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.shared.errors import (
    ConflictError,
    DomainError,
    NotFoundError,
    ValidationError,
)
from daikonstudio.domain.studies.study import Acted, Study, Turn


@dataclass(frozen=True)
class ApplyCommand:
    study_id: uuid.UUID
    proposal_id: uuid.UUID


def _rebuild(plain: dict[str, Any], study: Study) -> Action | None:
    """The proposal as the action it stands for, each step id back at its place in the plan as
    it is now, counted from 1; None when a step it names has left the plan."""
    places = {str(step.id): step.position + 1 for step in study.steps}
    refs = [plain["step_id"], plain["before_id"], plain["until_id"]]
    if any(ref is not None and ref not in places for ref in refs):
        return None
    step, before, until = (None if ref is None else places[ref] for ref in refs)
    return Action(
        does=plain["does"],
        step=step,
        tool_id=plain["tool_id"],
        before=before,
        purpose=plain["purpose"],
        settings=[Setting(name=name, value=value) for name, value in plain["settings"].items()],
        gated=plain["gated"],
        candidate=plain["candidate"],
        until=until,
    )


def _step_id(study: Study, place: int | None) -> uuid.UUID | None:
    """The step at a place in the plan, counted from 1; None where no place is named."""
    step = None if place is None else study.step_at(place - 1)
    return None if step is None else step.id


class ApplyProposal:
    def __init__(
        self, studies: StudyRepository, tools: ToolRegistry, store: BlobStore, advance: Advance
    ) -> None:
        self._studies = studies
        self._get = GetStudy(studies)
        self._tools = tools
        self._store = store
        self._advance = advance

    async def __call__(
        self, command: ApplyCommand, auth: AuthContext | None = None
    ) -> Result[Study, DomainError]:
        """Make the change a proposal names, if the plan still allows it.

        Nothing is changed while a run is under way: the run writes the same row. What can no
        longer be done is refused and the proposal left unapplied, so a press that did nothing
        is never recorded as having done something.
        """
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        found = await self._get(command.study_id, auth)
        if not isinstance(found, Success):
            return found
        study = found.unwrap()
        if in_flight(study.id):
            return Failure(ConflictError(RUNNING_MESSAGE))
        proposal = study.proposal(command.proposal_id)
        if proposal is None:
            return Failure(NotFoundError("Proposal", str(command.proposal_id)))
        if proposal.applied_at is not None:
            return Failure(ConflictError("This change has already been made."))

        action = _rebuild(proposal.action, study)
        # With no message there is nothing to quote: the press is the scientist's word.
        applied = (
            None if action is None else apply_actions(study, [action], self._tools, message=None)
        )
        if applied is None or applied.dropped:
            return Failure(
                ValidationError(
                    "This change can no longer be made.",
                    detail="The plan has changed since it was proposed.",
                )
            )

        acted = list(applied.acted)
        for step_id, candidate in applied.approvals:
            try:
                value = await approve_candidate(
                    study,
                    step_id,
                    candidate,
                    actor=auth.user_id,
                    workspace_id=study.workspace_id,
                    registry=self._tools,
                    store=self._store,
                )
            except ValidationError as error:
                return Failure(error)  # no finished result to choose from, or no such rank
            study.approve(step_id, value)
            chose = f"Chose candidate {candidate} of {_label(study.step(step_id))}."
            acted.append(Acted(does="approve", step=step_id, detail=chose, quoting=""))
        run = applied.run
        start: uuid.UUID | None = None
        until: uuid.UUID | None = None
        if run is not None:
            if run.step is None and study.next_step is None:
                return Failure(ValidationError("There is nothing left to run."))
            start, until = _step_id(study, run.step), _step_id(study, run.until)
            detail = _ran(study, start, until)
            acted.append(Acted(does="run", step=start, detail=detail, quoting=""))

        (done,) = acted  # one action makes one change
        now = datetime.now(UTC)
        event = Turn(
            role="event", kind="applied", say=f"Applied: {done.detail}", about=done.step, at=now
        )
        study.say(event)
        proposal.applied_at = now
        await self._studies.update(study)
        if run is not None:
            # After the save: the loop reads the study from storage. The event above is the
            # record of the run, so the loop is told not to write its own.
            started = await self._advance(
                study.id, auth, start=start, until=until, started_by="agent"
            )
            if isinstance(started, Failure):
                return started
        return Success(study)
