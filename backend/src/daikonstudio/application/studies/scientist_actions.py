"""What the scientist does to a step directly: point it at another tool, or choose a candidate.

A control on the step card is the scientist's word, as a pressed proposal is, so it goes
through the same edits and the same approval as an action the agent takes on a quote, with
nothing to quote. Each writes an event into the thread, so the record shows who changed the
plan, and how, between the agent's turns. Nothing is written while a run is in flight: the
run writes the same row.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.study_repository import StudyRepository
from daikonstudio.application.studies.advance_study import RUNNING_MESSAGE, in_flight
from daikonstudio.application.studies.approve import approve_candidate
from daikonstudio.application.studies.edits import _label, apply_actions
from daikonstudio.application.studies.get_study import GetStudy
from daikonstudio.application.studies.schemas import Action
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.shared.errors import (
    ConflictError,
    DomainError,
    NotFoundError,
    ValidationError,
)
from daikonstudio.domain.studies.study import Step, Study, Turn


async def _editable(
    get: GetStudy, study_id: uuid.UUID, step_id: uuid.UUID, auth: AuthContext | None
) -> Result[tuple[Study, Step], DomainError]:
    """The study and the step a control names, if the scientist may change it now."""
    require_authenticated(auth)
    require_editor(auth)
    found = await get(study_id, auth)
    if not isinstance(found, Success):
        return Failure(found.failure())
    study = found.unwrap()
    if in_flight(study.id):
        return Failure(ConflictError(RUNNING_MESSAGE))
    step = next((s for s in study.steps if s.id == step_id), None)
    if step is None:
        return Failure(NotFoundError("Step", str(step_id)))
    return Success((study, step))


class SwapTool:
    def __init__(self, studies: StudyRepository, tools: ToolRegistry) -> None:
        self._studies = studies
        self._get = GetStudy(studies)
        self._tools = tools

    async def __call__(
        self,
        study_id: uuid.UUID,
        step_id: uuid.UUID,
        tool_id: str,
        auth: AuthContext | None = None,
    ) -> Result[Study, DomainError]:
        """Point a step at one of its alternatives, which leaves it waiting to run.

        A setting the new tool needs and the old one lacked becomes the step's problem, for
        the card to show and the scientist to fill.
        """
        found = await _editable(self._get, study_id, step_id, auth)
        if not isinstance(found, Success):
            return Failure(found.failure())
        study, step = found.unwrap()
        swap = Action(does="swap", step=step.position + 1, tool_id=tool_id)
        applied = apply_actions(study, [swap], self._tools, message=None)
        if applied.dropped:
            return Failure(ValidationError("That tool cannot replace this step."))
        (done,) = applied.acted
        study.say(
            Turn(
                role="event", kind="swapped", say=done.detail, about=step_id, at=datetime.now(UTC)
            )
        )
        await self._studies.update(study)
        return Success(study)


class ApproveCandidate:
    def __init__(self, studies: StudyRepository, tools: ToolRegistry, store: BlobStore) -> None:
        self._studies = studies
        self._get = GetStudy(studies)
        self._tools = tools
        self._store = store

    async def __call__(
        self,
        study_id: uuid.UUID,
        step_id: uuid.UUID,
        candidate: int,
        auth: AuthContext | None = None,
    ) -> Result[Study, DomainError]:
        """Write the candidate the step reported under `candidate` as the value it resolves
        to, asserted by the scientist."""
        found = await _editable(self._get, study_id, step_id, auth)
        if not isinstance(found, Success):
            return Failure(found.failure())
        study, step = found.unwrap()
        assert auth is not None  # _editable has already rejected None
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
            return Failure(error)  # nothing finished to choose from, or no such rank
        study.approve(step_id, value)
        chose = f"Chose candidate {candidate} of {_label(step)}."
        study.say(
            Turn(role="event", kind="approved", say=chose, about=step_id, at=datetime.now(UTC))
        )
        if in_flight(study.id):
            # One began while the choice was being written: the run has the row now.
            return Failure(ConflictError(RUNNING_MESSAGE))
        await self._studies.update(study)
        return Success(study)
