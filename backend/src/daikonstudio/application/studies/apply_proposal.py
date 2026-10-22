"""Carrying out a proposal, once the scientist has confirmed it.

The confirmation is the whole point of the split. An agent that could do this
itself would be an agent that rewrites the plan to suit what it just saw, and the
plan's value is that it was written first. So the agent names the change exactly,
the scientist presses the button, and the change is recorded against the turn
that asked for it.

Applying is deliberately not running. The step is changed or added and left for
the play control that every other step has, so there is one path that starts
work and one place to watch it, and a confirmation never turns into a long job
the scientist did not know they were starting.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.ports.study_repository import StudyRepository
from daikonstudio.application.studies.get_study import GetStudy
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.shared.errors import ConflictError, DomainError, NotFoundError
from daikonstudio.domain.studies.study import StepStatus, Study


@dataclass(frozen=True)
class ApplyCommand:
    study_id: uuid.UUID
    proposal_id: uuid.UUID


class ApplyProposal:
    def __init__(self, studies: StudyRepository, tools: ToolRegistry) -> None:
        self._studies = studies
        self._get = GetStudy(studies)
        self._tools = tools

    async def __call__(
        self, command: ApplyCommand, auth: AuthContext | None = None
    ) -> Result[Study, DomainError]:
        require_authenticated(auth)
        found = await self._get(command.study_id, auth)
        if not isinstance(found, Success):
            return found
        study = found.unwrap()

        proposal = study.proposal(command.proposal_id)
        if proposal is None:
            return Failure(NotFoundError("Proposal", str(command.proposal_id)))
        if proposal.applied_at is not None:
            return Failure(ConflictError("This change has already been made."))

        if proposal.does == "rerun":
            step = next((s for s in study.steps if s.position == proposal.step), None)
            if step is None:
                return Failure(NotFoundError("Step", str(proposal.step)))
            if proposal.settings:
                # Settings were checked against the tool's own model before this
                # was offered, so what is stored is what that tool accepts.
                step.params = {**step.params, **proposal.settings}
            # Back to pending so the step reads as waiting to run rather than as
            # done with a result that no longer matches the settings beside it.
            step.status = StepStatus.PENDING
            step.error = None
        else:
            manifest = self._tools.get(proposal.tool_id or "")
            study.add_step(
                tool_id=manifest.id,
                tool_version=manifest.version,
                name=manifest.name,
                purpose=proposal.why,
                params=dict(proposal.settings),
            )

        proposal.applied_at = datetime.now(UTC)
        await self._studies.update(study)
        return Success(study)
