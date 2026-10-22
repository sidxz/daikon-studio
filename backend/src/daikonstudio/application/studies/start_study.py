"""Starting a study: the end of the draft, with the plan as it stands."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.planner.plan import handed_over
from daikonstudio.application.ports.study_repository import StudyRepository
from daikonstudio.application.studies.advance_study import in_flight
from daikonstudio.application.studies.edits import recheck, replace_automatic
from daikonstudio.application.studies.get_study import GetStudy
from daikonstudio.application.studies.open_study import RUNNING
from daikonstudio.application.tools.manifest import SlotKind
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.shared.errors import ConflictError, DomainError, ValidationError
from daikonstudio.domain.studies.study import Study, Turn


class StartStudy:
    def __init__(self, studies: StudyRepository, tools: ToolRegistry) -> None:
        self._studies = studies
        self._get = GetStudy(studies)
        self._tools = tools

    async def __call__(
        self, study_id: uuid.UUID, auth: AuthContext | None = None
    ) -> Result[Study, DomainError]:
        """Start the draft, if its plan can go ahead as it stands.

        Starting runs nothing: it ends the draft, after which the steps may be run. A plan
        that works on a compound table is not started without one, however it was written,
        since an upload is optional and the plan may have been drafted before its file. The
        plan is matched to what has been brought and checked once more, so a problem that was
        settled by a file, or one that a file's absence leaves, is read as it now stands.
        """
        require_authenticated(auth)
        require_editor(auth)
        found = await self._get(study_id, auth)
        if not isinstance(found, Success):
            return Failure(found.failure())
        study = found.unwrap()
        if in_flight(study.id):
            return Failure(ConflictError(RUNNING))
        wanted = handed_over(
            [step.tool_id for step in study.steps if not step.always], self._tools
        )
        brought = any(
            v.kind == SlotKind.COMPOUND_TABLE.value and v.produced_by is None for v in study.values
        )
        if SlotKind.COMPOUND_TABLE in wanted and not brought:
            return Failure(
                ValidationError(
                    "Attach the compounds file: this plan works on a compound table "
                    "and none was sent."
                )
            )
        replace_automatic(study, self._tools)
        recheck(study, self._tools)
        try:
            study.start()
        except ConflictError as error:
            return Failure(error)
        study.say(Turn(role="event", kind="started", say="Started.", at=datetime.now(UTC)))
        await self._studies.update(study)
        return Success(study)
