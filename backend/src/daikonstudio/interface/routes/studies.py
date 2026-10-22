"""Studies: an approved plan, saved, run a step at a time, and read back.

Starting a study runs nothing. Advancing one answers 202 at once and runs its steps
in the background until a gate, a failure or the end; running one step runs that
step alone. A client reads the study again to see where it got to.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, Json

from daikonstudio.application.studies.advance_study import AdvanceStudy, in_flight
from daikonstudio.application.studies.apply_proposal import ApplyCommand, ApplyProposal
from daikonstudio.application.studies.get_study import GetStudy, ListStudies
from daikonstudio.application.studies.read_value import ReadValue
from daikonstudio.application.studies.stage_chat import CHECKERS, SayCommand, StageChat
from daikonstudio.application.studies.start_study import StartStudy, StartStudyCommand, StepSpec
from daikonstudio.application.studies.study_chat import AskStudyCommand, StudyChat
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.domain.studies.study import Answered, SlotValue, Study, Thread
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies._core import AuthDep
from daikonstudio.interface.error_handlers import result_to_response

router = APIRouter(prefix="/api/v1/studies", tags=["studies"])

# The same ceiling as a plan upload, for the same reason: refused before it is read.
_MAX_UPLOAD_BYTES = 32 * 1024 * 1024

StartStudyDep = Annotated[StartStudy, Depends(use_case(StartStudy))]
AdvanceStudyDep = Annotated[AdvanceStudy, Depends(use_case(AdvanceStudy))]
GetStudyDep = Annotated[GetStudy, Depends(use_case(GetStudy))]
ListStudiesDep = Annotated[ListStudies, Depends(use_case(ListStudies))]
ReadValueDep = Annotated[ReadValue, Depends(use_case(ReadValue))]
StageChatDep = Annotated[StageChat, Depends(use_case(StageChat))]
StudyChatDep = Annotated[StudyChat, Depends(use_case(StudyChat))]
ApplyProposalDep = Annotated[ApplyProposal, Depends(use_case(ApplyProposal))]


class StartStepRequest(BaseModel):
    tool_id: str
    purpose: str
    params: dict[str, Any] = {}
    gated: bool
    always: bool = False
    expanded_from: str | None = None


class AnswerRequest(BaseModel):
    ask: str
    answer: str


class StartStudyRequest(BaseModel):
    goal: str
    restated_goal: str
    checks: list[str] = []
    stop_conditions: list[str] = []
    steps: list[StartStepRequest]
    # The planner's questions and what the scientist answered; "" when left blank.
    answers: list[AnswerRequest] = []
    # The plan it was started from, kept as sent so the study page can show it again.
    planning: dict[str, Any] = {}


class AnswerResponse(BaseModel):
    ask: str
    answer: str


class ValueSummaryResponse(BaseModel):
    id: str
    key: str
    kind: str
    summary: dict[str, Any]
    produced_by: int | None
    asserted_by: str | None

    @classmethod
    def from_value(cls, v: SlotValue) -> ValueSummaryResponse:
        return cls(
            id=str(v.id),
            key=v.key,
            kind=v.kind,
            summary=v.summary,
            produced_by=v.produced_by,
            asserted_by=str(v.asserted_by) if v.asserted_by else None,
        )


class StepStateResponse(BaseModel):
    position: int
    tool_id: str
    name: str
    purpose: str
    params: dict[str, Any]
    gated: bool
    always: bool
    expanded_from: str | None
    status: str
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None
    outputs: list[ValueSummaryResponse]
    # Whether this stage has an agent. One exists where its result can be checked
    # claim by claim, so the client asks rather than keeping its own list of kinds.
    discussable: bool


class StudyResponse(BaseModel):
    id: str
    goal: str
    restated_goal: str
    checks: list[str]
    stop_conditions: list[str]
    status: str
    # Whether steps are running in this process right now.
    running: bool
    steps: list[StepStateResponse]
    brought: list[ValueSummaryResponse]
    answers: list[AnswerResponse]
    planning: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_study(cls, study: Study) -> StudyResponse:
        return cls(
            id=str(study.id),
            goal=study.goal,
            restated_goal=study.restated_goal,
            checks=list(study.checks),
            stop_conditions=list(study.stop_conditions),
            status=study.status.value,
            running=in_flight(study.id),
            steps=[
                StepStateResponse(
                    position=s.position,
                    tool_id=s.tool_id,
                    name=s.name,
                    purpose=s.purpose,
                    params=s.params,
                    gated=s.gated,
                    always=s.always,
                    expanded_from=s.expanded_from,
                    status=s.status.value,
                    error=s.error,
                    started_at=s.started_at,
                    finished_at=s.finished_at,
                    outputs=[
                        ValueSummaryResponse.from_value(v)
                        for v in study.current_outputs_of(s.position)
                    ],
                    discussable=any(
                        v.kind in CHECKERS for v in study.current_outputs_of(s.position)
                    ),
                )
                for s in study.steps
            ],
            brought=[
                ValueSummaryResponse.from_value(v) for v in study.values if v.produced_by is None
            ],
            answers=[AnswerResponse(ask=a.ask, answer=a.answer) for a in study.answers],
            planning=study.planning,
            created_at=study.created_at,
        )


class ValueResponse(BaseModel):
    id: str
    key: str
    kind: str
    summary: dict[str, Any]
    data: Any


@router.post("", response_model=StudyResponse, status_code=201)
async def start_study(
    auth: AuthDep,
    service: StartStudyDep,
    plan: Annotated[Json[StartStudyRequest], Form()],
    file: Annotated[UploadFile | None, File()] = None,
) -> StudyResponse:
    """Save an approved plan as a study, with the compounds file if there is one.

    `plan` is a JSON `StartStudyRequest` sent as a form field, because the file
    travels in the same multipart request.
    """
    compounds = await file.read() if file is not None else None
    if compounds is not None and len(compounds) > _MAX_UPLOAD_BYTES:
        raise ValidationError(
            "That file is too large to read.",
            detail="A study keeps the whole table, and this one is past the limit for an upload.",
        )
    command = StartStudyCommand(
        goal=plan.goal,
        restated_goal=plan.restated_goal,
        checks=tuple(plan.checks),
        stop_conditions=tuple(plan.stop_conditions),
        steps=tuple(StepSpec(**s.model_dump()) for s in plan.steps),
        compounds=compounds,
        filename=file.filename if file is not None else None,
        answers=tuple(Answered(ask=a.ask, answer=a.answer) for a in plan.answers),
        planning=plan.planning,
    )
    return StudyResponse.from_study(result_to_response(await service(command, auth=auth)))


@router.get("", response_model=list[StudyResponse])
async def list_studies(auth: AuthDep, service: ListStudiesDep) -> list[StudyResponse]:
    """Your studies in this workspace, newest first."""
    return [StudyResponse.from_study(s) for s in result_to_response(await service(auth=auth))]


@router.get("/{study_id}", response_model=StudyResponse)
async def get_study(study_id: uuid.UUID, auth: AuthDep, service: GetStudyDep) -> StudyResponse:
    return StudyResponse.from_study(result_to_response(await service(study_id, auth=auth)))


@router.post("/{study_id}/advance", response_model=StudyResponse, status_code=202)
async def advance_study(
    study_id: uuid.UUID, auth: AuthDep, service: AdvanceStudyDep
) -> StudyResponse:
    """Run the study's next steps in the background, up to a gate, a failure or the end.

    Re-runs a failed step. Does nothing while the study is already running.
    """
    return StudyResponse.from_study(result_to_response(await service(study_id, auth=auth)))


@router.post("/{study_id}/steps/{position}/run", response_model=StudyResponse, status_code=202)
async def run_step(
    study_id: uuid.UUID, position: int, auth: AuthDep, service: AdvanceStudyDep
) -> StudyResponse:
    """Run one step alone in the background, whatever its gate says, and stop.

    Re-running a finished step adds newer values, which later steps then read; it does
    not reset those later steps. Does nothing while the study is already running.
    """
    return StudyResponse.from_study(
        result_to_response(await service(study_id, auth=auth, position=position))
    )


class SayRequest(BaseModel):
    message: str | None = None
    provider: str | None = None


class ProposalResponse(BaseModel):
    id: str
    does: str
    step: int | None
    tool_id: str | None
    settings: dict[str, Any]
    why: str
    applied_at: datetime | None


class PointerResponse(BaseModel):
    at: str
    label: str
    evidence: str
    cites: list[str]


class TurnResponse(BaseModel):
    role: str
    say: str
    points: list[PointerResponse]
    proposes: list[str]
    proposals: list[ProposalResponse]
    unresolved: list[str]
    dropped: int
    written_by: str | None
    at: datetime


class ThreadResponse(BaseModel):
    position: int | None
    handoff: str
    turns: list[TurnResponse]

    @classmethod
    def of(cls, thread: Thread) -> ThreadResponse:
        return cls(
            position=thread.position,
            handoff=thread.handoff,
            turns=[
                TurnResponse(
                    role=t.role,
                    say=t.say,
                    points=[
                        PointerResponse(
                            at=p.at, label=p.label, evidence=p.evidence, cites=list(p.cites)
                        )
                        for p in t.points
                    ],
                    proposes=list(t.proposes),
                    proposals=[
                        ProposalResponse(
                            id=str(p.id),
                            does=p.does,
                            step=p.step,
                            tool_id=p.tool_id,
                            settings=p.settings,
                            why=p.why,
                            applied_at=p.applied_at,
                        )
                        for p in t.proposals
                    ],
                    unresolved=list(t.unresolved),
                    dropped=t.dropped,
                    written_by=t.written_by,
                    at=t.at,
                )
                for t in thread.turns
            ],
        )


@router.post("/{study_id}/proposals/{proposal_id}/apply", response_model=StudyResponse)
async def apply_proposal(
    study_id: uuid.UUID, proposal_id: uuid.UUID, auth: AuthDep, service: ApplyProposalDep
) -> StudyResponse:
    """Make a change an agent proposed, now that the scientist has confirmed it.

    Changes the plan and stops there. The step is left for the play control the
    scientist already knows, so confirming a change never starts a long job they
    did not ask to start.
    """
    return StudyResponse.from_study(
        result_to_response(
            await service(ApplyCommand(study_id=study_id, proposal_id=proposal_id), auth=auth)
        )
    )


@router.get("/{study_id}/chat", response_model=ThreadResponse | None)
async def read_study_thread(
    study_id: uuid.UUID, auth: AuthDep, service: GetStudyDep
) -> ThreadResponse | None:
    """What has been said about the study as a whole. Null until anything has been."""
    study = result_to_response(await service(study_id, auth=auth))
    thread = study.thread(None)
    return None if thread is None else ThreadResponse.of(thread)


@router.post("/{study_id}/chat", response_model=ThreadResponse)
async def ask_study(
    study_id: uuid.UUID, body: SayRequest, auth: AuthDep, service: StudyChatDep
) -> ThreadResponse:
    """Say something to the agent that holds the whole study, or let it open.

    It reads each stage's summary rather than each stage's exchange, so what it
    knows about a result is what that stage's agent concluded about it.
    """
    thread = result_to_response(
        await service(
            AskStudyCommand(study_id=study_id, message=body.message, provider=body.provider),
            auth=auth,
        )
    )
    return ThreadResponse.of(thread)


@router.get("/{study_id}/steps/{position}/chat", response_model=ThreadResponse | None)
async def read_thread(
    study_id: uuid.UUID, position: int, auth: AuthDep, service: GetStudyDep
) -> ThreadResponse | None:
    """What has been said at one stage. Null until anything has been."""
    study = result_to_response(await service(study_id, auth=auth))
    thread = study.thread(position)
    return None if thread is None else ThreadResponse.of(thread)


@router.post("/{study_id}/steps/{position}/chat", response_model=ThreadResponse)
async def say(
    study_id: uuid.UUID,
    position: int,
    body: SayRequest,
    auth: AuthDep,
    service: StageChatDep,
) -> ThreadResponse:
    """Say something to the agent at one stage, or with no message, let it open.

    A POST because it costs a model call, and because the exchange is kept with
    the study: what the scientist was told before they decided is part of the
    record, the same reason a plan records what wrote it.
    """
    thread = result_to_response(
        await service(
            SayCommand(
                study_id=study_id,
                position=position,
                message=body.message,
                provider=body.provider,
            ),
            auth=auth,
        )
    )
    return ThreadResponse.of(thread)


@router.get("/{study_id}/values/{value_id}", response_model=ValueResponse)
async def read_value(
    study_id: uuid.UUID, value_id: uuid.UUID, auth: AuthDep, service: ReadValueDep
) -> ValueResponse:
    """One value, as much as a page shows: the first rows of a table, or a document
    without its raw text."""
    view = result_to_response(await service(study_id, value_id, auth=auth))
    v = view.value
    return ValueResponse(id=str(v.id), key=v.key, kind=v.kind, summary=v.summary, data=view.data)
