"""Studies: one conversation from the scientist's first message, the plan written in it, and
the steps run and read back.

A study opens as a draft on the first message, and its agent asks before it plans. Starting it
runs nothing. A run answers 202 at once and goes on in the background, the agent observing
every step that ends, and a client reads the study and its thread again to see where it got
to. Any change while a run is in flight answers 409: the run writes the same row.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, Field

from daikonstudio.application.ports.llm import LanguageModels
from daikonstudio.application.studies.advance_study import AdvanceStudy, in_flight
from daikonstudio.application.studies.agent import SayCommand, StudyAgent
from daikonstudio.application.studies.apply_proposal import ApplyCommand, ApplyProposal
from daikonstudio.application.studies.edits import alternatives
from daikonstudio.application.studies.get_study import GetStudy, ListStudies
from daikonstudio.application.studies.open_study import AttachFiles, OpenCommand, OpenStudy, Upload
from daikonstudio.application.studies.read_value import ReadValue
from daikonstudio.application.studies.scientist_actions import ApproveCandidate, SwapTool
from daikonstudio.application.studies.start_study import StartStudy
from daikonstudio.application.tools.manifest import ToolManifest
from daikonstudio.application.tools.registry import ToolRegistry, UnknownToolError
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.domain.studies.study import Answered, SlotValue, Step, Study, Thread, Turn
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies._core import AuthDep
from daikonstudio.interface.error_handlers import result_to_response

router = APIRouter(prefix="/api/v1/studies", tags=["studies"])

# Each file is read no further than this, so one past it is refused without being held whole.
_MAX_UPLOAD_BYTES = 32 * 1024 * 1024
# One file per kind is all a study takes: a compound table and a structure.
_MAX_FILES = 2
# A message, a question or an answer: room for a paragraph, not for a pasted file.
_MAX_TEXT = 4000
# Far more answers than the agent may ask questions in a whole study.
_MAX_ANSWERS = 20

OpenStudyDep = Annotated[OpenStudy, Depends(use_case(OpenStudy))]
AttachFilesDep = Annotated[AttachFiles, Depends(use_case(AttachFiles))]
StartStudyDep = Annotated[StartStudy, Depends(use_case(StartStudy))]
StudyAgentDep = Annotated[StudyAgent, Depends(use_case(StudyAgent))]
AdvanceStudyDep = Annotated[AdvanceStudy, Depends(use_case(AdvanceStudy))]
SwapToolDep = Annotated[SwapTool, Depends(use_case(SwapTool))]
ApproveCandidateDep = Annotated[ApproveCandidate, Depends(use_case(ApproveCandidate))]
ApplyProposalDep = Annotated[ApplyProposal, Depends(use_case(ApplyProposal))]
GetStudyDep = Annotated[GetStudy, Depends(use_case(GetStudy))]
ListStudiesDep = Annotated[ListStudies, Depends(use_case(ListStudies))]
ReadValueDep = Annotated[ReadValue, Depends(use_case(ReadValue))]
ToolRegistryDep = Annotated[ToolRegistry, Depends(use_case(ToolRegistry))]
LanguageModelsDep = Annotated[LanguageModels, Depends(use_case(LanguageModels))]


def _str(value: uuid.UUID | None) -> str | None:
    return None if value is None else str(value)


def _manifest(tools: ToolRegistry, tool_id: str) -> ToolManifest | None:
    try:
        return tools.get(tool_id)
    except UnknownToolError:
        return None  # dropped from the catalogue since: the step still reads, with no choices


class ProviderResponse(BaseModel):
    provider: str
    model: str


class AnswerResponse(BaseModel):
    ask: str
    answer: str


class ValueSummaryResponse(BaseModel):
    id: str
    key: str
    kind: str
    summary: dict[str, Any]
    # The id of the step that made it; null for a file the scientist brought.
    produced_by: str | None
    # Who chose it, for a candidate the scientist approved; null for anything computed.
    asserted_by: str | None

    @classmethod
    def from_value(cls, v: SlotValue) -> ValueSummaryResponse:
        return cls(
            id=str(v.id),
            key=v.key,
            kind=v.kind,
            summary=v.summary,
            produced_by=_str(v.produced_by),
            asserted_by=_str(v.asserted_by),
        )


class AlternativeResponse(BaseModel):
    tool_id: str
    name: str


class StepStateResponse(BaseModel):
    id: str
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
    # Why the plan cannot go ahead with this step as it stands. Start is refused while any
    # step has one.
    problem: str | None
    started_at: datetime | None
    finished_at: datetime | None
    outputs: list[ValueSummaryResponse]
    # The tool reports candidates for the scientist to choose between, so a finished result
    # of this step can be approved.
    judgment: bool
    # The tools that could take this one's place: the same inputs and the same results.
    alternatives: list[AlternativeResponse]

    @classmethod
    def from_step(cls, step: Step, study: Study, tools: ToolRegistry) -> StepStateResponse:
        manifest = _manifest(tools, step.tool_id)
        others = [] if manifest is None else alternatives(manifest, tools)
        return cls(
            id=str(step.id),
            position=step.position,
            tool_id=step.tool_id,
            name=step.name,
            purpose=step.purpose,
            params=step.params,
            gated=step.gated,
            always=step.always,
            expanded_from=step.expanded_from,
            status=step.status.value,
            error=step.error,
            problem=step.problem,
            started_at=step.started_at,
            finished_at=step.finished_at,
            outputs=[
                ValueSummaryResponse.from_value(v) for v in study.current_outputs_of(step.id)
            ],
            judgment=manifest is not None and manifest.judgment,
            alternatives=[AlternativeResponse(tool_id=t.id, name=t.name) for t in others],
        )


class StudyResponse(BaseModel):
    id: str
    goal: str
    restated_goal: str
    checks: list[str]
    stop_conditions: list[str]
    # "draft" until Start, then "open", "done" or "failed".
    status: str
    # Whether steps are running in this process right now.
    running: bool
    # When Start ended the draft; null while it is one.
    started_at: datetime | None
    mode: str
    # How many steps had run when the checks or stop conditions were last changed after
    # Start; null if they never were.
    amended_after_steps: int | None
    # Moves with every change, a turn in the thread included, so a client knows when to read
    # the thread again.
    updated_at: datetime
    created_at: datetime
    steps: list[StepStateResponse]
    brought: list[ValueSummaryResponse]
    answers: list[AnswerResponse]

    @classmethod
    def from_study(cls, study: Study, tools: ToolRegistry) -> StudyResponse:
        return cls(
            id=str(study.id),
            goal=study.goal,
            restated_goal=study.restated_goal,
            checks=list(study.checks),
            stop_conditions=list(study.stop_conditions),
            status=study.status.value,
            running=in_flight(study.id),
            started_at=study.started_at,
            mode=study.mode,
            amended_after_steps=study.amended_after_steps,
            updated_at=study.updated_at,
            created_at=study.created_at,
            steps=[StepStateResponse.from_step(s, study, tools) for s in study.steps],
            brought=[
                ValueSummaryResponse.from_value(v) for v in study.values if v.produced_by is None
            ],
            answers=[AnswerResponse(ask=a.ask, answer=a.answer) for a in study.answers],
        )


class QuestionResponse(BaseModel):
    ask: str
    why: str
    # The plausible answers when they enumerate; empty when the answer is open. A client adds
    # "not sure" itself, since it means something no answer does.
    options: list[str]
    # What the agent goes on if nobody answers; null when it cannot plan without an answer.
    default: str | None


class PointerResponse(BaseModel):
    # The id of the step whose result this is about; null for a claim about the plan.
    step: str | None
    at: str
    label: str
    evidence: str
    cites: list[str]
    # Where a criterion stands, as the agent declared it: "met", "not_met", "not_testable",
    # or "" on a claim about a result, which has no standing to report.
    state: str = ""


class ProposalResponse(BaseModel):
    id: str
    # The change as its button keeps it, every step by id, so a press after the plan was
    # edited still means the step it meant.
    action: dict[str, Any]
    why: str
    applied_at: datetime | None


class ActedResponse(BaseModel):
    does: str
    step: str | None
    detail: str
    # The scientist's words it carried out.
    quoting: str


class TurnResponse(BaseModel):
    # "scientist", "agent" or "event".
    role: str
    # An agent turn's: asking, planning, observe or reply; an event's: what happened.
    kind: str
    say: str
    # The study's answer to its goal, on the turn that observed the last step; empty
    # on every other turn.
    conclusion: str = ""
    # The id of the step the turn is about, if any.
    about: str | None
    questions: list[QuestionResponse]
    points: list[PointerResponse]
    proposals: list[ProposalResponse]
    unresolved: list[str]
    actions: list[ActedResponse]
    # What the agent asked to see of a table before it answered.
    looked: list[str]
    # How many of the agent's claims, proposals and actions did not survive checking.
    dropped: int
    # An observation with nothing to say: a tick on the step, not a line in the thread.
    silent: bool
    written_by: str | None
    at: datetime

    @classmethod
    def from_turn(cls, t: Turn) -> TurnResponse:
        return cls(
            role=t.role,
            kind=t.kind,
            say=t.say,
            conclusion=t.conclusion,
            about=_str(t.about),
            questions=[
                QuestionResponse(ask=q.ask, why=q.why, options=list(q.options), default=q.default)
                for q in t.questions
            ],
            points=[
                PointerResponse(
                    step=_str(p.step),
                    at=p.at,
                    label=p.label,
                    evidence=p.evidence,
                    state=p.state,
                    cites=list(p.cites),
                )
                for p in t.points
            ],
            proposals=[
                ProposalResponse(id=str(p.id), action=p.action, why=p.why, applied_at=p.applied_at)
                for p in t.proposals
            ],
            unresolved=list(t.unresolved),
            actions=[
                ActedResponse(does=a.does, step=_str(a.step), detail=a.detail, quoting=a.quoting)
                for a in t.actions
            ],
            looked=list(t.looked),
            dropped=t.dropped,
            silent=t.silent,
            written_by=t.written_by,
            at=t.at,
        )


class ThreadResponse(BaseModel):
    # Where the study stands, for someone opening it cold, as the agent last wrote it.
    summary: str
    turns: list[TurnResponse]

    @classmethod
    def of(cls, thread: Thread) -> ThreadResponse:
        return cls(summary=thread.summary, turns=[TurnResponse.from_turn(t) for t in thread.turns])


class ValueResponse(BaseModel):
    id: str
    key: str
    kind: str
    summary: dict[str, Any]
    data: Any


class AnswerRequest(BaseModel):
    ask: str = Field(max_length=_MAX_TEXT)
    answer: str = Field(max_length=_MAX_TEXT)


class SayRequest(BaseModel):
    # Neither a message nor answers asks the agent for the turn it owes.
    message: str | None = Field(default=None, max_length=_MAX_TEXT)
    # The answers to the agent's questions, as the cards send them, never read out of prose.
    answers: list[AnswerRequest] = Field(default_factory=list, max_length=_MAX_ANSWERS)
    # The id of the step the message is about, when it was sent from that step's card.
    about: uuid.UUID | None = None
    provider: str | None = None


class RunRequest(BaseModel):
    # The step to begin with, whatever its status; null means the next one waiting to run.
    start: uuid.UUID | None = None
    # The last step that may run; null means on to the next one that stops for a look.
    until: uuid.UUID | None = None


class SwapRequest(BaseModel):
    tool_id: str


class ApproveRequest(BaseModel):
    # The rank the step reported the candidate under.
    candidate: int


async def _uploads(files: list[UploadFile]) -> tuple[Upload, ...]:
    if len(files) > _MAX_FILES:  # refused before any of them is read
        raise ValidationError(
            "A study takes two files at most: one compound table and one structure."
        )
    uploads: list[Upload] = []
    for file in files:
        content = await file.read(_MAX_UPLOAD_BYTES + 1)
        if len(content) > _MAX_UPLOAD_BYTES:
            raise ValidationError(
                "That file is too large to read.",
                detail="A study keeps the whole file, and this one is past the upload limit.",
            )
        uploads.append(Upload(filename=file.filename or "", content=content))
    return tuple(uploads)


@router.get("/providers", response_model=list[ProviderResponse])
async def list_providers(auth: AuthDep, models: LanguageModelsDep) -> list[ProviderResponse]:
    """The model providers this deployment can actually reach.

    A provider with no key is left out rather than listed and broken, so a client can offer
    the choice honestly. An empty list means no agent is configured. Declared before
    `/{study_id}`, which would otherwise take "providers" for an id.
    """
    return [ProviderResponse(provider=c.provider, model=c.model) for c in models.available()]


@router.post("", response_model=StudyResponse, status_code=201)
async def open_study(
    auth: AuthDep,
    service: OpenStudyDep,
    tools: ToolRegistryDep,
    message: Annotated[str, Form(max_length=_MAX_TEXT)],
    provider: Annotated[str | None, Form(max_length=64)] = None,
    files: Annotated[list[UploadFile], File()] = [],  # noqa: B006 -- FastAPI copies a default
) -> StudyResponse:
    """Open a study on the scientist's first message, with whatever files came with it.

    The agent's first turn is taken before this answers, so the thread already holds the goal
    as understood and what it needs to ask. Multipart, because the files travel with the
    message.
    """
    command = OpenCommand(goal=message, provider=provider, files=await _uploads(files))
    return StudyResponse.from_study(result_to_response(await service(command, auth=auth)), tools)


@router.get("", response_model=list[StudyResponse])
async def list_studies(
    auth: AuthDep, service: ListStudiesDep, tools: ToolRegistryDep
) -> list[StudyResponse]:
    """Your studies in this workspace, drafts included, newest first."""
    studies = result_to_response(await service(auth=auth))
    return [StudyResponse.from_study(s, tools) for s in studies]


@router.get("/{study_id}", response_model=StudyResponse)
async def get_study(
    study_id: uuid.UUID, auth: AuthDep, service: GetStudyDep, tools: ToolRegistryDep
) -> StudyResponse:
    return StudyResponse.from_study(result_to_response(await service(study_id, auth=auth)), tools)


@router.get("/{study_id}/chat", response_model=ThreadResponse)
async def read_thread(study_id: uuid.UUID, auth: AuthDep, service: GetStudyDep) -> ThreadResponse:
    """The study's one conversation, from the agent's first turn on."""
    return ThreadResponse.of(result_to_response(await service(study_id, auth=auth)).thread)


@router.post("/{study_id}/chat", response_model=ThreadResponse)
async def say(
    study_id: uuid.UUID, body: SayRequest, auth: AuthDep, service: StudyAgentDep
) -> ThreadResponse:
    """Say something to the study's agent, or send the answers to its questions.

    It answers in the turn the study's phase calls for: before Start it writes and edits the
    plan; after, it replies, and does what the message asks only on the message's own words.
    A POST because it costs a model call, and because what was said is kept with the study.
    """
    command = SayCommand(
        study_id=study_id,
        message=body.message,
        answers=tuple(Answered(ask=a.ask, answer=a.answer) for a in body.answers),
        about=body.about,
        provider=body.provider,
    )
    return ThreadResponse.of(result_to_response(await service(command, auth=auth)))


@router.post("/{study_id}/files", response_model=StudyResponse)
async def attach_files(
    study_id: uuid.UUID,
    auth: AuthDep,
    service: AttachFilesDep,
    tools: ToolRegistryDep,
    files: Annotated[list[UploadFile], File()],
) -> StudyResponse:
    """Bring more files to a study, a draft or started, one of each kind."""
    study = result_to_response(await service(study_id, await _uploads(files), auth=auth))
    return StudyResponse.from_study(study, tools)


@router.post("/{study_id}/start", response_model=StudyResponse)
async def start_study(
    study_id: uuid.UUID, auth: AuthDep, service: StartStudyDep, tools: ToolRegistryDep
) -> StudyResponse:
    """End the draft, if its plan can go ahead as it stands. Runs nothing."""
    return StudyResponse.from_study(result_to_response(await service(study_id, auth=auth)), tools)


@router.post("/{study_id}/steps/{step_id}/run", response_model=StudyResponse, status_code=202)
async def run_step(
    study_id: uuid.UUID,
    step_id: uuid.UUID,
    auth: AuthDep,
    service: AdvanceStudyDep,
    tools: ToolRegistryDep,
) -> StudyResponse:
    """Run one step alone in the background, whatever its gate says, and stop.

    Re-running a finished step adds newer values, which later steps then read; it does not
    reset those later steps. Does nothing while the study is already running.
    """
    started = await service(study_id, auth=auth, start=step_id, until=step_id)
    return StudyResponse.from_study(result_to_response(started), tools)


@router.post("/{study_id}/run", response_model=StudyResponse, status_code=202)
async def run_study(
    study_id: uuid.UUID,
    body: RunRequest,
    auth: AuthDep,
    service: AdvanceStudyDep,
    tools: ToolRegistryDep,
) -> StudyResponse:
    """Run the steps in the background from `start` to `until`, a gate, a failure, a pause
    the agent asks for, or the end.

    Re-runs a failed step. Does nothing while the study is already running.
    """
    started = await service(study_id, auth=auth, start=body.start, until=body.until)
    return StudyResponse.from_study(result_to_response(started), tools)


@router.post("/{study_id}/steps/{step_id}/tool", response_model=StudyResponse)
async def swap_tool(
    study_id: uuid.UUID,
    step_id: uuid.UUID,
    body: SwapRequest,
    auth: AuthDep,
    service: SwapToolDep,
    tools: ToolRegistryDep,
) -> StudyResponse:
    """Point a step at one of its alternatives; the thread records the change."""
    study = result_to_response(await service(study_id, step_id, body.tool_id, auth=auth))
    return StudyResponse.from_study(study, tools)


@router.post("/{study_id}/steps/{step_id}/approve", response_model=StudyResponse)
async def approve_candidate(
    study_id: uuid.UUID,
    step_id: uuid.UUID,
    body: ApproveRequest,
    auth: AuthDep,
    service: ApproveCandidateDep,
    tools: ToolRegistryDep,
) -> StudyResponse:
    """Choose one of the candidates a finished step reported, as the scientist's own value.

    The thread records the choice, and the value names who made it.
    """
    study = result_to_response(await service(study_id, step_id, body.candidate, auth=auth))
    return StudyResponse.from_study(study, tools)


@router.post("/{study_id}/proposals/{proposal_id}/apply", response_model=StudyResponse)
async def apply_proposal(
    study_id: uuid.UUID,
    proposal_id: uuid.UUID,
    auth: AuthDep,
    service: ApplyProposalDep,
    tools: ToolRegistryDep,
) -> StudyResponse:
    """Make a change an agent proposed, now that the scientist has pressed its button.

    An edit changes the plan and leaves the step for its play control. A proposal to run
    starts the run its button names, in the background, and the study comes back as it was
    saved before the run began.
    """
    command = ApplyCommand(study_id=study_id, proposal_id=proposal_id)
    return StudyResponse.from_study(result_to_response(await service(command, auth=auth)), tools)


@router.get("/{study_id}/values/{value_id}", response_model=ValueResponse)
async def read_value(
    study_id: uuid.UUID, value_id: uuid.UUID, auth: AuthDep, service: ReadValueDep
) -> ValueResponse:
    """One value, as much as a page shows: the first rows of a table, or a document
    without its raw text."""
    view = result_to_response(await service(study_id, value_id, auth=auth))
    v = view.value
    return ValueResponse(id=str(v.id), key=v.key, kind=v.kind, summary=v.summary, data=view.data)
