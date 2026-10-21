"""Plans: turning a stated goal into a reviewable sequence of tool steps.

Synchronous on purpose, for now. A plan is one model call, the caller is a person
waiting on a page, and the alternative costs a table, a migration and a poller
before anyone knows whether the plans are any good.

ponytail: synchronous. If planning routinely runs past a proxy's patience, the
upgrade is the pattern `application/data/build_dataset.py` already uses -- write a
row, answer 202, poll it -- not a second mechanism.

Nothing here executes a plan. A plan is a proposal; running one needs a surface
for the gates it declares, which does not exist yet.
"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from daikonstudio.application.planner.ask_first import (
    QUESTION_BUDGET,
    AskFirst,
    AskFirstCommand,
    Questions,
)
from daikonstudio.application.planner.draft_plan import (
    DraftPlanCommand,
    PlanDraft,
    WritePlan,
)
from daikonstudio.application.planner.plan import Answer
from daikonstudio.application.ports.llm import LanguageModels
from daikonstudio.application.ports.table_profiler import TableProfile, TableProfiler
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies._core import AuthDep
from daikonstudio.interface.error_handlers import result_to_response

router = APIRouter(prefix="/api/v1/plans", tags=["plans"])

WritePlanDep = Annotated[WritePlan, Depends(use_case(WritePlan))]
AskFirstDep = Annotated[AskFirst, Depends(use_case(AskFirst))]
LanguageModelsDep = Annotated[LanguageModels, Depends(use_case(LanguageModels))]
TableProfilerDep = Annotated[TableProfiler, Depends(use_case(TableProfiler))]

# A file big enough to plan from is small. The cap is here rather than in the
# profiler so an oversized upload is refused before it is read into memory.
_MAX_UPLOAD_BYTES = 32 * 1024 * 1024


class QuestionResponse(BaseModel):
    ask: str
    why: str
    # The plausible answers when they enumerate, for a client to offer as
    # choices. Empty when the answer is open. A client adds "not sure" itself --
    # it means something different from any answer and must always be available.
    options: list[str] = []
    # What the planner will do if nobody answers. Null means it genuinely cannot
    # proceed either way, so this one blocks and is asked before planning.
    # Anything else is a refinement: the plan already handles it, and the
    # scientist may ignore it entirely.
    default: str | None = None


class QuestionsResponse(BaseModel):
    """What the planner needs to know, before it has planned anything."""

    goal: str
    restated_goal: str
    questions: list[QuestionResponse]
    written_by: str
    # The budget before any of these are counted, so a client can show how much
    # room the whole analysis has.
    questions_left: int

    @classmethod
    def from_asked(cls, asked: Questions) -> QuestionsResponse:
        return cls(
            goal=asked.goal,
            restated_goal=asked.asked.restated_goal,
            questions=[
                QuestionResponse(ask=q.ask, why=q.why, options=q.options, default=q.default)
                for q in asked.asked.questions
            ],
            written_by=str(asked.written_by),
            questions_left=max(QUESTION_BUDGET - len(asked.asked.questions), 0),
        )


class StepResponse(BaseModel):
    tool_id: str
    name: str
    purpose: str
    # Whether the plan stops here and shows you the result before going on.
    gated: bool
    # The composite this step came from, when the planner chose a sequence rather
    # than an individual tool. Lets a client group the steps the way the planner
    # thought about them while keeping each one separately recorded.
    expanded_from: str | None
    # Prepended because it always runs, not chosen by the planner. A client
    # can present these as preparation rather than as decisions.
    always: bool


class ProblemResponse(BaseModel):
    step: int | None
    message: str


class PlanResponse(BaseModel):
    goal: str
    restated_goal: str
    questions: list[QuestionResponse]
    steps: list[StepResponse]
    checks: list[str]
    stop_conditions: list[str]
    skipped: list[ProblemResponse]
    # Everything that would stop this plan running as written: an invented tool,
    # or a step reaching for something no earlier step produces. Empty does not
    # mean the plan is good, only that it is executable.
    problems: list[ProblemResponse]
    # Which model wrote it. A plan is a judgement, and rereading one fairly later
    # means knowing what wrote it.
    written_by: str
    # How many more questions this analysis may ask, in total across every round.
    # Shown so the back-and-forth is visibly bounded; at zero the planner must
    # proceed on stated assumptions.
    questions_left: int

    @classmethod
    def from_draft(cls, draft: PlanDraft) -> PlanResponse:
        return cls(
            goal=draft.goal,
            restated_goal=draft.plan.restated_goal,
            questions=[
                QuestionResponse(ask=q.ask, why=q.why, options=q.options, default=q.default)
                for q in draft.plan.questions
            ],
            steps=[
                StepResponse(
                    tool_id=s.tool_id,
                    name=s.name,
                    purpose=s.purpose,
                    gated=s.gated,
                    expanded_from=s.expanded_from,
                    always=s.always,
                )
                for s in draft.steps
            ],
            checks=draft.plan.checks,
            stop_conditions=draft.plan.stop_conditions,
            skipped=[
                ProblemResponse(step=None, message=f"{s.tool_id}: {s.why}")
                for s in draft.plan.skipped
            ],
            problems=[ProblemResponse(step=p.step, message=p.message) for p in draft.problems],
            written_by=str(draft.written_by),
            questions_left=draft.questions_left,
        )


class ProviderResponse(BaseModel):
    provider: str
    model: str


async def _profile_of(file: UploadFile | None, profiler: TableProfiler) -> TableProfile | None:
    """The shape of an uploaded file, or nothing.

    The file is read, described and dropped. Both routes take it, and the second
    takes it again rather than carrying a profile between calls: the browser
    already holds the file, profiling is deterministic from the same bytes, and
    keeping a profile server-side would mean deciding where a study's data lives,
    which is still open.
    """
    if file is None:
        return None
    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise ValidationError(
            "That file is too large to read.",
            detail="Plans are drafted from the shape of the data, so a sample is enough.",
        )
    return profiler.profile(content, filename=file.filename or "upload.csv")


@router.get("/providers", response_model=list[ProviderResponse])
async def list_providers(auth: AuthDep, models: LanguageModelsDep) -> list[ProviderResponse]:
    """The model providers this deployment can actually reach.

    A provider with no key is left out rather than listed and broken, so a client
    can offer the choice honestly. An empty list means no planner is configured.
    """
    return [ProviderResponse(provider=c.provider, model=c.model) for c in models.available()]


@router.post("/questions", response_model=QuestionsResponse)
async def ask_first(
    auth: AuthDep,
    service: AskFirstDep,
    profiler: TableProfilerDep,
    goal: Annotated[str, Form(max_length=4000)],
    provider: Annotated[str | None, Form(max_length=64)] = None,
    file: Annotated[UploadFile | None, File()] = None,
) -> QuestionsResponse:
    """What the planner needs to know, before it has planned anything.

    The first half of drafting. Asked to do both at once a model asks whether
    IC50 is the readout and, in the same answer, writes a plan that uses IC50
    throughout -- so the guess is load-bearing before anyone has read the
    question. One extra call makes the plan an answer to something.
    """
    profile = await _profile_of(file, profiler)
    asked = result_to_response(
        await service(AskFirstCommand(goal=goal, provider=provider, profile=profile), auth=auth)
    )
    return QuestionsResponse.from_asked(asked)


@router.post("", response_model=PlanResponse)
async def draft_plan(
    auth: AuthDep,
    service: WritePlanDep,
    profiler: TableProfilerDep,
    goal: Annotated[str, Form(max_length=4000)],
    provider: Annotated[str | None, Form(max_length=64)] = None,
    file: Annotated[UploadFile | None, File()] = None,
    answers: Annotated[str | None, Form()] = None,
) -> PlanResponse:
    """Draft a plan, given the goal, the shape of the data and whatever the
    scientist said to the questions.

    `answers` is a JSON array of `{ask, answer}` sent as a form field, because
    the rest of the request is multipart and carrying the file alongside it is
    what lets the plan be written against both. Answering nothing is allowed; the
    plan then has to say which assumptions it made.

    Writes nothing and runs nothing. A plan is a proposal.
    """
    profile = await _profile_of(file, profiler)

    replies: tuple[Answer, ...] = ()
    if answers:
        try:
            parsed = json.loads(answers)
            replies = tuple(Answer.model_validate(item) for item in parsed)
        except (json.JSONDecodeError, TypeError, PydanticValidationError) as exc:
            raise ValidationError(
                "The answers could not be read.",
                detail="Send them as a JSON array of objects with 'ask' and 'answer'.",
            ) from exc

    draft = result_to_response(
        await service(
            DraftPlanCommand(goal=goal, provider=provider, profile=profile, answers=replies),
            auth=auth,
        )
    )
    return PlanResponse.from_draft(draft)
