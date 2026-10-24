"""The one agent a study has, from the scientist's first message to the last step.

It keeps one conversation per study and answers in the schema of the phase the study is in,
which is what makes its rules mechanical rather than hoped for. Before there is a plan it may
only ask. While the plan is a draft it writes and edits it with the scientist, and no change
needs a quote, because nothing has run. Once the study has started it observes every step
that ends, unprompted, and acts only in reply to a message and on that message's words: an
observation has no field to act with, and an action that does not quote the message is
handed back as a proposal for the scientist to press.

Nothing the model says is stored as it said it. A claim about a result is kept only where the
reader for that kind of result finds it there; a claim about the plan only where it cites the
plan's own words; a proposal only where the study could carry it out; an edit only where the
catalogue and the plan agree. What fails is dropped and counted, never raised.

A turn that chooses tools reads a one-line index of the whole catalogue, sent as a stable prefix
the provider can cache, and a view of a few tools in full, chosen by the host from what the
model named last, what the plan uses and, while planning, what reaches the kinds the study
should end with. An observation reads neither: it cannot act.

A turn costs one model call, or two: when the model asks to look at a table, the rows it asked
for, capped, are put in front of it; when it names tools it was not shown in full, or a plan
the checker rejects, the view widens to them. It answers again, and that answer stands.

The scientist's own turn is stored before the model is asked, so a provider that fails leaves
the message in the thread, and the error reaches the scientist rather than their words being
lost.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import uuid
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.planner.catalogue import INDEX_HEADING, index
from daikonstudio.application.planner.plan import (
    DraftPlan,
    PlanProblem,
    Question,
    expand,
    incoherences,
)
from daikonstudio.application.planner.view import WANTED_CAP, outside, select, widen
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.llm import LanguageModel, LanguageModels
from daikonstudio.application.ports.study_repository import StudyRepository
from daikonstudio.application.studies.advance_study import (
    RUNNING_MESSAGE,
    StartedBy,
    _ran,
    in_flight,
)
from daikonstudio.application.studies.approve import approve_candidate
from daikonstudio.application.studies.edits import (
    _brought,
    _label,
    apply_actions,
    recheck,
    to_steps,
)
from daikonstudio.application.studies.get_study import GetStudy
from daikonstudio.application.studies.prompts import MAX_QUESTIONS, QUESTION_BUDGET, system
from daikonstudio.application.studies.proposals import vet
from daikonstudio.application.studies.read_value import decode_view
from daikonstudio.application.studies.readers import LOOK_ROWS_MAX, reader
from daikonstudio.application.studies.schemas import (
    AskingTurn,
    Claim,
    Look,
    ObserveTurn,
    ProgressClaim,
    ReplyTurn,
    planning_turn_schema,
)
from daikonstudio.application.studies.standing import (
    standing,
    trigger_for_observe,
    trigger_for_reply,
)
from daikonstudio.application.tools import codecs
from daikonstudio.application.tools.manifest import SlotKind, ToolManifest
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.shared.errors import ConflictError, DomainError, ValidationError
from daikonstudio.domain.studies.study import (
    Acted,
    Answered,
    Asked,
    Pointer,
    Proposed,
    SlotValue,
    Step,
    StepStatus,
    Study,
    Thread,
    Turn,
)

logger = logging.getLogger(__name__)

_RETRY = "\n\nYour previous plan could not run as written:\n"
_LOOKED = (
    "\n\nYou asked to see part of a result, and here it is. Answer again with it in front "
    "of you; a further look will not be answered.\n\n"
)
_WIDENED = (
    "\n\nThe tools you asked to read are now given in full above. Answer again with them in "
    "front of you."
)
_DRIVING = (
    "\n\nYou are driving this run on the scientist's instruction. To stop it before the "
    "next step, set `pause` and say why."
)


@dataclass(frozen=True)
class SayCommand:
    """What the scientist sent. With neither a message nor answers, the agent takes the turn
    it owes, as it does after a draft's first turn failed."""

    study_id: uuid.UUID
    message: str | None = None
    answers: tuple[Answered, ...] = ()
    about: uuid.UUID | None = None  # the step the message is about, if any
    provider: str | None = None


class Advance(Protocol):
    """Starting the run loop (`AdvanceStudy`): from `start`, or the next step waiting, to
    `until`, or the next step that stops for a look. A run the scientist starts is written
    into the thread as one; the agent's own turn already records the ones it starts, so it
    passes `started_by="agent"`."""

    async def __call__(
        self,
        study_id: uuid.UUID,
        auth: AuthContext | None = None,
        *,
        start: uuid.UUID | None = None,
        until: uuid.UUID | None = None,
        started_by: StartedBy = "scientist",
    ) -> Result[Study, DomainError]: ...


def _left(study: Study) -> int:
    """How many questions the whole analysis may still ask: each answer spent one."""
    return max(QUESTION_BUDGET - len(study.answers), 0)


def _index(registry: ToolRegistry) -> str:
    """The whole catalogue, a line a tool, as the prefix a tool-choosing turn reads first."""
    return f"{INDEX_HEADING}\n{index(registry.manifests())}"


def _steer(study: Study) -> tuple[list[str], set[SlotKind]]:
    """What steers the next view: the tools the agent's last turn named, and the kinds its
    latest turn that named any said the study should end with."""
    turns = [t for t in study.thread.turns if t.role == "agent"]
    wanted = list(turns[-1].wanted) if turns else []
    ends = next((t.target_kinds for t in reversed(turns) if t.target_kinds), ())
    return wanted, {kind for kind in SlotKind if kind.value in ends}


def _asked(questions: Sequence[Question]) -> tuple[Asked, ...]:
    """The questions as the thread keeps them, each asked once."""
    first: dict[str, Question] = {}
    for question in questions:
        first.setdefault(question.ask, question)
    return tuple(
        Asked(ask=q.ask, why=q.why, options=tuple(q.options), default=q.default or None)
        for q in first.values()
    )


def _said(command: SayCommand) -> str:
    """The scientist's turn as the thread shows it: the message, then a line per answer."""
    message = (command.message or "").strip()
    answers = [f"{a.ask}: {a.answer.strip() or '(no answer)'}" for a in command.answers]
    return "\n".join([message, *answers] if message else answers)


def _pressable(study: Study, proposals: Sequence[Proposed]) -> tuple[list[Proposed], int]:
    """The proposals whose steps are all still in the plan, each once, and how many named a
    step an edit in this turn took away. A button for a step that is gone could not be pressed."""
    live = {str(step.id) for step in study.steps}
    kept: list[Proposed] = []
    seen: set[str] = set()
    gone = 0
    for proposal in proposals:
        refs = (proposal.action[key] for key in ("step_id", "before_id", "until_id"))
        if any(ref is not None and ref not in live for ref in refs):
            gone += 1
        elif (same := json.dumps(proposal.action, sort_keys=True)) not in seen:
            seen.add(same)  # proposed and also acted on without a quote: one button
            kept.append(proposal)
    return kept, gone


async def _read(store: BlobStore, value: SlotValue, *, limit: int | None) -> Any:
    blob = await asyncio.to_thread(store.get_bytes, value.blob_key)
    return decode_view(value.kind, blob, limit=limit)


async def _whole(store: BlobStore, value: SlotValue, decoded: dict[uuid.UUID, Any]) -> Any:
    """A value in full, as a checker needs it, read once per turn however often it is wanted.

    ponytail: a table is decoded whole (0.4 s and 840 MB at 323k rows) even where only its
    summary is shown; decode only for a check if tables that size are studied.
    """
    if value.id not in decoded:
        decoded[value.id] = await _read(store, value, limit=None)
    return decoded[value.id]


async def _held(
    study: Study, store: BlobStore, decoded: dict[uuid.UUID, Any]
) -> dict[uuid.UUID, Any]:
    """What every step's current results hold, for the standing to describe them by. A table is
    described from its stored summary, so its rows are not read.

    ponytail: a structure is read whole for a description that needs only its meta; pass it
    over as a table is if structures grow large enough for that to show.
    """
    return {
        value.id: await _whole(store, value, decoded)
        for step in study.steps
        for value in study.current_outputs_of(step.id)
        if value.kind not in codecs.TABLE_KINDS
    }


async def _look(look: Look, study: Study, store: BlobStore) -> tuple[str, str | None]:
    """A look as the model is shown it, and as the turn records it; nothing where no table is."""
    step = study.step_at(look.step - 1)
    if step is not None:
        for value in study.current_outputs_of(step.id):
            if (show := reader(value.kind).look) is not None:
                rows = min(max(look.rows, 0), LOOK_ROWS_MAX)
                shown = show(await _read(store, value, limit=rows), look, value.summary)
                columns = f", columns {', '.join(look.columns)}" if look.columns else ""
                return (
                    f"Step {look.step}, {value.kind.replace('_', ' ')}:\n{shown}",
                    f"step {look.step}: {rows} rows{columns}",
                )
    return f"Step {look.step} has no table to look at.", None


async def _answer[T: (ObserveTurn, ReplyTurn)](
    model: LanguageModel,
    schema: type[T],
    study: Study,
    registry: ToolRegistry,
    store: BlobStore,
    *,
    held: Mapping[uuid.UUID, Any],
    trigger: str,
    stable: str = "",
    view: Sequence[ToolManifest] | None = None,
    in_hand: Set[SlotKind] = frozenset(),
) -> tuple[T, tuple[str, ...], frozenset[int]]:
    """The model's turn on a started study, what it looked at on the way, and the steps whose
    rows it was shown, by number as it counts them.

    The prompt is the standing, with `view` in full, then `trigger`. Without a view the turn
    reads no tools, and a tool it names widens nothing. A second call is made once, for a look
    or for tools named outside the view, and that answer stands with whatever it asks for
    ignored. A look's second call carries the widened view too.
    """
    instructions = system("running", registry)
    prompt = standing(study, registry, view=view or (), data=held) + trigger
    answer = await model.parse(system=instructions, prompt=prompt, schema=schema, stable=stable)
    named: list[str] = []
    if view is not None:
        named += getattr(answer, "wanted", [])
        named += [action.tool_id for action in getattr(answer, "actions", [])]
        named += [proposal.tool_id for proposal in answer.proposals]
    unseen = outside(registry, view or (), named)
    if unseen:
        wider = widen(registry, view or (), unseen, in_hand=set(in_hand))
        prompt = standing(study, registry, view=wider, data=held) + trigger
    looks: list[tuple[str, str | None]] = []
    if answer.look:
        looks = [await _look(look, study, store) for look in answer.look]
        prompt += _LOOKED + "\n\n".join(text for text, _ in looks)
    elif unseen:
        prompt += _WIDENED
    else:
        return answer, (), frozenset()
    shown = frozenset(
        look.step for look, (_, line) in zip(answer.look, looks, strict=True) if line is not None
    )
    answer = await model.parse(system=instructions, prompt=prompt, schema=schema, stable=stable)
    return answer, tuple(line for _, line in looks if line is not None), shown


async def _check(
    claims: Sequence[Claim],
    step: Step | None,
    study: Study,
    store: BlobStore,
    decoded: dict[uuid.UUID, Any],
) -> tuple[list[Pointer], int]:
    """Claims about one step, checked against its first result whose kind has a checker."""
    if step is not None:
        for value in study.current_outputs_of(step.id):
            if (check := reader(value.kind).check) is not None:
                kept, dropped = check(claims, await _whole(store, value, decoded), value.summary)
                # A reader knows what a result holds; which step it came from is ours to say.
                return [dataclasses.replace(p, step=step.id) for p in kept], dropped
    return [], len(claims)


def check_progress(claims: Sequence[ProgressClaim], study: Study) -> tuple[list[Pointer], int]:
    """Keep only the claims about where the study stands that its plan carries.

    A step is named by its number in the plan, and a check or a stop condition by its place in
    its list, each counted from 1 as the standing shows them, and a claim cites their words
    exactly. One citing a criterion the plan never wrote is dropped before anyone reads it,
    which is what makes it safe to put next to a decision.
    """
    kept: list[Pointer] = []
    dropped = 0
    sources: dict[str, list[str]] = {
        "step": [s.name for s in sorted(study.steps, key=lambda s: s.position)],
        "check": list(study.checks),
        "stop_condition": list(study.stop_conditions),
    }
    for claim in claims:
        available = sources[claim.about]
        indexes = [n - 1 for n in dict.fromkeys(claim.refers_to)]
        if not indexes or any(not 0 <= i < len(available) for i in indexes):
            dropped += 1
            continue
        named = {available[i].strip().lower() for i in indexes}
        if not claim.cites or any(c.strip().lower() not in named for c in claim.cites):
            dropped += 1
            continue
        where = ", ".join(str(n) for n in dict.fromkeys(claim.refers_to))
        kept.append(
            Pointer(
                step=None,
                at=f"{claim.about.replace('_', ' ')}{'s' if len(indexes) > 1 else ''} {where}",
                label=claim.label[:120],
                evidence=claim.about,
                cites=tuple(claim.cites),
            )
        )
    return kept, dropped


async def _checked(
    answer: ObserveTurn | ReplyTurn,
    study: Study,
    registry: ToolRegistry,
    store: BlobStore,
    decoded: dict[uuid.UUID, Any],
    shown: frozenset[int],
) -> tuple[list[Pointer], list[Proposed], int]:
    """What survives of a turn's claims and proposals, and how many did not.

    `shown` is the steps whose rows a look in this turn showed. A row is cited by its
    identifier, which only a look shows, so a claim citing rows of a step not looked at in
    this turn was guessed or remembered, and goes, however plausible the identifier: one that
    happens to exist would otherwise pass as a reading. A column is in the summary and needs
    no look.
    """
    by_step: dict[int, list[Claim]] = {}
    for claim in answer.claims:
        by_step.setdefault(claim.step, []).append(claim)
    points: list[Pointer] = []
    dropped = 0
    for number, claims in by_step.items():
        if number not in shown:
            dropped += sum(claim.evidence == "rows" for claim in claims)
            claims = [claim for claim in claims if claim.evidence != "rows"]
        if not claims:
            continue
        kept, lost = await _check(claims, study.step_at(number - 1), study, store, decoded)
        points += kept
        dropped += lost
    progress, unplanned = check_progress(answer.progress, study)
    proposals, unusable = vet(answer.proposals, study, registry)
    return [*points, *progress], proposals, dropped + unplanned + unusable


class StudyAgent:
    """The scientist's turns, and the agent's answer to each."""

    def __init__(
        self,
        studies: StudyRepository,
        tools: ToolRegistry,
        models: LanguageModels,
        store: BlobStore,
        advance: Advance,
    ) -> None:
        self._studies = studies
        self._get = GetStudy(studies)
        self._tools = tools
        self._models = models
        self._store = store
        self._advance = advance

    async def ask(self, study: Study, provider: str | None) -> Study:
        """The first turn on a draft: the goal as understood, and what to ask before planning.

        The restated goal is the turn's words, so the thread never shows the agent saying
        nothing. With nothing to ask it plans at once, so the scientist's first reply comes
        with a plan.
        """
        model = self._models.get(provider)
        left = _left(study)
        answer = await model.parse(
            system=system("asking", self._tools),
            prompt=standing(study, self._tools, left=left),
            schema=AskingTurn,
            stable=_index(self._tools),
        )
        study.restated_goal = answer.restated_goal.strip()
        # Trimmed here, not trusted to the prompt: a form nobody finishes is worse than a
        # shorter one, and the model has already ranked them.
        questions = _asked(answer.questions)[: min(MAX_QUESTIONS, left)]
        study.say(
            Turn(
                role="agent",
                kind="asking",
                say=study.restated_goal,
                questions=questions,
                wanted=tuple(answer.wanted[:WANTED_CAP]),
                target_kinds=tuple(kind.value for kind in answer.target_kinds),
                written_by=str(model.choice),
                at=datetime.now(UTC),
            )
        )
        await self._studies.update(study)
        if not questions:
            await self._plan(study, model)
        return study

    async def __call__(
        self, command: SayCommand, auth: AuthContext | None = None
    ) -> Result[Thread, DomainError]:
        """Take the scientist's turn, then the one the agent owes in the study's phase.

        A message can change the plan and start a run, so a viewer reads the thread but may
        not talk. Nothing is written while a run is in flight: the run writes the same row,
        and a write between its steps would have it record a finished step as failed.
        """
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        found = await self._get(command.study_id, auth)
        if not isinstance(found, Success):
            return Failure(found.failure())
        study = found.unwrap()
        if in_flight(study.id):
            return Failure(ConflictError(RUNNING_MESSAGE))
        said = _said(command)
        last = study.thread.turns[-1] if study.thread.turns else None
        # The same words again with no answer between are a retry of a send that failed: they
        # are in the thread already, and so are its answers.
        retry = last is not None and last.role == "scientist" and last.say == said
        if said and not retry:
            study.answers += command.answers
            study.say(Turn(role="scientist", say=said, about=command.about, at=datetime.now(UTC)))
            # Before the model is asked, so a provider that fails loses nothing they wrote.
            await self._studies.update(study)
        if study.phase == "asking":  # also a draft whose first turn failed
            await self.ask(study, command.provider)
            return Success(study.thread)
        model = self._models.get(command.provider)
        if study.phase == "planning":
            await self._plan(study, model)
            return Success(study.thread)
        return await self._reply(study, model, command, auth)

    async def _plan(self, study: Study, model: LanguageModel) -> None:
        """A planning turn: the plan written whole, or edited, and what is still to ask.

        The model reads the index and a view chosen from what its last turn named, what the
        plan uses and what reaches the kinds the study should end with. A plan built from
        tools it saw only in the index stands if the checker passes it.
        """
        left = _left(study)
        instructions = system("planning", self._tools)
        schema = planning_turn_schema(self._tools)
        wanted, targets = _steer(study)
        in_hand = _brought(study) | {SlotKind.COMPOUND_TABLE}
        in_plan = [s.tool_id for s in study.steps]
        view = select(
            self._tools, wanted=wanted, in_plan=in_plan, in_hand=in_hand, targets=targets
        )
        prompt = standing(study, self._tools, view=view, left=left)
        answer = await model.parse(
            system=instructions, prompt=prompt, schema=schema, stable=_index(self._tools)
        )
        # A plan with no steps is the schema's "no plan": the turn edits the one there is.
        problems = self._problems(answer.plan, study) if answer.plan.steps else []
        asked_first = not answer.plan.steps and not answer.actions
        if problems or (asked_first and outside(self._tools, view, answer.wanted)):
            # One more try, with the plan's own problems in front of it or the tools it asked
            # to read: a missing setting or a step with nothing to feed it is a defect the
            # check names exactly, and the model fixes a named defect far more often than it
            # avoids one unprompted. ponytail: one retry; the second answer stands whatever
            # its problems, and its steps show them.
            ids = [
                *answer.wanted,
                *(s.tool_id for s in answer.plan.steps),
                *(a.tool_id for a in answer.actions),
                *self._unmade(answer.plan, study),
            ]
            wider = widen(self._tools, view, ids, in_hand=in_hand)
            prompt = standing(study, self._tools, view=wider, left=left)
            if problems:
                prompt += _RETRY + "\n".join(f"- {p.message}" for p in problems)
                prompt += "\nReturn the corrected plan."
            else:
                prompt += _WIDENED
            answer = await model.parse(
                system=instructions, prompt=prompt, schema=schema, stable=_index(self._tools)
            )
        acted: list[Acted] = []
        if answer.plan.steps:
            # A whole plan replaces the steps any action beside it was numbered against.
            dropped = self._replace(study, answer.plan) + len(answer.actions)
        else:
            applied = apply_actions(study, answer.actions, self._tools, message=None)
            acted = applied.acted
            # A draft has run nothing: no candidate to choose yet, and nothing it may run.
            dropped = applied.dropped + len(applied.approvals) + int(applied.run is not None)
        offered = [*answer.questions, *answer.plan.questions]
        study.say(
            Turn(
                role="agent",
                kind="planning",
                say=answer.say.strip(),
                questions=_asked(offered)[:left],
                actions=tuple(acted),
                wanted=tuple(answer.wanted[:WANTED_CAP]),
                dropped=dropped,
                written_by=str(model.choice),
                at=datetime.now(UTC),
            )
        )
        await self._studies.update(study)

    def _problems(self, plan: DraftPlan, study: Study) -> list[PlanProblem]:
        """What would stop the plan as the model wrote it.

        Its automatic steps are left out: the model was told never to list them, so a problem
        pinned on one is not its to fix, and a structure nobody fetched surfaces on the step
        that wanted it. A compound table counts as in hand before its file arrives, since the
        file may follow; anything else has to be brought or made.
        """
        brought = _brought(study)
        chosen = [s for s in expand(plan, self._tools, starting_with=brought) if not s.always]
        in_hand = brought | {SlotKind.COMPOUND_TABLE}
        return incoherences(chosen, self._tools, starting_with=in_hand)

    def _unmade(self, plan: DraftPlan, study: Study) -> list[str]:
        """The tools that make what a step needs and no step before it made, walking the plan
        as `_problems` does, so a retry can read them in full."""
        brought = _brought(study)
        available = brought | {SlotKind.COMPOUND_TABLE}
        by_id = {m.id: m for m in self._tools.manifests()}
        ids: list[str] = []
        for step in expand(plan, self._tools, starting_with=brought):
            manifest = by_id.get(step.tool_id)
            if step.always or manifest is None:
                continue
            for slot in manifest.inputs:
                if not slot.optional and slot.kind not in available:
                    ids += [m.id for m in self._tools.producing(slot.kind)]
            available |= {slot.kind for slot in manifest.outputs}
            if manifest.resolves_to is not None:
                available.add(manifest.resolves_to)
        return ids

    def _replace(self, study: Study, plan: DraftPlan) -> int:
        """Put a whole plan in the draft's place; how many of its steps name no tool."""
        steps = expand(plan, self._tools, starting_with=_brought(study))
        known = {manifest.id for manifest in self._tools.manifests()}
        study.replace_steps(to_steps([s for s in steps if s.tool_id in known], self._tools))
        study.restated_goal = plan.restated_goal.strip()
        study.checks = tuple(plan.checks)
        study.stop_conditions = tuple(plan.stop_conditions)
        recheck(study, self._tools)
        return len(steps) - len(study.steps)

    async def _reply(
        self, study: Study, model: LanguageModel, command: SayCommand, auth: AuthContext
    ) -> Result[Thread, DomainError]:
        """The answer to a message on a started study: what it says, checked, and what it
        does on the scientist's words."""
        decoded: dict[uuid.UUID, Any] = {}
        holding = await _held(study, self._store, decoded)
        trigger = ""
        if about := next((s for s in study.steps if s.id == command.about), None):
            outputs = study.current_outputs_of(about.id)
            data = await _whole(self._store, outputs[0], decoded) if outputs else None
            trigger = "\n\n" + trigger_for_reply(study, about, data)
        # A reply edits a plan rather than building one: its tools, what was named last, and
        # the automatic ones. Anything else it names is read on the one second call.
        in_plan = [s.tool_id for s in study.steps]
        view = select(
            self._tools, wanted=_steer(study)[0], in_plan=in_plan, in_hand=set(), targets=set()
        )
        answer, looked, shown = await _answer(
            model,
            ReplyTurn,
            study,
            self._tools,
            self._store,
            held=holding,
            trigger=trigger,
            stable=_index(self._tools),
            view=view,
            in_hand=set(),
        )
        points, proposals, dropped = await _checked(
            answer, study, self._tools, self._store, decoded, shown
        )
        if in_flight(study.id):
            # One began while the model was answering: change nothing, write nothing.
            return Failure(ConflictError(RUNNING_MESSAGE))

        # The words each approval quoted, by the step as the model numbered it.
        numbered = {step.position + 1: step.id for step in study.steps}
        quotes = {
            (numbered.get(a.step or 0), a.candidate): a.quoting
            for a in answer.actions
            if a.does == "approve"
        }
        # With no message there is nothing to quote, so nothing is done and all is proposed.
        applied = apply_actions(study, answer.actions, self._tools, message=command.message or "")
        held, unusable = vet(applied.downgraded, study, self._tools)
        offered, gone = _pressable(study, [*proposals, *held])
        acted = list(applied.acted)
        dropped += applied.dropped + unusable + gone
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
            except ValidationError:
                dropped += 1  # no such rank, or nothing finished to choose from
                continue
            study.approve(step_id, value)
            chose = f"Chose candidate {candidate} of {_label(study.step(step_id))}."
            quoting = quotes.get((step_id, candidate), "")
            acted.append(Acted(does="approve", step=step_id, detail=chose, quoting=quoting))
        start: Step | None = None
        until: Step | None = None
        run = applied.run
        if run is not None and not run.step and study.next_step is None:
            run = None  # nothing is waiting to run
            dropped += 1
        if run is not None:
            start = study.step_at(run.step - 1) if run.step else None
            until = study.step_at(run.until - 1) if run.until else None
            ids = (None if start is None else start.id, None if until is None else until.id)
            acted.append(
                Acted(does="run", step=ids[0], detail=_ran(study, *ids), quoting=run.quoting)
            )

        study.thread.summary = answer.summary.strip()
        study.say(
            Turn(
                role="agent",
                kind="reply",
                say=answer.say.strip(),
                about=command.about,
                points=tuple(points),
                proposals=tuple(offered),
                unresolved=tuple(answer.unresolved),
                actions=tuple(acted),
                looked=looked,
                wanted=tuple(answer.wanted[:WANTED_CAP]),
                dropped=dropped,
                written_by=str(model.choice),
                at=datetime.now(UTC),
            )
        )
        await self._studies.update(study)
        if run is not None:
            started = await self._advance(
                study.id,
                auth,
                start=None if start is None else start.id,
                until=None if until is None else until.id,
                started_by="agent",
            )
            if isinstance(started, Failure):
                # The reply says it ran; the thread has to say it did not.
                study.say(
                    Turn(
                        role="event",
                        kind="run_refused",
                        say=f"The run did not start: {started.failure()}",
                        at=datetime.now(UTC),
                    )
                )
                await self._studies.update(study)
                return Failure(started.failure())
        return Success(study.thread)


class ObserveStep:
    """The turn the agent takes when a step ends: it observes, and that is all it can do.

    `ObserveTurn` has no field to change the plan or run anything, so a result can only be
    remarked on; the one thing an observation may do is stop a run the agent is driving, and
    only with a reason. It runs inside the run loop rather than for a request, so it never
    raises: a turn that could not be taken is recorded as an event and the loop goes on.
    """

    def __init__(
        self,
        studies: StudyRepository,
        tools: ToolRegistry,
        models: LanguageModels,
        store: BlobStore,
    ) -> None:
        self._studies = studies
        self._tools = tools
        self._models = models
        self._store = store

    async def __call__(
        self,
        workspace_id: uuid.UUID,
        study_id: uuid.UUID,
        step_id: uuid.UUID,
        *,
        driving: bool = False,
    ) -> bool:
        """Observe the step that just ended; whether to pause the run after it.

        `driving` says the agent is running the plan on the scientist's instruction, the only
        time a pause means anything.
        """
        try:
            return await self._observe(workspace_id, study_id, step_id, driving=driving)
        except Exception as error:
            logger.exception("Study %s could not observe step %s", study_id, step_id)
            await self._unobserved(workspace_id, study_id, step_id, error)
            return False

    async def _observe(
        self, workspace_id: uuid.UUID, study_id: uuid.UUID, step_id: uuid.UUID, *, driving: bool
    ) -> bool:
        study = await self._studies.get(workspace_id, study_id)
        step = None if study is None else next((s for s in study.steps if s.id == step_id), None)
        if study is None or step is None:
            return False  # gone since it ran: nothing to observe
        model = self._models.get(None)
        decoded: dict[uuid.UUID, Any] = {}
        outputs = [] if step.status is StepStatus.FAILED else study.current_outputs_of(step.id)
        data = {value.id: await _whole(self._store, value, decoded) for value in outputs}
        held = await _held(study, self._store, decoded)
        trigger = "\n\n" + trigger_for_observe(study, step, data)
        if driving:
            trigger += _DRIVING
        # No view and no index: an observation cannot act, so it has no tool to choose.
        answer, looked, shown = await _answer(
            model, ObserveTurn, study, self._tools, self._store, held=held, trigger=trigger
        )
        points, proposals, dropped = await _checked(
            answer, study, self._tools, self._store, decoded, shown
        )
        say = answer.say.strip()
        study.thread.summary = answer.summary.strip()
        study.say(
            Turn(
                role="agent",
                kind="observe",
                say=say,
                about=step.id,
                points=tuple(points),
                proposals=tuple(proposals),
                unresolved=tuple(answer.unresolved),
                looked=looked,
                dropped=dropped,
                silent=not (
                    say or answer.claims or answer.progress or answer.proposals or answer.pause
                ),
                written_by=str(model.choice),
                at=datetime.now(UTC),
            )
        )
        await self._studies.update(study)
        return answer.pause and bool(say)

    async def _unobserved(
        self, workspace_id: uuid.UUID, study_id: uuid.UUID, step_id: uuid.UUID, error: Exception
    ) -> None:
        """Say in the thread that the step went unobserved, and why, so the gap shows."""
        try:
            # Read afresh: the failed turn may have left its own copy half-written.
            study = await self._studies.get(workspace_id, study_id)
            if study is not None:
                said = str(error) or type(error).__name__
                study.say(
                    Turn(
                        role="event",
                        kind="observe_failed",
                        say=said,
                        about=step_id,
                        at=datetime.now(UTC),
                    )
                )
                await self._studies.update(study)
        except Exception:
            logger.exception(
                "Study %s could not record that step %s went unobserved", study_id, step_id
            )
