"""The conversation about the whole study, and how it knows where things stand.

A stage agent sees one result very closely and nothing else. Someone has to hold
the other view: which steps have run, what the plan said would count as a result,
which of those are met, and whether a stop condition has fired. That is a
different question from "which cavity", and asking the stage agent to answer it
would mean handing it the whole study and hoping.

It reads each stage's `handoff` rather than the stages' exchanges. That is what a
handoff is for: one paragraph per stage, rewritten each turn, so this agent's
context stays the size of the plan instead of the size of everything anyone has
said. A stage that has not been talked about contributes nothing, which is
honest -- nobody has read that result yet.

Bound by the same rule as a stage agent. It may not state a number, and every
claim it makes points at something in the study: a step, a check, or a stop
condition, cited by the exact words the plan used. A claim that cites a check the
plan does not contain is dropped before anyone reads it, which is the whole
reason this is safe to put next to a decision.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field
from returns.result import Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.ports.llm import LanguageModels
from daikonstudio.application.ports.study_repository import StudyRepository
from daikonstudio.application.studies.get_study import GetStudy
from daikonstudio.application.studies.proposals import Proposal, vet
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.shared.errors import DomainError
from daikonstudio.domain.studies.study import Pointer, StepStatus, Study, Thread, Turn

_CONTEXT_TURNS = 12

About = Literal["step", "check", "stop_condition"]


class ProgressClaim(BaseModel):
    """One thing the agent says about where the study stands."""

    about: About = Field(description="What this is about: a step, a check, or a stop condition.")
    refers_to: list[int] = Field(
        description=(
            "Which ones, by number. A step is its position. A check or a stop "
            "condition is its place in the list shown to you, counting from 1."
        )
    )
    label: str = Field(description="What you are saying about them, in a few words.")
    cites: list[str] = Field(
        description=(
            "The exact words you are relying on: the step's name, or the check or "
            "stop condition as the plan wrote it. Anything not found costs the claim."
        )
    )


class StudyTurn(BaseModel):
    """What the study agent may produce. No field here holds a number."""

    say: str = Field(description="What you want to tell the scientist.")
    claims: list[ProgressClaim]
    proposes: list[str] = Field(description="Tool ids worth running next, if any.")
    proposals: list[Proposal] = Field(
        default_factory=list,
        description=(
            "What you would like done about this, for the scientist to confirm. "
            "Propose only what the open question actually needs; an exchange that "
            "settles nothing needs none."
        ),
    )
    unresolved: list[str] = Field(description="What cannot be settled yet, one line each.")
    handoff: str = Field(
        description="Where the study stands, in one paragraph, for someone opening it cold."
    )


_SYSTEM = """\
You are the agent for a scientist's whole study in Daikon Studio. The stages have \
their own agents, each looking at one result. You are not looking at results. You \
are holding the plan: what was going to count as an answer, what has run, what \
each stage concluded, and whether this study is still on the path it set itself.

Rules, in order of importance.

1. You may not state a number. Not a score, not a count, not a measurement. Every \
number in this study belongs to a step that computed it, and a number repeated by \
you is one that can be wrong in a second way. Refer to a step by its position and \
to a check by its place in the list.

2. Every claim must survive checking. Each one says what it is about, which ones \
by number, and cites the exact words of the step name, check or stop condition it \
rests on. A claim citing a check the plan does not contain is dropped before the \
scientist sees it, and the drop is shown to them.

3. The checks were written before anything ran, so they are the only fair test of \
whether this study answered its question. Say plainly which are met, which are \
not, and which cannot be told yet. An unmet check is the most useful thing you \
can report, not a failure to soften.

4. A stop condition that has fired outranks everything else. Say so first, and say \
what it asks for.

5. What you know about a result is what that stage's agent wrote in its summary. \
Where a stage has no summary, nobody has read that result; say that rather than \
guessing what it showed.

6. Do not invent progress. A step that has not run has not run, however reasonable \
its result would be.
"""


def _check_progress(claims: Sequence[ProgressClaim], study: Study) -> tuple[list[Pointer], int]:
    """Keep only the claims the plan and the steps actually carry."""
    kept: list[Pointer] = []
    dropped = 0
    sources: dict[str, list[str]] = {
        "step": [s.name for s in sorted(study.steps, key=lambda s: s.position)],
        "check": list(study.checks),
        "stop_condition": list(study.stop_conditions),
    }
    for claim in claims:
        available = sources[claim.about]
        # Steps are cited by position, checks and stop conditions by their place
        # in the list the agent was shown, which starts at one.
        indexes = [n if claim.about == "step" else n - 1 for n in dict.fromkeys(claim.refers_to)]
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
                at=f"{claim.about.replace('_', ' ')}{'s' if len(indexes) > 1 else ''} {where}",
                label=claim.label[:120],
                evidence=claim.about,
                cites=tuple(claim.cites),
            )
        )
    return kept, dropped


def _standing(study: Study) -> str:
    """The study as the agent reads it: the plan, the steps, and what each stage said."""
    lines = [f"The goal, as Studio understood it: {study.restated_goal}", "", "The steps:"]
    for step in sorted(study.steps, key=lambda s: s.position):
        produced = [v.kind.replace("_", " ") for v in study.current_outputs_of(step.position)]
        lines.append(
            f"  {step.position}. {step.name} — {step.status.value}"
            + (f", produced {', '.join(produced)}" if produced else "")
            + (f", error: {step.error}" if step.error else "")
        )
    if study.checks:
        lines += ["", "What would count as a result, written before anything ran:"]
        lines += [f"  {i}. {c}" for i, c in enumerate(study.checks, start=1)]
    if study.stop_conditions:
        lines += ["", "When the plan said to stop instead:"]
        lines += [f"  {i}. {c}" for i, c in enumerate(study.stop_conditions, start=1)]

    lines += ["", "What each stage's agent concluded:"]
    summaries = [t for t in study.conversations if t.position is not None and t.handoff]
    if not summaries:
        lines.append("  Nothing. No result has been read by its stage agent yet.")
    for thread in sorted(summaries, key=lambda t: t.position or 0):
        named = next((s for s in study.steps if s.position == thread.position), None)
        lines.append(
            f"  Step {thread.position} ({named.name if named else '?'}): {thread.handoff}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class AskStudyCommand:
    study_id: uuid.UUID
    message: str | None = None
    provider: str | None = None


class StudyChat:
    def __init__(
        self, studies: StudyRepository, models: LanguageModels, tools: ToolRegistry
    ) -> None:
        self._studies = studies
        self._get = GetStudy(studies)
        self._models = models
        self._tools = tools

    async def __call__(
        self, command: AskStudyCommand, auth: AuthContext | None = None
    ) -> Result[Thread, DomainError]:
        require_authenticated(auth)
        found = await self._get(command.study_id, auth)
        if not isinstance(found, Success):
            return found  # type: ignore[return-value]
        study = found.unwrap()

        if command.message:
            study.say(
                None,
                Turn(role="scientist", say=command.message.strip(), at=datetime.now(UTC)),
            )

        model = self._models.get(command.provider)
        answer = await model.parse(system=_SYSTEM, prompt=self._prompt(study), schema=StudyTurn)

        points, dropped = _check_progress(answer.claims, study)
        offered, unusable = vet(answer.proposals, study, self._tools)
        dropped += unusable
        known = {m.id for m in self._tools.manifests()}
        proposes = tuple(t for t in answer.proposes if t in known)
        dropped += len(answer.proposes) - len(proposes)

        thread = study.say(
            None,
            Turn(
                role="agent",
                say=answer.say.strip(),
                points=tuple(points),
                proposes=proposes,
                unresolved=tuple(answer.unresolved),
                proposals=tuple(offered),
                dropped=dropped,
                written_by=str(model.choice),
                at=datetime.now(UTC),
            ),
        )
        thread.handoff = answer.handoff.strip()
        await self._studies.update(study)
        return Success(thread)

    def _prompt(self, study: Study) -> str:
        parts = [_standing(study)]
        done = sum(1 for s in study.steps if s.status is StepStatus.DONE)
        thread = study.thread(None)
        if thread and thread.turns:
            said = "\n".join(f"{t.role}: {t.say}" for t in thread.turns[-_CONTEXT_TURNS:])
            parts.append(f"What has been said about the study so far:\n{said}")
        else:
            parts.append(
                "Nothing has been said about the study yet. Open it: say where this "
                "study stands against the checks it set itself, and what it needs next."
                if done
                else "Nothing has run yet. Say what the plan will and will not settle."
            )
        return "\n\n".join(parts)
