"""The conversation at one stage of a study, and what it is allowed to say.

A judgement step stops and asks the scientist to decide. What it can hand them is
a table, and a table is not a decision: on the three targets this was tried
against, the tool's own ranking put the site a drug binds twelfth, first and
tenth, while surface grooves outscored all three. Telling them apart takes what
the tool does not have, which is knowing that His94 and His96 hold a zinc.

So there is an agent at each stage, bound by the rule every judgement step is
bound by: prose, choices and proposed steps, never numbers. A choice is expressed
as a pointer into the result -- which candidates, what the agent believes they
are, and the fields it read -- and every pointer is checked against that step's
own output before it is stored. A claim the result does not support is dropped
and counted, not shown.

Checking is the whole reason an agent is safe to have here, so a stage can be
talked about only where its result has a checker. One kind has one today, and
adding a kind means writing its checker, not loosening this.

The agent reads the value exactly as the page renders it, so it can never cite
something the scientist cannot see for themselves.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.ports.llm import LanguageModels
from daikonstudio.application.ports.study_repository import StudyRepository
from daikonstudio.application.studies.get_study import GetStudy
from daikonstudio.application.studies.proposals import Proposal, vet
from daikonstudio.application.studies.read_value import ReadValue
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.shared.errors import DomainError, NotFoundError, ValidationError
from daikonstudio.domain.studies.study import Pointer, Step, StepStatus, Study, Thread, Turn

# Enough of the exchange for the agent to not repeat itself, not so much that a
# long stage quietly becomes the most expensive call in the study.
_CONTEXT_TURNS = 12

Evidence = Literal["ligand_overlap", "annotated_site", "interface", "lining_residues"]


class Claim(BaseModel):
    """One thing the agent says about the result, as a reference into it."""

    candidates: list[int] = Field(
        description=(
            "The candidates this is about, by rank. Several ranks mean you are "
            "saying they are one site that the tool reported separately."
        )
    )
    label: str = Field(
        description="What this is, named the way a chemist would: 'the zinc cone'.",
    )
    evidence: Evidence = Field(description="Which column of the result carries this.")
    cites: list[str] = Field(
        description=(
            "The lining residues you are relying on, spelled exactly as the result "
            "spells them. Anything not found there costs you the whole claim."
        )
    )


class StageTurn(BaseModel):
    """What the agent may produce. There is no field here for a number."""

    say: str = Field(description="What you want to tell the scientist.")
    claims: list[Claim]
    proposes: list[str] = Field(
        description="Tool ids worth running next, if any. Must be tools that exist."
    )
    proposals: list[Proposal] = Field(
        default_factory=list,
        description=(
            "What you would like done about this, for the scientist to confirm. "
            "Propose only what the open question actually needs; an exchange that "
            "settles nothing needs none."
        ),
    )
    unresolved: list[str] = Field(
        description="What this result cannot settle, one short line each."
    )
    handoff: str = Field(
        description=(
            "The one paragraph another agent reads instead of this whole exchange: "
            "what this stage settled, on what evidence, and what it did not."
        ),
    )


_SYSTEM = """\
You are the agent at one stage of a scientist's study in Daikon Studio. A step has \
run and produced a result, and the scientist is about to decide something on the \
back of it. You are talking to them about that result and nothing else.

Rules, in order of importance.

1. You may not state a number. Not a score, not a volume, not a count. The result \
below already carries every number, the page already shows them, and a number \
repeated by you is a number that can be wrong in a second way. Refer to a \
candidate by its rank and let the table say the rest.

2. Every claim must survive checking. Each one names candidates by rank and cites \
lining residues spelled exactly as the result spells them. A claim citing a \
residue that candidate does not have is dropped before the scientist sees it, and \
the drop is shown to them. A guess costs you the claim and helps nobody.

3. Do not read the ranking as a recommendation. Rank and score come from the \
tool's geometric model, which does not know what the protein does. A surface \
groove routinely outranks the site a drug binds. Judge each candidate on the \
residues that line it, the ligands it touches and the annotated sites it contains.

4. Say when several candidates are one site. An extended cavity is often split \
across candidates sitting far apart in a ranked list, and a reader of the table \
alone will take them for separate sites. Put their ranks in one claim.

5. Claim nothing rather than something weak. An empty list is an answer, and a \
better one than naming the top candidate because it is top.

6. Write `handoff` for another agent, not for the scientist: what this stage \
settled, on what evidence, and what it could not. It is read instead of this \
exchange, so anything left out is lost.

7. Answer what the scientist actually asked. If they ask something this result \
cannot settle, say so and put it in `unresolved` rather than reaching.
"""


def _lining(pocket: Mapping[str, Any]) -> set[str]:
    rows = pocket.get("residues") or []
    return {_key(f"{r.get('resname', '')}{r.get('resseq', '')}") for r in rows}


def _key(name: str) -> str:
    """A residue as written anywhere: 'A:HIS94', 'His 94' and 'HIS94' are one."""
    return name.split(":")[-1].replace(" ", "").upper()


def _supported(evidence: str, pockets: Sequence[Mapping[str, Any]]) -> bool:
    if evidence == "ligand_overlap":
        return any(not o.get("additive") for p in pockets for o in (p.get("ligand_overlap") or []))
    if evidence == "annotated_site":
        return any(p.get("annotated") for p in pockets)
    if evidence == "interface":
        return any(p.get("interface") for p in pockets)
    return True  # lining_residues: the residues themselves are checked below


def _check_pockets(claims: Sequence[Claim], data: Mapping[str, Any]) -> tuple[list[Pointer], int]:
    """Keep only the claims this pocket result actually supports."""
    pockets = data.get("pockets") or []
    by_rank = {int(p["rank"]): p for p in pockets if "rank" in p}
    kept: list[Pointer] = []
    dropped = 0
    for claim in claims:
        named = [by_rank[r] for r in dict.fromkeys(claim.candidates) if r in by_rank]
        if not named or len(named) != len(set(claim.candidates)):
            dropped += 1
            continue
        lining: set[str] = set()
        for pocket in named:
            lining |= _lining(pocket)
        # A claim with no residues behind it is an opinion, not a reading.
        if not claim.cites or any(_key(c) not in lining for c in claim.cites):
            dropped += 1
            continue
        if not _supported(claim.evidence, named):
            dropped += 1
            continue
        ranks = ", ".join(str(r) for r in dict.fromkeys(claim.candidates))
        kept.append(
            Pointer(
                at=f"rank{'s' if len(named) > 1 else ''} {ranks}",
                label=claim.label,
                evidence=claim.evidence,
                cites=tuple(claim.cites),
            )
        )
    return kept, dropped


# A stage can be talked about only where its result can be checked. Adding a kind
# means writing its checker; there is deliberately no permissive default.
CHECKERS: dict[str, Callable[[Sequence[Claim], Mapping[str, Any]], tuple[list[Pointer], int]]] = {
    "pocket_candidates": _check_pockets,
}


def _describe(data: Mapping[str, Any]) -> str:
    """The result as the agent reads it: every candidate, with what lines it."""
    meta = data.get("meta") or {}
    pockets = data.get("pockets") or []
    lines = [
        f"Kept in the receptor: {', '.join(meta.get('receptor_ligands') or []) or 'nothing'}.",
        f"Numbering against UniProt: {meta.get('numbering') or 'not checked'}.",
        "",
    ]
    for pocket in pockets:
        overlap = ", ".join(
            f"{o.get('comp_id')}{' (additive)' if o.get('additive') else ''} "
            f"{o.get('shared')}/{o.get('of')}"
            for o in (pocket.get("ligand_overlap") or [])
        )
        annotated = "; ".join(
            f"{a.get('type')} {a.get('start')}" for a in (pocket.get("annotated") or [])
        )
        residues = ", ".join(
            f"{r.get('resname')}{r.get('resseq')}" for r in (pocket.get("residues") or [])
        )
        lines.append(
            f"Candidate {pocket.get('rank')}: druggability {pocket.get('druggability')}, "
            f"volume {pocket.get('volume')}, "
            f"interface {'yes' if pocket.get('interface') else 'no'}"
        )
        lines.append(f"  lines: {residues or '(none)'}")
        lines.append(f"  ligands: {overlap or '(none)'}")
        lines.append(f"  annotated: {annotated or '(none)'}")
    return "\n".join(lines)


@dataclass(frozen=True)
class SayCommand:
    study_id: uuid.UUID
    position: int
    message: str | None = None
    provider: str | None = None


class StageChat:
    def __init__(
        self,
        studies: StudyRepository,
        values: ReadValue,
        models: LanguageModels,
        tools: ToolRegistry,
    ) -> None:
        self._studies = studies
        self._get = GetStudy(studies)
        self._values = values
        self._models = models
        self._tools = tools

    async def __call__(
        self, command: SayCommand, auth: AuthContext | None = None
    ) -> Result[Thread, DomainError]:
        require_authenticated(auth)
        found = await self._get(command.study_id, auth)
        if not isinstance(found, Success):
            return found  # type: ignore[return-value]
        study = found.unwrap()

        step = next((s for s in study.steps if s.position == command.position), None)
        if step is None:
            return Failure(NotFoundError("Step", str(command.position)))
        if step.status is not StepStatus.DONE:
            return Failure(ValidationError("There is nothing to talk about until the step runs."))

        value = next(
            (
                v
                for v in sorted(study.values, key=lambda v: v.created_at)
                if v.produced_by == step.position
            ),
            None,
        )
        if value is None or value.kind not in CHECKERS:
            return Failure(
                ValidationError("This step's result cannot be checked, so it is not discussed.")
            )

        read = await self._values(command.study_id, value.id, auth=auth)
        if not isinstance(read, Success):
            return read  # type: ignore[return-value]
        data = read.unwrap().data
        if not isinstance(data, dict):
            return Failure(ValidationError("This step's result cannot be checked."))

        now = datetime.now(UTC)
        if command.message:
            study.say(
                command.position,
                Turn(role="scientist", say=command.message.strip(), at=now),
            )

        model = self._models.get(command.provider)
        answer = await model.parse(
            system=_SYSTEM,
            prompt=self._prompt(study, step, data),
            schema=StageTurn,
        )

        points, dropped = CHECKERS[value.kind](answer.claims, data)
        # Trimmed here, never constrained on the answer: the provider validates
        # the schema at its end, so a label one word too long would fail the
        # whole call rather than arrive slightly long.
        points = [
            Pointer(at=p.at, label=p.label[:120], evidence=p.evidence, cites=p.cites)
            for p in points
        ]
        offered, unusable = vet(answer.proposals, study, self._tools)
        dropped += unusable
        known = {m.id for m in self._tools.manifests()}
        proposes = tuple(t for t in answer.proposes if t in known)
        dropped += len(answer.proposes) - len(proposes)

        thread = study.say(
            command.position,
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

    def _prompt(self, study: Study, step: Step, data: Mapping[str, Any]) -> str:
        parts = [
            f"The study's goal, as Studio understood it:\n{study.restated_goal}",
            f"This stage is step {step.position}, {step.name}."
            f"\nIts purpose, as planned: {step.purpose}",
        ]
        if study.checks:
            parts.append(
                "What the plan said would count as a result, written before anything ran:\n"
                + "\n".join(f"- {c}" for c in study.checks)
            )
        if study.stop_conditions:
            parts.append(
                "When the plan said to stop instead:\n"
                + "\n".join(f"- {c}" for c in study.stop_conditions)
            )
        parts.append(f"The result, exactly as the scientist sees it:\n{_describe(data)}")
        thread = study.thread(step.position)
        if thread and thread.turns:
            said = "\n".join(f"{t.role}: {t.say}" for t in thread.turns[-_CONTEXT_TURNS:])
            parts.append(f"What has been said at this stage so far:\n{said}")
        else:
            parts.append(
                "Nothing has been said yet. Open the conversation: tell them what you "
                "make of this result and what you would look at first."
            )
        return "\n\n".join(parts)
