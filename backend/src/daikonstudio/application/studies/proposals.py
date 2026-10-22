"""What an agent may ask a study to do, and what is checked before it is offered.

An agent that can only talk ends every useful exchange the same way: it tells the
scientist to rerun preparation without the zinc, and the scientist scrolls off to
find the button. The step it wants is one it can name exactly, so it can hand
over a proposal instead.

A proposal is where this stops. The scientist confirms and the study acts; the
agent never acts on its own. That is the same rule as "a core is approved, never
computed", applied to the plan rather than to a value, and it is what keeps a
plan something written before the data rather than something that drifted to fit
it.

**Checks and stop conditions are not proposable at all.** They are the reason a
result from this study can be trusted, and a criterion rewritten after seeing the
data is not a criterion. If one is wrong the scientist amends it themselves, in
the open, and what that costs is theirs to judge.

A proposal is checked the way a claim is: the step must exist, the tool must be
one the catalogue has, its inputs must be things this study already holds, and
its settings must survive that tool's own parameter model. What fails is dropped
and counted rather than shown.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from daikonstudio.application.planner.plan import Setting
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.studies.study import Proposed, Study


class Proposal(BaseModel):
    """One thing the agent would like done, for the scientist to confirm."""

    does: Literal["rerun", "add"] = Field(
        description=(
            "'rerun' runs a step that is already in the plan, optionally with "
            "different settings. 'add' puts a new step at the end of the plan."
        )
    )
    step: int | None = Field(default=None, description="For a rerun: which step, by its position.")
    tool_id: str | None = Field(
        default=None,
        description="For an addition: the tool's id, exactly as the catalogue has it.",
    )
    settings: list[Setting] = Field(
        default_factory=list,
        description=(
            "The settings to run with, as name/value pairs. Leave empty to run a "
            "step exactly as it stands. Only names that tool actually takes."
        ),
    )
    why: str = Field(description="One sentence: what this would settle that is open now.")


def _settings(proposal: Proposal) -> dict[str, str]:
    return {s.name: s.value for s in proposal.settings}


def vet(
    proposals: Sequence[Proposal], study: Study, tools: ToolRegistry
) -> tuple[list[Proposed], int]:
    """Keep the proposals this study could actually carry out."""
    kept: list[Proposed] = []
    dropped = 0
    # What the study already holds, which is what a new step could read.
    available = {value.kind for value in study.values}

    for proposal in proposals:
        settings = _settings(proposal)
        if proposal.does == "rerun":
            step = next((s for s in study.steps if s.position == proposal.step), None)
            if step is None:
                dropped += 1
                continue
            tool_id = step.tool_id
        else:
            tool_id = proposal.tool_id or ""
            step = None

        try:
            manifest = tools.get(tool_id)
        except Exception:
            dropped += 1
            continue

        if proposal.does == "add":
            # Automatic steps are placed, never chosen -- by the planner or by an
            # agent. And a step whose inputs nothing has produced cannot run, so
            # offering it would be offering a failure.
            if manifest.automatic:
                dropped += 1
                continue
            if any(
                slot.kind.value not in available for slot in manifest.inputs if not slot.optional
            ):
                dropped += 1
                continue

        if settings:
            if manifest.params is None:
                dropped += 1
                continue
            # Names first: a model ignores what it does not declare, so
            # validation alone accepts an invented setting and the step runs
            # without it. The scientist would press a button that silently does
            # nothing of what it said.
            if any(name not in manifest.params.model_fields for name in settings):
                dropped += 1
                continue
            try:
                manifest.params.model_validate(settings)
            except ValidationError:
                dropped += 1
                continue

        kept.append(
            Proposed(
                id=uuid.uuid4(),
                does=proposal.does,
                step=step.position if step is not None else None,
                tool_id=tool_id,
                settings=settings,
                why=proposal.why,
            )
        )
    return kept, dropped
