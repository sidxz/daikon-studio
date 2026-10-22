"""What an agent may offer the study to do, and what is checked before it is offered.

An agent that can only talk ends every useful exchange the same way: it tells the
scientist to rerun preparation without the zinc, and the scientist scrolls off to
find the button. The change it wants is one it can name exactly, so it hands over a
proposal instead, and the scientist's press is the word that carries it out.

A proposal is checked by the same checks as the edit it would make, so a button is
never offered that pressing could not carry out: its step must be in the plan, a swap
must be to one of the step's alternatives, and settings must be ones the tool takes and
accepts. Two more hold because a proposal waits to be pressed. A new step must read only
what the study already holds, since a step whose inputs nothing has produced would be
offering a failure. And a candidate can be approved only on a finished result of a tool
that reports candidates. What fails is dropped and counted rather than shown.

**Checks and stop conditions are not proposable at all.** They are the reason a result
from this study can be trusted, and a criterion rewritten after seeing the data is not a
criterion. `Proposal` has no way to carry one.

A proposal keeps its steps by id rather than by number, so a button pressed after the
plan was edited still means the step it meant when it was offered.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from daikonstudio.application.studies.edits import _EDITS, _name, _Named, _tool
from daikonstudio.application.studies.schemas import Action, Proposal
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.studies.study import Proposed, Step, StepStatus, Study


def _offerable(named: _Named, study: Study, registry: ToolRegistry) -> bool:
    action, step = named.action, named.step
    if action.does == "run":
        return True  # its steps are in the plan; where it stops is the loop's to find
    if action.does == "approve":
        # Only a judgment tool says what a chosen candidate becomes.
        manifest = None if step is None else _tool(registry, step.tool_id)
        return (
            step is not None
            and manifest is not None
            and manifest.resolves_to is not None
            and action.candidate is not None
            and step.status is StepStatus.DONE
            and bool(study.outputs_of(step.id))
        )
    if action.does == "insert":
        manifest = _tool(registry, action.tool_id)
        held = {value.kind for value in study.values}
        if manifest is None or any(
            slot.kind.value not in held for slot in manifest.inputs if not slot.optional
        ):
            return False
    # The edit's own checks. They only build the change; nothing is made here.
    return _EDITS[action.does](study, registry, named) is not None


def _plain(named: _Named) -> dict[str, Any]:
    """The proposal as its button keeps it: every step by id, every setting by name."""
    action = named.action

    def ref(step: Step | None) -> str | None:
        return None if step is None else str(step.id)

    return {
        "does": action.does,
        "step_id": ref(named.step),
        "tool_id": action.tool_id,
        "before_id": ref(named.before),
        "purpose": action.purpose,
        "settings": {setting.name: setting.value for setting in action.settings},
        "gated": action.gated,
        "candidate": action.candidate,
        "until_id": ref(named.until),
    }


def vet(
    proposals: Sequence[Proposal], study: Study, registry: ToolRegistry
) -> tuple[list[Proposed], int]:
    """Keep the proposals this study could carry out as it stands, and count the rest.

    Positions mean the plan as it stands now, as the model was shown it.
    """
    shown = {step.position: step for step in study.steps}
    kept: list[Proposed] = []
    dropped = 0
    for proposal in proposals:
        named = _name(Action.model_validate(proposal.model_dump(exclude={"why"})), shown)
        if named is None or not _offerable(named, study, registry):
            dropped += 1
            continue
        kept.append(Proposed(id=uuid.uuid4(), action=_plain(named), why=proposal.why))
    return kept, dropped
