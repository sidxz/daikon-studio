"""Changing a plan: the edits an agent, a proposal and a button all share.

The model names steps by the number it was shown and tools by id, and none of it is
believed until the study and the catalogue agree. What does not hold is dropped and
counted, never raised, so a turn with six good edits and one bad one keeps the six.
Nothing here calls a model or reads a blob; approving a candidate and running need the
blob store and the run loop, so they are handed back for the caller.

Three rules keep an edit from doing more than was asked.

Positions mean the plan as the model was shown it. "Remove step 2 and gate step 3"
means the step that was 3 when it wrote that, so every position is resolved to its
step before the first edit is made. What is handed back is renumbered to where the
steps stand afterwards, since that is the plan the caller will look them up in.

An edit made on the scientist's word has to quote it. One that does not is handed back
as a proposal, for the scientist to press instead. That is a floor, not a proof ("ok"
quotes anything), but it keeps the agent to what was said rather than to its own last
suggestion.

Always-run steps are placed, never chosen: the model can neither insert one nor remove
one. After every batch they are matched to the plan again, put in where it now needs them
and taken out where it no longer does, and the plan is rechecked, so whatever an edit
broke is pinned to the step it broke.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from pydantic import ValidationError

from daikonstudio.application.planner.plan import (
    PlannedStep,
    automatic_steps,
    handed_over,
    incoherences,
)
from daikonstudio.application.studies.schemas import Action, Proposal
from daikonstudio.application.tools.manifest import SlotKind, ToolManifest
from daikonstudio.application.tools.registry import ToolRegistry, UnknownToolError
from daikonstudio.domain.shared.errors import ConflictError
from daikonstudio.domain.studies.study import Acted, Step, StepStatus, Study

# The steps still open: no result to keep, so an edit may take one out, and a problem may sit
# on one. Whatever the plan now says of a step that has run, or is running, it ran.
_OPEN = (StepStatus.PENDING, StepStatus.SKIPPED, StepStatus.FAILED)


def alternatives(manifest: ToolManifest, registry: ToolRegistry) -> list[ToolManifest]:
    """The tools that could take this one's place in a plan.

    Derived, never declared: the same required inputs and the same outputs, so whatever
    follows the step is fed either way, and nothing here restates the catalogue. An
    automatic tool is placed rather than chosen and a composite is a plan fragment
    rather than a step, so neither is offered.
    """

    def signature(tool: ToolManifest) -> tuple[set[SlotKind], set[SlotKind]]:
        return (
            {slot.kind for slot in tool.inputs if not slot.optional},
            {slot.kind for slot in tool.outputs},
        )

    wanted = signature(manifest)
    return [
        tool
        for tool in registry.manifests()
        if tool.id != manifest.id
        and not tool.automatic
        and not tool.expands_to
        and signature(tool) == wanted
    ]


def quoted(quoting: str, message: str) -> bool:
    """Whether `quoting` is words from `message`, ignoring case and spacing.

    Nothing quotes nothing, so an empty quote is never found.
    """
    words = _plain(quoting)
    return bool(words) and words in _plain(message)


def _plain(text: str) -> str:
    return " ".join(text.split()).casefold()


def to_steps(planned: Sequence[PlannedStep], registry: ToolRegistry) -> list[Step]:
    """Study steps for a plan that has passed `incoherences`.

    Fresh ids, and the name and version the catalogue gives rather than the view's, so a
    step records the tool it was made from. Every tool must be one the registry has.
    """
    steps: list[Step] = []
    for position, item in enumerate(planned):
        manifest = registry.get(item.tool_id)
        steps.append(
            Step(
                id=uuid.uuid4(),
                position=position,
                tool_id=manifest.id,
                tool_version=manifest.version,
                name=manifest.name,
                purpose=item.purpose,
                params=dict(item.params),
                gated=item.gated,
                always=item.always,
                expanded_from=item.expanded_from,
            )
        )
    return steps


def _brought(study: Study) -> set[SlotKind]:
    """The kinds the scientist brought: what a plan can lean on before any step has run."""
    kinds = {value.kind for value in study.values if value.produced_by is None}
    return {kind for kind in SlotKind if kind.value in kinds}


def _lacks(manifest: ToolManifest, available: set[SlotKind]) -> set[SlotKind]:
    """The kinds the tool must have and does not yet have in hand."""
    return {
        slot.kind for slot in manifest.inputs if not slot.optional and slot.kind not in available
    }


def replace_automatic(study: Study, registry: ToolRegistry) -> None:
    """Match the always-run steps to what the plan now has to prepare.

    `expand` places them in a fresh plan; an edit can change the need afterwards, and the
    model neither places them nor removes them. One the plan has come to need goes in
    right after the first step that makes what it needs, or at the front when the
    scientist brought it, so it runs before anything else touches the thing it prepares.
    One still waiting that has lost what it prepares, say because the step that fetched
    the structure was removed, comes out when no chosen step wants that kind any more: the
    model may not remove it, and it would only sit there with a problem. While one does,
    it stays, as `expand` places it for a file that has not arrived yet, and its problem
    shows until the file does. Whatever brings a file calls this again, so that the
    preparation the file makes possible is placed.
    """
    steps = list(study.steps)  # the plan as it was; changes are made as it is walked
    available = _brought(study)
    already = {step.tool_id for step in steps}
    # What the chosen steps expect to be handed over, read as `expand` reads a draft.
    wanted = handed_over([step.tool_id for step in steps if not step.always], registry)
    for index in range(-1, len(steps)):  # -1 is before the first step
        if index >= 0:
            step = steps[index]
            try:
                manifest = registry.get(step.tool_id)
            except UnknownToolError:
                continue  # `recheck` reports it; there is nothing of its to count
            lacks = _lacks(manifest, available)
            orphaned = bool(lacks) and not (lacks & wanted)  # and no chosen step expects it
            if step.always and step.status is StepStatus.PENDING and orphaned:
                study.remove_step(step.id)
                already.discard(step.tool_id)  # it may yet be placed where its input appears
                continue  # and what it would have made is not in hand
            available.update(slot.kind for slot in manifest.outputs)
            if manifest.resolves_to is not None:
                available.add(manifest.resolves_to)
        placed = automatic_steps(available, registry, already)
        already.update(item.tool_id for item in placed)
        # Whatever followed this step before anything was added still does: what has been
        # added went in after earlier steps.
        follower = steps[index + 1].id if index + 1 < len(steps) else None
        for step in to_steps(placed, registry):
            study.insert_step(step, before=follower)


def recheck(study: Study, registry: ToolRegistry) -> None:
    """Pin to each step the reason, if any, the plan cannot go ahead with it.

    `incoherences` reads the steps in order against what the scientist brought and says
    which wants something nothing before it makes, or has settings its tool refuses. Only
    a step still waiting (pending, skipped or failed) carries one: whatever the plan now
    says of a step that has run, or is running, it ran. Nor is anything pinned for the plan
    as a whole; "the plan has no steps" is not a step's.
    """
    view = [
        PlannedStep(
            tool_id=step.tool_id,
            name=step.name,
            purpose=step.purpose,
            gated=step.gated,
            params=step.params,
            expanded_from=step.expanded_from,
            always=step.always,
        )
        for step in study.steps
    ]
    found: dict[int, list[str]] = {}
    for problem in incoherences(view, registry, starting_with=_brought(study)):
        if problem.step is not None:
            found.setdefault(problem.step, []).append(problem.message)
    for position, step in enumerate(study.steps, start=1):
        messages = found.get(position) if step.status in _OPEN else None
        step.problem = " ".join(messages) if messages else None


@dataclass(frozen=True)
class Applied:
    """What `apply_actions` did, and what it leaves for the caller.

    `acted` is each change made, in order. `downgraded` is each valid change that did not
    quote the message, as a proposal. `approvals` is the `(step id, candidate)` of each
    approval to write, and `run` the one run to start; neither is done here. The steps in
    `downgraded` and `run` are numbered as the plan stands after the edits. `dropped`
    counts what could not be done at all.
    """

    acted: list[Acted]
    downgraded: list[Proposal]
    approvals: list[tuple[uuid.UUID, int]]
    run: Action | None
    dropped: int


@dataclass(frozen=True)
class _Named:
    """An action, and the steps its positions named in the plan as the model was shown it.

    `later` is the steps shown from `before` on, in order, for an insert whose step to go
    before an earlier edit in the same turn took out.
    """

    action: Action
    step: Step | None
    before: Step | None
    until: Step | None
    later: tuple[Step, ...] = ()


# A change that has been checked and is ready to make; None as the result is one with
# nothing to record here. Making it can still fail, if the step has run or an earlier
# edit in the same turn took it away.
type _Do = Callable[[], Acted | None]
type _Edit = Callable[[Study, ToolRegistry, _Named], _Do | None]


# The positions each kind of action reads. One it does not read is the model's slip, not a
# reason to throw the action away.
_READS: dict[str, frozenset[str]] = {
    "insert": frozenset({"before"}),
    "remove": frozenset({"step"}),
    "swap": frozenset({"step"}),
    "settings": frozenset({"step"}),
    "gate": frozenset({"step"}),
    "terms": frozenset(),
    "approve": frozenset({"step"}),
    "run": frozenset({"step", "until"}),
}


def _name(action: Action, shown: Mapping[int, Step]) -> _Named | None:
    """Find the steps the positions this action reads name; None if one names no step.

    0 names none, as the model is told; any other number has to be a step it was shown.
    """
    reads = _READS[action.does]
    positions = (
        action.step if "step" in reads else 0,
        action.before if "before" in reads else 0,
        action.until if "until" in reads else 0,
    )
    if any(p and p - 1 not in shown for p in positions):
        return None
    step, before, until = (shown[p - 1] if p else None for p in positions)
    # By the shown position, not the step's own: an earlier edit may have renumbered it.
    later = tuple(shown[i] for i in sorted(shown) if before and i >= positions[1] - 1)
    return _Named(action, step, before, until, later)


def _tool(registry: ToolRegistry, tool_id: str) -> ToolManifest | None:
    try:
        return registry.get(tool_id)
    except UnknownToolError:
        return None


def _given(action: Action) -> dict[str, str]:
    return {setting.name: setting.value for setting in action.settings}


def _takes(manifest: ToolManifest, params: dict[str, Any], given: dict[str, str]) -> bool:
    """Whether the tool takes the settings that were just given, as `params` would stand.

    Names first: a model ignores what it does not declare, so validation alone would accept
    an invented setting and the step would run without it. Then the values, judged only
    where they were just given. A required setting still missing is the plan check's to
    report on the step, so a swapped step can be mended one setting at a time.
    """
    if manifest.params is None:
        return not given
    if any(name not in manifest.params.model_fields for name in given):
        return False
    try:
        manifest.params.model_validate(params)
    except ValidationError as error:
        return not any(e["loc"] and e["loc"][0] in given for e in error.errors())
    return True


def _label(step: Step) -> str:
    return f"step {step.position + 1} ({step.name})"


def _listed(given: Mapping[str, str]) -> str:
    return ": " + ", ".join(f"{name}={value}" for name, value in given.items()) if given else ""


def _insert(study: Study, registry: ToolRegistry, named: _Named) -> _Do | None:
    action = named.action
    manifest = _tool(registry, action.tool_id)
    # Always-run steps are placed, never chosen.
    if manifest is None or manifest.automatic:
        return None
    # A composite goes in as its parts, in order, as `expand` writes one into a plan: left
    # whole it could never run. Each part takes the settings it declares; one no part
    # declares is invented.
    parts = [registry.get(part_id) for part_id in registry.expand(manifest.id)]
    given = _given(action)

    def taken_by(part: ToolManifest) -> dict[str, str]:
        takes = () if part.params is None else part.params.model_fields
        return {name: value for name, value in given.items() if name in takes}

    shares = [taken_by(part) for part in parts]
    if set(given) - {name for taken in shares for name in taken}:
        return None
    if not all(_takes(part, share, share) for part, share in zip(parts, shares, strict=True)):
        return None

    def do() -> Acted:
        planned = [
            PlannedStep(
                tool_id=part.id,
                name=part.name,
                purpose=action.purpose or manifest.description,
                # A tool that returns candidates stops for a person to choose between them; the
                # gate asked for is a stop after the last part.
                gated=part.judgment or (index == len(parts) - 1 and action.gated),
                params=share,
                expanded_from=manifest.id if manifest.expands_to else None,
            )
            for index, (part, share) in enumerate(zip(parts, shares, strict=True))
        ]
        # "Merge 4 and 5" comes as their removal and an insert before the first of them: it
        # goes where they were, before the next step shown that is still there, else at the end.
        live = {step.id for step in study.steps}
        before = next((step.id for step in named.later if step.id in live), None)
        steps = [study.insert_step(step, before=before) for step in to_steps(planned, registry)]
        added = ", ".join(_label(step) for step in steps)
        if manifest.expands_to:
            added = f"{manifest.name} as {added}"
        detail = f"Added {added}{_listed(given)}."
        return Acted(does="insert", step=steps[0].id, detail=detail, quoting=action.quoting)

    return do


def _remove(study: Study, registry: ToolRegistry, named: _Named) -> _Do | None:
    step = named.step
    # An always-run step is placed, not chosen: `replace_automatic` puts it in and takes it
    # out. One with a result is the record and stays, so it is not even offered as a button.
    if step is None or step.always or step.status not in _OPEN:
        return None

    def do() -> Acted:
        label = _label(step)
        study.remove_step(step.id)
        return Acted(
            does="remove", step=step.id, detail=f"Removed {label}.", quoting=named.action.quoting
        )

    return do


def _swap(study: Study, registry: ToolRegistry, named: _Named) -> _Do | None:
    action, step = named.action, named.step
    target = _tool(registry, action.tool_id)
    current = None if step is None else _tool(registry, step.tool_id)
    if step is None or target is None or current is None:
        return None
    if target.id not in {tool.id for tool in alternatives(current, registry)}:
        return None
    # Settings carry over by name; the rest belonged to the tool being left. Any the new
    # tool needs and the old one lacked are for the card to show, or for `settings` to give.
    takes = () if target.params is None else target.params.model_fields
    given = _given(action)
    params = {name: value for name, value in step.params.items() if name in takes} | given
    if not _takes(target, params, given):
        return None

    def do() -> Acted:
        was = f"step {step.position + 1} from {step.name}"
        study.set_tool(
            step.id,
            tool_id=target.id,
            tool_version=target.version,
            name=target.name,
            params=params,
        )
        return Acted(
            does="swap",
            step=step.id,
            detail=f"Changed {was} to {target.name}.",
            quoting=action.quoting,
        )

    return do


def _set_settings(study: Study, registry: ToolRegistry, named: _Named) -> _Do | None:
    action, step = named.action, named.step
    manifest = None if step is None else _tool(registry, step.tool_id)
    given = _given(action)
    # Nothing given is nothing to change; it would only put a finished step back to pending.
    if step is None or manifest is None or not given:
        return None
    params = {**step.params, **given}
    if not _takes(manifest, params, given):
        return None

    def do() -> Acted:
        study.set_params(step.id, params)
        detail = f"Changed the settings of {_label(step)}{_listed(given)}."
        return Acted(does="settings", step=step.id, detail=detail, quoting=action.quoting)

    return do


def _set_gate(study: Study, registry: ToolRegistry, named: _Named) -> _Do | None:
    step, gated = named.step, named.action.gated
    if step is None:
        return None

    def do() -> Acted:
        study.set_gate(step.id, gated)
        stops = "pause for result review" if gated else "continue without a review pause"
        return Acted(
            does="gate",
            step=step.id,
            detail=f"Configured {_label(step)} to {stops}.",
            quoting=named.action.quoting,
        )

    return do


def _terms(study: Study, registry: ToolRegistry, named: _Named) -> _Do | None:
    action = named.action
    # An empty list is not a replacement: a slip would wipe terms the scientist wrote.
    checks = tuple(action.checks) if action.checks else None
    stops = tuple(action.stop_conditions) if action.stop_conditions else None
    if checks is None and stops is None:
        return None
    changed = " and the ".join(
        label
        for label, value in (("checks", checks), ("stop conditions", stops))
        if value is not None
    )

    def do() -> Acted:
        study.amend_terms(checks, stops)
        return Acted(
            does="terms", step=None, detail=f"Changed the {changed}.", quoting=action.quoting
        )

    return do


_EDITS: dict[str, _Edit] = {
    "insert": _insert,
    "remove": _remove,
    "swap": _swap,
    "settings": _set_settings,
    "gate": _set_gate,
    "terms": _terms,
}


def _approval(registry: ToolRegistry, named: _Named) -> tuple[uuid.UUID, int] | None:
    """The approval to write, if the step is one that offers a choice and a rank was named."""
    step, candidate = named.step, named.action.candidate
    manifest = None if step is None else _tool(registry, step.tool_id)
    if step is None or not candidate or manifest is None or manifest.resolves_to is None:
        return None
    return step.id, candidate


def _now(step: Step | None) -> int:
    """Where a step stands in the plan now, counted from 1 as the model counts; 0 for none."""
    return 0 if step is None else step.position + 1


def _proposal(named: _Named) -> Proposal:
    """The action as a button: what it says without the quote, its steps numbered as they stand."""
    fields = named.action.model_dump(include=set(Proposal.model_fields))
    fields.update(
        step=_now(named.step),
        before=_now(named.before),
        until=_now(named.until),
        why=named.action.why or "Proposed, not done: it did not quote your message.",
    )
    return Proposal.model_validate(fields)


def apply_actions(
    study: Study,
    actions: Sequence[Action],
    registry: ToolRegistry,
    *,
    message: str | None,
) -> Applied:
    """Carry out what the model asked, as far as it holds and the scientist said so.

    `message` is the scientist's latest message. None means there is none to quote: the
    plan is being written together, or a button was pressed, and pressing it is the word.
    Otherwise an action must quote it. A valid one that does not is proposed instead; an
    invalid one is dropped first, since a button that could not be carried out is not
    offered. Changing the terms has no proposal to become, so without a quote it is
    dropped too.

    Approving and running are not done here, only checked and returned. Whatever the
    edits change, the always-run steps are then matched to the plan again
    (`replace_automatic`) and the plan is rechecked.
    """
    shown = {step.position: step for step in study.steps}
    acted: list[Acted] = []
    held: list[_Named] = []  # valid, but not on the scientist's word
    approvals: list[tuple[uuid.UUID, int]] = []
    runs: list[_Named] = []
    dropped = 0

    def ready(named: _Named) -> _Do | None:
        """The act that carries the action out, if it holds. Nothing is changed yet."""
        action = named.action
        if action.does == "approve":
            chosen = _approval(registry, named)
            return None if chosen is None else partial(approvals.append, chosen)
        if action.does == "run":
            return None if runs else partial(runs.append, named)  # one run in a turn
        return _EDITS[action.does](study, registry, named)

    for action in actions:
        named = _name(action, shown)
        do = None if named is None else ready(named)
        if named is None or do is None:
            dropped += 1
        elif message is not None and not quoted(action.quoting, message):
            if action.does == "terms":
                dropped += 1
            else:
                held.append(named)
        else:
            try:
                done = do()
            except ConflictError:
                dropped += 1  # the step has run, or an earlier edit in this turn took it away
            else:
                if done is not None:
                    acted.append(done)

    replace_automatic(study, registry)
    recheck(study, registry)

    live = {step.id for step in study.steps}

    def stands(named: _Named) -> bool:
        return all(
            step is None or step.id in live for step in (named.step, named.before, named.until)
        )

    # A step an edit in this turn removed cannot be proposed, approved or run.
    proposals = [named for named in held if stands(named)]
    dropped += len(held) - len(proposals)
    approved = [(step_id, rank) for step_id, rank in approvals if step_id in live]
    dropped += len(approvals) - len(approved)
    run = runs[0] if runs and stands(runs[0]) else None
    if runs and run is None:
        dropped += 1
    return Applied(
        acted=acted,
        downgraded=[_proposal(named) for named in proposals],
        approvals=approved,
        run=None
        if run is None
        else run.action.model_copy(update={"step": _now(run.step), "until": _now(run.until)}),
        dropped=dropped,
    )
