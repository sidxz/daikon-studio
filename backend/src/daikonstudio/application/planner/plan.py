"""What a planner returns, and what makes one coherent.

These are Pydantic models rather than dataclasses because they are the schema the
provider validates its own answer against. That is the point: a malformed plan
fails at the boundary instead of halfway through being rendered.

A plan is a *proposal*. Nothing here runs anything, and nothing here decides
anything -- the questions exist because the goal was underspecified, and the
gates exist because a person is supposed to look.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from daikonstudio.application.tools.manifest import SlotKind
from daikonstudio.application.tools.registry import ToolRegistry, UnknownToolError


class Question(BaseModel):
    """Something the planner needs answered before the plan means anything.

    The design's first move: a goal like "work out what's driving activity" is
    not actionable until someone says which readout counts and what to do with
    the compounds measured twice. Asking is the planner's highest-value act, so
    questions are a first-class part of a plan rather than prose in a preamble.
    """

    ask: str = Field(description="The question, addressed to the scientist.")
    why: str = Field(description="What changes about the analysis depending on the answer.")
    options: list[str] = Field(
        default_factory=list,
        description=(
            "The plausible answers, when they enumerate: which of three columns, "
            "whether a value is a bound or untested. Empty when the answer is "
            "open, such as the name of a target. Never include a 'not sure' "
            "option; one is offered on every question regardless."
        ),
    )
    default: str | None = Field(
        default=None,
        description=(
            "What you will do if nobody answers. Set it whenever you can defend "
            "a choice: that turns the question into an assumption the scientist "
            "can override, and the plan is complete without them. Leave it null "
            "only when the plan genuinely cannot be written either way -- which "
            "column is the readout, what an ambiguous code means. A question you "
            "could have defaulted is not a question, it is a decision you "
            "declined to take."
        ),
    )

    @property
    def blocking(self) -> bool:
        """No defensible default exists, so the plan cannot be written yet.

        The split that keeps this from becoming an interview. A blocking
        question is asked before planning; everything else rides alongside a
        finished plan as a refinement nobody has to touch.
        """
        return self.default is None


class Interrogation(BaseModel):
    """What the planner needs to know before a plan would mean anything.

    Separate from the plan on purpose. Asked and answered in one pass, the model
    both asks "is IC50 the readout?" and writes a plan that uses IC50 throughout:
    the guess is already load-bearing by the time the scientist reads the
    question, and the plan looks settled when it is not.
    """

    restated_goal: str = Field(
        description=(
            "The goal as understood: what is wanted, not how it will be done, so a "
            "mismatch is visible before anything else."
        )
    )
    questions: list[Question]


class Answer(BaseModel):
    ask: str = Field(description="The question, repeated verbatim.")
    answer: str = Field(description="What the scientist said, or chose.")


class Setting(BaseModel):
    """One of a step's settings, as a name/value pair.

    A pair rather than a dict entry because the providers' structured output
    closes every object: a `dict[str, Any]` field comes back empty however the
    model was told to fill it (verified with a probe against the configured
    model, 2026-10-21). A list of pairs is the shape every strict schema allows.
    """

    name: str = Field(description="The setting's name, exactly as the catalogue lists it.")
    value: str = Field(description="Its value, as text.")


class Step(BaseModel):
    tool_id: str = Field(description="The id of a tool from the catalogue. Never invent one.")
    purpose: str = Field(
        description=(
            "Why this step is here, in the scientist's terms, naming the specific "
            "values it uses: the id to fetch, the column to read, the chain to keep. "
            "Settings have their own field."
        )
    )
    gated: bool = Field(
        description="Whether to stop and show the result before going on to the next step."
    )
    settings: list[Setting] = Field(
        default_factory=list,
        description=(
            "The tool's settings, by the names the catalogue lists under 'settings'. "
            "Every required one must be present. Values come from the goal and the "
            "answers, never invented: an id the scientist named, a column the shape "
            "shows. Empty for a tool that lists none."
        ),
    )

    @property
    def params(self) -> dict[str, Any]:
        return {s.name: s.value for s in self.settings}


class Skipped(BaseModel):
    tool_id: str
    why: str = Field(description="Why this tool was considered and left out.")


class DraftPlan(BaseModel):
    """The planner's answer. Named a draft because nothing has approved it."""

    restated_goal: str = Field(
        description=(
            "The goal as understood: what is wanted, not how it will be done, so a "
            "mismatch is visible before anything runs."
        )
    )
    questions: list[Question] = Field(
        default_factory=list,
        description="What the planner needs answered. Derived from the goal and the data, "
        "never a generic checklist.",
    )
    steps: list[Step]
    checks: list[str] = Field(
        default_factory=list,
        description="What would count as a result, stated before anything runs.",
    )
    stop_conditions: list[str] = Field(
        default_factory=list,
        description="What would make the planner stop and report that the question cannot "
        "be answered this way, rather than answering it weakly.",
    )
    skipped: list[Skipped] = Field(
        default_factory=list,
        description=(
            "Tools a reader would expect here and deliberately left out. Not the "
            "ones that plainly do not apply."
        ),
    )


class PlannedStep(BaseModel):
    """A step as it will actually run.

    A composite the planner chose appears here as its parts, each naming the
    composite it came from, so the plan reads as the chemist's five decisions
    while the record keeps all twelve things that happened.
    """

    tool_id: str
    name: str
    purpose: str
    gated: bool
    params: dict[str, Any] = Field(default_factory=dict)
    expanded_from: str | None = None
    # Prepended because it always runs, not because the planner chose it.
    always: bool = False


class PlanProblem(BaseModel):
    """One reason a draft cannot be executed as written."""

    step: int | None = Field(description="1-based position of the offending step, if it is one.")
    message: str


def _handed_over(plan: DraftPlan, registry: ToolRegistry) -> set[SlotKind]:
    """What this plan expects someone to give it: consumed by a step, produced
    by none of them.

    Needed because an upload is optional. Someone can describe a series and
    attach the file afterwards, so "no file yet" does not mean "no compounds" --
    it means nobody has shown us. Reading the plan settles it without guessing:
    a plan that decomposes R-groups is a plan about compounds whether or not the
    file has arrived, and a plan that only touches a protein is not, however
    much chemistry the goal mentions.
    """
    consumed: set[SlotKind] = set()
    produced: set[SlotKind] = set()
    for step in plan.steps:
        try:
            manifest = registry.get(step.tool_id)
        except UnknownToolError:
            continue
        takes = {slot.kind for slot in manifest.inputs}
        consumed |= takes
        # Minus what it took: a tool that gives back the kind it was given has
        # cleaned something, not created it. Counting that as production makes a
        # plan look self-sufficient in the one slot the scientist has to supply.
        produced |= {slot.kind for slot in manifest.outputs} - takes
        if manifest.resolves_to is not None and manifest.resolves_to not in takes:
            produced.add(manifest.resolves_to)
    return consumed - produced


def expand(
    plan: DraftPlan,
    registry: ToolRegistry,
    *,
    starting_with: set[SlotKind] | None = None,
) -> list[PlannedStep]:
    """The planner's steps with composites replaced by their parts, and the
    always-run steps placed the moment what they need exists.

    `starting_with` is what the scientist brought. An automatic step is added
    only when what it needs is in hand: preparing a structure always happens
    before anything touches one, but a plan over a compound table alone has no
    structure to prepare, and prepending the step anyway would put a tool in
    the plan for a file that does not exist.

    "In hand" is not only "at the start". A structure fetched by id in step two
    needs preparing as much as one attached up front, so an automatic step goes
    in right after the step that first produces its input -- which is the
    prologue when the scientist brought it, and later when a step made it.

    Automatic steps are placed rather than chosen. Standardizing structures and
    merging duplicates do not depend on anything in the data or on what the
    scientist wants -- the same compound drawn two ways is two compounds in every
    table that ever arrives. A model asked to choose them can forget, and a plan
    missing one is silently wrong rather than visibly broken. If the model listed
    one anyway it is not run twice; its stated purpose is kept, because the model
    usually says something more specific than the manifest does.

    Gating inside a composite is not the planner's to decide: a step that asks a
    chemist something gates because of what it is, and the composite's own flag
    only decides whether to stop once at the end. Letting the model gate inner
    steps would mean a plan could quietly skip the pause on an approval.
    """
    steps: list[PlannedStep] = []
    available = set(starting_with) if starting_with is not None else {SlotKind.COMPOUND_TABLE}
    available |= _handed_over(plan, registry)
    chosen = {step.tool_id: step for step in plan.steps}
    automatic = [m for m in registry.manifests() if m.automatic]

    def place_automatic() -> None:
        placed = {s.tool_id for s in steps}
        for manifest in automatic:
            if manifest.id in placed:
                continue
            if any(slot.kind not in available for slot in manifest.inputs if not slot.optional):
                continue
            said = chosen.get(manifest.id)
            steps.append(
                PlannedStep(
                    tool_id=manifest.id,
                    name=manifest.name,
                    purpose=said.purpose if said else manifest.description,
                    gated=said.gated if said else False,
                    params=said.params if said else {},
                    always=True,
                )
            )
            available.update(slot.kind for slot in manifest.outputs)

    place_automatic()
    for step in plan.steps:
        try:
            manifest = registry.get(step.tool_id)
        except UnknownToolError:
            steps.append(
                PlannedStep(
                    tool_id=step.tool_id, name=step.tool_id, purpose=step.purpose, gated=False
                )
            )
            continue
        if manifest.automatic:
            # Placed above, or when its input appears; never where the model put it.
            continue
        parts = registry.expand(manifest.id)
        for position, part_id in enumerate(parts):
            part = registry.get(part_id)
            last = position == len(parts) - 1
            steps.append(
                PlannedStep(
                    tool_id=part.id,
                    name=part.name,
                    purpose=step.purpose,
                    gated=part.judgment or (last and step.gated),
                    params=dict(step.params),
                    expanded_from=manifest.id if manifest.expands_to else None,
                )
            )
            available.update(slot.kind for slot in part.outputs)
            if part.resolves_to is not None:
                available.add(part.resolves_to)
        place_automatic()
    return steps


def incoherences(
    steps: list[PlannedStep],
    registry: ToolRegistry,
    *,
    starting_with: set[SlotKind] | None = None,
) -> list[PlanProblem]:
    """Every way these steps would fail before they ran.

    This is the payoff for typing what tools consume and produce. A model writes
    plausible plans, and a plausible plan that reaches for an SAR table three
    steps before anything builds one reads perfectly and cannot execute. Catching
    that mechanically is cheaper and far more reliable than hoping a reviewer
    notices, and it is the one check on the planner that needs no judgement.

    Two failures are found here. A step naming a tool that does not exist -- the
    classic invented-tool answer. And a step whose required inputs nothing
    upstream has produced.

    A compound table is assumed available from the start: it is what the
    scientist brings. `resolves_to` counts as produced, because a judgement step
    resolves to a value once a person approves one of its candidates, and a plan
    that waits for that approval is correct rather than broken.

    `starting_with` overrides what is available before step one. A whole plan
    leaves it alone; checking a composite's own parts passes that composite's
    declared inputs, because by the time it runs it has been given them.
    """
    problems: list[PlanProblem] = []
    available: set[SlotKind] = (
        set(starting_with) if starting_with is not None else {SlotKind.COMPOUND_TABLE}
    )

    for position, step in enumerate(steps, start=1):
        try:
            manifest = registry.get(step.tool_id)
        except UnknownToolError:
            problems.append(
                PlanProblem(step=position, message=f"There is no tool called '{step.tool_id}'.")
            )
            continue

        missing = [
            slot for slot in manifest.inputs if not slot.optional and slot.kind not in available
        ]
        if missing:
            names = ", ".join(sorted({slot.kind.value.replace("_", " ") for slot in missing}))
            problems.append(
                PlanProblem(
                    step=position,
                    message=(
                        f"Step {position} ({manifest.name}) needs {names}, which no earlier "
                        f"step produces."
                    ),
                )
            )

        if manifest.params is not None:
            try:
                manifest.params.model_validate(step.params)
            except ValidationError as error:
                detail = "; ".join(
                    f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()
                )
                problems.append(
                    PlanProblem(
                        step=position,
                        message=f"Step {position} ({manifest.name}) settings: {detail}",
                    )
                )

        available.update(slot.kind for slot in manifest.outputs)
        if manifest.resolves_to is not None:
            available.add(manifest.resolves_to)

    if not any(not step.always for step in steps):
        # Only the always-run prologue. The scientist asked for an analysis and
        # got preparation, which is an empty plan however many rows it has.
        problems.append(PlanProblem(step=None, message="The plan has no steps."))
    return problems
