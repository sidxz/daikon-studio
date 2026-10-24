"""What the model says about a study: the changes it wants and the claims it makes.

One flat `Action` with a `does` literal rather than a model per edit: the providers'
structured output closes every object and has no union to offer, and one shape is one
thing for a prompt to describe. `does` says which of the other fields count.

Every position is 1-based and means the plan as the model was shown it; the server
resolves each to a step before it changes anything, so a number never drifts mid-turn.

The model fills every field of an action, a proposal and a look, and "none" is a value, not
a missing field: 0 for a step, a rank or a position, "" for a tool or a purpose, an empty
list for settings, checks and stop conditions. A field the model may leave out is a branch
in the grammar the provider compiles from the schema, and an `X | None` is two more, so a
reply carrying both actions and proposals grew past what Anthropic will compile ("The
compiled grammar is too large"). The defaults stay in Python, so code builds one by the
fields that matter.

A `Proposal` is an action the scientist has not yet confirmed. It leaves out what a
button must never carry: the checks and stop conditions, because a criterion rewritten
after the data is not a criterion, and the quote, because pressing the button is the
scientist's word.

A `Claim` is something the agent says about a result, as a reference into it; the reader
for that kind of result checks it before it is stored. A `Look` asks to see part of a
table, which the server answers with a bounded slice.

A turn is what the model fills in one phase of a study, and the schema for each phase is
what keeps the rules mechanical. Asking has no steps field, so nothing is planned before
the scientist has been asked; an event turn has no `actions` field, so a result can never
make the agent do anything; only a reply to a message can act, and it cannot pause. Nothing
here is a float: a number the agent writes is one no step computed.
"""

from __future__ import annotations

import functools
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

from daikonstudio.application.planner.plan import DraftPlan, Interrogation, Question, Step
from daikonstudio.application.planner.plan import Setting as Setting  # re-exported
from daikonstudio.application.tools.registry import ToolRegistry

Does = Literal["insert", "remove", "swap", "settings", "gate", "terms", "approve", "run"]
Proposable = Literal["insert", "remove", "swap", "settings", "gate", "approve", "run"]
Evidence = Literal[
    "ligand_overlap",
    "annotated_site",
    "interface",
    "lining_residues",
    "column",
    "rows",
    "coverage",
    "unmatched",
    "variable",
    "term",
    "fit",
    "unmade",
]
About = Literal["step", "check", "stop_condition"]

# What one look may show, however many rows it asks for. Defined here, where the model is told,
# and re-exported by the readers, which enforce it, so the two cannot drift apart.
LOOK_ROWS_MAX = 50

# Every field in the schema the model is given, defaults and all (see the module docstring).
_FILLED = ConfigDict(
    json_schema_mode_override="serialization", json_schema_serialization_defaults_required=True
)

_STEP = (
    "The step this is about, by its number in the plan you were shown (the first step is 1). "
    "Numbers stay as shown for the whole reply, however many steps are added or removed. "
    "For run: where to start, or 0 to start at the next step waiting to run. 0 for an "
    "action that names no step."
)
_TOOL = (
    "For insert and swap: the id of a tool from the catalogue. Never invent one. Empty "
    "for the rest."
)
_BEFORE = (
    "For insert: the number of the step to put the new one in front of, or 0 to add it at "
    "the end. 0 for the rest."
)
_PURPOSE = (
    "For insert: why the step is here, in the scientist's terms, naming the specific "
    "values it uses. Empty for the rest."
)
_SETTINGS = (
    "For insert, swap and settings: name/value pairs, by the names the catalogue lists "
    "under 'settings'. For settings, only the ones to change; the rest stay as they are. "
    "For swap, the settings the new tool takes that the old one did not. Empty for the rest."
)
_GATED = (
    "For gate: true to stop after this step for the scientist to look at its result, false "
    "to let it run on. For insert: true to stop after the new step; a tool that needs the "
    "scientist's decision stops either way. False for the rest."
)
_CANDIDATE = (
    "For approve: the rank of the candidate the step reported, as it reported it. 0 for the rest."
)
_UNTIL = (
    "For run: the number of the step to stop after, or 0 to run to the next step that "
    "stops for a look. 0 for the rest."
)


WANTED_TO_ANSWER = (
    "Tool ids from the catalogue you must read in full before you can answer. Return no plan "
    "and no actions and name them, and you will be asked once more with them in front of you. "
    "Otherwise leave this empty: naming tools here costs a second call."
)


class Action(BaseModel):
    """One change to the study, or an instruction to run it."""

    model_config = _FILLED

    does: Does = Field(
        description=(
            "What to do. insert: add a step (tool_id; before, purpose, settings and gated "
            "if needed). remove: take out a step that has not run (step). swap: point a "
            "step at another tool that takes the same inputs and gives the same results "
            "(step, tool_id). settings: change some of a step's settings (step, settings). "
            "gate: set whether a step stops for the scientist to look at its result (step, "
            "gated). terms: replace the checks or the stop conditions (checks, "
            "stop_conditions). approve: choose one of the candidates a step reported "
            "(step, candidate). run: run steps (step, until)."
        )
    )
    step: int = Field(default=0, description=_STEP)
    tool_id: str = Field(default="", description=_TOOL)
    before: int = Field(default=0, description=_BEFORE)
    purpose: str = Field(default="", description=_PURPOSE)
    settings: list[Setting] = Field(default_factory=list, description=_SETTINGS)
    gated: bool = Field(default=False, description=_GATED)
    checks: list[str] = Field(
        default_factory=list,
        description=(
            "For terms: the whole list of checks, replacing the current one. Empty keeps "
            "the current checks; the checks are never cleared."
        ),
    )
    stop_conditions: list[str] = Field(
        default_factory=list,
        description=(
            "For terms: the whole list of stop conditions, replacing the current one. Empty "
            "keeps the current stop conditions; they are never cleared."
        ),
    )
    candidate: int = Field(default=0, description=_CANDIDATE)
    until: int = Field(default=0, description=_UNTIL)
    quoting: str = Field(
        default="",
        description=(
            "The words of the scientist's latest message that you are acting on, copied "
            "exactly. Every change and every run made in reply to a message needs them; "
            "one that does not quote the message is only proposed, not done."
        ),
    )


class Proposal(BaseModel):
    """Something the agent would do, if the scientist says so."""

    model_config = _FILLED

    does: Proposable = Field(
        description="What you would like done, with the same meanings as in an action."
    )
    step: int = Field(default=0, description=_STEP)
    tool_id: str = Field(default="", description=_TOOL)
    before: int = Field(default=0, description=_BEFORE)
    purpose: str = Field(default="", description=_PURPOSE)
    settings: list[Setting] = Field(default_factory=list, description=_SETTINGS)
    gated: bool = Field(default=False, description=_GATED)
    candidate: int = Field(default=0, description=_CANDIDATE)
    until: int = Field(default=0, description=_UNTIL)
    why: str = Field(description="One sentence: what this would settle that is open now.")


class Claim(BaseModel):
    """One thing the agent says about a result, as a reference into it."""

    step: int = Field(
        description=(
            "The step whose result this is about, by its number in the plan you were "
            "shown (the first step is 1)."
        )
    )
    candidates: list[int] = Field(
        default_factory=list,
        description=(
            "For pockets: the candidates this is about, by rank. Several ranks mean you are "
            "saying they are one site that the tool reported separately. Leave empty for a "
            "table."
        ),
    )
    label: str = Field(
        description="What this is, named the way a chemist would: 'the zinc cone'.",
    )
    evidence: Evidence = Field(
        description=(
            "Which part of the result carries this. For pockets: ligand_overlap, "
            "annotated_site, interface or lining_residues. For a table: column, when you cite "
            "column names, or rows, when you cite values of its identifier column. For a core: "
            "coverage, unmatched when you cite compounds it misses, or variable when you cite "
            "a position it lets vary. For a model of substituents: term, fit or unmade."
        )
    )
    cites: list[str] = Field(
        description=(
            "What you are relying on, spelled exactly as the result spells it: the lining "
            "residues for pockets; for a table, the column names when the evidence is column, "
            "and the values of its identifier column when the evidence is rows; for a core, the "
            "identifiers of compounds it misses, or the positions or elements it lets vary; for "
            "a model of substituents, its positions, groups or compound identifiers. Anything "
            "not found there costs you the whole claim."
        )
    )


class Look(BaseModel):
    """A request to see part of a table, answered with a bounded slice."""

    model_config = _FILLED

    step: int = Field(
        description=(
            "The step whose table you want to see, by its number in the plan you were "
            "shown (the first step is 1)."
        )
    )
    rows: int = Field(
        default=20,
        description=f"How many rows from the top. At most {LOOK_ROWS_MAX} are shown.",
    )
    columns: list[str] = Field(
        default_factory=list,
        description=(
            "Which columns to see, by name; empty for all of them. Structure and "
            "sequence columns are never shown, and names the table lacks are ignored."
        ),
    )


class ProgressClaim(BaseModel):
    """One thing the agent says about where the study stands."""

    about: About = Field(description="What this is about: a step, a check, or a stop condition.")
    refers_to: list[int] = Field(
        description=(
            "Which ones, by number as shown to you, counting from 1. A step is its number "
            "in the plan. A check or a stop condition is its place in its list."
        )
    )
    label: str = Field(description="What you are saying about them, in a few words.")
    cites: list[str] = Field(
        description=(
            "The exact words you are relying on: the step's name, or the check or "
            "stop condition as the plan wrote it. Anything not found costs the claim."
        )
    )


# The turn before any plan: the restated goal and the questions, and nothing else.
AskingTurn = Interrogation


class PlanningTurn(BaseModel):
    """What the agent may produce while the plan is being written with the scientist."""

    say: str = Field(
        description=(
            "What you tell the scientist. A change to the plan goes in `plan` or `actions`; "
            "here say why, briefly, rather than describing the steps."
        )
    )
    # A plan with no steps is no plan: an empty one stands in for None, which would be a union.
    plan: DraftPlan = Field(
        default_factory=lambda: DraftPlan(restated_goal="", steps=[]),
        description=(
            "The whole plan. Only for the first draft, or when the scientist asks for a "
            "different approach and the steps are to be replaced. Leave it out to change a "
            "plan that exists; that is what `actions` are for."
        ),
    )
    actions: list[Action] = Field(
        default_factory=list,
        description=(
            "Changes to the plan that exists, by step number. The plan is still a draft, so "
            "nothing runs and a change needs no quote."
        ),
    )
    questions: list[Question] = Field(
        default_factory=list,
        description=(
            "What you still need answered, each with the default you will go on if you can "
            "defend one. They count against the questions the analysis may still ask."
        ),
    )
    wanted: list[str] = Field(default_factory=list, description=WANTED_TO_ANSWER)


@functools.cache
def _planning_turn_for(ids: tuple[str, ...]) -> type[PlanningTurn]:
    # Literal[*ids] renders as a string enum, no union. Only Step.tool_id is narrowed: edits
    # are checked by apply_actions and vet, and enumerating them again would double the tokens.
    enum_step = create_model(
        "Step",
        __base__=Step,
        tool_id=(Literal[*ids], Field(description=Step.model_fields["tool_id"].description)),
    )
    enum_draft = create_model("DraftPlan", __base__=DraftPlan, steps=(list[enum_step], ...))  # type: ignore[valid-type]
    return cast(
        "type[PlanningTurn]",
        create_model(
            "PlanningTurn",
            __base__=PlanningTurn,
            plan=(
                enum_draft,
                Field(
                    default_factory=lambda: enum_draft(restated_goal="", steps=[]),
                    description=PlanningTurn.model_fields["plan"].description,
                ),
            ),
        ),
    )


def planning_turn_schema(registry: ToolRegistry) -> type[PlanningTurn]:
    """PlanningTurn with `tool_id` constrained to this registry's ids; built once per registry."""
    return _planning_turn_for(tuple(m.id for m in registry.manifests()))


class _Reading(BaseModel):
    """What the agent says about a study that has started, whatever prompted the turn."""

    say: str = Field(
        description=(
            "What you want to tell the scientist. On an event, leave it empty when there "
            "is nothing worth their reading; silence is allowed."
        )
    )
    claims: list[Claim] = Field(
        default_factory=list,
        description=(
            "What you say about a result, each as a reference into it. Each is checked "
            "against the result before anyone reads it, and one it does not support is dropped."
        ),
    )
    progress: list[ProgressClaim] = Field(
        default_factory=list,
        description=(
            "What you say about where the study stands against its plan: a step, a check or "
            "a stop condition, citing the plan's own words."
        ),
    )
    unresolved: list[str] = Field(
        default_factory=list, description="What cannot be settled yet, one short line each."
    )
    proposals: list[Proposal] = Field(
        default_factory=list,
        description=(
            "What you would like done, for the scientist to confirm. Propose only what an "
            "open question actually needs; an exchange that settles nothing needs none."
        ),
    )
    look: list[Look] = Field(
        default_factory=list,
        description=(
            "Parts of a table result you want to see before you claim anything about its "
            "rows. You will be asked again with them."
        ),
    )


_SUMMARY = (
    "Where the study stands, in one paragraph, for someone opening it cold. It is read in "
    "place of the turns that have scrolled away, so rewrite it whole and keep what matters."
)


class ObserveTurn(_Reading):
    """What the agent may produce when a step has finished or failed.

    There is no field here to change the plan or run a step. An event can only be
    observed; the one thing it can do is end a run the agent was told to drive.
    """

    pause: bool = Field(
        default=False,
        description=(
            "When you are driving a run for the scientist: stop it after this step. Say why "
            "in `say`; a pause with nothing in `say` is ignored."
        ),
    )
    summary: str = Field(description=_SUMMARY)


class ReplyTurn(_Reading):
    """What the agent may produce in answer to the scientist's message.

    It may act, and only here. It cannot pause: nothing is running while the scientist
    is talking to it, and a run it starts is stopped by the next event, not by this turn.
    """

    actions: list[Action] = Field(
        default_factory=list,
        description=(
            "What you do on the scientist's word: change the plan, run steps, approve a "
            "candidate. Each quotes the words it carries out. Whatever they did not ask for "
            "goes in `proposals`, not here."
        ),
    )
    wanted: list[str] = Field(default_factory=list, description=WANTED_TO_ANSWER)
    summary: str = Field(description=_SUMMARY)
