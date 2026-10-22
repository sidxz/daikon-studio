"""What the model says about a study: the changes it wants and the claims it makes.

One flat `Action` with a `does` literal rather than a model per edit: the providers'
structured output closes every object and has no union to offer, and one shape is one
thing for a prompt to describe. `does` says which of the other fields count.

Every position is 1-based and means the plan as the model was shown it; the server
resolves each to a step before it changes anything, so a number never drifts mid-turn.

A `Proposal` is an action the scientist has not yet confirmed. It leaves out what a
button must never carry: the checks and stop conditions, because a criterion rewritten
after the data is not a criterion, and the quote, because pressing the button is the
scientist's word.

A `Claim` is something the agent says about a result, as a reference into it; the reader
for that kind of result checks it before it is stored. A `Look` asks to see part of a
table, which the server answers with a bounded slice.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from daikonstudio.application.planner.plan import Setting as Setting  # re-exported

Does = Literal["insert", "remove", "swap", "settings", "gate", "terms", "approve", "run"]
Proposable = Literal["insert", "remove", "swap", "settings", "gate", "approve", "run"]
Evidence = Literal[
    "ligand_overlap", "annotated_site", "interface", "lining_residues", "column", "rows"
]

_STEP = (
    "The step this is about, by its number in the plan you were shown (the first step is 1). "
    "Numbers stay as shown for the whole reply, however many steps are added or removed. "
    "For run: where to start; leave empty to start at the next step waiting to run."
)
_TOOL = "For insert and swap: the id of a tool from the catalogue. Never invent one."
_BEFORE = (
    "For insert: the number of the step to put the new one in front of. Leave empty to "
    "add it at the end."
)
_PURPOSE = (
    "For insert: why the step is here, in the scientist's terms, naming the specific "
    "values it uses."
)
_SETTINGS = (
    "For insert, swap and settings: name/value pairs, by the names the catalogue lists "
    "under 'settings'. For settings, only the ones to change; the rest stay as they are. "
    "For swap, the settings the new tool takes that the old one did not."
)
_GATED = (
    "For gate: whether the study stops after this step for the scientist to look at its "
    "result. For insert: the same for the new step; leave empty to let the tool decide."
)
_CANDIDATE = "For approve: the rank of the candidate the step reported, as it reported it."
_UNTIL = (
    "For run: the number of the step to stop after. Leave empty to run to the next step "
    "that stops for a look."
)


class Action(BaseModel):
    """One change to the study, or an instruction to run it."""

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
    step: int | None = Field(default=None, description=_STEP)
    tool_id: str | None = Field(default=None, description=_TOOL)
    before: int | None = Field(default=None, description=_BEFORE)
    purpose: str | None = Field(default=None, description=_PURPOSE)
    settings: list[Setting] = Field(default_factory=list, description=_SETTINGS)
    gated: bool | None = Field(default=None, description=_GATED)
    checks: list[str] | None = Field(
        default=None,
        description=(
            "For terms: the whole list of checks, replacing the current one. Leave null to "
            "keep the current checks; an empty list does not clear them."
        ),
    )
    stop_conditions: list[str] | None = Field(
        default=None,
        description=(
            "For terms: the whole list of stop conditions, replacing the current one. Leave "
            "null to keep the current stop conditions; an empty list does not clear them."
        ),
    )
    candidate: int | None = Field(default=None, description=_CANDIDATE)
    until: int | None = Field(default=None, description=_UNTIL)
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

    does: Proposable = Field(
        description="What you would like done, with the same meanings as in an action."
    )
    step: int | None = Field(default=None, description=_STEP)
    tool_id: str | None = Field(default=None, description=_TOOL)
    before: int | None = Field(default=None, description=_BEFORE)
    purpose: str | None = Field(default=None, description=_PURPOSE)
    settings: list[Setting] = Field(default_factory=list, description=_SETTINGS)
    gated: bool | None = Field(default=None, description=_GATED)
    candidate: int | None = Field(default=None, description=_CANDIDATE)
    until: int | None = Field(default=None, description=_UNTIL)
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
            "annotated_site, interface or lining_residues. For a table: column, or rows."
        )
    )
    cites: list[str] = Field(
        description=(
            "What you are relying on, spelled exactly as the result spells it: the lining "
            "residues for pockets; for a table, column names and values of its identifier "
            "column. Anything not found there costs you the whole claim."
        )
    )


class Look(BaseModel):
    """A request to see part of a table, answered with a bounded slice."""

    step: int = Field(
        description=(
            "The step whose table you want to see, by its number in the plan you were "
            "shown (the first step is 1)."
        )
    )
    rows: int = Field(default=20, description="How many rows from the top. At most 50 are shown.")
    columns: list[str] = Field(
        default_factory=list,
        description=(
            "Which columns to see, by name; leave empty for all of them. Structure and "
            "sequence columns are never shown, and names the table lacks are ignored."
        ),
    )
