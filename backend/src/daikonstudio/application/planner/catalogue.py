"""Rendering what the model is shown: the catalogue, and the shape of the data.

Shared by both halves of planning -- the questions it asks first and the plan it
writes afterwards -- so the two can never drift into describing the same tool or
the same column differently.
"""

from __future__ import annotations

from pydantic.fields import FieldInfo

from daikonstudio.application.ports.table_profiler import ColumnProfile, TableProfile
from daikonstudio.application.tools.manifest import SlotKind, ToolManifest

# How to read a file's shape. One text for both halves of planning, because the
# two copies it replaced had already drifted into saying the same thing two ways.
READING_THE_DATA = (
    "When a file's shape is shown below, read it first and do not ask what you can "
    "see. The shape already lists, for each column, how much of it is filled, what "
    "parses as what, the range of any numbers in it, and the values that do not fit "
    "the rest, so asking whether any of that is present is asking a question the "
    "page in front of you has answered. Ask instead about what the shape cannot "
    "tell you: what a column means, which of several plausible readouts is the one "
    "that is trusted, whether two batches were produced under the same conditions. "
    "Quote the number that prompted the question, so you are asking for a decision "
    "rather than setting homework. Nothing in the shape is interpreted for you: a "
    "column of numbers is a column of numbers, and whether it is a potency, a yield "
    "or a plate index is your inference and the scientist's to confirm. If the "
    "shape shows something the scientist did not mention and would want to know "
    "about, raise it unasked. Without a file, say in a question's `why` when "
    "reading the data would have settled it, so the scientist knows what attaching "
    "one would buy."
)

CATALOGUE_HEADING = "Catalogue of what this server can do:"
INDEX_HEADING = "Catalogue of what this server can do, one line each:"
VIEW_HEADING = (
    "In full, the tools you are most likely to need "
    "(name any other in `wanted` to read it in full):"
)


def _kind(kind: SlotKind) -> str:
    # One spelling of the vocabulary everywhere the model reads it. The enum
    # value with underscores is for code; the note on what was brought and the
    # coherence messages already use the spaced form.
    return kind.value.replace("_", " ")


def _settings(m: ToolManifest, by_id: dict[str, ToolManifest]) -> str:
    """The settings a step takes. A composite with no schema of its own shows the
    union of its parts', a required field winning over an optional one."""
    fields: dict[str, FieldInfo] = {}
    for part in [m] if m.params else [by_id[i] for i in m.expands_to if i in by_id]:
        for name, f in (part.params.model_fields if part.params else {}).items():
            if name not in fields or f.is_required():
                fields[name] = f
    return "; ".join(
        f"{name} ({'required' if f.is_required() else 'optional'}): {f.description}"
        for name, f in fields.items()
    )


def catalogue(manifests: list[ToolManifest]) -> str:
    """Composites first, each naming what it stands for.

    Order is the only steer: a model reading a list takes the earlier entry when
    both fit, and preferring "build the SAR table" over its two parts is exactly
    the behaviour the composites exist to get.

    What is rendered is what the planner can act on. Whether a tool runs outside
    Studio matters, because the gating rule asks for a pause after anything that
    can quietly fail. Which lane serves it does not: that is an operator's fact,
    and it stays on the catalogue page.
    """
    lines: list[str] = []
    by_id = {m.id: m for m in manifests}
    for m in sorted(manifests, key=lambda x: (not x.expands_to,)):
        takes = ", ".join(
            f"{_kind(s.kind)}{' (optional)' if s.optional else ''}" for s in m.inputs
        )
        gives = ", ".join(_kind(s.kind) for s in m.outputs)
        line = f"- {m.id}: {m.name}. {m.description}"
        line += f"\n  takes: {takes or 'nothing'}; gives: {gives}"
        if settings := _settings(m, by_id):
            line += f"\n  settings: {settings}"
        if m.expands_to:
            line += f"\n  one step that runs, in order: {', '.join(m.expands_to)}"
        if m.external:
            line += "; runs outside Studio, so it can fail or be unavailable"
        if m.judgment and m.resolves_to is not None:
            line += (
                f"; needs the scientist's decision, and what they approve becomes a "
                f"{_kind(m.resolves_to)}"
            )
        if m.automatic:
            line += "\n  added to every plan for you where its input first appears; "
            line += (
                "list it only to set its parameters, never to decide whether it runs"
                if m.params is not None
                else "never list it"
            )
        lines.append(line)
    return "\n".join(lines)


def index(manifests: list[ToolManifest]) -> str:
    """One line per tool, in the order `catalogue()` gives, for a model to choose from.

    The first sentence of a description is what separates one tool from another; the rest is
    what a model reads only once it has chosen, in `catalogue()`.
    """
    lines: list[str] = []
    for m in sorted(manifests, key=lambda x: (not x.expands_to,)):
        takes = ", ".join(
            f"{_kind(s.kind)}{' (optional)' if s.optional else ''}" for s in m.inputs
        )
        gives = ", ".join(_kind(s.kind) for s in m.outputs)
        first = m.description.split(". ", 1)[0].rstrip(".")
        marks = [m.area.value]
        if m.expands_to:
            marks.append("one step that runs several")
        if m.judgment and m.resolves_to is not None:
            marks.append("needs the scientist's decision")
        if m.automatic:
            marks.append("added for you")
        lines.append(
            f"- {m.id}: {m.name}. {first}. takes: {takes or 'nothing'}; gives: {gives} "
            f"[{'; '.join(marks)}]"
        )
    return "\n".join(lines)


def _column(c: ColumnProfile) -> str:
    bits = [f"{c.filled} filled", f"{c.distinct} distinct"]
    for parser, count in sorted(c.parses_as.items(), key=lambda kv: -kv[1]):
        if count >= c.filled * 0.5:
            bits.append(f"{count} parse as a {parser}")
    if c.minimum is not None and c.maximum is not None:
        bits.append(f"range {c.minimum:g} to {c.maximum:g}, median {c.median:g}")
    line = f"- {c.name}: " + ", ".join(bits)
    if c.odd_values:
        shown = ", ".join(repr(v) for v in c.odd_values)
        line += f"\n    does not fit the rest: {shown}"
    elif c.odd_count:
        line += f"\n    {c.odd_count} values do not fit the rest"
    if c.withheld:
        line += "\n    values not shown"
    elif c.examples:
        line += "\n    e.g. " + ", ".join(repr(v) for v in c.examples)
    if c.most_repeated is not None:
        value, times = c.most_repeated
        label = "a withheld value" if c.withheld else repr(value)
        line += f"\n    most repeated: {label}, {times} times"
    return line


def profile_text(profile: TableProfile) -> str:
    """The table's shape for the model, with no interpretation applied.

    Deliberately flat. Everything here is a counted fact, and every inference --
    which column is the readout, whether \'>50\' means a censored measurement,
    whether a two-valued column is the batch marker -- is the model's to make and
    to put to the scientist as a question. Writing those inferences here instead
    would be this one use case hardcoded into the plumbing.
    """
    header = f"The uploaded file has {profile.rows} rows"
    if profile.truncated:
        header += " (only the first rows were read, so counts are for those)"
    lines = [header + ":"] + [_column(c) for c in profile.columns]
    return "\n".join(lines)


def always_runs(manifests: list[ToolManifest]) -> str:
    """The steps that are added to every plan, named from the catalogue.

    Generated rather than written out, because `automatic` is a flag on a
    manifest and prose repeating today's answer becomes a lie the moment a
    second tool sets it or this one is renamed -- and the model would go on
    asserting it confidently. Anything the planner is told about the catalogue
    has to come from the catalogue.
    """
    names = [m.name for m in manifests if m.automatic]
    if not names:
        return "No steps are added for you; every step in your plan is one you chose."
    listed = ", ".join(names)
    text = (
        f"These are added to the plan for you, each right after the step that first "
        f"produces what it needs, or before your first step when the scientist "
        f"brought it: {listed}. Do not list them to decide whether they happen, and "
        f"do not ask whether to do them. They run whenever their input exists, "
        f"whatever the data and whatever the goal, so they are not judgement calls "
        f"and not yours to raise."
    )
    # Whether one of these runs is settled; how it runs may not be. A pH or a
    # ligand to strip is the scientist's call, and with no way to set it the
    # planner can only write down what should have happened and hope — which is
    # a plan describing something the study will not do.
    settable = [m.name for m in manifests if m.automatic and m.params is not None]
    if settable:
        text += (
            f" These do take parameters: {', '.join(settable)}. When an answer "
            f"settles one, list that step with the parameters only — its placement "
            f"and the fact that it runs are still ours, not yours."
        )
    return text


def brought(profile: TableProfile | None) -> set[SlotKind]:
    """What the scientist actually handed over.

    A file is the only thing anyone can hand this planner today, so the set is
    small, but deriving it matters more than its size. Asserting a compound
    table unconditionally told the planner about data that did not exist, and a
    model does the right thing with that: it asks what the table is and what to
    do with it. The question is unanswerable and the scientist is left thinking
    the thing hallucinated, when in fact it was told.

    Pasting a sequence or a PDB id into the goal brings nothing either. An id is
    not coordinates; a step in the catalogue turns one into the other, and the
    planner has to choose that step. Saying "you start with a structure" because
    a goal contains letters would be the same lie in the other direction.
    """
    return {SlotKind.COMPOUND_TABLE} if profile is not None else set()


def starting_note(available: set[SlotKind]) -> str:
    """What the scientist brought, said plainly to the planner.

    Without it a model reaches for a structure tool on a goal where nobody
    supplied a structure, and the coherence check rejects a plan that was
    reasonable to attempt. Saying what is in hand up front turns that into the
    planner telling the scientist what it would need instead.
    """
    tail = (
        "Nothing else exists until a step makes it. A step that takes nothing makes "
        "its output from what the goal names -- an identifier, a target, a search -- "
        "so look for one before deciding something is missing. If the goal needs "
        "something you were not given and no step can make, say so and name what "
        "you would need rather than planning around it or asking the scientist "
        "about data you do not have."
    )
    if not available:
        return f"You start with nothing but the goal: no file has been given to you. {tail}"
    listed = ", ".join(sorted(kind.value.replace("_", " ") for kind in available))
    return f"You start with: {listed}. {tail}"
