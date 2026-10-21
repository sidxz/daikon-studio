"""Rendering what the model is shown: the catalogue, and the shape of the data.

Shared by both halves of planning -- the questions it asks first and the plan it
writes afterwards -- so the two can never drift into describing the same tool or
the same column differently.
"""

from __future__ import annotations

from daikonstudio.application.ports.table_profiler import ColumnProfile, TableProfile
from daikonstudio.application.tools.manifest import ToolManifest


def catalogue(manifests: list[ToolManifest]) -> str:
    """Composites first, each naming what it stands for.

    Order is the only steer: a model reading a list takes the earlier entry when
    both fit, and preferring "build the SAR table" over its two parts is exactly
    the behaviour the composites exist to get.
    """
    lines: list[str] = []
    for m in sorted(manifests, key=lambda x: (not x.expands_to,)):
        takes = ", ".join(f"{s.kind.value}{' (optional)' if s.optional else ''}" for s in m.inputs)
        gives = ", ".join(s.kind.value for s in m.outputs)
        line = f"- {m.id}: {m.name}. {m.description}"
        line += f"\n  takes: {takes or 'nothing'}; gives: {gives}"
        if m.expands_to:
            line += f"\n  one step that runs, in order: {', '.join(m.expands_to)}"
        if m.judgment and m.resolves_to is not None:
            line += f"; needs a chemist, and what they approve becomes a {m.resolves_to.value}"
        lines.append(line)
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
    return (
        f"These always run and are added to the plan for you: {listed}. Do not list "
        f"them, do not ask whether to do them, and assume they have already happened "
        f"before your first step. They do not depend on the data or on what anyone "
        f"wants, so they are not judgement calls and not yours to raise."
    )
