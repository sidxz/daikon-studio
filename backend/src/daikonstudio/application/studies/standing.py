"""What the agent is shown each turn: the study as it stands, and what has just happened.

The model reads a study only through this text, so what is left out cannot be claimed about
and what is wrong is believed. One function writes the standing for every phase, because a
phase decides what the agent is asked to fill, not what it knows. Steps are numbered from 1,
the way the model's own numbers are resolved, so what it says maps back onto the plan it was
shown; a result is a short line here and never its rows, and the full description of one goes
only into the trigger of the turn it prompted.

Nothing here reads a blob or calls a model. A step, a turn or a pointer can name a step that
has since been removed from the plan, so a reference is looked up and never assumed.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from daikonstudio.application.planner.catalogue import (
    CATALOGUE_HEADING,
    catalogue,
    profile_text,
    starting_note,
)
from daikonstudio.application.ports.table_profiler import ColumnProfile, TableProfile
from daikonstudio.application.studies.readers import checkable, reader
from daikonstudio.application.tools.manifest import SlotKind
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.studies.study import Pointer, SlotValue, Step, StepStatus, Study, Turn

# How much of the conversation is written out: enough for the agent not to repeat itself.
# What scrolled out of the window is carried by `thread.summary`.
CONTEXT_TURNS = 20


def _kind(kind: str) -> str:
    # One spelling of the vocabulary everywhere the model reads it, as the catalogue spells it.
    return kind.replace("_", " ")


def one_line(kind: str, summary: Mapping[str, Any]) -> str:
    """A value in a few words: its kind, and how big it is where the summary says."""
    facts: list[str] = []
    if (rows := summary.get("rows")) is not None:
        facts.append(f"{rows} rows")
    if (columns := summary.get("columns")) is not None:
        facts.append(f"{len(columns)} columns")
    if (count := summary.get("count")) is not None:
        facts.append(f"count {count}")
    return f"{_kind(kind)} ({', '.join(facts)})" if facts else _kind(kind)


def _profile(raw: Mapping[str, Any]) -> TableProfile:
    """A table's profile as it was stored with the value: JSON made its columns dicts."""
    columns = tuple(ColumnProfile(**c) for c in raw.get("columns", ()))
    return TableProfile(**{**raw, "columns": columns})


def _questions(study: Study) -> str:
    replied = {item.ask for item in study.answers}
    lines: list[str] = []
    if study.answers:
        lines.append("You asked, and the scientist replied:")
        for item in study.answers:
            # An unanswered question and a "not sure" are different from each other and both
            # different from an answer, so they are written out rather than dropped.
            lines.append(f"- {item.ask}\n  -> {item.answer.strip() or '(no answer)'}")
    open_ = [q for turn in study.thread.turns for q in turn.questions if q.ask not in replied]
    if open_:
        lines.append("Asked, and not answered yet:")
        lines += [
            f"- {q.ask}" + (f" (your default: {q.default})" if q.default else "") for q in open_
        ]
    return "\n".join(lines)


def _terms(study: Study) -> str:
    lines: list[str] = []
    if study.checks:
        lines.append("What would count as a result, written before anything ran:")
        lines += [f"  {i}. {c}" for i, c in enumerate(study.checks, start=1)]
    if study.stop_conditions:
        lines.append("When the plan said to stop instead:")
        lines += [f"  {i}. {c}" for i, c in enumerate(study.stop_conditions, start=1)]
    if lines and (ran := study.amended_after_steps) is not None:
        when = f"step {ran} ran" if ran else "the study started, before any step ran"
        lines.append(f"These were amended after {when}.")
    return "\n".join(lines)


def _brought(study: Study) -> str:
    brought = [v for v in study.values if v.produced_by is None]
    lines: list[str] = []
    if brought:
        lines.append("What the scientist brought:")
    for value in brought:
        lines.append(f"- {one_line(value.kind, value.summary)}")
        # The shape of a table, so the model reads it before it asks about it.
        if profile := value.summary.get("profile"):
            lines.append(profile_text(_profile(profile)))
    if study.phase != "running":
        kinds = {value.kind for value in brought}
        lines.append(starting_note({kind for kind in SlotKind if kind.value in kinds}))
    return "\n".join(lines)


def _plan(study: Study) -> str:
    if not study.steps:
        return "The plan has no steps yet."
    lines = ["The steps, numbered as you refer to them (the first is 1):"]
    for step in sorted(study.steps, key=lambda s: s.position):
        facts = [step.status.value]
        if step.params:
            facts.append("settings " + ", ".join(f"{k}={v}" for k, v in step.params.items()))
        if step.gated:
            facts.append("gated")
        if step.problem:
            facts.append(f"problem: {step.problem}")
        if step.error:
            facts.append(f"error: {step.error}")
        if made := study.current_outputs_of(step.id):
            facts.append("produced: " + ", ".join(one_line(v.kind, v.summary) for v in made))
        lines.append(f"  {step.position + 1}. {step.name} [{step.tool_id}] — {'; '.join(facts)}")
    return "\n".join(lines)


def _step(numbers: Mapping[uuid.UUID, int], step_id: uuid.UUID) -> str:
    return f"step {numbers[step_id]}" if step_id in numbers else "a removed step"


def _point(point: Pointer, numbers: Mapping[uuid.UUID, int]) -> str:
    where = f"{_step(numbers, point.step)}, " if point.step is not None else ""
    return f"{where}{point.at} · {point.label}"


def _detail(turn: Turn, numbers: Mapping[uuid.UUID, int]) -> str:
    """What a turn pointed at and did, on one line."""
    parts: list[str] = []
    if turn.about is not None:
        parts.append(f"about {_step(numbers, turn.about)}")
    if turn.points:
        parts.append("points: " + "; ".join(_point(p, numbers) for p in turn.points))
    if turn.actions:
        parts.append("did: " + "; ".join(a.detail for a in turn.actions))
    return " | ".join(parts)


def _said(study: Study) -> str:
    # Silent turns go before the window is cut, so a tick costs no turn its place.
    turns = [t for t in study.thread.turns if not t.silent][-CONTEXT_TURNS:]
    if not turns:
        return ""
    numbers = {step.id: step.position + 1 for step in study.steps}
    lines = ["What has been said, most recent last:"]
    for turn in turns:
        lines.append(f"{turn.role}: {turn.say}")
        if detail := _detail(turn, numbers):
            lines.append(f"  {detail}")
    return "\n".join(lines)


def standing(
    study: Study,
    registry: ToolRegistry,
    *,
    memory: Sequence[str] = (),
    tool_notes: Mapping[str, str] = {},
) -> str:
    """The study as the agent reads it, whichever turn it is about to take.

    The catalogue comes first because the prompts point at it ("the catalogue below") and it
    is the part that does not change between turns. `memory` and `tool_notes` are the places
    for what the study recalls from earlier work and what a tool's manual says; nothing fills
    them yet, so nothing is written for them.
    """
    sections = [
        f"{CATALOGUE_HEADING}\n{catalogue(registry.manifests())}",
        f"The scientist's goal, in their words:\n{study.goal}",
        f"The goal as understood: {study.restated_goal}" if study.restated_goal else "",
        _questions(study),
        _terms(study),
        _brought(study),
        _plan(study),
        f"On this server, claims can be checked on: {', '.join(_kind(k) for k in checkable())}.",
        f"Where the study stands, as you last wrote it:\n{study.thread.summary}"
        if study.thread.summary
        else "",
        _said(study),
    ]
    return "\n\n".join(section for section in sections if section)


def _results(values: Sequence[SlotValue], data_of: Callable[[SlotValue], Any]) -> list[str]:
    """Each result as its reader describes it, with the notes that go with its kind."""
    blocks: list[str] = []
    for value in values:
        read = reader(value.kind)
        kind = _kind(value.kind)
        block = f"It produced {kind}:\n{read.summarize(data_of(value), value.summary)}"
        if read.notes:
            block += f"\n\nNotes on {kind}:\n{read.notes}"
        blocks.append(block)
    return blocks


def trigger_for_observe(study: Study, step: Step, data_by_value: Mapping[uuid.UUID, Any]) -> str:
    """What prompted an observing turn: the step that ended, and what it left behind.

    `data_by_value` holds the data of each of the step's current outputs, by value id.
    """
    named = f"{step.position + 1} ({step.name})"
    if step.status is StepStatus.FAILED:
        return f"Step {named} failed: {step.error}"
    results = _results(study.current_outputs_of(step.id), lambda v: data_by_value[v.id])
    return "\n\n".join([f"Step {named} finished.", *results])


def trigger_for_reply(study: Study, about: Step | None, data: Any | None) -> str:
    """The result a message is about, so a claim made in reply can be checked the same way.

    Nothing when the message is about no step in particular.
    """
    if about is None:
        return ""
    # ponytail: one `data`, so the first output; no tool gives a step two.
    results = _results(study.current_outputs_of(about.id)[:1], lambda _: data)
    return "\n\n".join(
        [f"The scientist's message is about step {about.position + 1} ({about.name}).", *results]
    )
