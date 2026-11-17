"""What the agent is shown of a study each turn.

The model reads a study only through this text, so what it leaves out cannot be claimed
about and what it gets wrong is believed: the plan is numbered the way the model's own
numbers are resolved, and the conversation is cut to a window with nothing silent in it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from daikonstudio.application.planner.catalogue import (
    CATALOGUE_HEADING,
    INDEX_HEADING,
    VIEW_HEADING,
    index,
)
from daikonstudio.application.studies.standing import CONTEXT_TURNS, standing
from daikonstudio.domain.studies.study import Pointer, SlotValue, Step, Study, Turn
from daikonstudio.infrastructure.tools.catalog import default_registry

REGISTRY = default_registry()
_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _study(*plan: str | tuple[str, dict[str, str]], **kw: Any) -> Study:
    """A draft whose steps are the tools named, in order, with their catalogue names."""
    steps = []
    for position, spec in enumerate(plan):
        tool_id, params = (spec, {}) if isinstance(spec, str) else spec
        manifest = REGISTRY.get(tool_id)
        steps.append(
            Step(
                id=uuid.uuid4(),
                position=position,
                tool_id=tool_id,
                tool_version=manifest.version,
                name=manifest.name,
                purpose="p",
                params=dict(params),
                gated=False,
                always=False,
                expanded_from=None,
            )
        )
    return Study(
        workspace_id=uuid.uuid4(),
        created_by=uuid.uuid4(),
        goal="g",
        restated_goal="rg",
        checks=kw.pop("checks", ()),
        stop_conditions=kw.pop("stop_conditions", ()),
        steps=steps,
        **kw,
    )


def _value(kind: str, produced_by: uuid.UUID, **summary: Any) -> SlotValue:
    return SlotValue(
        id=uuid.uuid4(),
        key=kind,
        kind=kind,
        blob_key=f"b/{kind}",
        summary=summary,
        produced_by=produced_by,
        created_at=_T0,
    )


def _finish(study: Study, step: Step, *values: SlotValue) -> None:
    study.start_step(step.id)
    study.finish_step(step.id, list(values))


def test_steps_are_numbered_from_one_and_name_their_outputs_by_kind() -> None:
    study = _study(
        "chembl_actives", ("fetch_pdb", {"pdb_id": "1CA2"}), "find_pockets", "property_confound"
    )
    actives, fetch, pockets, confound = study.steps
    _finish(
        study, actives, _value("compound_table", actives.id, rows=120, columns=list("abcdefgh"))
    )
    _finish(study, fetch, _value("structure", fetch.id, id="1CA2"))
    pockets.gated = True
    pockets.problem = "no structure has been prepared"
    # A result is described as its reader describes it, so a question typed with no step in hand
    # still sees every coefficient; a long description is cut, and the trigger has it whole.
    properties = [{"property": "logp", "r": 0.1237, "ci_low": -0.2131, "ci_high": 0.41, "n": 26}]
    properties += [
        {"property": f"descriptor_{i}", "r": 0.0, "ci_low": -0.1, "ci_high": 0.1, "n": 26}
        for i in range(60)
    ]
    correlation = _value("correlation", confound.id, activity_column="pIC50", excluded=0)
    _finish(study, confound, correlation)

    text = standing(study, REGISTRY, data={correlation.id: {"properties": properties}})

    assert "- logp: r 0.124 (-0.213 to 0.41), n 26" in text
    assert "descriptor_59" not in text and "(…)" in text
    assert "     id: 1CA2" in text  # under the step that made it

    assert (
        f"1. {actives.name} [chembl_actives]: done; produced: compound table (120 rows, 8 columns)"
    ) in text
    assert (
        f"2. {fetch.name} [fetch_pdb]: done; settings pdb_id=1CA2; produced: structure"
    ) in text
    assert (
        f"3. {pockets.name} [find_pockets]: pending; gated; "
        "problem: no structure has been prepared"
    ) in text


def test_silent_turns_and_old_turns_stay_out_of_the_standing() -> None:
    study = _study("fetch_pdb")
    for i in range(CONTEXT_TURNS + 3):
        study.say(Turn(role="scientist", say=f"said-{i:02d}", at=_T0))
    study.say(Turn(role="agent", say="a tick", silent=True, at=_T0))

    text = standing(study, REGISTRY)

    # The silent turn is dropped before the window is cut, so it costs no turn its place.
    assert "a tick" not in text
    assert "said-02" not in text
    assert "scientist: said-03" in text
    assert "scientist: said-22" in text


def test_an_amendment_is_shown_with_the_step_count() -> None:
    study = _study(
        "fetch_pdb", "prepare_structure", checks=("check one",), stop_conditions=("stop one",)
    )
    assert "amended" not in standing(study, REGISTRY)

    study.start()
    study.amend_terms(("check two",), None)
    assert "These were amended before any step ran." in standing(study, REGISTRY)

    # A count of the steps that had run, as the page says it, never a step's number.
    _finish(study, study.steps[1])
    study.amend_terms(("check three",), None)
    text = standing(study, REGISTRY)
    assert "These were amended after 1 step had run." in text
    assert "check three" in text


def test_a_removed_step_is_named_as_removed_not_looked_up() -> None:
    """Steps can be removed, and a turn written about one still names it. Reading that
    must not raise: the study would be unable to take another turn."""
    study = _study("fetch_pdb", "prepare_structure", "find_pockets")
    gone = study.steps[1]
    study.say(Turn(role="scientist", say="why this one?", about=gone.id, at=_T0))
    point = Pointer(step=gone.id, at="rank 1", label="the zinc cone", evidence="lining_residues")
    study.say(Turn(role="agent", say="Because.", points=(point,), at=_T0))
    study.remove_step(gone.id)

    text = standing(study, REGISTRY)

    assert "about a removed step" in text
    assert "a removed step, rank 1 · the zinc cone" in text


def test_standing_carries_no_catalogue_unless_given_a_view() -> None:
    text = standing(_study("standardize"), REGISTRY)
    assert CATALOGUE_HEADING not in text and INDEX_HEADING not in text and VIEW_HEADING not in text


def test_a_view_is_rendered_in_full_and_first() -> None:
    view = [REGISTRY.get("find_cores")]
    text = standing(_study(), REGISTRY, view=view, left=4)
    assert text.startswith(VIEW_HEADING)
    assert "- find_cores: " in text and "takes:" in text
    assert "You have 4 questions left" in text


def test_index_is_one_line_per_tool_in_catalogue_order() -> None:
    lines = index(REGISTRY.manifests()).splitlines()
    assert len(lines) == len(REGISTRY.manifests())
    assert "series (optional)" in next(line for line in lines if line.startswith("- find_cores:"))
    assert all(line.startswith("- ") and "takes:" in line and "gives:" in line for line in lines)
    assert REGISTRY.get(lines[0][2:].split(":")[0]).expands_to, (
        "composites come first, as in catalogue()"
    )


def test_open_items_are_carried_forward_from_the_latest_list_only() -> None:
    """The agent rewrites `unresolved` whole each turn, so an older list is a stale copy
    of the same items. Carried forward, a worry it already recorded reads as recorded
    rather than as something it still has to say, which is what made it say it again in
    different words every turn."""
    study = _study("fetch_pdb")
    study.say(_read("step 1 is in", "where the gap sits"))
    study.say(_read("step 2 is in", "no inhibitor here"))

    text = standing(study, REGISTRY)

    assert "- no inhibitor here" in text
    assert "- where the gap sits" not in text
    assert "Already recorded as open" in text


def test_an_item_the_latest_reading_left_out_is_not_carried_back() -> None:
    """Leaving an item out of `unresolved` is how a turn closes it. Looking back past an
    empty list for the newest non-empty one would hand the agent its own closed worry
    every turn after, which is the repetition this block exists to stop."""
    study = _study("fetch_pdb")
    study.say(_read("step 1 is in", "where the gap sits"))
    study.say(_read("step 2 is in"))
    # Events and the scientist carry no list of their own and must not reopen one.
    study.say(Turn(role="event", kind="ran", say="Started a run.", at=_T0))

    assert "Already recorded as open" not in standing(study, REGISTRY)


def test_nothing_is_carried_when_nothing_is_open() -> None:
    study = _study("fetch_pdb")
    study.say(Turn(role="agent", say="step 1 is in", at=_T0))

    assert "Already recorded as open" not in standing(study, REGISTRY)


def _read(say: str, *unresolved: str) -> Turn:
    """An agent turn of the kind that carries an open-items list."""
    return Turn(role="agent", kind="observe", say=say, unresolved=unresolved, at=_T0)


def test_the_standing_is_written_the_way_it_tells_the_model_to_write() -> None:
    """The prompts forbid dashes as punctuation, but the standing is prompt too, rebuilt and
    read on every turn. One in here undoes the rule by example, which is how the planner came
    to put them in the restated goal, the step purposes and the checks."""
    study = _study("fetch_pdb", "prepare_structure", checks=("c",), stop_conditions=("s",))
    study.say(Turn(role="scientist", say="go", at=_T0))

    text = standing(study, REGISTRY)

    assert "\u2014" not in text
    assert "\u2013" not in text
