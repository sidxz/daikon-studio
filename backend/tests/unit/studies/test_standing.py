"""What the agent is shown of a study each turn.

The model reads a study only through this text, so what it leaves out cannot be claimed
about and what it gets wrong is believed: the plan is numbered the way the model's own
numbers are resolved, and the conversation is cut to a window with nothing silent in it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

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
        f"1. {actives.name} [chembl_actives] — done; "
        "produced: compound table (120 rows, 8 columns)"
    ) in text
    assert (
        f"2. {fetch.name} [fetch_pdb] — done; settings pdb_id=1CA2; produced: structure"
    ) in text
    assert (
        f"3. {pockets.name} [find_pockets] — pending; gated; "
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
