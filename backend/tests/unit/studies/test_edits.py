"""Changing a plan: one vocabulary for the agent, a proposal and a button.

What these pin is the trust boundary. The model names steps by number and tools by id,
and none of it is believed until the study and the catalogue agree: a number that names
nothing is dropped and counted, never raised, and nothing the scientist did not say is
done.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from daikonstudio.application.planner.plan import Setting
from daikonstudio.application.studies.edits import (
    Applied,
    alternatives,
    apply_actions,
    quoted,
    recheck,
)
from daikonstudio.application.studies.schemas import Action, Proposal
from daikonstudio.application.tools.manifest import Slot, SlotKind, ToolArea, ToolManifest
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.studies.study import SlotValue, Step, StepStatus, Study
from daikonstudio.infrastructure.tools.catalog import default_registry

REGISTRY = default_registry()
_T0 = datetime(2026, 1, 1, tzinfo=UTC)

COMPOUNDS = (SlotKind.COMPOUND_TABLE,)
PDB = ("fetch_pdb", {"pdb_id": "1CA2"})
PROTEIN = (PDB, "prepare_structure", "find_pockets")
NOT_QUOTED = "Proposed, not done: it did not quote your message."


def _value(kind: SlotKind, *, produced_by: uuid.UUID | None = None) -> SlotValue:
    return SlotValue(
        id=uuid.uuid4(),
        key=kind.value,
        kind=kind.value,
        blob_key=f"b/{kind.value}",
        summary={},
        produced_by=produced_by,
        created_at=_T0,
    )


def _study(*plan: str | tuple[str, dict[str, str]], brought: tuple[SlotKind, ...] = ()) -> Study:
    """A draft whose steps are the tools named, as a plan from `expand` has them.

    A step is `always` when its tool is automatic. Nothing is rechecked, so each plan used
    here is written to be coherent with what is brought, and a test that wants a problem
    makes one.
    """
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
                always=manifest.automatic,
                expanded_from=None,
            )
        )
    study = Study(
        workspace_id=uuid.uuid4(),
        created_by=uuid.uuid4(),
        goal="g",
        restated_goal="rg",
        checks=("a check",),
        stop_conditions=("a stop",),
        steps=steps,
    )
    for kind in brought:
        study.bring(_value(kind))
    return study


def _tools(study: Study) -> list[str]:
    return [step.tool_id for step in study.steps]


def _finish(study: Study, step: Step) -> None:
    study.start_step(step.id)
    study.finish_step(step.id, [])


def _shape(study: Study) -> list[tuple[Any, ...]]:
    """Everything an edit could change on a step, to show that a refused one changed none."""
    return [(s.tool_id, s.params, s.status, s.gated, s.always, s.problem) for s in study.steps]


def _apply(study: Study, *actions: Action, message: str | None = None) -> Applied:
    return apply_actions(study, actions, REGISTRY, message=message)


def _setting(name: str, value: str) -> Setting:
    return Setting(name=name, value=value)


def _fake(tool_id: str, *, takes: tuple[Slot, ...] = (), **kw: Any) -> ToolManifest:
    """A made-up tool: it takes what it is told and gives a structure unless told otherwise."""
    gives = kw.pop("gives", SlotKind.STRUCTURE)
    return ToolManifest(
        id=tool_id,
        version="2",
        name=f"Tool {tool_id}",
        description=f"Does {tool_id}.",
        area=ToolArea.DATA,
        inputs=takes,
        outputs=(Slot(key="out", kind=gives, label="Out"),),
        library="test",
        **kw,
    )


def test_alternatives_are_derived_from_slot_signatures() -> None:
    table = Slot(key="t", kind=SlotKind.COMPOUND_TABLE, label="T")
    one, two = _fake("one"), _fake("two")
    # An optional input is a difference in what a tool can use, not in what it needs.
    optional = _fake("optional", takes=(Slot(key="t", kind=table.kind, label="T", optional=True),))
    needs_table = _fake("needs_table", takes=(table,))
    fake = ToolRegistry((one, two, optional, needs_table))

    assert [m.id for m in alternatives(one, fake)] == ["two", "optional"]
    assert alternatives(needs_table, fake) == []

    # The one pair the real catalogue has: both take nothing and give a structure.
    pdb, alphafold = REGISTRY.get("fetch_pdb"), REGISTRY.get("fetch_alphafold")
    assert [m.id for m in alternatives(pdb, REGISTRY)] == ["fetch_alphafold"]
    assert [m.id for m in alternatives(alphafold, REGISTRY)] == ["fetch_pdb"]


def test_a_composite_or_an_automatic_tool_is_never_an_alternative() -> None:
    plain = _fake("plain")
    placed = _fake("placed", automatic=True)
    composite = _fake("composite", expands_to=("plain",))
    assert alternatives(plain, ToolRegistry((plain, placed, composite))) == []


def test_quoted_collapses_whitespace_and_case_and_refuses_empty() -> None:
    message = "Please  re-run\n  preparation   WITHOUT the zinc, thanks"
    assert quoted("re-run preparation without the zinc", message)
    assert quoted("  RE-RUN   preparation without the ZINC ", message)
    assert not quoted("re-run preparation with the zinc", message)
    assert not quoted("", message)
    assert not quoted("   \n", message)  # nothing but spacing quotes nothing


def test_insert_places_the_step_and_an_automatic_one_after_it() -> None:
    study = _study("standardize", "property_profile", brought=COMPOUNDS)
    applied = _apply(
        study,
        Action(
            does="insert",
            tool_id="fetch_pdb",
            purpose="Get carbonic anhydrase II",
            settings=[_setting("pdb_id", "1CA2")],
            quoting="add the structure",
        ),
        message="Please add the structure 1CA2",
    )
    assert applied.dropped == 0 and applied.downgraded == []
    assert _tools(study) == ["standardize", "property_profile", "fetch_pdb", "prepare_structure"]
    assert [s.position for s in study.steps] == [0, 1, 2, 3]
    assert [s.always for s in study.steps] == [True, False, False, True]
    fetch, prepare = study.steps[2], study.steps[3]
    assert fetch.purpose == "Get carbonic anhydrase II" and fetch.params == {"pdb_id": "1CA2"}
    assert fetch.name == REGISTRY.get("fetch_pdb").name
    assert fetch.tool_version == REGISTRY.get("fetch_pdb").version
    assert prepare.purpose == REGISTRY.get("prepare_structure").description
    assert prepare.gated is False and prepare.params == {}
    assert all(s.problem is None for s in study.steps)  # nothing wanted is missing
    (acted,) = applied.acted
    assert (acted.does, acted.step, acted.quoting) == ("insert", fetch.id, "add the structure")
    assert acted.detail == "Added step 3 (Fetch a deposited structure): pdb_id=1CA2."

    # A composite goes in as its parts, in order, each naming it and carrying its purpose; the
    # gate asked for stops after the last, and each part takes the settings it declares. An
    # always-run tool is still never inserted.
    study = _study("standardize", brought=COMPOUNDS)
    applied = _apply(
        study,
        Action(
            does="insert",
            tool_id="check_properties",
            purpose="Whether potency tracks lipophilicity",
            settings=[_setting("activity_column", "pIC50")],
            gated=True,
        ),
        Action(does="insert", tool_id="standardize"),
    )
    assert applied.dropped == 1
    assert _tools(study) == ["standardize", "property_profile", "property_confound"]
    added = study.steps[1:]
    assert [(s.expanded_from, s.gated) for s in added] == [
        ("check_properties", False),
        ("check_properties", True),
    ]
    assert {s.purpose for s in added} == {"Whether potency tracks lipophilicity"}
    assert [s.params for s in added] == [{"activity_column": "pIC50"}] * 2
    assert all(s.problem is None for s in study.steps)
    assert [(a.step, a.detail) for a in applied.acted] == [
        (
            added[0].id,
            "Added Check the properties as step 2 (Physicochemical properties), "
            "step 3 (Property confound check): activity_column=pIC50.",
        )
    ]

    # "Merge 2 and 3" comes as their removal and an insert before the first of them: it goes
    # where they were, before the next step still there, rather than being dropped.
    study = _study(
        "standardize",
        "property_profile",
        "property_confound",
        "structural_alerts",
        brought=COMPOUNDS,
    )
    alerts = study.steps[3]
    applied = _apply(
        study,
        Action(does="remove", step=2),
        Action(does="remove", step=3),
        Action(
            does="insert",
            tool_id="check_properties",
            before=2,
            settings=[_setting("activity_column", "pIC50")],
        ),
    )
    assert applied.dropped == 0 and len(applied.acted) == 3
    assert _tools(study) == [
        "standardize",
        "property_profile",
        "property_confound",
        "structural_alerts",
    ]
    assert study.steps[1].expanded_from == "check_properties" and study.steps[3] is alerts


def test_remove_and_swap_renumber_and_recheck() -> None:
    study = _study(*PROTEIN)
    fetch, prepare, pockets = study.steps
    applied = _apply(study, Action(does="swap", step=1, tool_id="fetch_alphafold"))
    assert applied.dropped == 0
    assert (fetch.tool_id, fetch.name, fetch.tool_version) == (
        "fetch_alphafold",
        REGISTRY.get("fetch_alphafold").name,
        REGISTRY.get("fetch_alphafold").version,
    )
    assert fetch.params == {}  # pdb_id is not a setting of the tool it was swapped for
    assert fetch.status is StepStatus.PENDING
    assert fetch.problem is not None and "accession" in fetch.problem
    assert prepare.problem is None and pockets.problem is None
    (acted,) = applied.acted
    assert acted.detail == (
        "Changed step 1 from Fetch a deposited structure to Fetch a predicted structure."
    )

    # Taking out the fetch leaves preparation waiting on a structure that the pocket search
    # still wants, as in a plan drafted before its file is attached: it stays, problem showing.
    removed = _apply(study, Action(does="remove", step=1))
    assert removed.dropped == 0 and removed.acted[0].step == fetch.id
    assert _tools(study) == ["prepare_structure", "find_pockets"]
    assert [s.position for s in study.steps] == [0, 1]
    assert prepare.problem is not None and "structure" in prepare.problem
    assert pockets.problem is None  # preparation still stands in for the structure it wants

    # Take the search out too and nothing wants a structure: preparation goes with it, rather
    # than sit there with a problem that the model may not remove.
    gone = _apply(study, Action(does="remove", step=2))
    assert gone.dropped == 0 and gone.acted[0].step == pockets.id
    assert _tools(study) == []


def test_an_action_naming_a_position_out_of_range_is_dropped_not_raised() -> None:
    study = _study(*PROTEIN)
    before = _shape(study)
    applied = _apply(
        study,
        Action(does="remove", step=9),
        Action(does="gate", step=0, gated=True),  # 0 names no step, and a gate needs one
        Action(does="insert", tool_id="fetch_pdb", before=9),
        Action(does="run", until=9),
    )
    assert applied.dropped == 4
    assert applied.acted == [] and applied.downgraded == [] and applied.approvals == []
    assert applied.run is None
    assert _shape(study) == before


def test_an_unquoted_action_becomes_a_proposal() -> None:
    study = _study(*PROTEIN)
    before = _shape(study)
    applied = _apply(
        study,
        Action(
            does="settings",
            step=2,
            settings=[_setting("keep_ligands", "ZN")],
            quoting="without the zinc",  # not something the scientist said
        ),
        Action(does="gate", step=3, gated=True),  # said nothing at all
        message="Please look at the pockets",
    )
    assert applied.acted == [] and applied.dropped == 0
    assert _shape(study) == before  # nothing was done
    settings, gate = applied.downgraded
    assert settings == Proposal(
        does="settings", step=2, settings=[_setting("keep_ligands", "ZN")], why=NOT_QUOTED
    )
    assert (gate.does, gate.step, gate.gated, gate.why) == ("gate", 3, True, NOT_QUOTED)


def test_a_quoted_settings_change_puts_the_step_back_to_pending() -> None:
    study = _study(*PROTEIN)
    study.start()
    prepare = study.steps[1]
    prepare.params = {"ph": "6.5"}
    _finish(study, prepare)
    assert study.done_count == 1  # it has run to a result
    applied = _apply(
        study,
        Action(
            does="settings",
            step=2,
            settings=[_setting("keep_ligands", "ZN")],
            quoting="keep the zinc",
        ),
        message="Re-run preparation, but keep the zinc.",
    )
    assert applied.dropped == 0 and applied.downgraded == []
    assert prepare.status is StepStatus.PENDING
    assert prepare.params == {"ph": "6.5", "keep_ligands": "ZN"}  # a change, not a replacement
    (acted,) = applied.acted
    assert (
        acted.detail == "Changed the settings of step 2 (Prepare the structure): keep_ligands=ZN."
    )
    assert (acted.does, acted.step, acted.quoting) == ("settings", prepare.id, "keep the zinc")


def test_terms_change_records_the_amendment_and_never_from_a_button() -> None:
    study = _study(*PROTEIN)
    study.start()
    _finish(study, study.steps[0])
    applied = _apply(
        study,
        Action(
            does="terms",
            checks=["The pocket contains the annotated site"],
            quoting="change the check to",
        ),
        message="Change the check to: the pocket must contain the annotated site",
    )
    assert applied.dropped == 0
    assert study.checks == ("The pocket contains the annotated site",)
    assert study.stop_conditions == ("a stop",)  # not given, so not touched
    assert study.amended_after_steps == 1
    (acted,) = applied.acted
    assert (acted.does, acted.step, acted.detail) == ("terms", None, "Changed the checks.")

    # Without the scientist's words it is dropped, not offered: there is no button for it.
    unquoted = _apply(
        study,
        Action(does="terms", stop_conditions=["Never stop"], quoting="never stop"),
        message="Please carry on",
    )
    assert unquoted.dropped == 1 and unquoted.downgraded == [] and unquoted.acted == []
    assert study.stop_conditions == ("a stop",)

    # An empty list is not a replacement: a slip would wipe what the scientist wrote.
    quote = "change the check to"
    emptied = _apply(
        study,
        Action(does="terms", checks=[], stop_conditions=[], quoting=quote),
        Action(does="terms", checks=[], stop_conditions=["A new stop"], quoting=quote),
        message="Change the check to: the pocket must contain the annotated site",
    )
    assert emptied.dropped == 1  # the first has nothing to change
    assert [a.detail for a in emptied.acted] == ["Changed the stop conditions."]
    assert study.checks == ("The pocket contains the annotated site",)
    assert study.stop_conditions == ("A new stop",)
    with pytest.raises(ValidationError):
        Proposal(does="terms", why="x")  # type: ignore[arg-type]
    assert not {"checks", "stop_conditions", "quoting"} & set(Proposal.model_fields)


def test_recheck_pins_a_problem_to_the_step_that_wants_a_missing_input() -> None:
    study = _study("fetch_uniprot", "find_pockets", "predict_admet")
    fetch, pockets, admet = study.steps
    fetch.problem = "stale"  # a problem the plan no longer has goes away
    recheck(study, REGISTRY)
    assert fetch.problem is None
    assert pockets.problem is not None and "structure" in pockets.problem
    assert pockets.problem.startswith("Step 2 (Find binding pockets)")
    assert admet.problem is not None and "compound table" in admet.problem

    study.bring(_value(SlotKind.STRUCTURE))
    recheck(study, REGISTRY)  # what the scientist brought counts
    assert pockets.problem is None
    assert admet.problem is not None  # still no table


def test_a_position_means_the_step_the_model_saw_however_the_plan_shifts() -> None:
    """The one decision the single-action tests above cannot tell apart.

    "Remove step 2 and gate step 3" means the step that was 3 when the model wrote that.
    Taken one action at a time, the gate would land on whichever step had slid into 3 and
    the scientist's instruction would be carried out on the wrong step, quietly. And a
    proposal made for the step that was 4 has to be handed back under the number that step
    has now, or the button would be pressed on another.
    """
    study = _study("standardize", "property_profile", "structural_alerts", "predict_admet")
    study.bring(_value(SlotKind.COMPOUND_TABLE))
    _, profile, alerts, admet = study.steps
    applied = _apply(
        study,
        Action(does="remove", step=2, quoting="drop the profile"),
        Action(does="gate", step=3, gated=True, quoting="stop at the alerts"),
        Action(does="gate", step=4, gated=True, quoting="not said"),
        message="Drop the profile, and stop at the alerts.",
    )
    assert applied.dropped == 0
    assert profile not in study.steps
    assert (alerts.gated, admet.gated) == (True, False)  # the gate went where it was meant
    (proposal,) = applied.downgraded
    assert (proposal.does, proposal.step) == ("gate", 3)  # predict_admet, as numbered now
    assert study.steps[proposal.step - 1] is admet
