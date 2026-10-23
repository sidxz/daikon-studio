"""The one agent a study has: what it is shown, what it may do, and what is kept of a turn.

Every test scripts the model and asserts on what the study stored, because that is what the
scientist reads and what the next turn is built from. Nothing the model says is believed as
it said it: a claim is kept only where its result carries it, a change is made only on the
scientist's words, and a look costs exactly one more call.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

import polars as pl
import pytest
from returns.result import Result, Success

from daikonstudio.application.auth import AuthContext
from daikonstudio.application.planner.plan import DraftPlan, Interrogation, Question, Setting
from daikonstudio.application.planner.plan import Step as PlanStep
from daikonstudio.application.studies import advance_study
from daikonstudio.application.studies.agent import ObserveStep, SayCommand, StudyAgent
from daikonstudio.application.studies.prompts import MAX_QUESTIONS
from daikonstudio.application.studies.schemas import (
    Action,
    Claim,
    Look,
    ObserveTurn,
    PlanningTurn,
    ProgressClaim,
    ReplyTurn,
)
from daikonstudio.application.tools import codecs
from daikonstudio.domain.shared.errors import ConflictError, DomainError, ServiceUnavailableError
from daikonstudio.domain.studies.study import Answered, Asked, SlotValue, Step, Study, Turn
from daikonstudio.infrastructure.tools.catalog import default_registry
from tests.fakes.auth import FakeAuth
from tests.fakes.blob_store import InMemoryBlobStore
from tests.fakes.llm import ScriptedLanguageModel, ScriptedLanguageModels
from tests.fakes.studies import InMemoryStudies

REGISTRY = default_registry()
AUTH = FakeAuth()
CHECK = "Exactly one candidate is lined by His94 and His96."
PDB = ("fetch_pdb", {"pdb_id": "1CA2"})
PROTEIN = (PDB, "prepare_structure", "find_pockets")


def _pocket(rank: int, *residues: tuple[str, int]) -> dict[str, Any]:
    return {
        "rank": rank,
        "residues": [{"chain": "A", "resname": name, "resseq": seq} for name, seq in residues],
        "ligand_overlap": [],
        "annotated": [],
        "interface": False,
    }


POCKETS = {
    "meta": {"count": 2},
    "pockets": [_pocket(1, ("ALA", 128), ("LYS", 132)), _pocket(10, ("HIS", 94), ("HIS", 96))],
}


def _study(*plan: str | tuple[str, dict[str, str]], started: bool = True) -> Study:
    """The scientist's study, its steps the tools named, in order; started unless said not."""
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
        workspace_id=AUTH.workspace_id,
        created_by=AUTH.user_id,
        goal="Find where a drug could bind carbonic anhydrase II.",
        restated_goal="",
        checks=(CHECK,),
        stop_conditions=("No candidate is lined by a histidine.",),
        steps=steps,
    )
    if started:
        study.start()
    return study


def _done(study: Study, store: InMemoryBlobStore, number: int, kind: str, data: Any) -> SlotValue:
    """Step `number` (counted from 1) ran and left `data` in the store, as a tool leaves it."""
    step = study.steps[number - 1]
    blob, summary = codecs.encode(kind, data)
    value = SlotValue(
        id=uuid.uuid4(),
        key=kind,
        kind=kind,
        blob_key=f"values/{uuid.uuid4()}",
        summary=summary,
        produced_by=step.id,
        created_at=datetime.now(UTC),
    )
    store.put_bytes(value.blob_key, blob)
    study.start_step(step.id)
    study.finish_step(step.id, [value])
    return value


class _Advance:
    """The run loop as the agent calls it, recording each run it was asked to start."""

    def __init__(self) -> None:
        self.runs: list[tuple[uuid.UUID | None, uuid.UUID | None]] = []
        self.started_by: list[str] = []

    async def __call__(
        self,
        study_id: uuid.UUID,
        auth: AuthContext | None = None,
        *,
        start: uuid.UUID | None = None,
        until: uuid.UUID | None = None,
        started_by: str = "scientist",
    ) -> Result[Study, DomainError]:
        self.runs.append((start, until))
        self.started_by.append(started_by)
        return Success(_study(started=False))


def _agent(
    study: Study, store: InMemoryBlobStore, model: ScriptedLanguageModel | None
) -> tuple[StudyAgent, InMemoryStudies, _Advance]:
    studies = InMemoryStudies()
    studies.rows[study.id] = study
    advance = _Advance()
    agent = StudyAgent(studies, REGISTRY, ScriptedLanguageModels(model), store, advance)
    return agent, studies, advance


def _observer(
    study: Study, store: InMemoryBlobStore, model: ScriptedLanguageModel | None
) -> tuple[ObserveStep, InMemoryStudies]:
    studies = InMemoryStudies()
    studies.rows[study.id] = study
    return ObserveStep(studies, REGISTRY, ScriptedLanguageModels(model), store), studies


def _say(study: Study, message: str | None = None, **kw: Any) -> SayCommand:
    return SayCommand(study_id=study.id, message=message, **kw)


def _pockets_plan() -> DraftPlan:
    return DraftPlan(
        restated_goal="Where a drug could bind carbonic anhydrase II.",
        steps=[
            PlanStep(
                tool_id="fetch_pdb",
                purpose="Fetch 1CA2.",
                gated=False,
                settings=[Setting(name="pdb_id", value="1CA2")],
            ),
            PlanStep(tool_id="find_pockets", purpose="Find candidate sites.", gated=True),
        ],
        checks=[CHECK],
        stop_conditions=["No candidate is lined by a histidine."],
    )


async def test_the_first_turn_asks_and_restates_without_steps() -> None:
    study = _study(started=False)
    questions = [Question(ask=f"Question {i}?", why="w", default="d") for i in range(8)]
    restated = "Where a drug could bind carbonic anhydrase II."
    model = ScriptedLanguageModel(Interrogation(restated_goal=restated, questions=questions))
    agent, studies, _ = _agent(study, InMemoryBlobStore(), model)

    await agent.ask(study, None)

    stored = studies.rows[study.id]
    (turn,) = stored.thread.turns
    assert (turn.role, turn.kind, turn.say) == ("agent", "asking", restated)
    assert [q.ask for q in turn.questions] == [f"Question {i}?" for i in range(MAX_QUESTIONS)]
    assert stored.restated_goal == restated
    assert stored.steps == []
    assert model.schemas == [Interrogation]


async def test_zero_questions_chains_straight_into_a_plan() -> None:
    study = _study(started=False)
    model = ScriptedLanguageModel(
        Interrogation(restated_goal="rg", questions=[]),
        PlanningTurn(say="Here is a plan.", plan=_pockets_plan()),
    )
    agent, studies, _ = _agent(study, InMemoryBlobStore(), model)

    await agent.ask(study, None)

    stored = studies.rows[study.id]
    assert [t.kind for t in stored.thread.turns] == ["asking", "planning"]
    assert [s.tool_id for s in stored.steps] == ["fetch_pdb", "prepare_structure", "find_pockets"]
    assert model.schemas == [Interrogation, PlanningTurn]


async def test_a_plan_reply_expands_composites_and_pins_problems_to_steps() -> None:
    """Asked to check properties with no activity column, the plan comes back once with its
    problem in front of the model; the second answer stands and the problem sits on the step
    that has it, not on the composite the model wrote."""
    study = _study(started=False)
    study.bring(
        SlotValue(
            id=uuid.uuid4(),
            key="compound_table",
            kind="compound_table",
            blob_key="b",
            summary={"rows": 2, "columns": ["id", "smiles", "pIC50"]},
            produced_by=None,
            created_at=datetime.now(UTC),
        )
    )
    asked = "Which column is the readout?"
    study.say(
        Turn(
            role="agent",
            kind="asking",
            say="rg",
            questions=(Asked(ask=asked, why="w"),),
            at=datetime.now(UTC),
        )
    )
    plan = DraftPlan(
        restated_goal="Whether potency tracks lipophilicity.",
        steps=[PlanStep(tool_id="check_properties", purpose="Profile the series.", gated=False)],
        checks=["Potency does not track clogP."],
        stop_conditions=["Fewer than ten compounds have a potency."],
    )
    model = ScriptedLanguageModel(PlanningTurn(say="A plan.", plan=plan))
    agent, studies, _ = _agent(study, InMemoryBlobStore(), model)

    await agent(_say(study, answers=(Answered(ask=asked, answer="pIC50"),)), AUTH)

    stored = studies.rows[study.id]
    assert stored.answers == (Answered(ask=asked, answer="pIC50"),)
    assert stored.thread.turns[-2].say == "Which column is the readout? → pIC50"
    assert [s.tool_id for s in stored.steps] == [
        "standardize",
        "property_profile",
        "property_confound",
    ]
    assert [s.expanded_from for s in stored.steps] == [
        None,
        "check_properties",
        "check_properties",
    ]
    assert [s.problem is not None for s in stored.steps] == [False, False, True]
    assert "activity_column" in (stored.steps[2].problem or "")
    assert len(model.calls) == 2
    assert "could not run as written" in model.calls[1][1]
    assert (stored.restated_goal, stored.checks) == (plan.restated_goal, tuple(plan.checks))


async def test_a_planning_change_by_actions_needs_no_quote() -> None:
    study = _study(*PROTEIN, started=False)
    study.say(Turn(role="agent", kind="planning", say="A plan.", at=datetime.now(UTC)))
    model = ScriptedLanguageModel(
        PlanningTurn(say="It stops there now.", actions=[Action(does="gate", step=3, gated=True)])
    )
    agent, studies, _ = _agent(study, InMemoryBlobStore(), model)

    await agent(_say(study, "I want to look at the pockets before anything else."), AUTH)

    stored = studies.rows[study.id]
    assert stored.steps[2].gated is True
    turn = stored.thread.turns[-1]
    assert (turn.kind, turn.dropped) == ("planning", 0)
    assert [(a.does, a.quoting) for a in turn.actions] == [("gate", "")]


async def test_the_scientist_turn_is_stored_before_the_model_is_asked() -> None:
    study = _study(*PROTEIN)
    agent, studies, _ = _agent(study, InMemoryBlobStore(), None)

    with pytest.raises(ServiceUnavailableError):
        await agent(_say(study, "What did the pockets show?"), AUTH)

    turns = studies.rows[study.id].thread.turns
    assert [(t.role, t.say) for t in turns] == [("scientist", "What did the pockets show?")]


async def test_an_unsupported_claim_is_dropped_and_counted() -> None:
    """A result claim is checked by its result's reader, a progress claim by the plan's own
    words; what neither carries is dropped, and the drop is counted on the turn."""
    store = InMemoryBlobStore()
    study = _study(*PROTEIN)
    _done(study, store, 1, "structure", {"format": "pdb", "text": "ATOM", "meta": {"id": "1CA2"}})
    _done(study, store, 3, "pocket_candidates", POCKETS)
    zinc = {"candidates": [10], "label": "the zinc cone", "evidence": "lining_residues"}
    reply = ReplyTurn(
        say="Candidate 10 is the zinc site.",
        claims=[
            Claim(step=3, cites=["HIS94", "HIS96"], **zinc),
            Claim(step=3, cites=["HIS94", "TYR158"], **zinc),  # not a residue of candidate 10
            Claim(step=1, label="a deposited structure", evidence="column", cites=["id"]),
        ],
        progress=[
            ProgressClaim(about="check", refers_to=[1], label="met", cites=[CHECK]),
            ProgressClaim(
                about="check",
                refers_to=[1],
                label="not met",
                cites=["Exactly one candidate is lined by His119."],  # never in the plan
            ),
            ProgressClaim(
                about="step", refers_to=[3], label="ran", cites=["Find binding pockets"]
            ),
        ],
        summary="s",
    )
    agent, studies, _ = _agent(study, store, ScriptedLanguageModel(reply))

    await agent(_say(study, "Which pocket is the zinc site?"), AUTH)

    turn = studies.rows[study.id].thread.turns[-1]
    assert [(p.at, p.label) for p in turn.points] == [
        ("rank 10", "the zinc cone"),
        ("check 1", "met"),
        ("step 3", "ran"),
    ]
    assert turn.points[0].step == study.steps[2].id
    assert turn.dropped == 3


async def test_a_look_costs_one_more_call_and_the_rows_reach_the_second() -> None:
    store = InMemoryBlobStore()
    study = _study("standardize", "property_profile")
    table = pl.DataFrame({"id": ["a", "b"], "clogp": [1.5, 2.5]})
    _done(study, store, 2, "property_table", table)
    model = ScriptedLanguageModel(
        ReplyTurn(say="", look=[Look(step=2, rows=5)], summary="s"),
        ReplyTurn(
            say="One compound stands out.",
            claims=[Claim(step=2, label="the most lipophilic", evidence="rows", cites=["a"])],
            summary="s",
        ),
    )
    agent, studies, _ = _agent(study, store, model)

    await agent(_say(study, "Which compound is the greasiest?"), AUTH)

    assert len(model.calls) == 2
    assert "a: id=a, clogp=1.5" not in model.calls[0][1]
    assert "a: id=a, clogp=1.5" in model.calls[1][1]
    turn = studies.rows[study.id].thread.turns[-1]
    assert turn.looked == ("step 2: 5 rows",)
    assert [(p.step, p.cites) for p in turn.points] == [(study.steps[1].id, ("a",))]
    assert turn.dropped == 0


async def test_a_turn_never_makes_a_third_call() -> None:
    store = InMemoryBlobStore()
    study = _study("standardize", "property_profile")
    _done(study, store, 2, "property_table", pl.DataFrame({"id": ["a"], "clogp": [1.5]}))
    model = ScriptedLanguageModel(ReplyTurn(say="", look=[Look(step=2)], summary="s"))
    agent, studies, _ = _agent(study, store, model)

    await agent(_say(study, "Show me."), AUTH)

    assert len(model.calls) == 2
    assert studies.rows[study.id].thread.turns[-1].looked == ("step 2: 20 rows",)


async def test_a_quoted_run_starts_the_loop_and_an_unquoted_one_is_proposed() -> None:
    study = _study(*PROTEIN)
    pockets = study.steps[2]
    run = Action(does="run", step=3, until=3, quoting="run the pockets again")
    model = ScriptedLanguageModel(
        ReplyTurn(say="Running it.", actions=[run], summary="s"),
        ReplyTurn(say="I would run it again.", actions=[run], summary="s"),
    )
    agent, studies, advance = _agent(study, InMemoryBlobStore(), model)

    await agent(_say(study, "Please run the pockets again."), AUTH)
    assert advance.runs == [(pockets.id, pockets.id)]
    assert advance.started_by == ["agent"]  # its own turn records the run
    ran = studies.rows[study.id].thread.turns[-1]
    assert [(a.does, a.step, a.quoting) for a in ran.actions] == [
        ("run", pockets.id, "run the pockets again")
    ]

    await agent(_say(study, "What would you do next?"), AUTH)
    assert advance.runs == [(pockets.id, pockets.id)]
    proposed = studies.rows[study.id].thread.turns[-1]
    assert proposed.actions == ()
    assert [p.action for p in proposed.proposals] == [
        {
            "does": "run",
            "step_id": str(pockets.id),
            "tool_id": None,
            "before_id": None,
            "purpose": None,
            "settings": {},
            "gated": None,
            "candidate": None,
            "until_id": str(pockets.id),
        }
    ]


async def test_approval_writes_an_asserted_pocket_from_the_rank() -> None:
    store = InMemoryBlobStore()
    study = _study(*PROTEIN)
    candidates = _done(study, store, 3, "pocket_candidates", POCKETS)
    approve = Action(does="approve", step=3, candidate=10, quoting="go with the zinc cone")
    again = Action(does="approve", step=3, candidate=1, quoting="take the first one instead")
    model = ScriptedLanguageModel(
        ReplyTurn(say="Done.", actions=[approve], summary="s"),
        ReplyTurn(say="Changed.", actions=[again], summary="s"),
    )
    agent, studies, _ = _agent(study, store, model)

    await agent(_say(study, "Go with the zinc cone."), AUTH)

    stored = studies.rows[study.id]
    pocket = stored.latest("pocket")
    assert pocket is not None
    assert (pocket.asserted_by, pocket.produced_by) == (AUTH.user_id, study.steps[2].id)
    meta = {"rank": 10, "from": str(candidates.id)}
    assert pocket.summary == meta
    assert json.loads(store.get_bytes(pocket.blob_key)) == {"meta": meta, **POCKETS["pockets"][1]}
    acted = stored.thread.turns[-1].actions
    assert [(a.does, a.quoting) for a in acted] == [("approve", "go with the zinc cone")]

    # Choosing again reads the candidates the step reported, not the last choice, and wins.
    await agent(_say(study, "Take the first one instead."), AUTH)

    stored = studies.rows[study.id]
    newer = stored.latest("pocket")
    assert newer is not None
    assert newer.summary == {"rank": 1, "from": str(candidates.id)}
    assert stored.thread.turns[-1].dropped == 0


async def test_a_message_while_a_run_is_in_flight_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _study(*PROTEIN)
    model = ScriptedLanguageModel(ReplyTurn(say="x", summary="s"))
    agent, studies, _ = _agent(study, InMemoryBlobStore(), model)
    monkeypatch.setitem(advance_study._in_flight, study.id, asyncio.current_task())

    result = await agent(_say(study, "Stop and rerun the pockets."), AUTH)

    assert isinstance(result.failure(), ConflictError)
    assert studies.rows[study.id].thread.turns == []
    assert model.calls == []


async def test_observe_stores_a_silent_turn_and_returns_no_pause() -> None:
    store = InMemoryBlobStore()
    study = _study(*PROTEIN)
    _done(study, store, 3, "pocket_candidates", POCKETS)
    model = ScriptedLanguageModel(ObserveTurn(say="", summary="Two pockets; nothing to add."))
    observe, studies = _observer(study, store, model)

    paused = await observe(study.workspace_id, study.id, study.steps[2].id)

    assert paused is False
    stored = studies.rows[study.id]
    (turn,) = stored.thread.turns
    assert (turn.role, turn.kind, turn.silent, turn.about) == (
        "agent",
        "observe",
        True,
        study.steps[2].id,
    )
    assert stored.thread.summary == "Two pockets; nothing to add."
    assert "Candidate 10" in model.calls[0][1]


async def test_observe_returns_pause_only_with_a_reason() -> None:
    store = InMemoryBlobStore()
    study = _study(*PROTEIN)
    _done(study, store, 3, "pocket_candidates", POCKETS)
    model = ScriptedLanguageModel(
        ObserveTurn(say="", pause=True, summary="s"),
        ObserveTurn(say="Neither histidine lines one pocket.", pause=True, summary="s"),
    )
    observe, _ = _observer(study, store, model)
    step = study.steps[2]

    assert await observe(study.workspace_id, study.id, step.id, driving=True) is False
    assert await observe(study.workspace_id, study.id, step.id, driving=True) is True
    assert "driving this run" in model.calls[0][1]


async def test_observe_survives_a_model_failure() -> None:
    study = _study(*PROTEIN)
    observe, studies = _observer(study, InMemoryBlobStore(), None)

    assert await observe(study.workspace_id, study.id, study.steps[0].id) is False

    (turn,) = studies.rows[study.id].thread.turns
    assert (turn.role, turn.kind, turn.about) == ("event", "observe_failed", study.steps[0].id)
    assert turn.say == "No anthropic model is configured on this server."
