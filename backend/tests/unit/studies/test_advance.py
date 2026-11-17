"""The run loop: run steps in order, let the agent look at each one that ends, and stop at a
gate, at a failure, where it was told to, or where the agent says so."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import Mapping
from typing import Any

import pytest

from daikonstudio.application.studies import advance_study
from daikonstudio.application.studies.advance_study import AdvanceStudy
from daikonstudio.application.studies.agent import ObserveStep
from daikonstudio.application.studies.schemas import ObserveTurn
from daikonstudio.domain.shared.errors import ConflictError, NotFoundError, ValidationError
from daikonstudio.domain.studies.study import Step, StepStatus, Study, StudyStatus
from daikonstudio.infrastructure.tools.catalog import default_registry
from tests.fakes.auth import FakeAuth
from tests.fakes.blob_store import InMemoryBlobStore
from tests.fakes.llm import ScriptedLanguageModel, ScriptedLanguageModels
from tests.fakes.studies import InMemoryStudies
from tests.fakes.tool_executor import InProcessToolExecutor

STRUCTURE = {"format": "pdb", "text": "ATOM", "meta": {"uniprot": ["P0AE18"]}}
QUIET = ObserveTurn(say="", summary="Nothing to add.")
PDB = "Fetch a deposited structure"


def _setup(
    impls: Mapping[str, Any],
    *steps: tuple[str, dict[str, Any], bool],
    observing: ScriptedLanguageModel | None = None,
):
    """A started study with these steps, and a loop whose observations come from `observing`
    (a model with nothing to say, unless a test scripts one)."""
    auth = FakeAuth()
    registry = default_registry()
    studies = InMemoryStudies()
    store = InMemoryBlobStore()
    executor = InProcessToolExecutor(store, impls)
    study = Study(
        workspace_id=auth.workspace_id,
        created_by=auth.user_id,
        goal="g",
        restated_goal="rg",
        checks=(),
        stop_conditions=(),
        steps=[
            Step(
                id=uuid.uuid4(),
                position=i,
                tool_id=tool_id,
                tool_version="1",
                name=registry.get(tool_id).name,
                purpose="p",
                params=params,
                gated=gated,
                always=registry.get(tool_id).automatic,
                expanded_from=None,
            )
            for i, (tool_id, params, gated) in enumerate(steps)
        ],
    )
    study.start()
    studies.rows[study.id] = study
    models = ScriptedLanguageModels(observing or ScriptedLanguageModel(QUIET))
    observe = ObserveStep(studies, registry, models, store)
    advance = AdvanceStudy(studies, registry, executor, store, observe)
    return auth, studies, executor, advance, study.id


def _ids(studies: InMemoryStudies, study_id: uuid.UUID) -> list[uuid.UUID]:
    return [s.id for s in studies.rows[study_id].steps]


def _observed(studies: InMemoryStudies, study_id: uuid.UUID) -> list[uuid.UUID | None]:
    """The steps the agent looked at, in the order it looked."""
    return [t.about for t in studies.rows[study_id].thread.turns if t.kind == "observe"]


async def _advance(advance: AdvanceStudy, study_id: uuid.UUID, auth: FakeAuth, **run: Any) -> None:
    assert (await advance(study_id, auth=auth, **run)).unwrap()
    await asyncio.gather(*advance_study._running)


async def _run_one(
    advance: AdvanceStudy, study_id: uuid.UUID, auth: FakeAuth, step_id: uuid.UUID
) -> None:
    await _advance(advance, study_id, auth, start=step_id, until=step_id)


async def test_runs_to_the_gate_then_on_to_the_end() -> None:
    seen: list[dict[str, object]] = []

    def uniprot(_p: object, inputs: Mapping[str, object]) -> dict[str, object]:
        seen.append(dict(inputs))
        return {"annotations": {"meta": {"accession": "P0AE18"}}}

    auth, studies, _, advance, sid = _setup(
        {"fetch_pdb": lambda _p, _i: {"structure": STRUCTURE}, "fetch_uniprot": uniprot},
        ("fetch_pdb", {"pdb_id": "1abc"}, True),
        ("prepare_structure", {}, False),
        ("fetch_uniprot", {}, False),
    )
    fetch, prepare, annotate = _ids(studies, sid)

    await _advance(advance, sid, auth)
    study = studies.rows[sid]
    assert [s.status for s in study.steps] == [
        StepStatus.DONE,
        StepStatus.PENDING,
        StepStatus.PENDING,
    ]
    assert [v.kind for v in study.outputs_of(fetch)] == ["structure"]
    assert study.outputs_of(fetch)[0].blob_key.startswith(
        f"{auth.workspace_id}/studies/{sid}/values/"
    )

    await _advance(advance, sid, auth)
    study = studies.rows[sid]
    assert study.step(prepare).status is StepStatus.SKIPPED
    assert study.step(prepare).error == "No runner is serving the chem-tools lane."
    assert study.step(annotate).status is StepStatus.DONE
    assert seen == [{"structure": STRUCTURE}]
    assert study.status is StudyStatus.DONE
    assert _observed(studies, sid) == [fetch, annotate]  # a skipped step is not looked at


async def test_a_failing_tool_fails_its_step_and_stops() -> None:
    def missing(_p: object, _i: object) -> dict[str, object]:
        raise ValueError("PDB entry 4XXX was not found.")

    auth, studies, executor, advance, sid = _setup(
        {"fetch_pdb": missing, "fetch_uniprot": lambda _p, _i: {"annotations": {}}},
        ("fetch_pdb", {"pdb_id": "4xxx"}, False),
        ("fetch_uniprot", {}, False),
    )
    fetch, _ = _ids(studies, sid)

    await _advance(advance, sid, auth)
    study = studies.rows[sid]
    assert study.steps[0].status is StepStatus.FAILED
    assert "PDB entry 4XXX was not found." in (study.steps[0].error or "")
    assert study.steps[1].status is StepStatus.PENDING
    assert study.status is StudyStatus.FAILED
    assert [c[0] for c in executor.calls] == ["fetch_pdb"]
    assert _observed(studies, sid) == [fetch]  # a failure is looked at too


async def test_a_step_left_running_is_run_again() -> None:
    auth, studies, _, advance, sid = _setup(
        {"fetch_pdb": lambda _p, _i: {"structure": STRUCTURE}},
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
    )
    (fetch,) = _ids(studies, sid)
    studies.rows[sid].start_step(fetch)

    await _advance(advance, sid, auth)
    assert studies.rows[sid].step(fetch).status is StepStatus.DONE


async def test_two_advances_run_a_step_once() -> None:
    auth, _, executor, advance, sid = _setup(
        {"fetch_pdb": lambda _p, _i: {"structure": STRUCTURE}},
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
    )
    await advance(sid, auth=auth)
    await advance(sid, auth=auth)
    await asyncio.gather(*advance_study._running)
    assert len(executor.calls) == 1


async def test_a_required_input_nobody_made_fails_the_step() -> None:
    auth, studies, executor, advance, sid = _setup(
        {"find_pockets": lambda _p, _i: {"pockets": {}}},
        ("find_pockets", {}, False),
    )
    await _advance(advance, sid, auth)
    step = studies.rows[sid].steps[0]
    assert step.status is StepStatus.FAILED
    assert "Required input unavailable: structure" in (step.error or "")
    assert executor.calls == []


async def test_a_tool_that_overruns_is_failed_with_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(advance_study, "TOOL_TIMEOUT_SECONDS", 0.05)

    def slow(_p: object, _i: object) -> dict[str, object]:
        time.sleep(1)
        return {"structure": STRUCTURE}

    auth, studies, _, advance, sid = _setup(
        {"fetch_pdb": slow}, ("fetch_pdb", {"pdb_id": "1abc"}, False)
    )
    await _advance(advance, sid, auth)
    step = studies.rows[sid].steps[0]
    assert step.status is StepStatus.FAILED
    assert (step.error or "").startswith("Timed out")


async def test_a_step_cancelled_by_shutdown_is_recorded_as_interrupted() -> None:
    started = threading.Event()

    def blocking(_p: object, _i: object) -> dict[str, object]:
        started.set()
        time.sleep(0.5)
        return {"structure": STRUCTURE}

    auth, studies, _, advance, sid = _setup(
        {"fetch_pdb": blocking}, ("fetch_pdb", {"pdb_id": "1abc"}, False)
    )
    assert (await advance(sid, auth=auth)).unwrap()
    task = advance_study._in_flight[sid]
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    step = studies.rows[sid].steps[0]
    assert step.status is StepStatus.FAILED
    assert step.error == "Interrupted by a restart."


async def test_run_one_step_observes_it_and_stops() -> None:
    model = ScriptedLanguageModel(QUIET)
    auth, studies, executor, advance, sid = _setup(
        {
            "fetch_pdb": lambda _p, _i: {"structure": STRUCTURE},
            "fetch_uniprot": lambda _p, _i: {"annotations": {}},
        },
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
        ("fetch_uniprot", {}, False),
        observing=model,
    )
    fetch, _ = _ids(studies, sid)

    await _run_one(advance, sid, auth, fetch)

    study = studies.rows[sid]
    assert [s.status for s in study.steps] == [StepStatus.DONE, StepStatus.PENDING]
    assert [c[0] for c in executor.calls] == ["fetch_pdb"]
    assert _observed(studies, sid) == [fetch]
    assert len(model.calls) == 1
    (ran,) = (t for t in study.thread.turns if t.role == "event")
    assert (ran.kind, ran.say, ran.about) == ("ran", f"Started a run of step 1 ({PDB}).", fetch)
    assert not advance_study.in_flight(sid)


async def test_the_loop_stops_where_the_agent_pauses() -> None:
    model = ScriptedLanguageModel(
        ObserveTurn(say="Neither pocket is lined by a histidine.", pause=True, summary="s")
    )
    auth, studies, executor, advance, sid = _setup(
        {
            "fetch_pdb": lambda _p, _i: {"structure": STRUCTURE},
            "fetch_uniprot": lambda _p, _i: {"annotations": {}},
        },
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
        ("fetch_uniprot", {}, False),
        observing=model,
    )
    fetch, _ = _ids(studies, sid)

    await _advance(advance, sid, auth)

    study = studies.rows[sid]
    assert [s.status for s in study.steps] == [StepStatus.DONE, StepStatus.PENDING]
    assert [c[0] for c in executor.calls] == ["fetch_pdb"]
    assert [t.kind for t in study.thread.turns] == ["ran", "observe", "paused"]
    paused = study.thread.turns[-1]
    assert (paused.role, paused.about) == ("event", fetch)
    assert paused.say == f"Execution paused automatically after step 1 ({PDB})."
    assert "driving this run" in model.calls[0][1]


async def test_a_pause_without_a_reason_is_ignored() -> None:
    model = ScriptedLanguageModel(ObserveTurn(say="", pause=True, summary="s"))
    auth, studies, _, advance, sid = _setup(
        {
            "fetch_pdb": lambda _p, _i: {"structure": STRUCTURE},
            "fetch_uniprot": lambda _p, _i: {"annotations": {}},
        },
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
        ("fetch_uniprot", {}, False),
        observing=model,
    )

    await _advance(advance, sid, auth)

    study = studies.rows[sid]
    assert [s.status for s in study.steps] == [StepStatus.DONE, StepStatus.DONE]
    assert [t.kind for t in study.thread.turns] == ["ran", "observe", "observe"]


async def test_the_loop_stops_at_until_and_at_a_gate() -> None:
    model = ScriptedLanguageModel(QUIET)
    auth, studies, executor, advance, sid = _setup(
        {
            "fetch_pdb": lambda _p, _i: {"structure": STRUCTURE},
            "fetch_uniprot": lambda _p, _i: {"annotations": {}},
        },
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
        ("fetch_uniprot", {}, False),
        ("fetch_pdb", {"pdb_id": "1abc"}, True),
        ("fetch_uniprot", {}, False),
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
        observing=model,
    )
    ids = _ids(studies, sid)

    def statuses() -> list[str]:
        return [s.status.value for s in studies.rows[sid].steps]

    await _advance(advance, sid, auth, until=ids[1])
    assert statuses() == ["done", "done", "pending", "pending", "pending"]

    await _advance(advance, sid, auth, start=ids[2])  # the gated step runs, and the loop stops
    assert statuses() == ["done", "done", "done", "pending", "pending"]

    await _advance(advance, sid, auth)
    assert statuses() == ["done"] * 5

    # The agent hears that it is driving only where the loop would go on to another step.
    assert ["driving this run" in prompt for _, prompt in model.calls] == [
        True,
        False,  # until
        False,  # the gate
        True,
        False,  # the last
    ]
    assert [t.say for t in studies.rows[sid].thread.turns if t.kind == "ran"] == [
        "Started a run from step 1 toward step 2.",
        "Started a run from step 3 to the next stop.",  # the gate on step 3 ended it there
        "Started a run from step 4 to the next stop.",
    ]

    turns = len(studies.rows[sid].thread.turns)
    await _advance(advance, sid, auth)  # nothing is left to run
    assert len(studies.rows[sid].thread.turns) == turns

    # `until` bounds the run by its place in the plan, so a step already done bounds it too:
    # the pending step before it runs, then the done step runs again, as everything after a
    # step that ran again does, and the pending step beyond it does not run.
    for waiting in (ids[0], ids[2]):
        studies.rows[sid].set_params(waiting, {"pdb_id": "1abc"})
    assert statuses() == ["pending", "done", "pending", "done", "done"]
    await _advance(advance, sid, auth, until=ids[1])
    assert statuses() == ["done", "done", "pending", "done", "done"]
    assert len(executor.calls) == 7
    assert "driving this run" not in model.calls[-1][1]  # it was told nothing more would run

    # With the next step waiting already beyond `until`, there is nothing to run.
    await _advance(advance, sid, auth, until=ids[1])
    assert statuses() == ["done", "done", "pending", "done", "done"]
    assert len(executor.calls) == 7

    # Playing one step stops after it, though an earlier step is still waiting.
    await _advance(advance, sid, auth, start=ids[3], until=ids[3])
    assert statuses() == ["done", "done", "pending", "done", "done"]
    assert len(executor.calls) == 8

    # A step whose settings changed is run again with what follows it, done or not: from step
    # 1 through the done step 2 to the gate on step 3, with no `until`.
    studies.rows[sid].set_params(ids[0], {"pdb_id": "1abc"})
    assert statuses() == ["pending", "done", "pending", "done", "done"]
    await _advance(advance, sid, auth, start=ids[0])
    assert statuses() == ["done"] * 5
    assert len(executor.calls) == 11
    assert _observed(studies, sid)[-3:] == ids[:3]


async def test_a_run_the_agent_starts_writes_no_ran_event() -> None:
    auth, studies, _, advance, sid = _setup(
        {"fetch_pdb": lambda _p, _i: {"structure": STRUCTURE}},
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
    )
    await _advance(advance, sid, auth, started_by="agent")  # its own turn says what it ran
    assert [t.kind for t in studies.rows[sid].thread.turns] == ["observe"]


async def test_a_draft_cannot_run() -> None:
    auth, studies, executor, advance, sid = _setup(
        {"fetch_pdb": lambda _p, _i: {"structure": STRUCTURE}},
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
    )
    studies.rows[sid].started_at = None

    result = await advance(sid, auth=auth)

    assert isinstance(result.failure(), ConflictError)
    assert str(result.failure()) == "This study has not started."
    assert executor.calls == []
    assert studies.rows[sid].thread.turns == []
    assert not advance_study.in_flight(sid)


async def test_running_a_step_whose_input_is_missing_fails_it() -> None:
    auth, studies, executor, advance, sid = _setup(
        {"find_pockets": lambda _p, _i: {"pockets": {}}},
        ("find_pockets", {}, False),
    )
    (pockets,) = _ids(studies, sid)
    await _run_one(advance, sid, auth, pockets)
    step = studies.rows[sid].steps[0]
    assert step.status is StepStatus.FAILED
    assert "Required input unavailable: structure" in (step.error or "")
    assert executor.calls == []


async def test_rerunning_a_done_step_adds_a_newer_value() -> None:
    auth, studies, _, advance, sid = _setup(
        {"fetch_pdb": lambda _p, _i: {"structure": STRUCTURE}},
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
    )
    (fetch,) = _ids(studies, sid)
    await _run_one(advance, sid, auth, fetch)
    await _run_one(advance, sid, auth, fetch)
    study = studies.rows[sid]
    made = study.outputs_of(fetch)
    assert len(made) == 2
    assert study.step(fetch).status is StepStatus.DONE
    assert study.latest("structure") == max(made, key=lambda v: v.created_at)
    assert study.latest("structure") == made[1]
    assert _observed(studies, sid) == [fetch, fetch]


async def test_a_second_click_while_running_is_a_no_op() -> None:
    auth, studies, executor, advance, sid = _setup(
        {"fetch_pdb": lambda _p, _i: {"structure": STRUCTURE}},
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
    )
    (fetch,) = _ids(studies, sid)
    await advance(sid, auth=auth, start=fetch, until=fetch)
    await advance(sid, auth=auth, start=fetch, until=fetch)
    await asyncio.gather(*advance_study._running)
    assert len(executor.calls) == 1
    assert len(studies.rows[sid].outputs_of(fetch)) == 1


async def test_running_a_step_that_does_not_exist_is_refused() -> None:
    auth, studies, executor, advance, sid = _setup(
        {"fetch_pdb": lambda _p, _i: {"structure": STRUCTURE}},
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
    )
    for run in ({"start": uuid.uuid4()}, {"until": uuid.uuid4()}):
        result = await advance(sid, auth=auth, **run)
        assert isinstance(result.failure(), NotFoundError)
        assert str(result.failure()) == "Step not found."
    assert executor.calls == []

    # Nor is a run that would begin at a step with a problem: running it could only fail.
    (fetch,) = studies.rows[sid].steps
    fetch.problem = "Step 1 (Fetch a deposited structure) settings: pdb_id: Field required"
    for run in ({"start": fetch.id}, {}):
        refused = (await advance(sid, auth=auth, **run)).failure()
        assert isinstance(refused, ValidationError)
        assert str(refused) == f"The step cannot run yet: {fetch.problem}"
    assert executor.calls == [] and studies.rows[sid].thread.turns == []


async def test_a_default_lane_tool_with_no_implementation_says_so() -> None:
    auth, studies, _, advance, sid = _setup({}, ("fetch_pdb", {"pdb_id": "1abc"}, False))
    await _advance(advance, sid, auth)
    assert (
        studies.rows[sid].steps[0].error
        == "This analysis tool is unavailable on the current server."
    )


async def test_an_executor_failure_is_recorded_on_the_step() -> None:
    auth, studies, executor, advance, sid = _setup(
        {"fetch_pdb": lambda _p, _i: {"structure": STRUCTURE}},
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
    )

    async def runner_failed(**_kw: object) -> list[object]:
        raise RuntimeError("The runner ran out of memory.")

    executor.run = runner_failed  # type: ignore[method-assign,assignment]
    await _advance(advance, sid, auth)
    step = studies.rows[sid].steps[0]
    assert step.status is StepStatus.FAILED
    assert step.error == "RuntimeError: The runner ran out of memory."
