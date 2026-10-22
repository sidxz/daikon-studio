"""The advance loop: run steps in order, stop at a gate, fail where it broke."""

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
from daikonstudio.domain.studies.study import Step, StepStatus, Study, StudyStatus
from daikonstudio.infrastructure.tools.catalog import default_registry
from tests.fakes.auth import FakeAuth
from tests.fakes.blob_store import InMemoryBlobStore
from tests.fakes.studies import InMemoryStudies
from tests.fakes.tool_executor import InProcessToolExecutor

STRUCTURE = {"format": "pdb", "text": "ATOM", "meta": {"uniprot": ["P0AE18"]}}


def _setup(impls: Mapping[str, Any], *steps: tuple[str, dict[str, Any], bool]):
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
    studies.rows[study.id] = study
    advance = AdvanceStudy(studies, registry, executor, store)
    return auth, studies, executor, advance, study.id


async def _advance(advance: AdvanceStudy, study_id: uuid.UUID, auth: FakeAuth) -> None:
    assert (await advance(study_id, auth=auth)).unwrap()
    await asyncio.gather(*advance_study._running)


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

    await _advance(advance, sid, auth)
    study = studies.rows[sid]
    assert [s.status for s in study.steps] == [
        StepStatus.DONE,
        StepStatus.PENDING,
        StepStatus.PENDING,
    ]
    assert [v.kind for v in study.outputs_of(0)] == ["structure"]
    assert study.outputs_of(0)[0].blob_key.startswith(f"{auth.workspace_id}/studies/{sid}/values/")

    await _advance(advance, sid, auth)
    study = studies.rows[sid]
    assert study.steps[1].status is StepStatus.SKIPPED
    assert study.steps[1].error == "No runner is serving the chem-tools lane."
    assert study.steps[2].status is StepStatus.DONE
    assert seen == [{"structure": STRUCTURE}]
    assert study.status is StudyStatus.DONE


async def test_a_failing_tool_fails_its_step_and_stops() -> None:
    def missing(_p: object, _i: object) -> dict[str, object]:
        raise ValueError("PDB entry 4XXX was not found.")

    auth, studies, executor, advance, sid = _setup(
        {"fetch_pdb": missing, "fetch_uniprot": lambda _p, _i: {"annotations": {}}},
        ("fetch_pdb", {"pdb_id": "4xxx"}, False),
        ("fetch_uniprot", {}, False),
    )

    await _advance(advance, sid, auth)
    study = studies.rows[sid]
    assert study.steps[0].status is StepStatus.FAILED
    assert "PDB entry 4XXX was not found." in (study.steps[0].error or "")
    assert study.steps[1].status is StepStatus.PENDING
    assert study.status is StudyStatus.FAILED
    assert [c[0] for c in executor.calls] == ["fetch_pdb"]


async def test_a_step_left_running_is_run_again() -> None:
    auth, studies, _, advance, sid = _setup(
        {"fetch_pdb": lambda _p, _i: {"structure": STRUCTURE}},
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
    )
    studies.rows[sid].start_step(0)

    await _advance(advance, sid, auth)
    assert studies.rows[sid].steps[0].status is StepStatus.DONE


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
    assert "Needs structure" in (step.error or "")
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


async def _run_one(advance: AdvanceStudy, study_id: uuid.UUID, auth: FakeAuth, n: int) -> None:
    assert (await advance(study_id, auth=auth, position=n)).unwrap()
    await asyncio.gather(*advance_study._running)


async def test_running_one_step_runs_only_that_step() -> None:
    auth, studies, executor, advance, sid = _setup(
        {
            "fetch_pdb": lambda _p, _i: {"structure": STRUCTURE},
            "fetch_uniprot": lambda _p, _i: {"annotations": {}},
        },
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
        ("fetch_uniprot", {}, False),
    )
    await _run_one(advance, sid, auth, 0)
    study = studies.rows[sid]
    assert [s.status for s in study.steps] == [StepStatus.DONE, StepStatus.PENDING]
    assert [c[0] for c in executor.calls] == ["fetch_pdb"]
    assert not advance_study.in_flight(sid)


async def test_running_a_step_whose_input_is_missing_fails_it() -> None:
    auth, studies, executor, advance, sid = _setup(
        {"find_pockets": lambda _p, _i: {"pockets": {}}},
        ("find_pockets", {}, False),
    )
    await _run_one(advance, sid, auth, 0)
    step = studies.rows[sid].steps[0]
    assert step.status is StepStatus.FAILED
    assert "Needs structure" in (step.error or "")
    assert executor.calls == []


async def test_running_a_done_step_again_makes_a_newer_value() -> None:
    auth, studies, _, advance, sid = _setup(
        {"fetch_pdb": lambda _p, _i: {"structure": STRUCTURE}},
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
    )
    await _run_one(advance, sid, auth, 0)
    await _run_one(advance, sid, auth, 0)
    study = studies.rows[sid]
    made = study.outputs_of(0)
    assert len(made) == 2
    assert study.steps[0].status is StepStatus.DONE
    assert study.latest("structure") == max(made, key=lambda v: v.created_at)
    assert study.latest("structure") == made[1]


async def test_a_second_click_while_running_is_a_no_op() -> None:
    auth, studies, executor, advance, sid = _setup(
        {"fetch_pdb": lambda _p, _i: {"structure": STRUCTURE}},
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
    )
    await advance(sid, auth=auth, position=0)
    await advance(sid, auth=auth, position=0)
    await asyncio.gather(*advance_study._running)
    assert len(executor.calls) == 1
    assert len(studies.rows[sid].outputs_of(0)) == 1


async def test_running_a_step_that_does_not_exist_is_refused() -> None:
    auth, _, executor, advance, sid = _setup(
        {"fetch_pdb": lambda _p, _i: {"structure": STRUCTURE}},
        ("fetch_pdb", {"pdb_id": "1abc"}, False),
    )
    result = await advance(sid, auth=auth, position=9)
    assert "No step 9" in str(result.failure())
    assert executor.calls == []


async def test_a_default_lane_tool_with_no_implementation_says_so() -> None:
    auth, studies, _, advance, sid = _setup({}, ("fetch_pdb", {"pdb_id": "1abc"}, False))
    await _advance(advance, sid, auth)
    assert studies.rows[sid].steps[0].error == "Not available on this server yet."


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
