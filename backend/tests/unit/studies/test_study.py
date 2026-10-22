from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from daikonstudio.domain.shared.errors import ConflictError
from daikonstudio.domain.studies.study import (
    SlotValue,
    Step,
    StepStatus,
    Study,
    StudyStatus,
)

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _study(*tool_ids: str) -> Study:
    return Study(
        workspace_id=uuid.uuid4(),
        created_by=uuid.uuid4(),
        goal="g",
        restated_goal="rg",
        checks=(),
        stop_conditions=(),
        steps=[
            Step(
                position=i,
                tool_id=t,
                tool_version="1",
                name=t,
                purpose="p",
                params={},
                gated=False,
                always=False,
                expanded_from=None,
            )
            for i, t in enumerate(tool_ids)
        ],
    )


def _value(*, kind: str, produced_by: int | None, at: int = 0, **kw: Any) -> SlotValue:
    return SlotValue(
        id=uuid.uuid4(),
        key=kind,
        kind=kind,
        blob_key=f"b/{uuid.uuid4()}",
        summary={},
        produced_by=produced_by,
        created_at=_T0 + timedelta(seconds=at),
        **kw,
    )


def test_a_new_study_is_open_and_its_first_step_is_next() -> None:
    study = _study("fetch_pdb", "find_pockets")
    assert study.status is StudyStatus.OPEN
    assert study.next_step is not None and study.next_step.position == 0


def test_finishing_a_step_records_its_values_and_moves_on() -> None:
    study = _study("fetch_pdb", "find_pockets")
    study.start_step(0)
    study.finish_step(0, [_value(kind="structure", produced_by=0)])
    latest = study.latest("structure")
    assert latest is not None and latest.produced_by == 0
    assert study.next_step is not None and study.next_step.position == 1
    assert len(study.outputs_of(0)) == 1


def test_a_value_must_name_the_step_that_made_it() -> None:
    study = _study("fetch_pdb")
    study.start_step(0)
    with pytest.raises(ConflictError):
        study.finish_step(0, [_value(kind="structure", produced_by=None)])


def test_a_failed_step_fails_the_study_and_can_be_restarted() -> None:
    study = _study("fetch_pdb")
    study.start_step(0)
    study.fail_step(0, "no such entry")
    assert study.status is StudyStatus.FAILED
    assert study.next_step is not None and study.next_step.position == 0
    study.start_step(0)
    assert study.steps[0].error is None and study.steps[0].finished_at is None


def test_an_interrupted_step_is_still_next() -> None:
    study = _study("fetch_pdb", "find_pockets")
    study.start_step(0)
    assert study.next_step is not None and study.next_step.position == 0


def test_skipped_steps_do_not_block_completion() -> None:
    study = _study("prepare_structure")
    study.skip_step(0, "Not available on this server yet.")
    assert study.status is StudyStatus.DONE and study.next_step is None


def test_latest_prefers_the_newest_value_of_a_kind() -> None:
    study = _study("a")
    study.bring(_value(kind="structure", produced_by=None, at=5))
    newer = _value(kind="structure", produced_by=None, at=9)
    study.bring(newer)
    study.bring(_value(kind="structure", produced_by=None, at=1))
    assert study.latest("structure") == newer
    assert study.latest("nothing") is None


def test_a_rerun_step_is_not_handed_its_own_last_output() -> None:
    """Preparing a structure produces a structure. Without a position bound, the
    second run would prepare the already-prepared file instead of the deposited
    one — and quietly, because the result still looks like a structure."""
    study = _study("fetch_pdb", "prepare_structure")
    deposited = _value(kind="structure", produced_by=0, at=1)
    study.start_step(0)
    study.finish_step(0, [deposited])
    study.start_step(1)
    study.finish_step(1, [_value(kind="structure", produced_by=1, at=2)])
    # Run step 1 again: its own output is newer, and must not be what it reads.
    assert study.latest("structure", before=1) == deposited
    assert study.latest("structure") != deposited


def test_the_scientist_can_bring_a_value_but_not_claim_a_step_made_it() -> None:
    study = _study("a")
    study.bring(_value(kind="structure", produced_by=None))
    with pytest.raises(ConflictError):
        study.bring(_value(kind="structure", produced_by=0))


def test_a_done_step_can_be_run_again() -> None:
    study = _study("fetch_pdb")
    study.start_step(0)
    first = _value(kind="structure", produced_by=0)
    study.finish_step(0, [first])
    study.start_step(0)
    assert study.steps[0].status is StepStatus.RUNNING
    assert study.steps[0].finished_at is None
    assert study.outputs_of(0) == [first]


def test_current_outputs_show_only_the_newest_value_per_key() -> None:
    """A re-run step reads like a notebook cell: one output, the latest. The
    earlier value is still in the record."""
    from datetime import UTC, datetime, timedelta

    study = _study("fetch_pdb")
    study.start_step(0)
    first = _value(kind="structure", produced_by=0)
    study.finish_step(0, [first])
    study.start_step(0)
    later = SlotValue(
        id=uuid.uuid4(),
        key=first.key,
        kind="structure",
        blob_key="b2",
        summary={},
        produced_by=0,
        created_at=datetime.now(UTC) + timedelta(seconds=1),
    )
    study.finish_step(0, [later])
    assert [v.id for v in study.current_outputs_of(0)] == [later.id]
    assert len(study.outputs_of(0)) == 2
