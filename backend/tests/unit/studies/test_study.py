from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from daikonstudio.domain.shared.errors import ConflictError
from daikonstudio.domain.studies.study import (
    Proposed,
    SlotValue,
    Step,
    StepStatus,
    Study,
    StudyStatus,
    Thread,
    Turn,
)

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _step(position: int, tool_id: str) -> Step:
    return Step(
        id=uuid.uuid4(),
        position=position,
        tool_id=tool_id,
        tool_version="1",
        name=tool_id,
        purpose="p",
        params={},
        gated=False,
        always=False,
        expanded_from=None,
    )


def _study(*tool_ids: str, **kw: Any) -> Study:
    """A draft whose steps are the tools named, in order."""
    return Study(
        workspace_id=uuid.uuid4(),
        created_by=uuid.uuid4(),
        goal="g",
        restated_goal="rg",
        checks=(),
        stop_conditions=(),
        steps=[_step(i, t) for i, t in enumerate(tool_ids)],
        **kw,
    )


def _started(*tool_ids: str) -> Study:
    study = _study(*tool_ids)
    study.start()
    return study


def _value(*, kind: str, produced_by: uuid.UUID | None, at: int = 0, **kw: Any) -> SlotValue:
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


def _turn(role: str, kind: str = "", **kw: Any) -> Turn:
    return Turn(role=role, kind=kind, say="s", at=_T0, **kw)


def _finish(study: Study, step: Step, *values: SlotValue) -> None:
    """Run a step to done, as the run loop does."""
    study.start_step(step.id)
    study.finish_step(step.id, list(values))


def test_a_study_is_a_draft_until_started() -> None:
    study = _study("standardize")
    assert study.status is StudyStatus.DRAFT
    study.start()
    assert study.status is StudyStatus.OPEN
    assert study.started_at is not None


def test_start_refuses_a_step_with_a_problem() -> None:
    study = _study("standardize")
    step = study.steps[0]
    step.problem = "needs core"
    with pytest.raises(ConflictError):
        study.start()
    assert study.status is StudyStatus.DRAFT  # a refused start leaves the draft as it was
    step.problem = None
    study.start()
    assert study.status is StudyStatus.OPEN


def test_start_refuses_an_empty_plan_and_a_second_start() -> None:
    with pytest.raises(ConflictError):
        _study().start()
    started = _started("standardize")
    with pytest.raises(ConflictError):
        started.start()


def test_phase_follows_the_thread_and_the_start() -> None:
    study = _study("standardize")
    assert study.phase == "asking"
    study.say(_turn("scientist"))
    study.say(_turn("event", "attached"))
    assert study.phase == "asking"  # only the agent's own turn moves a draft on
    study.say(_turn("agent", "asking"))
    assert study.phase == "planning"
    study.start()
    assert study.phase == "running"


def test_a_fresh_study_has_its_own_empty_thread_and_the_run_mode() -> None:
    one, other = _study("a"), _study("a")
    assert one.thread.turns == [] and one.thread.summary == ""
    assert one.thread is not other.thread
    assert (one.mode, one.started_at, one.amended_after_steps) == ("run", None, None)


def test_a_study_rebuilt_from_storage_keeps_its_thread_and_start() -> None:
    saved = Thread(turns=[_turn("agent", "asking")], summary="so far")
    study = _study("a", thread=saved, started_at=_T0, amended_after_steps=2)
    assert study.thread is saved
    assert study.started_at == _T0 and study.amended_after_steps == 2
    assert study.phase == "running"


def test_say_adds_to_the_one_thread_and_finds_its_proposals() -> None:
    study = _study("a")
    proposed = Proposed(id=uuid.uuid4(), action={"does": "swap"}, why="it takes a sequence")
    first = _turn("scientist")
    study.say(first)
    thread = study.say(_turn("agent", "planning", proposals=(proposed,)))
    assert thread is study.thread
    assert len(thread.turns) == 2 and thread.turns[0] is first
    assert study.proposal(proposed.id) is proposed
    assert study.proposal(uuid.uuid4()) is None


def test_a_step_is_found_by_id_or_position() -> None:
    study = _study("a", "b")
    first, second = study.steps
    assert study.step(second.id) is second
    assert study.step_at(0) is first
    assert study.step_at(5) is None
    with pytest.raises(ConflictError):
        study.step(uuid.uuid4())


def test_replace_steps_is_for_a_draft_only() -> None:
    study = _study("a")
    study.replace_steps([_step(7, "b"), _step(7, "c")])
    assert [s.tool_id for s in study.steps] == ["b", "c"]
    assert [s.position for s in study.steps] == [0, 1]
    study.start()
    with pytest.raises(ConflictError):
        study.replace_steps([_step(0, "d")])
    assert [s.tool_id for s in study.steps] == ["b", "c"]


def test_insert_renumbers_and_keeps_ids() -> None:
    study = _study("a", "b")
    _, second = study.steps
    new = _step(99, "c")  # the position it arrives with is not trusted
    assert study.insert_step(new, before=second.id) is new
    assert [s.tool_id for s in study.steps] == ["a", "c", "b"]
    assert [s.position for s in study.steps] == [0, 1, 2]
    assert study.step(second.id) is second and second.position == 2  # moved, same id
    study.insert_step(_step(0, "d"), before=None)
    assert [s.tool_id for s in study.steps] == ["a", "c", "b", "d"]
    assert [s.position for s in study.steps] == [0, 1, 2, 3]
    with pytest.raises(ConflictError):
        study.insert_step(_step(0, "e"), before=uuid.uuid4())
    assert len(study.steps) == 4


def test_remove_refuses_a_done_step() -> None:
    study = _started("a", "b", "c")
    done, pending, last = study.steps
    _finish(study, done)
    with pytest.raises(ConflictError):
        study.remove_step(done.id)
    assert len(study.steps) == 3
    study.remove_step(pending.id)
    assert [s.id for s in study.steps] == [done.id, last.id]
    assert [s.position for s in study.steps] == [0, 1]


def test_a_running_step_cannot_be_removed() -> None:
    study = _started("a")
    study.start_step(study.steps[0].id)
    with pytest.raises(ConflictError):
        study.remove_step(study.steps[0].id)


@pytest.mark.parametrize("status", [StepStatus.PENDING, StepStatus.SKIPPED, StepStatus.FAILED])
def test_a_pending_skipped_or_failed_step_can_be_removed(status: StepStatus) -> None:
    study = _started("a", "b")
    study.steps[0].status = status
    study.remove_step(study.steps[0].id)
    assert [s.tool_id for s in study.steps] == ["b"]
    assert study.steps[0].position == 0


def test_set_tool_puts_the_step_back_to_pending_and_clears_its_problem() -> None:
    study = _started("fetch_pdb")
    step = study.steps[0]
    study.start_step(step.id)
    study.fail_step(step.id, "no such entry")
    step.problem = "needs an accession"
    swapped = study.set_tool(
        step.id,
        tool_id="fetch_alphafold",
        tool_version="2",
        name="Fetch AlphaFold",
        params={"accession": "P12345"},
    )
    assert swapped is step
    assert (step.tool_id, step.tool_version, step.name) == (
        "fetch_alphafold",
        "2",
        "Fetch AlphaFold",
    )
    assert step.params == {"accession": "P12345"}
    assert step.status is StepStatus.PENDING
    assert step.error is None and step.problem is None
    assert study.next_step is step


def test_set_params_and_set_gate_edit_a_step_in_place() -> None:
    study = _started("a")
    step = study.steps[0]
    step.params = {"old": 1}
    study.start_step(step.id)
    study.fail_step(step.id, "no such entry")
    step.problem = "needs an accession"
    assert study.set_gate(step.id, True) is step
    assert step.gated is True and step.status is StepStatus.FAILED  # a gate leaves the outcome
    assert study.set_params(step.id, {"k": 2}) is step
    assert step.params == {"k": 2}  # replaced, not merged: the caller owns the merge
    assert step.status is StepStatus.PENDING  # the old result no longer matches its settings
    assert step.error is None  # a pending step must not wear the last run's failure
    assert step.problem == "needs an accession"  # only the plan check knows if this fixes it


def test_amend_terms_records_how_many_steps_had_run() -> None:
    study = _started("a", "b")
    study.checks, study.stop_conditions = ("a check",), ("a stop",)
    _finish(study, study.steps[0])
    study.amend_terms(("a new check",), None)
    assert study.checks == ("a new check",)
    assert study.stop_conditions == ("a stop",)  # not given, so not touched
    assert study.amended_after_steps == 1

    draft = _study("a")
    draft.checks, draft.stop_conditions = ("a check",), ("a stop",)
    draft.amend_terms(None, ("a new stop",))
    assert draft.stop_conditions == ("a new stop",)
    assert draft.checks == ("a check",)
    assert draft.amended_after_steps is None  # nothing had run, so there is nothing to mark


def test_amend_terms_given_nothing_marks_nothing() -> None:
    study = _started("a", "b")
    _finish(study, study.steps[0])
    study.amend_terms(None, None)
    assert study.amended_after_steps is None  # nothing changed, so there is nothing to mark


def test_done_count_counts_finished_steps_not_skipped_ones() -> None:
    study = _started("a", "b", "c")
    first, second, _ = study.steps
    _finish(study, first)
    study.skip_step(second.id, "Not available on this server yet.")
    assert study.done_count == 1


def test_approve_needs_an_asserted_value_from_that_step() -> None:
    study = _started("find_pockets", "dock")
    pockets, dock = study.steps
    _finish(study, pockets, _value(kind="pocket_candidates", produced_by=pockets.id))
    chooser = uuid.uuid4()
    with pytest.raises(ConflictError):  # nobody asserted it: a tool computed it
        study.approve(pockets.id, _value(kind="pocket", produced_by=pockets.id))
    with pytest.raises(ConflictError):  # asserted, but it settles another step
        study.approve(pockets.id, _value(kind="pocket", produced_by=dock.id, asserted_by=chooser))
    missing = uuid.uuid4()
    with pytest.raises(ConflictError):  # asserted and consistent, but there is no such step
        study.approve(missing, _value(kind="pocket", produced_by=missing, asserted_by=chooser))
    assert study.latest("pocket") is None
    approved = _value(kind="pocket", produced_by=pockets.id, asserted_by=chooser)
    study.approve(pockets.id, approved)
    assert study.latest("pocket") == approved


def test_latest_before_a_step_compares_positions_not_ids() -> None:
    study = _started("a", "b", "c")
    # Ids carry no order. Make them run backwards, so comparing ids would call the last
    # step the first one and the answer below would be wrong, not just slow.
    for step, n in zip(study.steps, (30, 20, 10), strict=True):
        step.id = uuid.UUID(int=n)
    a, b, c = study.steps
    brought = _value(kind="structure", produced_by=None, at=0)
    older = _value(kind="structure", produced_by=a.id, at=1)
    newer = _value(kind="structure", produced_by=b.id, at=2)
    last = _value(kind="structure", produced_by=c.id, at=3)
    study.bring(brought)
    _finish(study, a, older)
    _finish(study, b, newer)
    _finish(study, c, last)
    assert study.latest("structure") == last
    assert study.latest("structure", before=c.id) == newer
    assert study.latest("structure", before=b.id) == older
    assert study.latest("structure", before=a.id) == brought  # only what was brought precedes it
    with pytest.raises(ConflictError):
        study.latest("structure", before=uuid.uuid4())


def test_a_removed_steps_values_stay_in_the_record_but_not_in_a_later_steps_scope() -> None:
    """Edit a done step and it is pending again with its old result still held. Remove it
    and that result belongs to no step: it is record, and it must not feed the next one."""
    study = _started("fetch_pdb", "prepare_structure", "dock")
    fetch, prepare, dock = study.steps
    deposited = _value(kind="structure", produced_by=fetch.id, at=1)
    prepared = _value(kind="structure", produced_by=prepare.id, at=2)
    _finish(study, fetch, deposited)
    _finish(study, prepare, prepared)
    study.set_params(prepare.id, {"ph": 7.0})
    study.remove_step(prepare.id)
    assert study.latest("structure", before=dock.id) == deposited
    assert study.latest("structure") == prepared
    assert study.outputs_of(prepare.id) == [prepared]


def test_finishing_a_step_records_its_values_and_moves_on() -> None:
    study = _started("fetch_pdb", "find_pockets")
    first, second = study.steps
    study.start_step(first.id)
    study.finish_step(first.id, [_value(kind="structure", produced_by=first.id)])
    latest = study.latest("structure")
    assert latest is not None and latest.produced_by == first.id
    assert study.next_step is second
    assert len(study.outputs_of(first.id)) == 1


def test_a_started_study_is_open_and_its_first_step_is_next() -> None:
    study = _started("fetch_pdb", "find_pockets")
    assert study.status is StudyStatus.OPEN
    assert study.next_step is not None and study.next_step.position == 0


def test_a_value_must_name_the_step_that_made_it() -> None:
    study = _started("fetch_pdb")
    step = study.steps[0]
    study.start_step(step.id)
    with pytest.raises(ConflictError):
        study.finish_step(step.id, [_value(kind="structure", produced_by=None)])


def test_a_failed_step_fails_the_study_and_can_be_restarted() -> None:
    study = _started("fetch_pdb")
    step = study.steps[0]
    study.start_step(step.id)
    study.fail_step(step.id, "no such entry")
    assert study.status is StudyStatus.FAILED
    assert study.next_step is step
    study.start_step(step.id)
    assert step.error is None and step.finished_at is None


def test_an_interrupted_step_is_still_next() -> None:
    study = _started("fetch_pdb", "find_pockets")
    study.start_step(study.steps[0].id)
    assert study.next_step is study.steps[0]


def test_skipped_steps_do_not_block_completion() -> None:
    study = _started("prepare_structure")
    study.skip_step(study.steps[0].id, "Not available on this server yet.")
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
    study = _started("fetch_pdb", "prepare_structure")
    fetch, prepare = study.steps
    deposited = _value(kind="structure", produced_by=fetch.id, at=1)
    study.start_step(fetch.id)
    study.finish_step(fetch.id, [deposited])
    study.start_step(prepare.id)
    study.finish_step(prepare.id, [_value(kind="structure", produced_by=prepare.id, at=2)])
    # Run the second step again: its own output is newer, and must not be what it reads.
    assert study.latest("structure", before=prepare.id) == deposited
    assert study.latest("structure") != deposited


def test_the_scientist_can_bring_a_value_but_not_claim_a_step_made_it() -> None:
    study = _study("a")
    study.bring(_value(kind="structure", produced_by=None))
    with pytest.raises(ConflictError):
        study.bring(_value(kind="structure", produced_by=study.steps[0].id))


def test_a_done_step_can_be_run_again() -> None:
    study = _started("fetch_pdb")
    step = study.steps[0]
    study.start_step(step.id)
    first = _value(kind="structure", produced_by=step.id)
    study.finish_step(step.id, [first])
    study.start_step(step.id)
    assert step.status is StepStatus.RUNNING
    assert step.finished_at is None
    assert study.outputs_of(step.id) == [first]


def test_current_outputs_show_only_the_newest_value_per_key() -> None:
    """A re-run step reads like a notebook cell: one output, the latest. The
    earlier value is still in the record."""
    from datetime import UTC, datetime, timedelta

    study = _started("fetch_pdb")
    step = study.steps[0]
    study.start_step(step.id)
    first = _value(kind="structure", produced_by=step.id)
    study.finish_step(step.id, [first])
    study.start_step(step.id)
    later = SlotValue(
        id=uuid.uuid4(),
        key=first.key,
        kind="structure",
        blob_key="b2",
        summary={},
        produced_by=step.id,
        created_at=datetime.now(UTC) + timedelta(seconds=1),
    )
    study.finish_step(step.id, [later])
    assert [v.id for v in study.current_outputs_of(step.id)] == [later.id]
    assert len(study.outputs_of(step.id)) == 2
