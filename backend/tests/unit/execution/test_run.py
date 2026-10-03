"""Guard coverage beyond tests/integration/test_run_lifecycle.py's five cases,
plus `compute_cache_key`'s determinism -- the property Task 17 relies on to
recognise a repeated request as a cache hit rather than compute a fresh key
that happens to look different for identical inputs.
"""

import uuid

import pytest

from daikonstudio.domain.execution.run import Run, RunKind, RunStatus, compute_cache_key
from daikonstudio.domain.shared.errors import ConflictError


def _pending(**overrides: object) -> Run:
    defaults: dict[str, object] = dict(
        kind=RunKind.TRAINING,
        workspace_id=uuid.uuid4(),
        requested_by=uuid.uuid4(),
        cache_key="deadbeef",
    )
    defaults.update(overrides)
    return Run(**defaults)  # type: ignore[arg-type]


def test_compute_cache_key_is_deterministic_and_order_independent():
    a = compute_cache_key(dataset_id="d1", engine_id="e1", conditions={"k": 1})
    b = compute_cache_key(engine_id="e1", conditions={"k": 1}, dataset_id="d1")
    assert a == b
    assert len(a) == 64  # sha256 hex digest


def test_compute_cache_key_differs_when_a_value_changes():
    a = compute_cache_key(dataset_id="d1", engine_id="e1")
    b = compute_cache_key(dataset_id="d2", engine_id="e1")
    assert a != b


def test_compute_cache_key_handles_uuid_parts():
    key = compute_cache_key(dataset_id=uuid.uuid4())
    assert isinstance(key, str) and len(key) == 64


def test_report_progress_requires_running():
    run = _pending()
    with pytest.raises(ConflictError):
        run.report_progress(0.5, phase="x")


def test_succeed_requires_running():
    run = _pending()
    with pytest.raises(ConflictError):
        run.succeed("s3://x")


def test_fail_is_allowed_from_pending():
    """An enqueue that never reaches the worker (e.g. Redis unreachable) fails
    the Run while it is still pending -- fail() must not demand `running`."""
    run = _pending()
    run.fail("could not enqueue")
    assert run.status is RunStatus.FAILED
    assert run.error_message == "could not enqueue"


def test_fail_on_a_terminal_run_raises():
    run = _pending()
    run.cancel()
    with pytest.raises(ConflictError):
        run.fail("too late")


def test_cancel_from_running_flips_status_but_cannot_stop_the_worker():
    run = _pending()
    run.start()
    run.cancel()
    assert run.status is RunStatus.CANCELLED


def test_link_protocol_records_the_protocol_a_training_run_produced():
    run = _pending()
    assert run.protocol_id is None

    protocol_id = uuid.uuid4()
    run.link_protocol(protocol_id)
    assert run.protocol_id == protocol_id


def test_relinking_the_same_protocol_is_a_no_op():
    """A worker retry that re-runs the tail of a handler must not be punished
    for arriving at the same answer twice."""
    run = _pending()
    protocol_id = uuid.uuid4()
    run.link_protocol(protocol_id)
    run.link_protocol(protocol_id)
    assert run.protocol_id == protocol_id


def test_relinking_a_different_protocol_raises():
    """Write-once: a Run concerns exactly one Protocol, and re-pointing a
    completed Run would silently rewrite history for anyone already citing it."""
    run = _pending()
    run.link_protocol(uuid.uuid4())
    with pytest.raises(ConflictError):
        run.link_protocol(uuid.uuid4())


def test_start_on_a_running_run_restarts_it():
    """`run_job` (`infrastructure/jobs.py`) calls `start()` unconditionally on
    whatever it loads, with no separate branch for "already running" --
    defense against any redelivery of the same run_id into a fresh execution
    without the row passing back through PENDING first. The runner queue's
    own redelivery path (`RunQueue.sweep` resetting an expired lease) already
    flips the row to 'pending' before anyone can reclaim it, so this is a
    domain-level invariant, not a mechanism this codebase's queue exercises
    today: with no checkpoints, restart-from-zero is the designed recovery
    either way, so a call landing on RUNNING is a legitimate restart, and
    stale progress from whatever produced it is wiped."""
    run = _pending()
    run.start()
    run.report_progress(0.66, phase="training baseline")

    run.start()  # redelivery after a worker crash

    assert run.status is RunStatus.RUNNING
    assert run.progress == 0.0
    assert run.phase is None


def test_start_on_a_terminal_run_still_raises():
    """Cancelled-while-queued (or already finished) runs must not restart --
    ConflictError from start() is how the worker knows to drop a redelivery."""
    run = _pending()
    run.cancel()
    with pytest.raises(ConflictError):
        run.start()


def test_retry_returns_a_failed_run_to_pending_and_clears_the_failure():
    run = _pending()
    run.start()
    run.report_progress(0.4, phase="training")
    run.fail("boom")
    run.record_prediction_counts(uploaded_rows=4, scored_rows=3)
    run.retry()
    assert (run.status, run.progress, run.phase, run.error_message, run.metrics) == (
        RunStatus.PENDING,
        0.0,
        None,
        None,
        None,
    )


def test_retry_is_allowed_from_cancelled_too():
    """A worker crash leaves a run RUNNING; the user cancels it (legal from
    RUNNING) and retries. That is what makes a crash recoverable by hand."""
    run = _pending()
    run.cancel()
    run.retry()
    assert run.status is RunStatus.PENDING


@pytest.mark.parametrize(
    "prepare",
    [
        lambda run: None,
        lambda run: run.start(),
        lambda run: (run.start(), run.succeed("uri")),
    ],
    ids=["pending", "running", "ready"],
)
def test_retry_refuses_pending_running_and_ready(prepare):
    """`running` is the one that matters: a retry from there would start a
    second fit beside one that may still be alive."""
    run = _pending()
    prepare(run)
    with pytest.raises(ConflictError):
        run.retry()
