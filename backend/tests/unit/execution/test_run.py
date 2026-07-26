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
