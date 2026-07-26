import uuid

import pytest

from daikonstudio.domain.execution.run import Run, RunKind, RunStatus
from daikonstudio.domain.shared.errors import ConflictError


def _pending() -> Run:
    return Run(
        kind=RunKind.TRAINING,
        workspace_id=uuid.uuid4(),
        requested_by=uuid.uuid4(),
        cache_key="deadbeef",
    )


def test_run_starts_pending():
    run = _pending()
    assert run.status is RunStatus.PENDING
    assert run.progress == 0.0


def test_lifecycle_transitions_to_ready():
    run = _pending()
    run.start()
    assert run.status is RunStatus.RUNNING
    run.report_progress(0.5, phase="training baseline")
    assert run.progress == 0.5 and run.phase == "training baseline"
    run.succeed("s3://results.parquet")
    assert run.status is RunStatus.READY and run.result_uri.endswith(".parquet")


def test_failure_records_the_message():
    run = _pending()
    run.start()
    run.fail("engine raised ValueError")
    assert run.status is RunStatus.FAILED
    assert run.error_message == "engine raised ValueError"


def test_a_terminal_run_cannot_restart():
    run = _pending()
    run.start()
    run.succeed("s3://x")
    with pytest.raises(ConflictError):
        run.start()


def test_cancel_is_allowed_from_pending_and_running_only():
    run = _pending()
    run.cancel()
    assert run.status is RunStatus.CANCELLED
    with pytest.raises(ConflictError):
        run.cancel()
