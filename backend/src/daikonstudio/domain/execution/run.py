"""The Run aggregate -- one execution: training a model or making predictions.

Status is a one-way lattice: `pending -> running -> {ready, failed,
cancelled}`, with `pending -> cancelled` as the only shortcut and
`running -> running` allowed as a restart (at-least-once redelivery after a
worker crash -- see `start()`). `_TERMINAL`
gates every mutating method uniformly, so "a terminal Run cannot change again"
is one check reused everywhere rather than a rule re-derived per method.

`cancel()` on a `running` Run flips the row and nothing else; the worker is a
separate process and the row is the only channel to it. What makes that
actually stop work is cooperative and lives outside this aggregate -- see the
method's docstring.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from daikonstudio.domain.shared.entity import AggregateRoot
from daikonstudio.domain.shared.errors import ConflictError


class RunKind(StrEnum):
    TRAINING = "training"
    PREDICTION = "prediction"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL = {RunStatus.READY, RunStatus.FAILED, RunStatus.CANCELLED}


def compute_cache_key(**parts: object) -> str:
    """A stable fingerprint of whatever inputs determine identical work --
    Task 17 uses this so a repeated prediction request against the same
    protocol and inputs returns the cached Run's result instead of
    re-running the engine.

    Deliberately not `hash()`: Python randomises string hashing per process
    (`PYTHONHASHSEED`), so the same call in the web process and in a
    runner-agent process would produce two different keys for identical inputs.
    `hashlib.sha256` over a JSON encoding with sorted keys is the same value
    everywhere, forever, given the same `parts` -- which is the entire point
    of a cache key that one process writes and another must look up.
    """
    canonical = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class Run(AggregateRoot):
    def __init__(
        self,
        *,
        kind: RunKind,
        workspace_id: uuid.UUID,
        requested_by: uuid.UUID,
        cache_key: str,
        params: Mapping[str, Any] | None = None,
        protocol_id: uuid.UUID | None = None,
        status: RunStatus = RunStatus.PENDING,
        progress: float = 0.0,
        phase: str | None = None,
        result_uri: str | None = None,
        error_message: str | None = None,
        id: uuid.UUID | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        version: int = 1,
    ) -> None:
        super().__init__(id=id, created_at=created_at, updated_at=updated_at, version=version)
        self.kind = kind
        self.workspace_id = workspace_id
        self.requested_by = requested_by
        self.cache_key = cache_key
        # What this Run was asked to do, in whatever shape its kind needs (a
        # training run: dataset, engine, conditions). A runner's claim response
        # hands it a bare `run_id`, so without this the handler has no way back
        # to the request that created the row. Write-once by convention: `update()` never
        # persists it, so a handler cannot rewrite its own instructions mid-flight.
        self.params: dict[str, Any] = dict(params or {})
        # The Protocol this Run concerns. Known at creation for a prediction
        # (you pick the Protocol to run) but not for a training run, where the
        # Protocol does not exist until the work finishes -- `link_protocol`
        # fills it in then.
        #
        # A real column rather than a `params` key, because `params` is
        # write-once by convention: `update()` deliberately never persists it,
        # so that a handler cannot rewrite its own instructions mid-flight. An
        # outcome is not an instruction, and needs a field that survives an
        # update.
        self.protocol_id = protocol_id
        self.status = status
        self.progress = progress
        self.phase = phase
        self.result_uri = result_uri
        self.error_message = error_message

    def _touch(self) -> None:
        self.updated_at = datetime.now(UTC)

    def link_protocol(self, protocol_id: uuid.UUID) -> None:
        """Record the Protocol a training run produced.

        Without this a finished training Run is a dead end: the client that
        submitted it holds a run id, polls it to `ready`, and has no way to
        reach the Scorecard it just paid for -- which is the single most
        important transition in the application.

        Write-once. A Run concerns exactly one Protocol, and re-pointing a
        completed Run at a different one would silently rewrite history for
        anyone already citing it.
        """
        if self.protocol_id is not None and self.protocol_id != protocol_id:
            raise ConflictError(
                f"Run '{self.id}' is already linked to protocol '{self.protocol_id}'"
            )
        self.protocol_id = protocol_id
        self._touch()

    def start(self) -> None:
        """`pending -> running` normally. `running -> running` is also legal:
        a lease-expiry requeue is at-least-once, so a runner crash mid-job
        puts the same run_id back in front of a fresh claimant, which lands
        here with the row already RUNNING. With no checkpoints, restart-from-zero is the designed
        recovery, so the redelivery restarts the run and wipes the dead
        attempt's stale progress. Terminal runs still refuse -- a redelivery
        for a run that was cancelled (or somehow finished) while queued must
        be dropped by the caller, not restarted."""
        if self.status in _TERMINAL:
            raise ConflictError(f"Cannot start run '{self.id}' in status '{self.status}'")
        self.status = RunStatus.RUNNING
        self.progress = 0.0
        self.phase = None
        self._touch()

    def report_progress(self, fraction: float, *, phase: str) -> None:
        if self.status is not RunStatus.RUNNING:
            raise ConflictError(
                f"Cannot report progress on run '{self.id}' in status '{self.status}'"
            )
        self.progress = fraction
        self.phase = phase
        self._touch()

    def succeed(self, result_uri: str) -> None:
        if self.status is not RunStatus.RUNNING:
            raise ConflictError(f"Cannot succeed run '{self.id}' in status '{self.status}'")
        self.status = RunStatus.READY
        self.result_uri = result_uri
        self.progress = 1.0
        self._touch()

    def fail(self, message: str) -> None:
        """Allowed from `pending` as well as `running`: an enqueue that never
        lands (e.g. the write that sets `lane` fails after the Run row itself
        was created) is a failure of the Run just as much as a crash mid-execution,
        and both must be reachable
        from whatever status the Run was in when it broke."""
        if self.status in _TERMINAL:
            raise ConflictError(f"Cannot fail run '{self.id}' in terminal status '{self.status}'")
        self.status = RunStatus.FAILED
        self.error_message = message
        self._touch()

    def cancel(self) -> None:
        """`pending -> cancelled` genuinely stops the work: the job never
        runs. `running -> cancelled` cannot -- the worker is a separate
        process already executing `handler(ctx, run)`, and this method has no
        channel to it.

        The row *is* the channel, though. A worker checkpoints against it
        (`RunTraining._checkpoint`) and stops the fit when it no longer reads
        RUNNING, and `run_job` re-reads before `succeed()` so a cancellation
        landing inside the last checkpoint window is not overwritten with
        READY. So this stops the work as promptly as the engine calls
        `ctx.report` -- and for an engine that never calls it, only when the
        current fit returns.
        """
        if self.status in _TERMINAL:
            raise ConflictError(
                f"Cannot cancel run '{self.id}' in terminal status '{self.status}'"
            )
        self.status = RunStatus.CANCELLED
        self._touch()
