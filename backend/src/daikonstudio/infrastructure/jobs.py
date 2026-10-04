"""Job-execution core: one function that runs a queued Run, dispatching on
`Run.kind` -- prot-cellar's `run_import` shape -- plus the two `JobEnqueuer`
implementations that feed it and the one place a ctx gets assembled from
SqlAlchemy repositories.

`_HANDLERS` is the seam Tasks 14 (training) and 17 (prediction) fill in:
`_HANDLERS[RunKind.TRAINING]` / `_HANDLERS[RunKind.PREDICTION]` are each a
`JobHandler` that does the real work and returns a result URI. Both kinds are
wired now. A future `RunKind` hitting an unfilled slot would fail with
`KeyError`, caught by the same `except (Exception, SystemExit)` as any other
handler failure and recorded on the Run rather than crashing the caller.

`run_job` itself never touches SQLAlchemy: `_load`/`_save`/`_train`/`_predict`
pull repositories straight off `ctx` -- port objects (`ctx["runs"]`,
`ctx["datasets"]`, `ctx["protocols"]`, `ctx["store"]`), not a sessionmaker.
`build_sqlalchemy_ctx` is the one place that trio gets assembled from a real
`async_sessionmaker`, used by `InlineEnqueuer` to run a job in its own
process; a self-hosted runner instead claims a run_id via `RunQueue` and
builds its own ctx the same way. The engine registry and the structure
normalizer are stateless and take no configuration, so they are constructed
here rather than carried on `ctx`.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from daikonstudio.application.engines.checkpoints import DEFAULT_INTERVAL_SECONDS
from daikonstudio.application.engines.context import RunInterrupted
from daikonstudio.application.engines.manifest import DEFAULT_LANE
from daikonstudio.application.execution.failure_message import user_facing_error
from daikonstudio.application.execution.predict_with_protocol import RunPrediction
from daikonstudio.application.execution.train_protocol import RunTraining
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.run_queue import RunQueue
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.domain.execution.run import Run, RunKind, RunStatus
from daikonstudio.domain.shared.errors import ConflictError
from daikonstudio.infrastructure.chem.chemical_space import UmapLayout
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer
from daikonstudio.infrastructure.engines.registry import default_registry
from daikonstudio.infrastructure.persistence.sqlalchemy.catalog.repository import (
    SqlAlchemyProtocolRepository,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.data.repository import (
    SqlAlchemyDatasetRepository,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.repository import (
    SqlAlchemyRunRepository,
)

_logger = structlog.get_logger(__name__)

JobHandler = Callable[[dict[str, Any], Run], Awaitable[str]]


def build_sqlalchemy_ctx(
    sessions: async_sessionmaker[AsyncSession],
    store: BlobStore,
    *,
    job_deadline_seconds: int | None = None,
) -> dict[str, Any]:
    """The one place the SqlAlchemy repository trio is assembled into a ctx --
    used by `InlineEnqueuer`, and by nothing else once arq is gone (a
    self-hosted runner builds its own the same way once it has claimed a
    run_id)."""
    return {
        "runs": SqlAlchemyRunRepository(sessions),
        "datasets": SqlAlchemyDatasetRepository(sessions),
        "protocols": SqlAlchemyProtocolRepository(sessions),
        "store": store,
        # `None` unless a caller passes one: InlineEnqueuer builds a ctx with no lane
        # deadline to enforce, since dev-mode jobs have no lane to enforce one for.
        "job_deadline_seconds": job_deadline_seconds,
    }


async def _train(ctx: dict[str, Any], run: Run) -> str:
    return await RunTraining(
        ctx["datasets"],
        ctx["protocols"],
        ctx["runs"],
        ctx["store"],
        default_registry(),
        RdkitStructureNormalizer(),
        # `.get`, not `[...]`: a ctx built by hand in a test may carry neither key.
        deadline_seconds=ctx.get("job_deadline_seconds"),
        layout=UmapLayout(),
        checkpoint_interval_seconds=ctx.get(
            "checkpoint_interval_seconds", DEFAULT_INTERVAL_SECONDS
        ),
    )(run)


async def _predict(ctx: dict[str, Any], run: Run) -> str:
    return await RunPrediction(
        ctx["protocols"],
        ctx["store"],
        default_registry(),
        RdkitStructureNormalizer(),
    )(run)


_HANDLERS: dict[RunKind, JobHandler] = {RunKind.TRAINING: _train, RunKind.PREDICTION: _predict}


async def _load(ctx: dict[str, Any], run_id: uuid.UUID) -> Run:
    runs: RunRepository = ctx["runs"]
    run = await runs.get_by_id(run_id)
    if run is None:
        raise RuntimeError(f"Run '{run_id}' not found")
    return run


async def _save(ctx: dict[str, Any], run: Run) -> None:
    runs: RunRepository = ctx["runs"]
    await runs.update(run)


async def fail_run(ctx: dict[str, Any], run_id: uuid.UUID, message: str) -> None:
    """Record FAILED on a run from outside its handler.

    The runner agent's hard kill: a fit that never returns cannot be stopped
    (a thread cannot be killed), so the agent writes the failure and exits the
    process. Raises ConflictError if the run went terminal in the meantime."""
    run = await _load(ctx, run_id)
    run.fail(message)
    await _save(ctx, run)


async def run_job(ctx: dict[str, Any], run_id: uuid.UUID) -> None:
    """Execute one queued Run: load it, start it, run its handler, record the
    outcome.

    Catches `(Exception, SystemExit)` -- deliberately not `asyncio.CancelledError`
    or `KeyboardInterrupt`, which must propagate uncaught so the caller's own
    shutdown handling still works. A caught failure is recorded on the Run
    (`fail()`, persisted) before the exception is re-raised.

    ponytail: for a plain `Exception` that re-raise lets the caller log the
    failure and move on to the next job in the same process. For `SystemExit`
    it does not -- CPython's `asyncio/tasks.py` special-cases `SystemExit`
    (and `KeyboardInterrupt`) by propagating them past every application-level
    handler, including the caller's own, all the way out of the event loop.
    So a `SystemExit` raised inside a handler (plausible once a training/
    prediction library wraps one that calls `sys.exit()` internally) still
    kills the whole process here, taking any other jobs running concurrently
    in it down too. Catching it doesn't prevent that -- nothing can, short of
    not letting it happen in the first place -- but it does guarantee the row
    is persisted as `FAILED` before the process dies, rather than leaving it
    RUNNING until redelivery restarts the whole job from zero (see the
    comment on `run.start()` below). Upgrade path if this bites: one job per
    process, or a supervisor that restarts the process on exit.
    """
    run = await _load(ctx, run_id)
    # At-least-once delivery: a crash mid-job can redeliver this run_id with the
    # row already RUNNING, and start() treats that as a restart-from-zero (see
    # Run.start). A ConflictError here therefore means the run went terminal
    # while queued -- cancelled, most likely -- so the redelivered job is
    # nobody's work anymore: return without saving, and without raising, so the
    # caller treats the job as done rather than retrying it.
    try:
        run.start()
    except ConflictError:
        return
    await _save(ctx, run)
    try:
        handler = _HANDLERS[run.kind]
        result_uri = await handler(ctx, run)
    except RunInterrupted as interrupted:
        # Deliberately NOT re-raised, unlike a handler failure below: this exception
        # means the work was stopped on purpose, and a propagating exception is how a
        # redelivering caller would come to retry it -- restarting a fit the user
        # cancelled, on the same GPU.
        if not interrupted.cancelled:
            run.fail(interrupted.reason)
            await _save(ctx, run)
        # A cancellation needs no write: the row is already CANCELLED, which is
        # precisely why `report` raised.
        return
    except (Exception, SystemExit) as exc:
        run.fail(user_facing_error(exc))
        await _save(ctx, run)
        raise  # re-raise: FAILED is persisted above regardless of what happens next

    # Re-read before claiming success. `RunTraining`'s reporter only consults the row
    # every `_PROGRESS_INTERVAL_SECONDS`, and an engine that never calls `report` never
    # consults it at all -- so without this a cancellation landing after the last
    # checkpoint would be silently overwritten with READY, which is exactly the lie
    # `Run.cancel()` used to have to admit to in its docstring.
    current = await _load(ctx, run.id)
    if current.status is not RunStatus.RUNNING:
        return
    # As every progress write does: a write whose answer was lost leaves this copy behind.
    run.version = current.version
    run.succeed(result_uri)
    await _save(ctx, run)


class DbEnqueuer:
    """Enqueues a run by claiming its lane in the database -- the production
    `JobEnqueuer`. Self-hosted runners poll `RunQueue.claim_next` for pending
    work; enqueuing is nothing more than making the row visible to that poll
    on the right lane."""

    def __init__(self, queue: RunQueue) -> None:
        self._queue = queue

    async def enqueue(self, run_id: uuid.UUID, lane: str = DEFAULT_LANE) -> None:
        await self._queue.set_lane(run_id, lane)


class InlineEnqueuer:
    """Runs `run_job` directly in the caller's own process instead of handing it
    off to a self-hosted runner. Selected by `STUDIO_INLINE_JOBS=1` so tests and
    local dev need no runner at all -- the dev-mode twin of `DbEnqueuer`, chosen
    at container-build time (chem-cellar's `NullJobOrchestrator` trick): both are
    the same `JobEnqueuer` port from the caller's point of view.

    On a handler failure, `run_job()` has already called `run.fail()` and
    persisted it before re-raising (so a real runner would still log it). Here
    the caller is an HTTP request, and the real async path via `DbEnqueuer`
    never lets a job failure reach that request at all -- it returns 202
    before any runner even claims the row, and the failure only shows up
    later, on the row, when the client polls. Logging, not re-raising, after
    `run_job()` records it keeps both paths behaviourally identical: `enqueue()`
    returns normally either way, and a caller written for the async contract
    doesn't need a `try/except` it will only ever exercise in dev mode.
    `asyncio.CancelledError` and `KeyboardInterrupt` are not caught here
    either, for the same shutdown-must-propagate reason `run_job()` doesn't
    catch them.
    """

    def __init__(self, sessions: async_sessionmaker[AsyncSession], store: BlobStore) -> None:
        # Same shape a self-hosted runner builds for itself after claiming a run_id
        # (see `build_sqlalchemy_ctx`), so a handler cannot tell which enqueuer it is
        # running under.
        self._ctx: dict[str, Any] = build_sqlalchemy_ctx(sessions, store)

    async def enqueue(self, run_id: uuid.UUID, lane: str = DEFAULT_LANE) -> None:
        # `lane` is ignored on purpose: running the job in the caller's own process
        # has no queue to route it to. Accepting the argument is what keeps the two
        # implementations interchangeable from a caller's point of view.
        try:
            await run_job(self._ctx, run_id)
        except (Exception, SystemExit):
            # run_job already persisted FAILED on the row; this is so the operator
            # sees the traceback, exactly as the runner agent logs the same failure.
            _logger.exception("inline job failed", run_id=str(run_id))
