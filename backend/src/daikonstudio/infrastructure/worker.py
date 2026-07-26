"""arq worker: one task, dispatching on `Run.kind` -- prot-cellar's `run_import`
shape, and the two `JobEnqueuer` implementations that feed it.

Entrypoint for arq::

    uv run arq daikonstudio.infrastructure.worker.WorkerSettings

`_HANDLERS` is the seam Tasks 14 (training) and 17 (prediction) fill in:
`_HANDLERS[RunKind.TRAINING]` / `_HANDLERS[RunKind.PREDICTION]` are each a
`JobHandler` that does the real work and returns a result URI. Both kinds are
wired now. A future `RunKind` hitting an unfilled slot would fail with
`KeyError`, caught by the same `except (Exception, SystemExit)` as any other
handler failure and recorded on the Run rather than crashing the worker.

This module is also the worker's composition root. Handlers are application-
layer objects with no idea where their collaborators come from, so the small
`_train` adapter below builds them per job from `ctx`: repositories over the
process-wide session factory, and the blob store the process was configured
with. The engine registry and the structure normalizer are stateless and take
no configuration, so they are constructed here rather than carried on `ctx`.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

import arq
from arq.connections import ArqRedis, RedisSettings
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from daikonstudio.application.execution.predict_with_protocol import RunPrediction
from daikonstudio.application.execution.train_protocol import RunTraining
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.domain.execution.run import Run, RunKind
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
from daikonstudio.infrastructure.storage.fsspec_blob_store import FsspecBlobStore
from daikonstudio.settings import Settings

JobHandler = Callable[[dict[str, Any], Run], Awaitable[str]]


async def _train(ctx: dict[str, Any], run: Run) -> str:
    sessions = ctx["sessions"]
    return await RunTraining(
        SqlAlchemyDatasetRepository(sessions),
        SqlAlchemyProtocolRepository(sessions),
        SqlAlchemyRunRepository(sessions),
        ctx["store"],
        default_registry(),
        RdkitStructureNormalizer(),
    )(run)


async def _predict(ctx: dict[str, Any], run: Run) -> str:
    sessions = ctx["sessions"]
    return await RunPrediction(
        SqlAlchemyProtocolRepository(sessions),
        ctx["store"],
        default_registry(),
        RdkitStructureNormalizer(),
    )(run)


_HANDLERS: dict[RunKind, JobHandler] = {RunKind.TRAINING: _train, RunKind.PREDICTION: _predict}


async def _load(ctx: dict[str, Any], run_id: uuid.UUID) -> Run:
    repository = SqlAlchemyRunRepository(ctx["sessions"])
    run = await repository.get_by_id(run_id)
    if run is None:
        raise RuntimeError(f"Run '{run_id}' not found")
    return run


async def _save(ctx: dict[str, Any], run: Run) -> None:
    await SqlAlchemyRunRepository(ctx["sessions"]).update(run)


async def run_job(ctx: dict[str, Any], run_id: uuid.UUID) -> None:
    """Execute one queued Run: load it, start it, run its handler, record the
    outcome.

    Catches `(Exception, SystemExit)` -- deliberately not `asyncio.CancelledError`
    or `KeyboardInterrupt`, which must propagate uncaught so arq's own shutdown
    handling still works. A caught failure is recorded on the Run (`fail()`,
    persisted) before the exception is re-raised.

    ponytail: for a plain `Exception` that re-raise lets arq log the failure
    and move on to the next job in the same process. For `SystemExit` it does
    not -- CPython's `asyncio/tasks.py` special-cases `SystemExit` (and
    `KeyboardInterrupt`) by propagating them past every application-level
    handler, including arq's own, all the way out of the event loop. So a
    `SystemExit` raised inside a handler (plausible once Tasks 14/17 wrap a
    training/prediction library that calls `sys.exit()` internally) still
    kills the whole worker process here, taking any other jobs running
    concurrently in it down too. Catching it doesn't prevent that -- nothing
    can, short of not letting it happen in the first place -- but it does
    guarantee the row is persisted as `FAILED` before the process dies, which
    is strictly better than the alternative (see the note on `run.start()`
    below for what "not catching it" would leave behind instead). Upgrade
    path if this bites: one job per worker process (arq's `max_jobs=1` or a
    process-per-job deployment), or a supervisor that restarts the worker on
    exit.
    """
    run = await _load(ctx, run_id)
    # ponytail: if a previous attempt at this same run_id crashed the worker
    # process after start() but before finishing (including the SystemExit
    # case above), arq's at-least-once delivery redelivers the job to a fresh
    # process. That redelivery lands here with the row already RUNNING, not
    # PENDING -- start() raises ConflictError, that raise is outside the
    # try/except below, fail() never runs, and the row is stuck at RUNNING
    # forever with no worker left executing it. Upgrade path: a reaper that
    # fails Runs stuck in RUNNING past some staleness window, or relaxing
    # start() to allow a RUNNING -> RUNNING restart specifically for
    # redelivery (distinguishable from a genuine double-start by a jobs table
    # arq itself doesn't expose here).
    run.start()
    await _save(ctx, run)
    try:
        handler = _HANDLERS[run.kind]
        result_uri = await handler(ctx, run)
        run.succeed(result_uri)
    except (Exception, SystemExit) as exc:
        run.fail(repr(exc))
        await _save(ctx, run)
        raise  # re-raise: FAILED is persisted above regardless of what happens next
    await _save(ctx, run)


async def _on_startup(ctx: dict[str, Any]) -> None:
    """Build one engine/session factory per worker process and stash it on
    `ctx` -- the worker has no request/response cycle to hang a per-call
    session on the way the FastAPI app does. Built directly (not via
    `infrastructure/persistence/session.py`'s `create_session_factory`) so
    `_on_shutdown` gets back the actual engine object to dispose, rather than
    reaching into `async_sessionmaker`'s private `.kw`.
    """
    settings = Settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    ctx["engine"] = engine
    ctx["sessions"] = async_sessionmaker(engine, expire_on_commit=False)
    ctx["store"] = FsspecBlobStore(settings.blob_base_url)


async def _on_shutdown(ctx: dict[str, Any]) -> None:
    engine: AsyncEngine | None = ctx.get("engine")
    if engine is not None:
        await engine.dispose()


class WorkerSettings:
    """arq WorkerSettings -- mirrors the lifespan wiring in `interface/app.py`."""

    functions: ClassVar[list[Any]] = [run_job]
    redis_settings = RedisSettings.from_dsn(Settings().redis_url)
    on_startup = _on_startup
    on_shutdown = _on_shutdown
    # ponytail: 1800s covers CPU engines comfortably. Climb to a durable engine
    # (Temporal) only when a GPU training run genuinely needs multi-hour execution.
    job_timeout = 1800


class ArqEnqueuer:
    """Enqueues a `run_job` job via arq/Redis -- the production `JobEnqueuer`.

    The connection pool is created lazily on the first `enqueue()` call and
    cached for reuse (mirrors prot-cellar's `ArqJobEnqueuer`), so building the
    DI container never requires Valkey to be reachable.
    """

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._pool: ArqRedis | None = None
        self._lock = asyncio.Lock()

    async def _get_pool(self) -> ArqRedis:
        if self._pool is None:
            async with self._lock:
                if self._pool is None:
                    self._pool = await arq.create_pool(RedisSettings.from_dsn(self._redis_url))
        return self._pool

    async def enqueue(self, run_id: uuid.UUID) -> None:
        pool = await self._get_pool()
        await pool.enqueue_job("run_job", run_id)

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.aclose()
            self._pool = None


class InlineEnqueuer:
    """Runs `run_job` directly in the caller's own process instead of pushing
    to Redis. Selected by `STUDIO_INLINE_JOBS=1` so tests and local dev need
    no Valkey at all -- the dev-mode twin of `ArqEnqueuer`, chosen at
    container-build time (chem-cellar's `NullJobOrchestrator` trick): both are
    the same `JobEnqueuer` port from the caller's point of view.

    On a handler failure, `run_job()` has already called `run.fail()` and
    persisted it before re-raising (so a real arq worker would still log it).
    Here the caller is an HTTP request, and the real async path via
    `ArqEnqueuer` never lets a job failure reach that request at all -- it
    returns 202 before the worker even starts, and the failure only shows up
    later, on the row, when the client polls. Swallowing the exception after
    `run_job()` records it keeps both paths behaviourally identical: `enqueue()`
    returns normally either way, and a caller written for the async contract
    doesn't need a `try/except` it will only ever exercise in dev mode.
    `asyncio.CancelledError` and `KeyboardInterrupt` are not caught here
    either, for the same shutdown-must-propagate reason `run_job()` doesn't
    catch them.
    """

    def __init__(self, sessions: async_sessionmaker[AsyncSession], store: BlobStore) -> None:
        # The same two entries `_on_startup` puts on a real worker's ctx, so a
        # handler cannot tell which enqueuer it is running under.
        self._ctx: dict[str, Any] = {"sessions": sessions, "store": store}

    async def enqueue(self, run_id: uuid.UUID) -> None:
        with contextlib.suppress(Exception, SystemExit):
            await run_job(self._ctx, run_id)
