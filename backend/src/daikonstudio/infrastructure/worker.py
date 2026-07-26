"""arq worker: one task, dispatching on `Run.kind` -- prot-cellar's `run_import`
shape, and the two `JobEnqueuer` implementations that feed it.

Entrypoint for arq::

    uv run arq daikonstudio.infrastructure.worker.WorkerSettings

`_HANDLERS` is the seam Tasks 14 (training) and 17 (prediction) fill in:
`_HANDLERS[RunKind.TRAINING]` / `_HANDLERS[RunKind.PREDICTION]` are each a
`JobHandler` that does the real work and returns a result URI. Left empty
here deliberately -- this task builds the seam, not what runs through it. A
real job hitting the empty dict fails with `KeyError`, caught by the same
`except (Exception, SystemExit)` as any other handler failure and recorded on
the Run rather than crashing the worker.
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

from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.repository import (
    SqlAlchemyRunRepository,
)
from daikonstudio.settings import Settings

JobHandler = Callable[[dict[str, Any], Run], Awaitable[str]]

_HANDLERS: dict[RunKind, JobHandler] = {}


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
    persisted) and then re-raised so arq still logs it as a failed job.
    """
    run = await _load(ctx, run_id)
    run.start()
    await _save(ctx, run)
    try:
        handler = _HANDLERS[run.kind]
        result_uri = await handler(ctx, run)
        run.succeed(result_uri)
    except (Exception, SystemExit) as exc:
        run.fail(repr(exc))
        await _save(ctx, run)
        raise  # re-raise so arq logs it
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

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._ctx: dict[str, Any] = {"sessions": sessions}

    async def enqueue(self, run_id: uuid.UUID) -> None:
        with contextlib.suppress(Exception, SystemExit):
            await run_job(self._ctx, run_id)
