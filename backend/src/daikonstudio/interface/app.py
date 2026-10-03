import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from daikonstudio.infrastructure.di.container import create_container
from daikonstudio.infrastructure.duar.auth import (
    get_duar,
    log_effective_scope,
    register_service_actions,
)
from daikonstudio.interface.error_handlers import register_error_handlers
from daikonstudio.interface.middleware import RequestIdMiddleware
from daikonstudio.interface.routes.collections import router as collections_router
from daikonstudio.interface.routes.datasets import router as datasets_router
from daikonstudio.interface.routes.engines import router as engines_router
from daikonstudio.interface.routes.protocols import router as protocols_router
from daikonstudio.interface.routes.runner_api import router as runner_api_router
from daikonstudio.interface.routes.runners import router as runners_router
from daikonstudio.interface.routes.runs import router as runs_router
from daikonstudio.interface.routes.sweeps import router as sweeps_router
from daikonstudio.logging import configure_logging
from daikonstudio.settings import Settings


async def check_database(sessions: async_sessionmaker[AsyncSession]) -> str | None:
    """None when `SELECT 1` answers within 3 s, else the failure's class name.

    The readiness half of `/health`: the liveness probe stays unconditional so
    a database blip does not restart a healthy API process, while this one tells
    a load balancer or `docker compose` whether requests can actually succeed.
    """
    try:
        async with asyncio.timeout(3), sessions() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:  # deliberately broad: a probe reports, it never raises
        return type(exc).__name__
    return None


def create_app() -> FastAPI:
    settings = Settings()
    # First, so the Duar scope line below and every later INFO actually land
    # somewhere -- without this the root logger has no handler at all.
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    # get_duar() is the one process-wide Duar instance (shared with
    # interface/dependencies/_core.py's get_auth — see get_duar's docstring
    # for why two instances would silently check permissions under the wrong
    # realm scope). Unconfigured/misconfigured Duar settings raise ValueError
    # here, deliberately: a service that can't authenticate must fail at boot,
    # not boot healthy with the auth middleware silently absent.
    duar = get_duar()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # duar.lifespan fetches the JWKS signing key — fatal if it fails,
        # since auth cannot work at all without it. Action registration is
        # best-effort and must never block boot (see register_service_actions).
        #
        # Ignore comment below: the SDK's `lifespan` property is annotated as
        # returning Callable[[FastAPI], AsyncIterator[None]] — the undecorated
        # inner function's own signature — without accounting for the
        # @asynccontextmanager decorator that actually wraps it into something
        # `async with`-able. Confirmed against the installed duar_auth 0.17.2
        # source: a type-annotation bug in the SDK, not a real incompatibility.
        async with duar.lifespan(app):  # type: ignore[attr-defined]
            # After the SDK lifespan, which is what runs fetch_whoami(): the scope
            # it resolved decides whether any authenticated request can succeed, and
            # the SDK logs nothing either way. See log_effective_scope.
            log_effective_scope(duar)
            await register_service_actions(duar)
            yield

    app = FastAPI(title="daikon-studio", version="0.1.0", lifespan=lifespan)

    # Built here rather than in the lifespan: every binding is lazy, so this
    # touches no database, filesystem or network, and an app that has not been
    # started (a test driving it over ASGITransport) still resolves use cases.
    app.state.container = create_container(settings)

    # duar.protect() MUST be added before CORSMiddleware. Starlette applies
    # middleware LIFO (last added = outermost), so this order makes CORS the
    # outer layer: a 401 raised by auth still passes back out through CORS and
    # keeps its headers. Reversed, the browser sees an opaque network error
    # instead of a 401 — do not "tidy" this order.
    # "/api/v1/runner" is the self-hosted-runner protocol -- authenticated by
    # its own runner-token dependency (interface/dependencies/runner_auth.py),
    # not Duar: a runner process carries no IdP/Duar token pair at
    # all. exclude_paths matches on a path-segment boundary (exact match or
    # `path + "/"` prefix -- see duar_auth.authz_middleware), so this
    # cannot also swallow "/api/v1/runners" (human-facing runner management,
    # `interface/routes/runners.py`), which stays Duar-protected.
    duar.protect(
        app,
        exclude_paths=[
            "/health",
            "/ready",
            "/version",
            "/docs",
            "/openapi.json",
            "/api/v1/runner",
        ],
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Outermost (added last), so the request id exists before auth or CORS can
    # reject anything and is stamped on every response, including a 401.
    app.add_middleware(RequestIdMiddleware)

    register_error_handlers(app, cors_origins=settings.cors_origins)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        failure = await check_database(app.state.container[async_sessionmaker])
        if failure is not None:
            return JSONResponse(
                status_code=503, content={"status": "unavailable", "detail": failure}
            )
        return JSONResponse({"status": "ready"})

    @app.get("/version")
    async def version() -> dict[str, str]:
        return {"service": settings.service_name, "version": app.version}

    app.include_router(collections_router)
    app.include_router(datasets_router)
    app.include_router(engines_router)
    app.include_router(protocols_router)
    app.include_router(runner_api_router)
    app.include_router(runners_router)
    app.include_router(runs_router)
    app.include_router(sweeps_router)

    return app


# The ASGI entry point: `uvicorn daikonstudio.interface.app:app` (make dev) and
# the OpenAPI snapshot (make generate-api) both import this name. Constructing at
# import time means an unconfigured Duar raises here rather than at first
# request, which is the intended behaviour — a service that cannot authenticate
# must fail at boot, not serve traffic unprotected. Both Makefile targets source
# backend/.env first. Same shape as prot-cellar's interface/app.py.
app = create_app()
