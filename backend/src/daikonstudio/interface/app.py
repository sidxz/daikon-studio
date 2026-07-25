from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from daikonstudio.infrastructure.sentinel.auth import get_sentinel, register_service_actions
from daikonstudio.settings import Settings


def create_app() -> FastAPI:
    settings = Settings()
    # get_sentinel() is the one process-wide Sentinel instance (shared with
    # interface/dependencies/_core.py's get_auth — see get_sentinel's docstring
    # for why two instances would silently check permissions under the wrong
    # realm scope). Unconfigured/misconfigured Sentinel settings raise ValueError
    # here, deliberately: a service that can't authenticate must fail at boot,
    # not boot healthy with the auth middleware silently absent.
    sentinel = get_sentinel()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # sentinel.lifespan fetches the JWKS signing key — fatal if it fails,
        # since auth cannot work at all without it. Action registration is
        # best-effort and must never block boot (see register_service_actions).
        #
        # Ignore comment below: the SDK's `lifespan` property is annotated as
        # returning Callable[[FastAPI], AsyncIterator[None]] — the undecorated
        # inner function's own signature — without accounting for the
        # @asynccontextmanager decorator that actually wraps it into something
        # `async with`-able. Confirmed against the installed sentinel_auth 0.17.2
        # source: a type-annotation bug in the SDK, not a real incompatibility.
        async with sentinel.lifespan(app):  # type: ignore[attr-defined]
            await register_service_actions(sentinel)
            yield

    app = FastAPI(title="daikon-studio", version="0.1.0", lifespan=lifespan)

    # sentinel.protect() MUST be added before CORSMiddleware. Starlette applies
    # middleware LIFO (last added = outermost), so this order makes CORS the
    # outer layer: a 401 raised by auth still passes back out through CORS and
    # keeps its headers. Reversed, the browser sees an opaque network error
    # instead of a 401 — do not "tidy" this order.
    sentinel.protect(app, exclude_paths=["/health", "/version", "/docs", "/openapi.json"])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    async def version() -> dict[str, str]:
        return {"service": settings.service_name, "version": app.version}

    return app
