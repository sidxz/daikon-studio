from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from daikonstudio.infrastructure.sentinel.auth import build_sentinel, register_service_actions
from daikonstudio.settings import Settings


def create_app() -> FastAPI:
    settings = Settings()
    # Sentinel is "configured" only when the service key is explicitly set (see
    # Settings.sentinel_service_key). No protected routes exist yet in Phase 1
    # Task 4; once they land, get_auth's reject-all stub
    # (interface/dependencies/_core.py) still guards them even when this
    # middleware is absent, so an unconfigured Sentinel never becomes a bypass.
    sentinel = build_sentinel(settings) if settings.sentinel_service_key else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if sentinel is None:
            yield
            return
        # sentinel.lifespan fetches the JWKS signing key — fatal if it fails,
        # since auth cannot work at all without it. Action registration is
        # best-effort and must never block boot (see register_service_actions).
        async with sentinel.lifespan(app):
            await register_service_actions(sentinel)
            yield

    app = FastAPI(title="daikon-studio", version="0.1.0", lifespan=lifespan)

    if sentinel is not None:
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
