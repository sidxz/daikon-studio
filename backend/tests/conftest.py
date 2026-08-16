import asyncio
import os
from collections.abc import AsyncIterator

# Set fake Duar env before any daikonstudio import: get_duar() (a
# process-wide @lru_cache singleton, shared by interface/app.py and
# interface/dependencies/_core.py) constructs a real Duar() on first call,
# which raises ValueError without a service key and IdP audience. create_app()
# calls it unconditionally and deliberately fails loud on a real misconfigured
# deploy — but the test suite isn't a real deploy, so it needs values here, not
# a bypass.
#
# Every Duar/IdP setting is pinned here, not just the two that raise on a
# missing value. `Settings` reads `backend/.env` (env_file in its model_config),
# and env vars set here take precedence over that file -- so pinning the whole
# group is what actually makes the suite independent of whatever a developer has
# in `.env`. It was hermetic only by accident while no `.env` existed; the moment
# setup docs told developers to create one, an unpinned STUDIO_DUAR_SERVICE_NAME
# made the app validate tokens against a different service than the harness minted
# them for, and 45 tests failed with "Authz token was issued for a different service".
#
# STUDIO_DUAR_URL is pinned to an unroutable address on purpose: nothing in the
# suite should reach a live Duar, and a developer's `.env` legitimately points at
# the real one with a real key.
os.environ["STUDIO_DUAR_SERVICE_KEY"] = "test-key-for-api-tests"
os.environ["STUDIO_IDP_AUDIENCE"] = "test-audience.apps.googleusercontent.com"
os.environ["STUDIO_DUAR_URL"] = "http://127.0.0.1:1"
os.environ["STUDIO_SERVICE_NAME"] = "daikon-studio"
os.environ["STUDIO_DUAR_SERVICE_NAME"] = "daikon-studio"
os.environ["STUDIO_IDP_JWKS_URL"] = "http://127.0.0.1:1/.well-known/jwks.json"
os.environ["STUDIO_IDP_ISSUER"] = "https://accounts.google.com"

import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import NullPool
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from testcontainers.community.postgres import PostgresContainer

# Two-tier fixture design — do not collapse back into one:
#
# `_migrated_engine` (session scope) owns the genuinely expensive, one-time
# work: start the container, run migrations once, build one engine. It uses
# NullPool so it never holds a live DBAPI connection between tests — every
# `.connect()` opens a fresh asyncpg connection under whatever event loop is
# currently running, so the engine is safe to share across the per-test
# loops pytest-asyncio creates by default (no loop-scope marker needed).
#
# `migrated_session` (function scope, the fixture tests actually use) opens
# one connection + one outer transaction per test and rolls it back at
# teardown, so writes in one test (Task 11+) never leak into the next. It
# uses SQLAlchemy's documented "join a Session into an external transaction"
# recipe (`join_transaction_mode="create_savepoint"`) so even a nested
# `session.commit()` in application code only commits to a savepoint —
# the final `connection.rollback()` still discards everything.


@pytest_asyncio.fixture(scope="session")
async def _migrated_engine() -> AsyncIterator[AsyncEngine]:
    with PostgresContainer("postgres:16-alpine") as postgres:
        sync_url = postgres.get_connection_url()
        async_url = sync_url.replace("postgresql+psycopg2", "postgresql+asyncpg")
        config = Config("alembic.ini")
        config.set_main_option("sqlalchemy.url", async_url)
        # command.upgrade() drives alembic/env.py, whose online path calls
        # asyncio.run(). That can't nest inside the loop pytest-asyncio is
        # already running this fixture on, so push the blocking call onto a
        # worker thread, which has no running loop to collide with.
        await asyncio.to_thread(command.upgrade, config, "head")
        engine = create_async_engine(async_url, poolclass=NullPool)
        yield engine
        await engine.dispose()


@pytest_asyncio.fixture
async def migrated_session(_migrated_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    async with _migrated_engine.connect() as connection:
        await connection.begin()
        async with AsyncSession(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        ) as session:
            yield session
        await connection.rollback()
