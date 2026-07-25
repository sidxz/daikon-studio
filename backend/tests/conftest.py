import asyncio

import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def migrated_session():
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
        engine = create_async_engine(async_url)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            yield session
        await engine.dispose()
