import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Import every model module so Base.metadata is complete for --autogenerate.
# A model that is never imported is invisible to autogenerate and silently
# missing from the schema; append new ones here.
from daikonstudio.infrastructure.persistence.sqlalchemy.base import Base
from daikonstudio.infrastructure.persistence.sqlalchemy.catalog import (
    models as catalog_models,  # noqa: F401
)
from daikonstudio.infrastructure.persistence.sqlalchemy.data import models  # noqa: F401
from daikonstudio.infrastructure.persistence.sqlalchemy.execution import (
    models as execution_models,  # noqa: F401
)
from daikonstudio.infrastructure.persistence.sqlalchemy.runners import (
    models as runners_models,  # noqa: F401
)

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    # disable_existing_loggers=False: fileConfig's default disables every logger
    # that already exists, which silenced the app's own loggers for the rest of
    # the process whenever migrations ran in-process (the test session does).
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Set sqlalchemy.url from the environment when it isn't already set on the
# config (e.g. programmatically, as the integration test fixture does).
if not config.get_main_option("sqlalchemy.url"):
    database_url = os.environ.get("STUDIO_DATABASE_URL")
    if not database_url:
        raise RuntimeError("STUDIO_DATABASE_URL environment variable is required")
    config.set_main_option("sqlalchemy.url", database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
