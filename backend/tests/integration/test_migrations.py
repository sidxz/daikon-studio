import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import text

from daikonstudio.infrastructure.persistence.sqlalchemy.base import Base
from daikonstudio.infrastructure.persistence.sqlalchemy.catalog import (
    models as _catalog_models,  # noqa: F401
)
from daikonstudio.infrastructure.persistence.sqlalchemy.data import (
    models as _data_models,  # noqa: F401
)
from daikonstudio.infrastructure.persistence.sqlalchemy.execution import (
    models as _execution_models,  # noqa: F401
)
from daikonstudio.infrastructure.persistence.sqlalchemy.runners import (
    models as _runners_models,  # noqa: F401
)


@pytest.mark.asyncio
async def test_migrations_apply_cleanly(migrated_session):
    result = await migrated_session.execute(text("SELECT version_num FROM alembic_version"))
    assert result.scalar_one() is not None


@pytest.mark.asyncio
async def test_no_table_exists_only_in_the_orm_or_only_in_the_migrations(_migrated_engine):
    """`alembic revision --autogenerate` emits `drop_table` for any table the
    migrations created that `Base.metadata` does not know about -- which is what
    happens the moment a model module is not imported in `alembic/env.py`. The
    imports above are the same four `env.py` must carry; this pins that the
    migrated schema and the ORM agree on which tables exist at all."""

    def table_diffs(connection):
        diffs = compare_metadata(MigrationContext.configure(connection), Base.metadata)
        return [diff for diff in diffs if diff[0] in ("add_table", "remove_table")]

    async with _migrated_engine.connect() as connection:
        diffs = await connection.run_sync(table_diffs)
    assert diffs == [], diffs
