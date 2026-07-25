import pytest
from sqlalchemy import text


@pytest.mark.asyncio(loop_scope="session")
async def test_migrations_apply_cleanly(migrated_session):
    result = await migrated_session.execute(text("SELECT version_num FROM alembic_version"))
    assert result.scalar_one() is not None
