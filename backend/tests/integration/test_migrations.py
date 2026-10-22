import asyncio
import importlib.util
import json
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import text

from daikonstudio.infrastructure.persistence.sqlalchemy import pages as _pages  # noqa: F401
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


def _load_migration(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(f"alembic/versions/{name}.py"))
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


async def _json(session, sql: str, **params) -> object:
    # `::text` then json.loads: independent of whichever jsonb codec the driver uses.
    return json.loads(await session.scalar(text(sql), params))


_SINGLE_TARGET = {"column": "y", "kind": "numeric", "unit": "nM", "direction": "low"}
_REPORT_012 = {
    "total_rows": 3,
    "valid_rows": 3,
    "invalid": [],
    "conflicting": [{"structure": "CCO", "values": [0, 1], "row_numbers": [1, 2]}],
    "duplicates_collapsed": 1,
    "salts_flagged": 0,
    "duplicate_spread": 0.25,
}


@pytest.mark.asyncio
async def test_011_backfills_each_protocols_creator_from_its_training_run(migrated_session):
    migration = _load_migration("011_created_by")

    workspace, protocol, trainer = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await migrated_session.execute(
        text(
            "INSERT INTO protocols (id, workspace_id, name, dataset_id, engine_id, artifact_uri,"
            " readouts, conditions, status, protocol_version, version, created_at, updated_at)"
            " VALUES (:id, :ws, 'p', :ds, 'rf', 'x', '[]', '{}', 'draft', 1, 1, now(), now())"
        ),
        {"id": protocol, "ws": workspace, "ds": uuid.uuid4()},
    )
    await migrated_session.execute(
        text(
            "INSERT INTO runs (id, workspace_id, kind, requested_by, cache_key, params,"
            " protocol_id, status, progress, attempts, version, created_at, updated_at)"
            " VALUES (:id, :ws, 'training', :by, 'k', '{}', :p, 'ready', 1, 0, 1, now(), now())"
        ),
        {"id": uuid.uuid4(), "ws": workspace, "by": trainer, "p": protocol},
    )

    await migrated_session.execute(text(migration.BACKFILL_PROTOCOL_CREATORS))

    created_by = await migrated_session.scalar(
        text("SELECT created_by FROM protocols WHERE id = :id"), {"id": protocol}
    )
    assert created_by == trainer


@pytest.mark.asyncio
async def test_013_moves_a_single_target_into_a_list_and_back(migrated_session):
    migration = _load_migration("013_dataset_targets")
    # Re-create the 012 column beside the 013 one, inside this test's rolled-back
    # transaction, so both directions run against real rows.
    await migrated_session.execute(text("ALTER TABLE datasets ADD COLUMN target JSONB"))
    dataset_id = uuid.uuid4()
    await migrated_session.execute(
        text(
            "INSERT INTO datasets (id, workspace_id, name, structure_column, target, targets,"
            " split, content_hash, snapshot_uri, row_count, validation_report, version,"
            " created_at, updated_at) VALUES (:id, :ws, 'd', 'smiles',"
            " CAST(CAST(:target AS text) AS jsonb), '[]', '{}', 'h', 'x', 3,"
            " CAST(CAST(:report AS text) AS jsonb), 1, now(), now())"
        ),
        {
            "id": dataset_id,
            "ws": uuid.uuid4(),
            "target": json.dumps(_SINGLE_TARGET),
            "report": json.dumps(_REPORT_012),
        },
    )

    for statement in (
        migration.TARGETS_FROM_TARGET,
        migration.KEY_SPREAD_BY_TARGET,
        migration.TAG_CONFLICT_COLUMNS,
    ):
        await migrated_session.execute(text(statement))

    assert await _json(
        migrated_session, "SELECT targets::text FROM datasets WHERE id = :id", id=dataset_id
    ) == [_SINGLE_TARGET]
    report = await _json(
        migrated_session,
        "SELECT validation_report::text FROM datasets WHERE id = :id",
        id=dataset_id,
    )
    assert report["duplicate_spread"] == {"y": 0.25}
    assert report["conflicting"][0]["column"] == "y"

    await migrated_session.execute(
        text("UPDATE datasets SET target = NULL WHERE id = :id"), {"id": dataset_id}
    )
    for statement in (
        migration.TARGET_FROM_TARGETS,
        migration.UNKEY_SPREAD,
        migration.UNTAG_CONFLICT_COLUMNS,
    ):
        await migrated_session.execute(text(statement))

    assert (
        await _json(
            migrated_session, "SELECT target::text FROM datasets WHERE id = :id", id=dataset_id
        )
        == _SINGLE_TARGET
    )
    assert (
        await _json(
            migrated_session,
            "SELECT validation_report::text FROM datasets WHERE id = :id",
            id=dataset_id,
        )
        == _REPORT_012
    )


@pytest.mark.asyncio
async def test_013_downgrade_counts_the_datasets_it_would_truncate(migrated_session):
    migration = _load_migration("013_dataset_targets")
    # Relative, not absolute: the container is shared with every other integration
    # test, and some of them commit multi-target datasets of their own.
    before = await migrated_session.scalar(text(migration.COUNT_MULTI_TARGET))
    await migrated_session.execute(
        text(
            "INSERT INTO datasets (id, workspace_id, name, structure_column, targets, split,"
            " content_hash, snapshot_uri, row_count, validation_report, version, created_at,"
            " updated_at) VALUES (:id, :ws, 'd', 'smiles',"
            ' \'[{"column": "a", "kind": "binary"}, {"column": "b", "kind": "binary"}]\','
            " '{}', 'h2', 'x', 3, '{}', 1, now(), now())"
        ),
        {"id": uuid.uuid4(), "ws": uuid.uuid4()},
    )
    assert await migrated_session.scalar(text(migration.COUNT_MULTI_TARGET)) == before + 1


@pytest.mark.asyncio
async def test_013_nests_a_training_runs_headline_under_its_datasets_target(migrated_session):
    migration = _load_migration("013_dataset_targets")
    workspace, dataset_id, run_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await migrated_session.execute(
        text(
            "INSERT INTO datasets (id, workspace_id, name, structure_column, targets, split,"
            " content_hash, snapshot_uri, row_count, validation_report, version, created_at,"
            ' updated_at) VALUES (:id, :ws, \'d\', \'smiles\', \'[{"column": "y", "kind":'
            " \"numeric\"}]', '{}', 'h3', 'x', 3, '{}', 1, now(), now())"
        ),
        {"id": dataset_id, "ws": workspace},
    )
    flat = {"primary_metric": "rmse", "value": 0.5, "baseline_value": 0.7}
    await migrated_session.execute(
        text(
            "INSERT INTO runs (id, workspace_id, kind, requested_by, cache_key, params, status,"
            " progress, attempts, version, created_at, updated_at, metrics) VALUES (:id, :ws,"
            " 'training', :by, 'k', CAST(CAST(:params AS text) AS jsonb), 'ready', 1, 0, 1,"
            " now(), now(), CAST(CAST(:metrics AS text) AS jsonb))"
        ),
        {
            "id": run_id,
            "ws": workspace,
            "by": uuid.uuid4(),
            "params": json.dumps({"dataset_id": str(dataset_id)}),
            "metrics": json.dumps(flat),
        },
    )

    await migrated_session.execute(text(migration.NEST_RUN_METRICS))
    assert await _json(
        migrated_session, "SELECT metrics::text FROM runs WHERE id = :id", id=run_id
    ) == {"targets": [{"column": "y", **flat}]}

    await migrated_session.execute(text(migration.FLATTEN_RUN_METRICS))
    assert (
        await _json(migrated_session, "SELECT metrics::text FROM runs WHERE id = :id", id=run_id)
        == flat
    )


@pytest.mark.asyncio
async def test_two_apis_starting_at_once_migrate_an_empty_database_once(_migrated_engine):
    """The API migrates itself at start (infrastructure/persistence/migrate.py), so a
    second replica or a rolling update's overlap runs the same upgrade at the same time.
    The advisory lock in env.py makes the second wait and then find nothing to do.

    Two processes, as two APIs are: alembic keeps its context in process-wide globals,
    so two upgrades on threads of one process tangle each other and hang -- which no
    deploy does."""
    import asyncio
    import sys

    from alembic.script import ScriptDirectory
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    from daikonstudio.infrastructure.persistence.migrate import SCRIPT_LOCATION

    fresh = f"fresh_{uuid.uuid4().hex}"
    admin = create_async_engine(
        _migrated_engine.url, poolclass=NullPool, isolation_level="AUTOCOMMIT"
    )
    async with admin.connect() as connection:
        await connection.execute(text(f'CREATE DATABASE "{fresh}"'))
    url = _migrated_engine.url.set(database=fresh).render_as_string(hide_password=False)
    start_an_api = (
        "import sys; from daikonstudio.infrastructure.persistence.migrate import "
        "upgrade_to_head; upgrade_to_head(sys.argv[1])"
    )
    try:
        apis = [
            await asyncio.create_subprocess_exec(sys.executable, "-c", start_an_api, url)
            for _ in range(2)
        ]
        codes = await asyncio.wait_for(asyncio.gather(*(api.wait() for api in apis)), timeout=120)
        assert codes == [0, 0]

        engine = create_async_engine(url, poolclass=NullPool)
        async with engine.connect() as connection:
            version = (
                await connection.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one()
        await engine.dispose()
        assert version == ScriptDirectory(str(SCRIPT_LOCATION)).get_current_head()
    finally:
        async with admin.connect() as connection:
            await connection.execute(text(f'DROP DATABASE IF EXISTS "{fresh}" WITH (FORCE)'))
        await admin.dispose()


@pytest.mark.asyncio
async def test_migrating_in_process_leaves_the_apps_logging_alone(_migrated_engine):
    """Duar lost every structured log line when it began migrating in-process: alembic's
    env.py ran fileConfig and replaced the app's handlers. `upgrade_to_head` builds its
    config with no ini file, so env.py never does."""
    import asyncio
    import logging

    from daikonstudio.infrastructure.persistence.migrate import upgrade_to_head

    root = logging.getLogger()
    before = (list(root.handlers), root.level)

    url = _migrated_engine.url.render_as_string(hide_password=False)
    await asyncio.wait_for(asyncio.to_thread(upgrade_to_head, url), timeout=120)

    assert (list(root.handlers), root.level) == before


_BUILD_COLUMNS = text("""
    SELECT column_name, is_nullable, data_type FROM information_schema.columns
    WHERE table_name = 'runners' AND column_name LIKE 'build\\_%'
    ORDER BY column_name
""")


@pytest.mark.asyncio
async def test_019_adds_the_runner_build_columns_and_takes_them_away_again(_migrated_engine):
    """Both directions, against a real database. A column migration that cannot
    downgrade strands an operator on the release they are trying to roll back,
    and `op.add_column` is exactly the kind of statement that looks reversible
    and is not. Runs on its own fresh database: downgrading the session's shared
    one would pull the schema out from under every other test.
    """
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    fresh = f"migration_019_{uuid.uuid4().hex}"
    admin = create_async_engine(
        _migrated_engine.url, poolclass=NullPool, isolation_level="AUTOCOMMIT"
    )
    async with admin.connect() as connection:
        await connection.execute(text(f'CREATE DATABASE "{fresh}"'))
    url = _migrated_engine.url.set(database=fresh).render_as_string(hide_password=False)
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", url)
    engine = create_async_engine(url, poolclass=NullPool)

    async def build_columns() -> list[tuple[str, str, str]]:
        async with engine.connect() as connection:
            return [tuple(row) for row in (await connection.execute(_BUILD_COLUMNS)).all()]

    try:
        # alembic's env.py runs its own event loop -- hence the thread, same as
        # the `_migrated_engine` fixture.
        await asyncio.to_thread(command.upgrade, config, "head")
        assert await build_columns() == [
            ("build_date", "YES", "text"),
            ("build_git_sha", "YES", "text"),
            ("build_version", "YES", "text"),
        ]

        await asyncio.to_thread(command.downgrade, config, "018")
        assert await build_columns() == []

        await asyncio.to_thread(command.upgrade, config, "019")
        assert [name for name, _, _ in await build_columns()] == [
            "build_date",
            "build_git_sha",
            "build_version",
        ]
    finally:
        await engine.dispose()
        async with admin.connect() as connection:
            await connection.execute(text(f'DROP DATABASE IF EXISTS "{fresh}" WITH (FORCE)'))
        await admin.dispose()


_STUDY_TABLE = text("SELECT count(*) FROM information_schema.tables WHERE table_name = 'studies'")


@pytest.mark.asyncio
async def test_020_studies_round_trips(_migrated_engine):
    """The studies table comes and goes with the migration, on its own fresh database."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    fresh = f"migration_020_{uuid.uuid4().hex}"
    admin = create_async_engine(
        _migrated_engine.url, poolclass=NullPool, isolation_level="AUTOCOMMIT"
    )
    async with admin.connect() as connection:
        await connection.execute(text(f'CREATE DATABASE "{fresh}"'))
    url = _migrated_engine.url.set(database=fresh).render_as_string(hide_password=False)
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", url)
    engine = create_async_engine(url, poolclass=NullPool)

    async def has_table() -> bool:
        async with engine.connect() as connection:
            return (await connection.execute(_STUDY_TABLE)).scalar_one() == 1

    try:
        await asyncio.to_thread(command.upgrade, config, "head")
        assert await has_table()
        await asyncio.to_thread(command.downgrade, config, "019")
        assert not await has_table()
        await asyncio.to_thread(command.upgrade, config, "020")
        assert await has_table()
    finally:
        await engine.dispose()
        async with admin.connect() as connection:
            await connection.execute(text(f'DROP DATABASE IF EXISTS "{fresh}" WITH (FORCE)'))


_ANSWERS_COLUMN = text("""
    SELECT is_nullable, data_type FROM information_schema.columns
    WHERE table_name = 'studies' AND column_name = 'answers'
""")


@pytest.mark.asyncio
async def test_021_study_answers_round_trips(_migrated_engine):
    """A study saved before 021 reads back with no answers; the column comes and goes."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    fresh = f"migration_021_{uuid.uuid4().hex}"
    admin = create_async_engine(
        _migrated_engine.url, poolclass=NullPool, isolation_level="AUTOCOMMIT"
    )
    async with admin.connect() as connection:
        await connection.execute(text(f'CREATE DATABASE "{fresh}"'))
    url = _migrated_engine.url.set(database=fresh).render_as_string(hide_password=False)
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", url)
    engine = create_async_engine(url, poolclass=NullPool)

    async def answers_column() -> list[tuple[str, str]]:
        async with engine.connect() as connection:
            return [tuple(row) for row in (await connection.execute(_ANSWERS_COLUMN)).all()]

    try:
        await asyncio.to_thread(command.upgrade, config, "020")
        study_id = uuid.uuid4()
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO studies (id, workspace_id, version, created_by, goal,"
                    " restated_goal, checks, stop_conditions, steps, values)"
                    " VALUES (:id, :ws, 1, :by, 'g', 'rg', '[]', '[]', '[]', '[]')"
                ),
                {"id": study_id, "ws": uuid.uuid4(), "by": uuid.uuid4()},
            )
        await asyncio.to_thread(command.upgrade, config, "021")
        assert await answers_column() == [("NO", "jsonb")]
        async with engine.connect() as connection:
            answers = (
                await connection.execute(
                    text("SELECT answers FROM studies WHERE id = :id"), {"id": study_id}
                )
            ).scalar_one()
        assert answers == []

        await asyncio.to_thread(command.downgrade, config, "020")
        assert await answers_column() == []
        await asyncio.to_thread(command.upgrade, config, "021")
        assert await answers_column() == [("NO", "jsonb")]
    finally:
        await engine.dispose()
        async with admin.connect() as connection:
            await connection.execute(text(f'DROP DATABASE IF EXISTS "{fresh}" WITH (FORCE)'))
        await admin.dispose()


_RUNS_KIND_CHECK = text(
    "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'ck_runs_kind'"
)


@pytest.mark.asyncio
async def test_022_tool_runs_round_trips(_migrated_engine):
    """022 lets a run be a tool step; downgrading takes that back."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    fresh = f"migration_022_{uuid.uuid4().hex}"
    admin = create_async_engine(
        _migrated_engine.url, poolclass=NullPool, isolation_level="AUTOCOMMIT"
    )
    async with admin.connect() as connection:
        await connection.execute(text(f'CREATE DATABASE "{fresh}"'))
    url = _migrated_engine.url.set(database=fresh).render_as_string(hide_password=False)
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", url)
    engine = create_async_engine(url, poolclass=NullPool)

    async def kind_check() -> str:
        async with engine.connect() as connection:
            return (await connection.execute(_RUNS_KIND_CHECK)).scalar_one()

    try:
        await asyncio.to_thread(command.upgrade, config, "021")
        assert "'tool'" not in await kind_check()
        await asyncio.to_thread(command.upgrade, config, "head")
        assert "'tool'" in await kind_check()
        await asyncio.to_thread(command.downgrade, config, "021")
        assert "'tool'" not in await kind_check()
        assert "'prediction'" in await kind_check()
    finally:
        await engine.dispose()
        async with admin.connect() as connection:
            await connection.execute(text(f'DROP DATABASE IF EXISTS "{fresh}" WITH (FORCE)'))
        await admin.dispose()


_PLANNING_COLUMN = text("""
    SELECT is_nullable, data_type FROM information_schema.columns
    WHERE table_name = 'studies' AND column_name = 'planning'
""")


@pytest.mark.asyncio
async def test_023_study_planning_round_trips(_migrated_engine):
    """A study saved before 023 reads back with no planning; the column comes and goes."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    fresh = f"migration_023_{uuid.uuid4().hex}"
    admin = create_async_engine(
        _migrated_engine.url, poolclass=NullPool, isolation_level="AUTOCOMMIT"
    )
    async with admin.connect() as connection:
        await connection.execute(text(f'CREATE DATABASE "{fresh}"'))
    url = _migrated_engine.url.set(database=fresh).render_as_string(hide_password=False)
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", url)
    engine = create_async_engine(url, poolclass=NullPool)

    async def planning_column() -> list[tuple[str, str]]:
        async with engine.connect() as connection:
            return [tuple(row) for row in (await connection.execute(_PLANNING_COLUMN)).all()]

    try:
        await asyncio.to_thread(command.upgrade, config, "022")
        study_id = uuid.uuid4()
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO studies (id, workspace_id, version, created_by, goal,"
                    " restated_goal, checks, stop_conditions, steps, values)"
                    " VALUES (:id, :ws, 1, :by, 'g', 'rg', '[]', '[]', '[]', '[]')"
                ),
                {"id": study_id, "ws": uuid.uuid4(), "by": uuid.uuid4()},
            )
        await asyncio.to_thread(command.upgrade, config, "023")
        assert await planning_column() == [("NO", "jsonb")]
        async with engine.connect() as connection:
            planning = (
                await connection.execute(
                    text("SELECT planning FROM studies WHERE id = :id"), {"id": study_id}
                )
            ).scalar_one()
        assert planning == {}

        await asyncio.to_thread(command.downgrade, config, "022")
        assert await planning_column() == []
        await asyncio.to_thread(command.upgrade, config, "023")
        assert await planning_column() == [("NO", "jsonb")]
    finally:
        await engine.dispose()
        async with admin.connect() as connection:
            await connection.execute(text(f'DROP DATABASE IF EXISTS "{fresh}" WITH (FORCE)'))
        await admin.dispose()


_STUDY_SHAPE_COLUMNS = text("""
    SELECT column_name, is_nullable, data_type FROM information_schema.columns
    WHERE table_name = 'studies' AND column_name IN
        ('started_at', 'mode', 'amended_after_steps', 'thread', 'planning', 'conversations')
    ORDER BY column_name
""")
_OLD_SHAPE = [("conversations", "NO", "jsonb"), ("planning", "NO", "jsonb")]
_NEW_SHAPE = [
    ("amended_after_steps", "YES", "integer"),
    ("mode", "NO", "text"),
    ("started_at", "YES", "timestamp with time zone"),
    ("thread", "NO", "jsonb"),
]
_INSERT_STUDY = text(
    "INSERT INTO studies (id, workspace_id, version, created_by, goal, restated_goal,"
    " checks, stop_conditions, steps, values) VALUES (:id, :ws, 1, :by, 'g', 'rg',"
    " '[]', '[]', '[]', '[]')"
)
_INSERT_THREADED_STUDY = text(
    "INSERT INTO studies (id, workspace_id, version, created_by, goal, restated_goal,"
    " checks, stop_conditions, steps, values, thread) VALUES (:id, :ws, 1, :by, 'g', 'rg',"
    " '[]', '[]', '[]', '[]', CAST(:thread AS jsonb))"
)


@pytest.mark.asyncio
async def test_025_one_thread_round_trips(_migrated_engine):
    """025 gives a study one thread, a start time and a mode in place of the plan it
    was started from and its per-stage conversations. It empties the table first: the
    table was never released, and a row in the old shape cannot be read by the new
    repository. Both directions, on its own fresh database."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    fresh = f"migration_025_{uuid.uuid4().hex}"
    admin = create_async_engine(
        _migrated_engine.url, poolclass=NullPool, isolation_level="AUTOCOMMIT"
    )
    async with admin.connect() as connection:
        await connection.execute(text(f'CREATE DATABASE "{fresh}"'))
    url = _migrated_engine.url.set(database=fresh).render_as_string(hide_password=False)
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", url)
    engine = create_async_engine(url, poolclass=NullPool)

    async def shape() -> list[tuple[str, str, str]]:
        async with engine.connect() as connection:
            return [tuple(row) for row in (await connection.execute(_STUDY_SHAPE_COLUMNS)).all()]

    async def study_count() -> int:
        async with engine.connect() as connection:
            return (await connection.execute(text("SELECT count(*) FROM studies"))).scalar_one()

    async def insert(statement, **params) -> uuid.UUID:
        study_id = uuid.uuid4()
        async with engine.begin() as connection:
            await connection.execute(
                statement, {"id": study_id, "ws": uuid.uuid4(), "by": uuid.uuid4(), **params}
            )
        return study_id

    try:
        await asyncio.to_thread(command.upgrade, config, "024")
        assert await shape() == _OLD_SHAPE
        await insert(_INSERT_STUDY)

        await asyncio.to_thread(command.upgrade, config, "025")
        assert await study_count() == 0  # the old-shape row is gone
        assert await shape() == _NEW_SHAPE  # and planning and conversations with it

        # A row that names none of the new columns gets their defaults.
        bare = await insert(_INSERT_STUDY)
        async with engine.connect() as connection:
            started_at, mode, amended, thread = (
                await connection.execute(
                    text(
                        "SELECT started_at, mode, amended_after_steps, thread::text"
                        " FROM studies WHERE id = :id"
                    ),
                    {"id": bare},
                )
            ).one()
        assert (started_at, mode, amended, json.loads(thread)) == (None, "run", None, {})

        # A row with a thread reads it back.
        stored = {
            "summary": "Wants the pocket docked.",
            "turns": [{"role": "scientist", "say": "Dock it.", "at": "2026-01-01T00:00:00+00:00"}],
        }
        threaded = await insert(_INSERT_THREADED_STUDY, thread=json.dumps(stored))
        async with engine.connect() as connection:
            thread = (
                await connection.execute(
                    text("SELECT thread::text FROM studies WHERE id = :id"), {"id": threaded}
                )
            ).scalar_one()
        assert json.loads(thread) == stored

        await asyncio.to_thread(command.downgrade, config, "024")
        assert await shape() == _OLD_SHAPE
        assert await study_count() == 0  # a new-shape row cannot be read back either
        await asyncio.to_thread(command.upgrade, config, "025")
        assert await shape() == _NEW_SHAPE
    finally:
        await engine.dispose()
        async with admin.connect() as connection:
            await connection.execute(text(f'DROP DATABASE IF EXISTS "{fresh}" WITH (FORCE)'))
        await admin.dispose()
