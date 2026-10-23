"""A tool step queued as a Run executes on a runner: claimed over the runner protocol
and driven by `run_job`, with its input read and its outputs written through the
runner's blob endpoint. Same `app` fixture and NullPool reasoning as
`test_runner_full_loop.py`."""

from __future__ import annotations

import uuid

import httpx
import polars as pl
import pytest_asyncio
from lagom import Container, Singleton
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.protocol_access import ProtocolAccess
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.application.ports.tool_executor import ToolInput
from daikonstudio.application.tools.codecs import encode
from daikonstudio.application.tools.tool_run import (
    OUTPUTS_MANIFEST,
    ToolRunParams,
    read_outputs,
    tool_run_prefix,
)
from daikonstudio.domain.execution.run import Run, RunKind, RunStatus
from daikonstudio.infrastructure.di.container import create_container
from daikonstudio.infrastructure.jobs import run_job
from daikonstudio.infrastructure.runner.ports import build_http_ctx
from daikonstudio.interface.app import create_app
from daikonstudio.settings import Settings
from tests.helpers.runner_fixtures import (
    claim,
    cleanup_registered_runners,
    register_runner,
    seed_run,
)
from tests.helpers.sync_asgi import SyncAsgiTransport


@pytest_asyncio.fixture
async def app(tmp_path, _migrated_engine: AsyncEngine, protocol_access):
    application = create_app()
    container = Container(
        create_container(
            Settings(
                inline_jobs=False,
                blob_base_url=f"file://{tmp_path}",
                database_url=str(_migrated_engine.url),
            )
        )
    )
    container.define(
        async_sessionmaker,
        Singleton(lambda: async_sessionmaker(bind=_migrated_engine, expire_on_commit=False)),
    )
    container.define(ProtocolAccess, Singleton(lambda: protocol_access))  # type: ignore[type-abstract]
    application.state.container = container
    async with cleanup_registered_runners(_migrated_engine):
        yield application


async def _run_tool_step(app, anonymous_client, tool_id: str) -> tuple[Run, str]:
    workspace_id, study_id, run_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    store: BlobStore = app.state.container[BlobStore]
    input_key = f"{workspace_id}/studies/{study_id}/values/x.parquet"
    blob, _ = encode("compound_table", pl.DataFrame({"SMILES": ["CCO", "c1ccccc1", "CCN"]}))
    store.put_bytes(input_key, blob)
    prefix = tool_run_prefix(workspace_id, study_id, run_id)
    params = ToolRunParams(
        tool_id=tool_id,
        lane="default",
        study_id=study_id,
        step_id=uuid.uuid4(),
        params={},
        inputs=(ToolInput(key="compounds", kind="compound_table", blob_key=input_key),),
        output_prefix=prefix,
    )
    await seed_run(app, workspace_id, kind=RunKind.TOOL, params=params.to_params(), id=run_id)
    _, headers = await register_runner(app, ["default"])
    claimed = await claim(anonymous_client, headers)
    assert claimed["run"]["id"] == str(run_id)

    token = headers["Authorization"].removeprefix("Bearer ")
    ctx = build_http_ctx(
        "http://testserver",
        token,
        run_id,
        deadline_seconds=claimed["deadline_seconds"],
        async_transport=httpx.ASGITransport(app=app),
        sync_transport=SyncAsgiTransport(app),
    )
    try:
        await run_job(ctx, run_id)
    except Exception:
        pass  # run_job re-raises a handler failure after recording it on the row
    finally:
        await ctx["_client"].aclose()
    run = await app.state.container[RunRepository].get_by_id(run_id)
    assert run is not None
    return run, prefix


async def test_a_tool_run_executes_on_a_runner(app, anonymous_client):
    run, prefix = await _run_tool_step(app, anonymous_client, "standardize")

    assert run.status is RunStatus.READY, run.error_message
    assert run.result_uri is not None and run.result_uri.endswith(OUTPUTS_MANIFEST)
    store: BlobStore = app.state.container[BlobStore]
    [output] = read_outputs(store, prefix)
    assert output.blob_key == f"{prefix}compounds.parquet"
    assert output.summary["rows"] == 3
    assert store.exists(output.blob_key)


async def test_an_unknown_tool_fails_the_run_naming_it(app, anonymous_client):
    run, _ = await _run_tool_step(app, anonymous_client, "no_such_tool")

    assert run.status is RunStatus.FAILED
    assert run.error_message is not None and "no_such_tool" in run.error_message
