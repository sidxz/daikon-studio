"""Coverage for the runner-side HTTP port implementations (Task 8):
`RunnerApiClient` and the four `run_job`-facing ports it backs
(`HttpRunRepository`, `HttpDatasetRepository`, `HttpProtocolRepository`,
`HttpBlobStore`).

Run against the real app fixture over `httpx.ASGITransport`/`SyncAsgiTransport`
-- no mocking of the server, same recipe `test_runner_protocol.py` uses to
prove the routes themselves. This module proves the *client* side: that these
ports round-trip through those exact routes the way `jobs.run_job` needs.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
import pytest_asyncio
from lagom import Container, Singleton
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from tests.helpers.runner_fixtures import (
    claim,
    cleanup_registered_runners,
    register_runner,
    seed_run,
)
from tests.helpers.sync_asgi import SyncAsgiTransport

from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from daikonstudio.domain.catalog.readout import Readout, ReadoutType
from daikonstudio.domain.execution.run import RunKind, RunStatus
from daikonstudio.domain.shared.errors import ConcurrencyConflictError
from daikonstudio.infrastructure.di.container import create_container
from daikonstudio.infrastructure.runner.ports import (
    HttpBlobStore,
    HttpDatasetRepository,
    HttpProtocolRepository,
    HttpRunRepository,
    RunnerApiClient,
    build_http_ctx,
)
from daikonstudio.interface.app import create_app
from daikonstudio.settings import Settings

SOLUBILITY_CSV = b"smiles,y\nCCO,1.0\nc1ccccc1,5.0\nCCN,2.0\nc1ccncc1,6.0\n"


def _bearer_token(headers: dict[str, str]) -> str:
    return headers["Authorization"].removeprefix("Bearer ")


def _build_client(app, headers: dict[str, str], run_id: uuid.UUID) -> RunnerApiClient:
    return RunnerApiClient(
        "http://testserver",
        _bearer_token(headers),
        run_id,
        async_transport=httpx.ASGITransport(app=app),
        sync_transport=SyncAsgiTransport(app),
    )


# --------------------------------------------------------------------------
# HttpRunRepository
# --------------------------------------------------------------------------


async def test_get_by_id_returns_a_run_matching_the_envelope_fields(
    anonymous_client, app, workspace_id
):
    run = await seed_run(app, workspace_id, params={"foo": "bar"})
    _, headers = await register_runner(app, ["default"])
    await claim(anonymous_client, headers)
    client = _build_client(app, headers, run.id)

    fetched = await HttpRunRepository(client).get_by_id(run.id)

    assert fetched is not None
    assert fetched.id == run.id
    assert fetched.kind == run.kind
    assert fetched.workspace_id == run.workspace_id
    assert fetched.requested_by == run.requested_by
    assert fetched.cache_key == run.cache_key
    assert fetched.params == {"foo": "bar"}
    assert fetched.protocol_id == run.protocol_id
    assert fetched.status == run.status
    assert fetched.progress == run.progress
    assert fetched.phase == run.phase
    assert fetched.result_uri == run.result_uri
    assert fetched.error_message == run.error_message
    assert fetched.version == run.version
    await client.aclose()


async def test_get_by_id_of_an_unknown_run_is_none(anonymous_client, app):
    _, headers = await register_runner(app, ["default"])
    # No claim needed: an id nobody ever claimed 404s before the claim check
    # even runs (see test_runner_protocol.py's own proof of that ordering).
    client = _build_client(app, headers, uuid.uuid4())

    assert await HttpRunRepository(client).get_by_id(uuid.uuid4()) is None
    await client.aclose()


async def test_update_after_start_persists_running_and_bumps_version(
    anonymous_client, app, workspace_id
):
    run = await seed_run(app, workspace_id)
    _, headers = await register_runner(app, ["default"])
    claimed = await claim(anonymous_client, headers)
    client = _build_client(app, headers, run.id)
    runs = HttpRunRepository(client)

    local = await runs.get_by_id(run.id)
    assert local is not None
    starting_version = local.version
    assert starting_version == claimed["run"]["version"]

    local.start()
    await runs.update(local)

    assert local.status is RunStatus.RUNNING
    assert local.version == starting_version + 1

    fetched = await runs.get_by_id(run.id)
    assert fetched is not None
    assert fetched.status is RunStatus.RUNNING
    assert fetched.version == starting_version + 1
    await client.aclose()


async def test_update_with_a_stale_local_copy_raises_the_concurrency_error(
    anonymous_client, app, workspace_id
):
    run = await seed_run(app, workspace_id)
    _, headers = await register_runner(app, ["default"])
    await claim(anonymous_client, headers)
    client = _build_client(app, headers, run.id)
    runs = HttpRunRepository(client)

    fresh = await runs.get_by_id(run.id)
    stale = await runs.get_by_id(run.id)
    assert fresh is not None and stale is not None

    fresh.start()
    await runs.update(fresh)  # wins the race, bumps the row's version

    stale.start()
    with pytest.raises(ConcurrencyConflictError):
        await runs.update(stale)
    await client.aclose()


# --------------------------------------------------------------------------
# HttpBlobStore
# --------------------------------------------------------------------------


@pytest_asyncio.fixture
async def blob_app(tmp_path, _migrated_engine: AsyncEngine):
    """A dedicated app instance for `HttpBlobStore`, not the shared `app`
    fixture: `SyncAsgiTransport` runs each request to completion on its own
    fresh event loop (see its docstring), and an asyncpg connection can only
    ever be driven from the loop that opened it. `app`'s `session_factory`
    hands every route the same *one* pinned connection for the whole test
    (so a savepoint rollback can undo it) -- exactly what breaks the moment a
    request lands on a different loop. Binding straight to the session-scoped
    `NullPool` engine instead means every request opens (and closes) its own
    connection, correctly scoped to whichever loop is asking.

    Trade-off: no automatic per-test rollback, unlike every other test's
    `app` -- this test's writes are its own run/blob rows under a fresh
    random `workspace_id`, and the Postgres testcontainer itself is torn down
    with the session regardless, so those are fine left behind. The one
    exception is the runner it registers: `runners` is instance-level, not
    workspace-scoped, so `cleanup_registered_runners` deletes it explicitly
    -- see that helper's own docstring for why.
    """
    application = create_app()
    container = Container(
        create_container(Settings(blob_base_url=f"file://{tmp_path}", inline_jobs=True))
    )
    container.define(
        async_sessionmaker,
        Singleton(lambda: async_sessionmaker(bind=_migrated_engine, expire_on_commit=False)),
    )
    application.state.container = container
    async with cleanup_registered_runners(_migrated_engine):
        yield application


async def test_blob_put_then_get_round_trips_bytes(blob_app, workspace_id):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=blob_app), base_url="http://testserver"
    ) as anon:
        run = await seed_run(blob_app, workspace_id)
        _, headers = await register_runner(blob_app, ["default"])
        await claim(anon, headers)
        client = _build_client(blob_app, headers, run.id)
        store = HttpBlobStore(client)

        key = f"{workspace_id}/uploads/x.csv"
        uri = store.put_bytes(key, b"hello runner")
        assert isinstance(uri, str)
        assert uri

        assert store.get_bytes(key) == b"hello runner"
        await client.aclose()


async def test_a_missing_blob_is_a_missing_file(blob_app, workspace_id):
    """A best-effort read (`_train_structures`) catches FileNotFoundError, which
    is what the inline store raises; the runner store must raise the same, or the
    read that degrades gracefully inline takes the whole prediction down."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=blob_app), base_url="http://testserver"
    ) as anon:
        run = await seed_run(blob_app, workspace_id)
        _, headers = await register_runner(blob_app, ["default"])
        await claim(anon, headers)
        client = _build_client(blob_app, headers, run.id)
        store = HttpBlobStore(client)

        with pytest.raises(FileNotFoundError):
            store.get_bytes(f"{workspace_id}/protocols/{uuid.uuid4()}/scorecard-inputs.json")
        await client.aclose()


# --------------------------------------------------------------------------
# HttpProtocolRepository
# --------------------------------------------------------------------------


async def test_protocol_add_then_get_roundtrips(anonymous_client, app, workspace_id):
    run = await seed_run(app, workspace_id, kind=RunKind.TRAINING)
    _, headers = await register_runner(app, ["default"])
    await claim(anonymous_client, headers)
    client = _build_client(app, headers, run.id)
    protocols = HttpProtocolRepository(client)
    runs = HttpRunRepository(client)

    protocol = InSilicoProtocol(
        workspace_id=workspace_id,
        name="runner-http-protocol",
        dataset_id=uuid.uuid4(),
        engine_id="ecfp4-randomforest",
        artifact_uri="file:///nowhere/model.joblib",
        readouts=(
            Readout(name="y", type=ReadoutType.NUMERIC, unit=None, direction=None, description=""),
        ),
        conditions={"folds": 5},
    )
    await protocols.add(protocol)

    # GET /runs/{id}/protocol reads the *run's* protocol_id, so link it first --
    # the same two-step the brief calls out. `start()` first: the update route
    # only accepts a mutable status ("running"/"ready"/"failed"), never the
    # "pending" a freshly seeded run carries.
    local = await runs.get_by_id(run.id)
    assert local is not None
    local.start()
    local.link_protocol(protocol.id)
    await runs.update(local)

    fetched = await protocols.get(workspace_id, protocol.id)
    assert fetched is not None
    assert fetched.id == protocol.id
    assert fetched.name == protocol.name
    assert fetched.dataset_id == protocol.dataset_id
    assert fetched.engine_id == protocol.engine_id
    assert fetched.artifact_uri == protocol.artifact_uri
    assert fetched.conditions == {"folds": 5}
    assert len(fetched.readouts) == 1
    assert fetched.readouts[0].name == "y"
    await client.aclose()


async def test_protocol_get_before_the_run_links_one_is_none(anonymous_client, app, workspace_id):
    run = await seed_run(app, workspace_id, kind=RunKind.TRAINING)
    _, headers = await register_runner(app, ["default"])
    await claim(anonymous_client, headers)
    client = _build_client(app, headers, run.id)

    assert await HttpProtocolRepository(client).get(workspace_id, uuid.uuid4()) is None
    await client.aclose()


# --------------------------------------------------------------------------
# HttpDatasetRepository
# --------------------------------------------------------------------------


async def test_dataset_get_returns_the_dataset_a_training_run_carries(
    anonymous_client, app, workspace_id, client, csv_upload
):
    upload_ref = await csv_upload(SOLUBILITY_CSV)
    created = await client.post(
        "/api/v1/datasets",
        json={
            "name": "solubility",
            "upload_ref": upload_ref,
            "structure_column": "smiles",
            "targets": [{"column": "y", "kind": "numeric"}],
            "split": {"strategy": "random", "seed": 1},
        },
    )
    assert created.status_code == 201, created.text
    dataset_id = created.json()["id"]

    run = await seed_run(
        app, workspace_id, kind=RunKind.TRAINING, params={"dataset_id": dataset_id}
    )
    _, headers = await register_runner(app, ["default"])
    await claim(anonymous_client, headers)
    runner_client = _build_client(app, headers, run.id)

    fetched = await HttpDatasetRepository(runner_client).get(workspace_id, uuid.UUID(dataset_id))

    assert fetched is not None
    assert str(fetched.id) == dataset_id
    await runner_client.aclose()


# --------------------------------------------------------------------------
# build_http_ctx -- the drop-in-replacement-for-build_sqlalchemy_ctx contract
# --------------------------------------------------------------------------


async def test_build_http_ctx_assembles_a_working_run_job_ctx(anonymous_client, app, workspace_id):
    run = await seed_run(app, workspace_id, params={"foo": "bar"})
    _, headers = await register_runner(app, ["default"])
    await claim(anonymous_client, headers)

    ctx = build_http_ctx(
        "http://testserver",
        _bearer_token(headers),
        run.id,
        deadline_seconds=42,
        async_transport=httpx.ASGITransport(app=app),
        sync_transport=SyncAsgiTransport(app),
    )

    assert isinstance(ctx["runs"], HttpRunRepository)
    assert isinstance(ctx["datasets"], HttpDatasetRepository)
    assert isinstance(ctx["protocols"], HttpProtocolRepository)
    assert isinstance(ctx["store"], HttpBlobStore)
    assert ctx["job_deadline_seconds"] == 42

    # The ports it wires up actually work, the same way `_load` in
    # `infrastructure.jobs` would call them.
    fetched = await ctx["runs"].get_by_id(run.id)
    assert fetched is not None
    assert fetched.params == {"foo": "bar"}
    await ctx["_client"].aclose()
