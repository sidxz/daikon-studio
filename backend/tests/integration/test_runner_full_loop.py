"""The proof of the whole self-hosted-runners phase (Task 10): a training Run
submitted through the ordinary human-facing API executes to completion with
*no* handler code changed -- only claimed and driven over the HTTP runner
protocol (`build_http_ctx` + `jobs.run_job`, Task 8's ports) instead of the
in-process `InlineEnqueuer` every other integration test relies on.

Uses its own `app` fixture, `inline_jobs=False`, rather than the shared one
`tests/integration/conftest.py` re-exports from `tests/api/conftest.py`: a
module-level fixture of the same name overrides the conftest one for every
test in this file (documented pytest fixture-override behaviour), so
`client`/`anonymous_client`/`csv_upload` -- all defined elsewhere but
parameterised on `app` -- pick up this one too, with no need to redefine any
of them here.

Bound to the session-scoped `NullPool` engine, not a savepoint-pinned
per-test connection -- same reason `test_runner_ports.py`'s `blob_app`
fixture gives: `SyncAsgiTransport` (used by the runner's blob traffic) runs
each request to completion on its own fresh event loop, and an asyncpg
connection can only ever be driven from the loop that opened it. The shared
`app` fixture hands every route the same *one* pinned connection for the
whole test so a savepoint rollback can undo it -- exactly what breaks the
moment a request lands on a different loop.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import httpx
import pytest_asyncio
from lagom import Container, Singleton
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from daikonstudio.infrastructure.di.container import create_container
from daikonstudio.infrastructure.jobs import run_job
from daikonstudio.infrastructure.runner.ports import build_http_ctx
from daikonstudio.interface.app import create_app
from daikonstudio.settings import Settings
from tests.helpers.runner_fixtures import claim, cleanup_registered_runners
from tests.helpers.sync_asgi import SyncAsgiTransport

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "pains_sample.csv"


@pytest_asyncio.fixture
async def app(tmp_path, _migrated_engine: AsyncEngine):
    """Overrides `tests.api.conftest.app` for this module -- `inline_jobs=False`
    so `POST /api/v1/protocols` only enqueues (leaves the Run `pending`); this
    test drives it to completion itself, over the runner protocol.
    """
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
    application.state.container = container

    # `runners` is instance-level, not workspace-scoped, so this fixture's
    # NullPool trade-off (see the module docstring) leaves the one runner it
    # registers behind unless cleaned up explicitly -- see
    # `cleanup_registered_runners`'s own docstring for why that matters.
    async with cleanup_registered_runners(_migrated_engine):
        yield application


async def test_a_training_run_executes_through_the_runner_protocol_unmodified(
    client, anonymous_client, csv_upload, app
):
    # --- Dataset: the same upload + create calls test_full_loop.py makes ---
    upload_ref = await csv_upload(_FIXTURE.read_bytes())
    dataset_response = await client.post(
        "/api/v1/datasets",
        json={
            "name": "PAINS",
            "upload_ref": upload_ref,
            "structure_column": "smiles",
            "target": {"column": "is_pains", "kind": "binary"},
            "split": {"strategy": "scaffold", "seed": 42},
        },
    )
    assert dataset_response.status_code == 201, f"[dataset] {dataset_response.text}"
    dataset_id = dataset_response.json()["id"]

    # --- Training: submitted, but nothing executes it -- no runner has claimed ---
    train_response = await client.post(
        "/api/v1/protocols",
        json={
            "name": "PAINS xgb via runner",
            "dataset_id": dataset_id,
            "engine_id": "ecfp4-xgboost",
            "conditions": {},
        },
    )
    assert train_response.status_code == 202, f"[training] {train_response.text}"
    run_id = train_response.json()["id"]

    pending = await client.get(f"/api/v1/runs/{run_id}")
    assert pending.status_code == 200, f"[training] {pending.text}"
    assert pending.json()["status"] == "pending", (
        f"[training] expected nothing to execute inline: {pending.json()}"
    )

    # --- Runner: register over the real endpoint, grab its bearer token ---
    runner_response = await client.post(
        "/api/v1/runners", json={"name": f"e2e-runner-{uuid.uuid4()}", "lanes": ["default"]}
    )
    assert runner_response.status_code == 201, f"[runner] {runner_response.text}"
    token = runner_response.json()["token"]
    runner_headers = {"Authorization": f"Bearer {token}"}

    # --- Claim + execute: exactly what `agent.poll_once` does, over test transports ---
    claimed = await claim(anonymous_client, runner_headers)
    assert claimed["run"]["id"] == run_id, f"[claim] {claimed}"

    ctx = build_http_ctx(
        "http://testserver",
        token,
        uuid.UUID(run_id),
        deadline_seconds=claimed["deadline_seconds"],
        async_transport=httpx.ASGITransport(app=app),
        sync_transport=SyncAsgiTransport(app),
    )
    try:
        await run_job(ctx, uuid.UUID(run_id))
    finally:
        await ctx["_client"].aclose()

    # --- Assert via the USER api: the runner's work is visible server-side ---
    finished = await client.get(f"/api/v1/runs/{run_id}")
    assert finished.status_code == 200, f"[result] {finished.text}"
    run = finished.json()
    assert run["status"] == "ready", f"[result] run failed: {run.get('error_message')}"
    assert run["protocol_id"] is not None, f"[result] no protocol linked: {run}"

    scorecard_response = await client.get(f"/api/v1/protocols/{run['protocol_id']}/scorecard")
    assert scorecard_response.status_code == 200, f"[scorecard] {scorecard_response.text}"
    card = scorecard_response.json()
    assert card["primary_metric"] == "mcc", card
    assert card["baseline_metrics"]["mcc"] is not None, f"[scorecard] baseline undefined: {card}"

    # --- Queue drained: a second claim finds nothing left ---
    second_claim = await anonymous_client.post("/api/v1/runner/claim", headers=runner_headers)
    assert second_claim.status_code == 204, f"[claim] {second_claim.text}"
