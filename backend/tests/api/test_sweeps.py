"""The sweeps HTTP contract."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from lagom import Container, Singleton
from sqlalchemy.ext.asyncio import async_sessionmaker

from daikonstudio.infrastructure.di.container import create_container
from daikonstudio.interface.app import create_app
from daikonstudio.settings import Settings


@pytest.fixture
def app(tmp_path, session_factory):
    """Overrides `tests.api.conftest.app`: `inline_jobs=False` so a submitted
    sweep's runs stay `pending` instead of each executing a real fit -- this
    file is testing the endpoint's wiring (status codes, shapes, grouping),
    not that N models can finish training inside one request. Same
    `session_factory` savepoint recipe as the fixture it replaces; no runner
    protocol is involved here, so there's no cross-event-loop concern like
    `test_runner_full_loop.py`'s NullPool override has.
    """
    application = create_app()
    container = Container(
        create_container(Settings(blob_base_url=f"file://{tmp_path}", inline_jobs=False))
    )
    container.define(async_sessionmaker, Singleton(lambda: session_factory))
    application.state.container = container
    return application


# Twenty distinct compounds -- same shape `test_protocols.py` uses: an 80/10/10
# random split over four rows can leave a single-row test partition, which is
# a degenerate frame `CreateDataset` rejects.
_STRUCTURES = (
    "CCO",
    "CCN",
    "CCCO",
    "CCCCO",
    "CCCCCO",
    "c1ccccc1",
    "Cc1ccccc1",
    "c1ccncc1",
    "c1ccsc1",
    "c1cc[nH]c1",
    "C1CCCCC1",
    "C1CCNCC1",
    "C1CCOC1",
    "C1CCCC1",
    "C1CC1",
    "c1ccc2ccccc2c1",
    "c1ccc2[nH]ccc2c1",
    "C1CCC2CCCCC2C1",
    "c1cnc2ccccc2c1",
    "O=C1CCCCC1",
)


def _csv() -> bytes:
    rows = "\n".join(f"{smiles},{1.0 + 0.37 * index}" for index, smiles in enumerate(_STRUCTURES))
    return f"smiles,y\n{rows}\n".encode()


@pytest_asyncio.fixture
async def dataset_id(client, csv_upload) -> str:
    upload_ref = await csv_upload(_csv())
    response = await client.post(
        "/api/v1/datasets",
        json={
            "name": "solubility",
            "upload_ref": upload_ref,
            "structure_column": "smiles",
            "target": {"column": "y", "kind": "numeric", "unit": "logS", "direction": "high"},
            "split": {"strategy": "random", "seed": 1},
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


@pytest.mark.asyncio
async def test_submit_returns_202_with_every_run(client, dataset_id) -> None:
    response = await client.post(
        "/api/v1/sweeps",
        json={
            "name": "BBBP comparison",
            "dataset_id": str(dataset_id),
            "configs": [
                {"engine_id": "ecfp4-randomforest", "conditions": {}},
                {"engine_id": "ecfp4-xgboost", "conditions": {}},
            ],
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert len(body["runs"]) == 2
    assert body["name"] == "BBBP comparison"
    assert all(run["sweep_id"] == body["sweep_id"] for run in body["runs"])
    assert body["runs"][0]["engine_id"] == "ecfp4-randomforest"


@pytest.mark.asyncio
async def test_unknown_engine_is_404_and_creates_nothing(client, dataset_id) -> None:
    response = await client.post(
        "/api/v1/sweeps",
        json={
            "name": "doomed",
            "dataset_id": str(dataset_id),
            "configs": [{"engine_id": "no-such-engine", "conditions": {}}],
        },
    )
    assert response.status_code == 404
    assert (await client.get("/api/v1/sweeps")).json()["items"] == []


@pytest.mark.asyncio
async def test_empty_configs_is_422(client, dataset_id) -> None:
    response = await client.post(
        "/api/v1/sweeps",
        json={"name": "empty", "dataset_id": str(dataset_id), "configs": []},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_then_read_then_cancel(client, dataset_id) -> None:
    submitted = (
        await client.post(
            "/api/v1/sweeps",
            json={
                "name": "round trip",
                "dataset_id": str(dataset_id),
                "configs": [{"engine_id": "ecfp4-randomforest", "conditions": {}}],
            },
        )
    ).json()
    sweep_id = submitted["sweep_id"]

    listed = (await client.get("/api/v1/sweeps")).json()["items"]
    assert [item["sweep_id"] for item in listed] == [sweep_id]
    assert listed[0]["total"] == 1

    detail = (await client.get(f"/api/v1/sweeps/{sweep_id}")).json()
    assert len(detail["runs"]) == 1

    assert (await client.post(f"/api/v1/sweeps/{sweep_id}/cancel")).status_code == 204
    after = (await client.get(f"/api/v1/sweeps/{sweep_id}")).json()
    assert after["runs"][0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_unknown_sweep_is_404(client) -> None:
    assert (await client.get(f"/api/v1/sweeps/{uuid.uuid4()}")).status_code == 404
