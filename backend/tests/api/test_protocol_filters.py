import uuid
from datetime import UTC, datetime, timedelta

import pytest

from daikonstudio.domain.catalog.protocol import InSilicoProtocol, ProtocolStatus
from daikonstudio.domain.catalog.readout import Readout, ReadoutType
from daikonstudio.infrastructure.persistence.sqlalchemy.catalog.repository import (
    SqlAlchemyProtocolRepository,
)


async def test_protocol_filters_search_all_pages_and_preserve_visibility(
    client,
    other_editor_client,
    other_workspace_client,
    session_factory,
    workspace_id,
    client_user_id,
    protocol_access,
):
    response = await client.post("/api/v1/folders", json={"kind": "protocol", "name": "Research"})
    assert response.status_code == 201, response.text
    folder_id = uuid.UUID(response.json()["id"])
    repository = SqlAlchemyProtocolRepository(session_factory)
    other_owner = uuid.uuid4()
    specs = [
        (
            "Solubility reference",
            "logS",
            "ecfp4-xgboost",
            ProtocolStatus.DRAFT,
            folder_id,
            client_user_id,
        ),
        (
            "hERG panel",
            "herg_blocker",
            "chemprop-dmpnn",
            ProtocolStatus.PUBLISHED,
            folder_id,
            other_owner,
        ),
        (
            "Solubility comparison",
            "logS",
            "ecfp4-xgboost",
            ProtocolStatus.DRAFT,
            folder_id,
            client_user_id,
        ),
        ("assay_100%", "pIC50", "ecfp4-xgboost", ProtocolStatus.PUBLISHED, None, client_user_id),
        (
            "Private solubility",
            "logS",
            "ecfp4-xgboost",
            ProtocolStatus.DRAFT,
            folder_id,
            other_owner,
        ),
        ("Unrelated", "Ki", "chemprop-dmpnn", ProtocolStatus.DRAFT, folder_id, client_user_id),
    ]
    protocols = []
    for index, (name, target, engine, status, folder, owner) in enumerate(specs):
        protocol = InSilicoProtocol(
            workspace_id=workspace_id,
            name=name,
            dataset_id=uuid.uuid4(),
            engine_id=engine,
            artifact_uri="memory://filter-test",
            readouts=(
                Readout(
                    name=target,
                    type=ReadoutType.NUMERIC,
                    unit=None,
                    direction=None,
                    description=f"Predicted {target}",
                ),
            ),
            conditions={},
            status=status,
            folder_id=folder,
            created_by=owner,
            created_at=datetime(2026, 10, 1, tzinfo=UTC) + timedelta(days=index),
        )
        await repository.add(protocol)
        await protocol_access.register(protocol)
        protocols.append(protocol)
    foreign = InSilicoProtocol(
        workspace_id=uuid.uuid4(),
        name="Solubility foreign",
        dataset_id=uuid.uuid4(),
        engine_id="ecfp4-xgboost",
        artifact_uri="memory://foreign",
        readouts=protocols[0].readouts,
        conditions={},
        created_by=client_user_id,
    )
    await repository.add(foreign)
    await protocol_access.register(foreign)

    async def ids(http_client=client, **params):
        response = await http_client.get("/api/v1/protocols", params=params)
        assert response.status_code == 200, response.text
        return {item["id"] for item in response.json()["items"]}

    assert await ids(q="  LOGs  ") == {str(protocols[0].id), str(protocols[2].id)}
    assert await ids(q="SOLUBILITY") == {str(protocols[0].id), str(protocols[2].id)}
    assert await ids(q="HERG_BLOCKER") == {str(protocols[1].id)}
    assert await ids(q="%") == {str(protocols[3].id)}
    assert await ids(q="assay_100%") == {str(protocols[3].id)}
    assert await ids(q="missing") == set()
    assert await ids(status="published") == {str(protocols[1].id), str(protocols[3].id)}
    assert await ids(engine_id="chemprop-dmpnn", status="draft") == {str(protocols[5].id)}
    assert await ids(engine_id="unknown") == set()
    assert await ids(folder_id=str(folder_id), q="assay") == set()
    assert await ids(mine="true", status="published") == {str(protocols[3].id)}
    assert await ids(other_editor_client, q="logS", status="draft") == set()
    assert await ids(other_editor_client, q="herg", status="published") == {str(protocols[1].id)}
    assert await ids(other_workspace_client, q="logS") == set()

    params = {
        "q": "logS",
        "engine_id": "ecfp4-xgboost",
        "status": "draft",
        "folder_id": str(folder_id),
        "mine": "true",
        "limit": 1,
    }
    first = await client.get("/api/v1/protocols", params=params)
    assert first.status_code == 200, first.text
    assert [item["id"] for item in first.json()["items"]] == [str(protocols[2].id)]
    assert first.json()["next_cursor"]
    second = await client.get(
        "/api/v1/protocols", params={**params, "cursor": first.json()["next_cursor"]}
    )
    assert second.status_code == 200, second.text
    assert [item["id"] for item in second.json()["items"]] == [str(protocols[0].id)]
    assert second.json()["next_cursor"] is None


@pytest.mark.parametrize(
    "params", [{"status": "unknown"}, {"q": "x" * 257}, {"engine_id": "x" * 257}]
)
async def test_protocol_filters_reject_invalid_parameters(client, params):
    response = await client.get("/api/v1/protocols", params=params)
    assert response.status_code == 422, response.text
