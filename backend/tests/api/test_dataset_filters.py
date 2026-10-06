import uuid
from datetime import UTC, datetime, timedelta

import pytest

from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.data.target import TargetKind, TargetSpec
from daikonstudio.domain.data.validation import ValidationReport
from daikonstudio.infrastructure.persistence.sqlalchemy.data.repository import (
    SqlAlchemyDatasetRepository,
)


async def test_dataset_search_and_filters_apply_before_pagination_and_stay_workspace_scoped(
    client, other_workspace_client, session_factory, workspace_id
):
    created_folder = await client.post(
        "/api/v1/folders", json={"kind": "dataset", "name": "Research"}
    )
    assert created_folder.status_code == 201, created_folder.text
    folder_id = uuid.UUID(created_folder.json()["id"])
    repository = SqlAlchemyDatasetRepository(session_factory)
    specs = [
        (
            "Solubility reference",
            [("logS", TargetKind.NUMERIC)],
            SplitStrategy.SCAFFOLD,
            folder_id,
        ),
        ("hERG panel", [("herg_blocker", TargetKind.BINARY)], SplitStrategy.RANDOM, folder_id),
        (
            "Mixed screen",
            [("logS", TargetKind.NUMERIC), ("reactive", TargetKind.BINARY)],
            SplitStrategy.SCAFFOLD,
            folder_id,
        ),
        ("assay_100%", [("pIC50", TargetKind.NUMERIC)], SplitStrategy.RANDOM, None),
        ("Unrelated", [("Ki", TargetKind.NUMERIC)], SplitStrategy.RANDOM, folder_id),
    ]
    datasets = []
    for index, (name, targets, strategy, folder) in enumerate(specs):
        dataset = Dataset(
            workspace_id=workspace_id,
            name=name,
            structure_column="smiles",
            targets=tuple(TargetSpec(column=column, kind=kind) for column, kind in targets),
            split=SplitSpec(strategy=strategy, seed=42),
            content_hash=str(uuid.uuid4()),
            snapshot_uri="memory://filter-test",
            row_count=10,
            validation_report=ValidationReport(total_rows=10, valid_rows=10),
            folder_id=folder,
            created_at=datetime(2026, 10, 1, tzinfo=UTC) + timedelta(days=index),
        )
        await repository.add(dataset)
        datasets.append(dataset)
    await repository.add(
        Dataset(
            workspace_id=uuid.uuid4(),
            name="hERG private",
            structure_column="smiles",
            targets=(TargetSpec(column="herg_blocker", kind=TargetKind.BINARY),),
            split=SplitSpec(strategy=SplitStrategy.RANDOM, seed=42),
            content_hash=str(uuid.uuid4()),
            snapshot_uri="memory://foreign",
            row_count=10,
            validation_report=ValidationReport(total_rows=10, valid_rows=10),
        )
    )

    async def ids(**params):
        response = await client.get("/api/v1/datasets", params=params)
        assert response.status_code == 200, response.text
        return {item["id"] for item in response.json()["items"]}

    assert await ids(q="  LOGs  ") == {str(datasets[0].id), str(datasets[2].id)}
    assert await ids(q="HERG") == {str(datasets[1].id)}
    assert await ids(q="%") == {str(datasets[3].id)}
    assert await ids(q="assay_100%") == {str(datasets[3].id)}
    assert await ids(q="missing") == set()
    assert await ids(target_kind="binary") == {str(datasets[1].id), str(datasets[2].id)}
    assert await ids(target_kind="numeric") == {
        str(dataset.id) for index, dataset in enumerate(datasets) if index != 1
    }
    assert await ids(split_strategy="scaffold", target_kind="binary") == {str(datasets[2].id)}
    assert await ids(folder_id=str(folder_id), q="assay") == set()

    params = {
        "q": "logS",
        "target_kind": "numeric",
        "split_strategy": "scaffold",
        "folder_id": str(folder_id),
        "limit": 1,
    }
    first = await client.get("/api/v1/datasets", params=params)
    assert first.status_code == 200, first.text
    assert [item["id"] for item in first.json()["items"]] == [str(datasets[2].id)]
    assert first.json()["next_cursor"]
    second = await client.get(
        "/api/v1/datasets", params={**params, "cursor": first.json()["next_cursor"]}
    )
    assert second.status_code == 200, second.text
    assert [item["id"] for item in second.json()["items"]] == [str(datasets[0].id)]
    assert second.json()["next_cursor"] is None
    foreign = await other_workspace_client.get("/api/v1/datasets", params={"q": "logS"})
    assert foreign.status_code == 200, foreign.text
    assert foreign.json()["items"] == []


@pytest.mark.parametrize(
    "params", [{"target_kind": "unknown"}, {"split_strategy": "unknown"}, {"q": "x" * 257}]
)
async def test_dataset_filters_reject_invalid_parameters(client, params):
    response = await client.get("/api/v1/datasets", params=params)
    assert response.status_code == 422, response.text
