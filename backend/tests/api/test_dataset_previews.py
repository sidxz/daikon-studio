"""Review does not create a dataset; freezing uses the exact reviewed bytes."""

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta

from tests.api.test_datasets import SOLUBILITY_CSV, create_body

from daikonstudio.application.data import preview_dataset
from daikonstudio.application.data.create_dataset import upload_key
from daikonstudio.application.data.preview_dataset import preview_key
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_repository import DatasetRepository


async def prepare(client, csv_upload, data=SOLUBILITY_CSV):
    ref = await csv_upload(data)
    response = await client.post("/api/v1/datasets/previews", json=create_body(ref))
    assert response.status_code == 202, response.text
    build_id = response.json()["id"]
    await asyncio.gather(*preview_dataset._running)
    finished = await client.get(f"/api/v1/datasets/previews/{build_id}")
    assert finished.status_code == 200, finished.text
    return ref, finished.json()


async def test_review_prepares_exact_counts_without_creating_a_dataset(
    app, client, csv_upload, workspace_id
):
    _, preview = await prepare(client, csv_upload)
    assert preview["status"] == "succeeded", preview
    assert preview["dataset_id"] is None
    readiness = preview["preparation"]["readiness"]
    assert readiness["row_count"] == 4
    assert sum(readiness["partition_counts"].values()) == 4
    repository = app.state.container[DatasetRepository]
    assert await repository.get(workspace_id, uuid.UUID(preview["id"])) is None
    assert (await client.get("/api/v1/datasets")).json()["items"] == []


async def test_freeze_uses_reviewed_bytes_even_if_the_upload_changes(
    app, client, csv_upload, workspace_id
):
    ref, preview = await prepare(client, csv_upload)
    store = app.state.container[BlobStore]
    store.put_bytes(upload_key(workspace_id, uuid.UUID(ref)), b"smiles,y\ninvalid,99\n")
    response = await client.post(
        f"/api/v1/datasets/previews/{preview['id']}/freeze", json={"name": "Reviewed dataset"}
    )
    assert response.status_code == 201, response.text
    dataset = response.json()
    assert dataset["id"] == preview["id"]
    assert dataset["name"] == "Reviewed dataset"
    assert dataset["row_count"] == 4
    assert dataset["validation_report"] == preview["preparation"]["validation_report"]
    readiness = await client.get(f"/api/v1/datasets/{dataset['id']}/readiness")
    assert readiness.status_code == 200, readiness.text
    assert readiness.json() == preview["preparation"]["readiness"]
    repeated = await client.post(
        f"/api/v1/datasets/previews/{preview['id']}/freeze", json={"name": "Reviewed dataset"}
    )
    assert repeated.status_code == 201
    assert repeated.json()["id"] == dataset["id"]
    assert len((await client.get("/api/v1/datasets")).json()["items"]) == 1


async def test_failed_review_preserves_the_report_and_creates_nothing(client, csv_upload):
    _, preview = await prepare(client, csv_upload, b"smiles,y\ninvalid,1\nalso-invalid,2\n")
    assert preview["status"] == "failed"
    assert preview["preparation"] is None
    assert preview["error"]["detail"]["total_rows"] == 2
    response = await client.post(
        f"/api/v1/datasets/previews/{preview['id']}/freeze", json={"name": "No dataset"}
    )
    assert response.status_code == 409
    assert (await client.get("/api/v1/datasets")).json()["items"] == []


async def test_reviews_and_freeze_are_scoped_to_the_workspace(
    client, other_workspace_client, csv_upload
):
    _, preview = await prepare(client, csv_upload)
    path = f"/api/v1/datasets/previews/{preview['id']}"
    assert (await other_workspace_client.get(path)).status_code == 404
    assert (
        await other_workspace_client.post(f"{path}/freeze", json={"name": "Other"})
    ).status_code == 404


async def test_viewers_cannot_prepare_or_freeze(client, viewer_client, csv_upload):
    ref, preview = await prepare(client, csv_upload)
    assert (
        await viewer_client.post("/api/v1/datasets/previews", json=create_body(ref))
    ).status_code == 403
    assert (
        await viewer_client.post(
            f"/api/v1/datasets/previews/{preview['id']}/freeze", json={"name": "Viewer"}
        )
    ).status_code == 403


async def test_expired_review_cannot_be_frozen(app, client, csv_upload, workspace_id):
    _, preview = await prepare(client, csv_upload)
    store = app.state.container[BlobStore]
    key = preview_key(workspace_id, uuid.UUID(preview["id"]), "manifest.json")
    manifest = json.loads(store.get_bytes(key))
    manifest["expires_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    store.put_bytes(key, json.dumps(manifest).encode())
    path = f"/api/v1/datasets/previews/{preview['id']}"
    assert (await client.get(path)).status_code == 409
    assert (await client.post(f"{path}/freeze", json={"name": "Expired"})).status_code == 409


async def test_freeze_cannot_change_the_reviewed_specification(client, csv_upload):
    _, preview = await prepare(client, csv_upload)
    path = f"/api/v1/datasets/previews/{preview['id']}/freeze"
    response = await client.post(path, json={"name": "Changed", "split": {"strategy": "scaffold"}})
    assert response.status_code == 422
    assert (await client.post(path, json={"name": "   "})).status_code == 422


async def test_identical_reviews_do_not_create_duplicate_datasets(client, csv_upload):
    _, first = await prepare(client, csv_upload)
    _, second = await prepare(client, csv_upload)
    assert (
        await client.post(
            f"/api/v1/datasets/previews/{first['id']}/freeze", json={"name": "First"}
        )
    ).status_code == 201
    duplicate = await client.post(
        f"/api/v1/datasets/previews/{second['id']}/freeze", json={"name": "Second"}
    )
    assert duplicate.status_code == 409


async def test_existing_dataset_readiness_is_cached_and_scoped(
    app, client, other_workspace_client, csv_upload, workspace_id, monkeypatch
):
    ref = await csv_upload(SOLUBILITY_CSV)
    response = await client.post("/api/v1/datasets", json=create_body(ref))
    assert response.status_code == 201
    dataset_id = response.json()["id"]
    path = f"/api/v1/datasets/{dataset_id}/readiness"
    first = await client.get(path)
    assert first.status_code == 200
    store = app.state.container[BlobStore]
    original = store.get_bytes
    read_keys = []

    def record(key):
        read_keys.append(key)
        return original(key)

    monkeypatch.setattr(store, "get_bytes", record)
    second = await client.get(path)
    assert second.json() == first.json()
    assert not any(key.endswith("snapshot.parquet") for key in read_keys)
    assert (await other_workspace_client.get(path)).status_code == 404


async def test_all_conflicting_compounds_return_a_report_instead_of_a_split_error(
    client, csv_upload
):
    ref = await csv_upload(b"smiles,y\nCCO,0\nCCO,1\nCCN,0\nCCN,1\n")
    response = await client.post(
        "/api/v1/datasets/previews",
        json=create_body(ref, targets=[{"column": "y", "kind": "binary"}]),
    )
    assert response.status_code == 202
    await asyncio.gather(*preview_dataset._running)
    reviewed = (await client.get(f"/api/v1/datasets/previews/{response.json()['id']}")).json()
    assert reviewed["status"] == "failed"
    assert reviewed["error"]["error"] == "InvalidDatasetError"
    assert len(reviewed["error"]["detail"]["conflicting"]) == 2
