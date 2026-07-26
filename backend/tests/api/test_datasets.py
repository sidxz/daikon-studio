"""End-to-end tests for the dataset feature: upload a CSV, get back a frozen,
validated, split, content-addressed Dataset -- or a rejection that says why."""

from __future__ import annotations

SOLUBILITY_CSV = b"smiles,y\nCCO,1.0\nc1ccccc1,5.0\nCCN,2.0\nc1ccncc1,6.0\n"

# Nine benzene analogues and one piperidine: one scaffold family owns 90% of the
# rows, so a scaffold split cannot honour an 80/10/10 request without straddling
# it. Task 10 fails loudly on that rather than returning an untrainable dataset.
CONGENERIC_CSV = (
    b"smiles,y\n"
    b"c1ccccc1,1.0\nCc1ccccc1,1.1\nCCc1ccccc1,1.2\nCCCc1ccccc1,1.3\nCCCCc1ccccc1,1.4\n"
    b"CCCCCc1ccccc1,1.5\nCCCCCCc1ccccc1,1.6\nFc1ccccc1,1.7\nClc1ccccc1,1.8\nC1CCNCC1,9.0\n"
)

NUMERIC_TARGET = {"column": "y", "kind": "numeric"}
RANDOM_SPLIT = {"strategy": "random", "seed": 1}


def create_body(upload_ref: str, **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "name": "solubility",
        "upload_ref": upload_ref,
        "structure_column": "smiles",
        "target": NUMERIC_TARGET,
        "split": RANDOM_SPLIT,
    }
    body.update(overrides)
    return body


async def test_create_dataset_returns_201_with_validation_report(client, csv_upload):
    upload_ref = await csv_upload(SOLUBILITY_CSV)
    response = await client.post(
        "/api/v1/datasets",
        json=create_body(
            upload_ref,
            target={"column": "y", "kind": "numeric", "unit": "logS", "direction": "high"},
            split={"strategy": "scaffold", "seed": 42},
        ),
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["row_count"] == 4
    assert body["validation_report"]["invalid"] == []
    assert body["validation_report"]["total_rows"] == 4
    assert body["content_hash"]
    assert body["snapshot_uri"]
    assert body["version"] == 1


async def test_create_dataset_rejects_a_frame_with_no_valid_structures(client, csv_upload):
    """The 422 carries the whole report, not a bare message: which rows failed and
    why is the useful part of a rejection."""
    upload_ref = await csv_upload(b"smiles,y\nnope,1.0\nalso-nope,2.0\n")
    response = await client.post("/api/v1/datasets", json=create_body(upload_ref, name="broken"))
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert len(detail["invalid"]) == 2
    assert detail["invalid"][0] == {
        "row_number": 1,
        "value": "nope",
        "reason": "invalid structure",
    }
    assert detail["total_rows"] == 2
    assert detail["valid_rows"] == 0


async def test_validation_report_records_collapsed_duplicates_and_assay_spread(client, csv_upload):
    """Two SMILES for the same molecule collapse into one row, and the spread
    between the replicate measurements survives into the stored report."""
    upload_ref = await csv_upload(b"smiles,y\nCCO,1.0\nOCC,3.0\nc1ccccc1,5.0\nCCN,2.0\n")
    response = await client.post("/api/v1/datasets", json=create_body(upload_ref))
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["row_count"] == 3
    assert body["validation_report"]["duplicates_collapsed"] == 1
    assert body["validation_report"]["duplicate_spread"] == 2.0


async def test_dataset_is_scoped_to_the_callers_workspace(
    client, other_workspace_client, csv_upload
):
    upload_ref = await csv_upload(SOLUBILITY_CSV)
    created = (
        await client.post("/api/v1/datasets", json=create_body(upload_ref, name="private"))
    ).json()
    assert (await other_workspace_client.get(f"/api/v1/datasets/{created['id']}")).status_code == (
        404
    )
    assert (await client.get(f"/api/v1/datasets/{created['id']}")).status_code == 200


async def test_list_datasets_never_leaks_another_workspaces_rows(
    client, other_workspace_client, csv_upload
):
    upload_ref = await csv_upload(SOLUBILITY_CSV)
    await client.post("/api/v1/datasets", json=create_body(upload_ref, name="private"))

    mine = (await client.get("/api/v1/datasets")).json()
    theirs = (await other_workspace_client.get("/api/v1/datasets")).json()
    assert [item["name"] for item in mine["items"]] == ["private"]
    assert theirs["items"] == []


async def test_workspace_id_comes_from_the_token_not_the_body(client, csv_upload):
    """Supplying a workspace_id is not silently ignored -- it is refused, so no
    client can ever come to believe it works."""
    upload_ref = await csv_upload(SOLUBILITY_CSV)
    response = await client.post(
        "/api/v1/datasets",
        json=create_body(upload_ref, workspace_id="00000000-0000-0000-0000-000000000001"),
    )
    assert response.status_code == 422, response.text


async def test_upload_ref_from_another_workspace_is_not_readable(
    client, other_workspace_client, csv_upload
):
    """An upload_ref is resolved under the caller's own workspace prefix, so a
    stolen ref reads nothing."""
    upload_ref = await csv_upload(SOLUBILITY_CSV)
    response = await other_workspace_client.post("/api/v1/datasets", json=create_body(upload_ref))
    assert response.status_code == 404, response.text


async def test_upload_ref_cannot_escape_the_workspace_prefix(client):
    response = await client.post("/api/v1/datasets", json=create_body("../../etc/passwd"))
    assert response.status_code == 422, response.text


async def test_re_uploading_identical_data_conflicts_and_names_the_existing_dataset(
    client, csv_upload
):
    """Content addressing means identical data is one Dataset. A second upload is
    refused with the id of the one that already holds it, so a browser refresh is
    recoverable without silently discarding the new name/units."""
    first_ref = await csv_upload(SOLUBILITY_CSV)
    first = await client.post("/api/v1/datasets", json=create_body(first_ref))
    assert first.status_code == 201, first.text

    second_ref = await csv_upload(SOLUBILITY_CSV)
    second = await client.post(
        "/api/v1/datasets", json=create_body(second_ref, name="solubility-again")
    )
    assert second.status_code == 409, second.text
    assert second.json()["existing_dataset_id"] == first.json()["id"]


async def test_identical_data_in_a_different_workspace_is_not_a_conflict(
    client, other_workspace_client, csv_upload
):
    """The uniqueness is per workspace: two tenants uploading the same public
    dataset must not collide."""
    mine = await client.post(
        "/api/v1/datasets", json=create_body(await csv_upload(SOLUBILITY_CSV))
    )
    assert mine.status_code == 201, mine.text

    upload = await other_workspace_client.post(
        "/api/v1/datasets/uploads", files={"file": ("data.csv", SOLUBILITY_CSV, "text/csv")}
    )
    theirs = await other_workspace_client.post(
        "/api/v1/datasets", json=create_body(upload.json()["upload_ref"])
    )
    assert theirs.status_code == 201, theirs.text
    assert theirs.json()["content_hash"] == mine.json()["content_hash"]


async def test_scaffold_split_failure_reaches_the_scientist_intact(client, csv_upload):
    """A congeneric series cannot be scaffold-split 80/10/10. That is a real
    outcome, and the message telling the scientist what to do instead must
    survive the trip to the client."""
    upload_ref = await csv_upload(CONGENERIC_CSV)
    response = await client.post(
        "/api/v1/datasets",
        json=create_body(upload_ref, split={"strategy": "scaffold", "seed": 7}),
    )
    assert response.status_code == 422, response.text
    message = response.json()["message"]
    assert "RANDOM" in message
    assert "scaffold family" in message


async def test_missing_target_column_is_rejected(client, csv_upload):
    upload_ref = await csv_upload(SOLUBILITY_CSV)
    response = await client.post(
        "/api/v1/datasets",
        json=create_body(upload_ref, target={"column": "potency", "kind": "numeric"}),
    )
    assert response.status_code == 422, response.text
    assert "potency" in response.json()["message"]


async def test_unknown_upload_ref_is_a_404(client):
    response = await client.post(
        "/api/v1/datasets", json=create_body("11111111-1111-1111-1111-111111111111")
    )
    assert response.status_code == 404, response.text


async def test_viewer_cannot_create_a_dataset(viewer_client, csv_upload):
    upload_ref = await csv_upload(SOLUBILITY_CSV)
    response = await viewer_client.post("/api/v1/datasets", json=create_body(upload_ref))
    assert response.status_code == 403, response.text


async def test_unauthenticated_request_is_rejected(anonymous_client):
    """The real AuthzMiddleware, not a stub, is what turns this away."""
    assert (await anonymous_client.get("/api/v1/datasets")).status_code == 401
    assert (await anonymous_client.post("/api/v1/datasets", json={})).status_code == 401
    assert (await anonymous_client.post("/api/v1/datasets/uploads")).status_code == 401


async def test_a_dataset_survives_the_round_trip_to_the_database(client, csv_upload):
    """GET reads back what POST wrote -- the report, the specs and the hash all
    round-trip through JSONB rather than being reconstructed from the request."""
    upload_ref = await csv_upload(SOLUBILITY_CSV)
    created = (
        await client.post(
            "/api/v1/datasets",
            json=create_body(
                upload_ref,
                target={"column": "y", "kind": "numeric", "unit": "logS", "direction": "high"},
                split={"strategy": "random", "seed": 99, "fractions": [0.6, 0.2, 0.2]},
            ),
        )
    ).json()

    fetched = (await client.get(f"/api/v1/datasets/{created['id']}")).json()
    assert fetched == created
    assert fetched["target"] == {
        "column": "y",
        "kind": "numeric",
        "unit": "logS",
        "direction": "high",
    }
    assert fetched["split"] == {"strategy": "random", "seed": 99, "fractions": [0.6, 0.2, 0.2]}
