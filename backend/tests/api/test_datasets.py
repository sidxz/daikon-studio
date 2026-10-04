"""End-to-end tests for the dataset feature: upload a CSV, get back a frozen,
validated, split, content-addressed Dataset -- or a rejection that says why."""

from __future__ import annotations

import re
import uuid

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
        "targets": [NUMERIC_TARGET],
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
            targets=[{"column": "y", "kind": "numeric", "unit": "logS", "direction": "high"}],
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
        "reason": "SMILES could not be parsed",
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
    assert body["validation_report"]["duplicate_spread"] == {"y": 2.0}


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


async def test_paging_through_more_datasets_than_the_limit_terminates(client, csv_upload):
    """The regression the cursor bug hid: nothing previously asked for a second
    page, so `next_cursor` was `None` in every test and the round trip was never
    exercised at all.

    The cursor is interpolated straight into the query string rather than handed
    to httpx's `params=`, deliberately. `params=` percent-encodes, which would
    paper over the actual defect -- a `+` in an ISO timestamp arriving as a space,
    failing to parse, and silently restarting the listing. Interpolating is what a
    client following `next_cursor` from a JSON body does.
    """
    created = []
    for index in range(5):
        csv = f"smiles,y\nCCO,{index}.0\nc1ccccc1,5.0\nCCN,2.0\nc1ccncc1,6.0\n".encode()
        response = await client.post(
            "/api/v1/datasets", json=create_body(await csv_upload(csv), name=f"dataset-{index}")
        )
        assert response.status_code == 201, response.text
        created.append(response.json()["id"])

    seen: list[str] = []
    cursor: str | None = None
    pages = 0
    while True:
        url = "/api/v1/datasets?limit=2"
        if cursor is not None:
            url += f"&cursor={cursor}"
        response = await client.get(url)
        assert response.status_code == 200, response.text
        page = response.json()
        pages += 1
        seen.extend(item["id"] for item in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
        # The property that keeps the token intact across a query string: an
        # alphabet with nothing for a parser to reinterpret.
        assert re.fullmatch(r"[A-Za-z0-9_=-]+", cursor), cursor
        assert pages < 10, f"cursor pagination did not terminate: {pages} pages, saw {seen}"

    assert pages == 3
    assert len(seen) == 5
    assert len(set(seen)) == 5
    assert set(seen) == set(created)


async def test_a_malformed_cursor_is_rejected_rather_than_restarting_the_listing(client):
    """Silently serving page one in answer to a corrupt cursor is how the
    infinite loop hid. The second case is the exact string a query-string parser
    produced from the old unencoded cursor: the `+` arrived as a space."""
    for bad_cursor in ("not-a-real-cursor", f"2026-07-26T02:28:32.079357 00:00|{uuid.uuid4()}"):
        response = await client.get("/api/v1/datasets", params={"cursor": bad_cursor})
        assert response.status_code == 422, response.text
        assert response.json()["message"] == "Invalid pagination cursor"


async def test_the_split_is_inside_the_content_hash(client, csv_upload):
    """The same data under a different seed is a different hash and a separate
    Dataset -- the unique index enforces "same data, split the same way", not
    "same data". Pinned because later tasks cite Datasets for reproducibility."""
    first = await client.post(
        "/api/v1/datasets",
        json=create_body(
            await csv_upload(SOLUBILITY_CSV), split={"strategy": "random", "seed": 1}
        ),
    )
    second = await client.post(
        "/api/v1/datasets",
        json=create_body(
            await csv_upload(SOLUBILITY_CSV), split={"strategy": "random", "seed": 2}
        ),
    )
    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text
    assert first.json()["content_hash"] != second.json()["content_hash"]


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
    assert "random split" in message
    assert "Bemis–Murcko scaffold" in message  # noqa: RUF001


async def test_missing_target_column_is_rejected(client, csv_upload):
    upload_ref = await csv_upload(SOLUBILITY_CSV)
    response = await client.post(
        "/api/v1/datasets",
        json=create_body(upload_ref, targets=[{"column": "potency", "kind": "numeric"}]),
    )
    assert response.status_code == 422, response.text
    assert "potency" in response.json()["message"]


async def test_a_reserved_target_column_name_is_rejected(client, csv_upload):
    """C1 (whole-branch review, Critical): a target named `uncertainty`
    means `predict_with_protocol.py`'s own ensemble-spread column silently
    overwrites the served prediction; a target named `structure` overwrites
    compound identity instead. Both are columns the pipeline itself injects
    downstream (predictions, exports, the train/test split), so the guard
    has to sit here, at creation, where the export-time collision guard is
    already too late -- by then the bad column has already trained and
    published a Protocol."""
    upload_ref = await csv_upload(
        b"smiles,uncertainty\nCCO,1.0\nc1ccccc1,5.0\nCCN,2.0\nc1ccncc1,6.0\n"
    )
    response = await client.post(
        "/api/v1/datasets",
        json=create_body(upload_ref, targets=[{"column": "uncertainty", "kind": "numeric"}]),
    )
    assert response.status_code == 422, response.text
    assert "uncertainty" in response.json()["message"]


async def test_every_reserved_target_column_name_is_rejected(client, csv_upload):
    for name in (
        "structure",
        "uncertainty",
        "applicability",
        "generation_method",
        "row_id",
        "split",
    ):
        upload_ref = await csv_upload(f"smiles,{name}\nCCO,1.0\nc1ccccc1,5.0\n".encode())
        response = await client.post(
            "/api/v1/datasets",
            json=create_body(upload_ref, targets=[{"column": name, "kind": "numeric"}]),
        )
        assert response.status_code == 422, response.text
        assert name in response.json()["message"]


async def test_a_single_class_train_partition_is_rejected_before_training(client, csv_upload):
    """I1 (whole-branch review, Important): `assign_split` only raises when a
    requested partition comes back *empty*, not when it is merely
    single-class -- so a binary dataset whose train split holds one class
    used to create 201, train `ready`, and produce a Protocol whose
    predictions come out as one class with `uncertainty` exactly `0.0` on
    every compound: maximally confident, and worthless. Ten rows, all the
    same class, guarantee a single-class train partition (8 of 10 rows)
    regardless of split seed or strategy."""
    rows = "\n".join(
        f"{smiles},1"
        for smiles in (
            "CCO",
            "CCN",
            "CCCO",
            "CCCCO",
            "CCCCCO",
            "c1ccccc1",
            "Cc1ccccc1",
            "c1ccncc1",
            "c1ccsc1",
            "C1CCCCC1",
        )
    )
    upload_ref = await csv_upload(f"smiles,active\n{rows}\n".encode())
    response = await client.post(
        "/api/v1/datasets",
        json=create_body(upload_ref, targets=[{"column": "active", "kind": "binary"}]),
    )
    assert response.status_code == 422, response.text
    assert "train" in response.json()["message"]


async def test_a_constant_train_partition_is_rejected_for_regression_too(client, csv_upload):
    """The regression twin (I1): `r2_score` returns `0.0`, not NaN, on a
    constant target, so a regression Scorecard would show "R2 = 0.0" as if
    it were measured, with `metrics_undefined` silent about it -- the
    classification path already handles its equivalent explicitly. A
    constant-valued partition is the regression analogue of single-class,
    and the check must cover it the same way, on the same all-identical-value
    dataset shape as the classification case above."""
    rows = "\n".join(
        f"{smiles},5.0"
        for smiles in (
            "CCO",
            "CCN",
            "CCCO",
            "CCCCO",
            "CCCCCO",
            "c1ccccc1",
            "Cc1ccccc1",
            "c1ccncc1",
            "c1ccsc1",
            "C1CCCCC1",
        )
    )
    upload_ref = await csv_upload(f"smiles,y\n{rows}\n".encode())
    response = await client.post("/api/v1/datasets", json=create_body(upload_ref))
    assert response.status_code == 422, response.text
    assert "train" in response.json()["message"]


# Ten distinct structures with a controlled RANDOM split (seed=1, explicit
# 50/50/0 fractions): train lands on indices 0, 1, 4, 7, 8 and test on
# 2, 3, 5, 6, 9 -- verified directly against `assign_split`. Shared by the
# two tests below, which put a varied/constant target on opposite sides of
# that exact split to isolate "train is fine, test alone is degenerate" from
# "test is fine, train alone is degenerate".
_TEN_STRUCTURES = (
    "CCO",
    "CCN",
    "CCCO",
    "CCCCO",
    "CCCCCO",
    "c1ccccc1",
    "Cc1ccccc1",
    "c1ccncc1",
    "c1ccsc1",
    "C1CCCCC1",
)
_CONTROLLED_SPLIT = {"strategy": "random", "seed": 1, "fractions": [0.5, 0.0, 0.5]}


async def test_a_varied_train_with_a_constant_test_partition_is_rejected(client, csv_upload):
    """I1's own named regression scenario (whole-branch review, re-review):
    both tests above use a wholly single-class/constant dataset, so the
    "train" then "test" loop always returns on `train` and never actually
    exercises the test-partition branch. This isolates it: train (indices 0,
    1, 4, 7, 8) gets five distinct values; test (2, 3, 5, 6, 9) gets the same
    value five times over -- a real, varied, trainable dataset whose *test*
    partition alone is degenerate, which is exactly the case `r2_score`
    silently mismeasures as `R2 = 0.0`."""
    values = [1.0, 2.0, 9.9, 9.9, 3.0, 9.9, 9.9, 4.0, 5.0, 9.9]
    rows = "\n".join(f"{s},{v}" for s, v in zip(_TEN_STRUCTURES, values, strict=True))
    upload_ref = await csv_upload(f"smiles,y\n{rows}\n".encode())
    response = await client.post(
        "/api/v1/datasets", json=create_body(upload_ref, split=_CONTROLLED_SPLIT)
    )
    assert response.status_code == 422, response.text
    assert "test" in response.json()["message"]


async def test_a_single_class_test_partition_is_accepted_for_binary_classification(
    client, csv_upload
):
    """The narrowing (whole-branch review, re-review): a single-class *test*
    partition on a BINARY target must NOT be rejected -- `_scoring.py`'s
    single-class branch already reports every classification metric as
    undefined, and `train_protocol.py`'s `_undefined_reasons` turns that into
    an actionable `metrics_undefined` message on the Scorecard. Refusing to
    even train would replace that honest answer with an over-eager hard
    refusal for a dataset that is otherwise perfectly trainable -- exactly
    what a 25-seed sweep against balanced and imbalanced binary datasets
    showed the pre-narrowing guard doing on 9-15 of 25 seeds, always on
    `test`, never on `train`.

    Same controlled split as the regression test above, values chosen so
    train (indices 0, 1, 4, 7, 8) holds both classes and test (2, 3, 5, 6, 9)
    holds only one.
    """
    values = [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]
    rows = "\n".join(f"{s},{v}" for s, v in zip(_TEN_STRUCTURES, values, strict=True))
    upload_ref = await csv_upload(f"smiles,active\n{rows}\n".encode())
    response = await client.post(
        "/api/v1/datasets",
        json=create_body(
            upload_ref,
            targets=[{"column": "active", "kind": "binary"}],
            split=_CONTROLLED_SPLIT,
        ),
    )
    assert response.status_code == 201, response.text


async def test_an_overlong_name_is_a_422_not_an_asyncpg_500(client, csv_upload):
    """I3 (whole-branch review, Important): `name` has no `max_length`
    against `DatasetModel.name`'s `String(256)` column, so an over-long value
    used to reach asyncpg and come back as an unmapped
    `StringDataRightTruncationError` (500) instead of a 422 naming the
    field."""
    upload_ref = await csv_upload(SOLUBILITY_CSV)
    response = await client.post("/api/v1/datasets", json=create_body(upload_ref, name="x" * 257))
    assert response.status_code == 422, response.text


async def test_an_overlong_structure_column_is_a_422_not_an_asyncpg_500(client, csv_upload):
    """Same finding, `DatasetModel.structure_column`'s `String(128)` column.

    The uploaded file's own header is the same 129-character name the request
    names as `structure_column`, so this reaches every application-level
    check (the column really is present) and would only be stopped by the
    database's column width without this fix -- not by the unrelated
    "column not present" 422 a nonexistent column name would trigger for a
    less careful test.
    """
    long_column = "x" * 129
    upload_ref = await csv_upload(f"{long_column},y\nCCO,1.0\nc1ccccc1,5.0\n".encode())
    response = await client.post(
        "/api/v1/datasets", json=create_body(upload_ref, structure_column=long_column)
    )
    assert response.status_code == 422, response.text


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
                targets=[{"column": "y", "kind": "numeric", "unit": "logS", "direction": "high"}],
                split={"strategy": "random", "seed": 99, "fractions": [0.6, 0.2, 0.2]},
            ),
        )
    ).json()

    fetched = (await client.get(f"/api/v1/datasets/{created['id']}")).json()
    assert fetched == created
    assert fetched["targets"] == [
        {"column": "y", "kind": "numeric", "unit": "logS", "direction": "high"}
    ]
    assert fetched["split"] == {"strategy": "random", "seed": 99, "fractions": [0.6, 0.2, 0.2]}


TWO_TARGET_CSV = (
    b"smiles,solubility,reactive\n"
    b"CCO,1.0,0\nc1ccccc1,5.0,1\nCCN,2.0,0\nc1ccncc1,6.0,1\nCCCO,1.5,1\n"
    b"Cc1ccccc1,5.5,0\nCCCN,2.5,1\nc1ccsc1,6.5,0\nCCCCO,1.2,0\nC1CCCCC1,4.0,1\n"
)


async def test_a_dataset_keeps_several_targets_in_the_order_chosen(client, csv_upload):
    upload_ref = await csv_upload(TWO_TARGET_CSV)
    response = await client.post(
        "/api/v1/datasets",
        json=create_body(
            upload_ref,
            targets=[
                {"column": "reactive", "kind": "binary"},
                {"column": "solubility", "kind": "numeric", "unit": "logS", "direction": "high"},
            ],
        ),
    )
    assert response.status_code == 201, response.text
    assert [t["column"] for t in response.json()["targets"]] == ["reactive", "solubility"]


async def test_a_dataset_is_refused_when_any_one_target_is_degenerate(client, csv_upload):
    header, *rows = TWO_TARGET_CSV.decode().strip().split("\n")
    csv = "\n".join([f"{header},flag", *(f"{row},0" for row in rows)]) + "\n"
    upload_ref = await csv_upload(csv.encode())
    response = await client.post(
        "/api/v1/datasets",
        json=create_body(
            upload_ref,
            targets=[
                {"column": "solubility", "kind": "numeric"},
                {"column": "flag", "kind": "binary"},
            ],
        ),
    )
    assert response.status_code == 422, response.text
    assert "'flag'" in response.text


async def test_the_structure_column_cannot_also_be_a_target(client, csv_upload):
    upload_ref = await csv_upload(TWO_TARGET_CSV)
    response = await client.post(
        "/api/v1/datasets",
        json=create_body(upload_ref, targets=[{"column": "smiles", "kind": "numeric"}]),
    )
    assert response.status_code == 422, response.text


async def test_compounds_carry_every_target_and_sort_by_any_one(client, csv_upload):
    upload_ref = await csv_upload(TWO_TARGET_CSV)
    created = await client.post(
        "/api/v1/datasets",
        json=create_body(
            upload_ref,
            targets=[
                {"column": "solubility", "kind": "numeric"},
                {"column": "reactive", "kind": "binary"},
            ],
        ),
    )
    dataset_id = created.json()["id"]
    response = await client.get(
        f"/api/v1/datasets/{dataset_id}/compounds",
        params={"sort": "target", "target": 1, "sort_dir": "desc"},
    )
    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert set(items[0]["targets"]) == {"solubility", "reactive"}
    assert items[0]["targets"]["reactive"] == 1.0
    assert items[-1]["targets"]["reactive"] == 0.0

    out_of_range = await client.get(
        f"/api/v1/datasets/{dataset_id}/compounds", params={"sort": "target", "target": 2}
    )
    assert out_of_range.status_code == 422
