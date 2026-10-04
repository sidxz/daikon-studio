"""The dataset profile is computed once, in the background, and never twice at once.

A 400k-compound profile takes minutes. Computing it inside the request meant a
reload, a second tab or a second reader started another full computation beside
the first. Now the first request starts it and answers 202; every later request
joins the same computation until the result is saved, then reads the saved copy.
"""

import asyncio
import threading

from tests.api import test_protocols

from daikonstudio.application.data import get_dataset_profile

# A 20-compound dataset with a random split, shared with the protocol suite.
_create_dataset = test_protocols._create_dataset


async def _until_ready(client, dataset_id: str, attempts: int = 200, params: dict | None = None):
    for _ in range(attempts):
        response = await client.get(f"/api/v1/datasets/{dataset_id}/profile", params=params)
        if response.status_code != 202:
            return response
        await asyncio.sleep(0.05)
    raise AssertionError("the profile never finished")


async def test_the_first_request_starts_the_computation_and_says_so(client, csv_upload):
    dataset_id = await _create_dataset(client, csv_upload)

    first = await client.get(f"/api/v1/datasets/{dataset_id}/profile")

    assert first.status_code == 202, first.text
    body = first.json()
    assert body["status"] == "computing"
    assert body["compounds"] == 20
    assert body["started_at"]

    ready = await _until_ready(client, dataset_id)
    assert ready.status_code == 200, ready.text
    assert "scaffolds" in ready.json()


async def test_concurrent_requests_join_one_computation(client, csv_upload, monkeypatch):
    dataset_id = await _create_dataset(client, csv_upload)
    release = threading.Event()
    calls = 0
    real = get_dataset_profile.build_profile

    def slow_build(**kwargs):
        nonlocal calls
        calls += 1
        release.wait(timeout=10)
        return real(**kwargs)

    monkeypatch.setattr(get_dataset_profile, "build_profile", slow_build)

    first = await client.get(f"/api/v1/datasets/{dataset_id}/profile")
    second = await client.get(f"/api/v1/datasets/{dataset_id}/profile")
    assert first.status_code == second.status_code == 202
    assert first.json()["started_at"] == second.json()["started_at"]

    release.set()
    ready = await _until_ready(client, dataset_id)
    assert ready.status_code == 200
    assert calls == 1


async def test_a_failed_computation_is_reported_once_then_retried(client, csv_upload, monkeypatch):
    dataset_id = await _create_dataset(client, csv_upload)

    def broken(**kwargs):
        raise RuntimeError("descriptor table exploded")

    monkeypatch.setattr(get_dataset_profile, "build_profile", broken)
    assert (await client.get(f"/api/v1/datasets/{dataset_id}/profile")).status_code == 202

    failed = await _until_ready(client, dataset_id)
    assert failed.status_code == 409
    assert "could not be computed" in failed.json()["message"]

    # Reported once: the next request starts a fresh attempt.
    monkeypatch.setattr(get_dataset_profile, "build_profile", get_dataset_profile.build_profile)
    retried = await client.get(f"/api/v1/datasets/{dataset_id}/profile")
    assert retried.status_code == 202


async def test_each_target_has_its_own_profile(client, csv_upload):
    from tests.api.test_datasets import TWO_TARGET_CSV, create_body

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

    first = await _until_ready(client, dataset_id)
    second = await _until_ready(client, dataset_id, params={"target": 1})

    assert first.json()["target_kind"] == "numeric"
    assert second.json()["target_kind"] == "binary"
