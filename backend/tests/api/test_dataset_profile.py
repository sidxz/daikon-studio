"""The dataset profile is computed once, in the background, and never twice at once.

A 400k-compound profile takes minutes. Computing it inside the request meant a
reload, a second tab or a second reader started another full computation beside
the first. Now the first request starts it and answers 202; every later request
joins the same computation until the result is saved, then reads the saved copy.

The computation itself is a `ProfileComputer`, and here it is the in-process fake --
these tests are about the endpoint, not about where the work runs. The child process
that production uses has its own tests in `tests/unit/chem/test_profile_process.py`.
"""

import asyncio

from tests.api import test_protocols

# A 20-compound dataset with a random split, shared with the protocol suite.
_create_dataset = test_protocols._create_dataset


async def _until_ready(client, dataset_id: str, attempts: int = 200, params: dict | None = None):
    for _ in range(attempts):
        response = await client.get(f"/api/v1/datasets/{dataset_id}/profile", params=params)
        if response.status_code != 202:
            return response
        await asyncio.sleep(0.05)
    raise AssertionError("the profile never finished")


async def _create_two_target_dataset(client, csv_upload) -> str:
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
    assert created.status_code == 201, created.text
    return str(created.json()["id"])


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


async def test_concurrent_requests_join_one_computation(client, csv_upload, profile_computer):
    dataset_id = await _create_dataset(client, csv_upload)
    profile_computer.release = asyncio.Event()

    first = await client.get(f"/api/v1/datasets/{dataset_id}/profile")
    second = await client.get(f"/api/v1/datasets/{dataset_id}/profile")
    assert first.status_code == second.status_code == 202
    assert first.json()["started_at"] == second.json()["started_at"]

    profile_computer.release.set()
    ready = await _until_ready(client, dataset_id)
    assert ready.status_code == 200
    assert profile_computer.calls == 1


async def test_a_failed_computation_is_reported_once_then_retried(
    client, csv_upload, profile_computer
):
    """A computation that dies -- in production, a child process that failed or was
    killed -- must reach the reader as the 409 retry, not leave them polling 202."""
    dataset_id = await _create_dataset(client, csv_upload)
    profile_computer.error = RuntimeError("the profile child exited")

    assert (await client.get(f"/api/v1/datasets/{dataset_id}/profile")).status_code == 202

    failed = await _until_ready(client, dataset_id)
    assert failed.status_code == 409
    assert "could not be computed" in failed.json()["message"]

    # Reported once: the next request starts a fresh attempt.
    profile_computer.error = None
    retried = await client.get(f"/api/v1/datasets/{dataset_id}/profile")
    assert retried.status_code == 202


async def test_two_targets_are_profiled_one_after_the_other(client, csv_upload, profile_computer):
    """Not a nicety: `_RUNNING` is keyed per target, so a four-target dataset opened
    once would start four profiles, and four at 3-4 GB exceeds the API's 8 GiB limit.
    An OOM-killed API is a worse failure than a profile that waits its turn."""
    dataset_id = await _create_two_target_dataset(client, csv_upload)
    profile_computer.release = asyncio.Event()

    for target in (0, 1):
        started = await client.get(
            f"/api/v1/datasets/{dataset_id}/profile", params={"target": target}
        )
        assert started.status_code == 202, started.text

    profile_computer.release.set()
    assert (await _until_ready(client, dataset_id)).status_code == 200
    assert (await _until_ready(client, dataset_id, params={"target": 1})).status_code == 200

    assert profile_computer.calls == 2
    assert profile_computer.peak == 1


async def test_each_target_has_its_own_profile(client, csv_upload):
    dataset_id = await _create_two_target_dataset(client, csv_upload)

    first = await _until_ready(client, dataset_id)
    second = await _until_ready(client, dataset_id, params={"target": 1})

    assert first.json()["target_kind"] == "numeric"
    assert second.json()["target_kind"] == "binary"
