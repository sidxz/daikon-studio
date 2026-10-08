"""Lab-notebook pages on datasets, protocols and runs: visibility, revisions, images."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from tests.api.test_protocols import _create_dataset, _train

from daikonstudio.domain.shared.errors import ConflictError
from daikonstudio.domain.shared.page import page_body
from daikonstudio.infrastructure.persistence.sqlalchemy.pages import SqlAlchemyPageRepository

_P = "/api/v1/pages"
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _doc(words: str) -> dict:
    return {"type": "doc", "content": [{"type": "paragraph", "text": words}]}


async def _page(http, owner_kind: str, owner_id: str, title: str = "Notes") -> dict:
    response = await http.post(
        _P, json={"title": title, "owner_kind": owner_kind, "owner_id": owner_id}
    )
    assert response.status_code == 201, response.text
    return response.json()


async def _revise(http, page: dict, words: str) -> dict:
    response = await http.patch(
        f"{_P}/{page['id']}", json={"body": _doc(words), "expected_version": page["version"]}
    )
    assert response.status_code == 200, response.text
    return response.json()


async def _trained(client, dataset_id: str) -> tuple[str, str]:
    response = await _train(client, dataset_id)
    assert response.status_code == 202, response.text
    run = (await client.get(f"/api/v1/runs/{response.json()['id']}")).json()
    return run["id"], run["protocol_id"]


async def _rows(session_factory, page_id: str) -> int:
    async with session_factory() as session:
        return await session.scalar(
            text("SELECT count(*) FROM pages WHERE id = :id"), {"id": page_id}
        )


async def test_pages_on_each_kind_of_owner(client, csv_upload):
    dataset_id = await _create_dataset(client, csv_upload)
    run_id, protocol_id = await _trained(client, dataset_id)
    for kind, owner_id in (("dataset", dataset_id), ("protocol", protocol_id), ("run", run_id)):
        page = await _page(client, kind, owner_id, title=f" {kind} notes ")
        assert page["title"] == f"{kind} notes"
        assert (page["version"], page["head_sha256"], page["revision_count"]) == (0, None, 0)
        assert (page["can_edit"], page["can_archive"], page["can_delete"]) == (True, True, True)
        listed = await client.get(_P, params={"owner_kind": kind, "owner_id": owner_id})
        assert [p["id"] for p in listed.json()] == [page["id"]]
        assert (await client.get(f"{_P}/{page['id']}")).json()["title"] == f"{kind} notes"


async def test_revising_keeps_history_and_refuses_a_stale_version(client, csv_upload):
    dataset_id = await _create_dataset(client, csv_upload)
    page = await _page(client, "dataset", dataset_id)
    first = await _revise(client, page, "one")
    second = await _revise(client, first, "two")
    assert (second["version"], second["revision_count"]) == (2, 2)
    assert second["last_edited_by"] == second["author_id"]

    stale = await client.patch(
        f"{_P}/{page['id']}", json={"body": _doc("three"), "expected_version": 1}
    )
    assert stale.status_code == 409
    unchanged = await _revise(client, second, "two")
    assert unchanged["version"] == 2

    revisions = (await client.get(f"{_P}/{page['id']}/revisions")).json()
    assert [r["revision_no"] for r in revisions] == [1, 2]
    assert revisions[1]["sha256"] == second["head_sha256"]
    content = await client.get(f"{_P}/{page['id']}/content/{revisions[0]['sha256']}")
    assert content.json() == _doc("one")

    other = await _revise(client, await _page(client, "dataset", dataset_id), "elsewhere")
    foreign = await client.get(f"{_P}/{page['id']}/content/{other['head_sha256']}")
    assert foreign.status_code == 404

    not_a_doc = await client.patch(
        f"{_P}/{page['id']}", json={"body": {"type": "x"}, "expected_version": 2}
    )
    assert not_a_doc.status_code == 422


async def test_an_archived_page_is_locked_until_restored(client, other_editor_client, csv_upload):
    dataset_id = await _create_dataset(client, csv_upload)
    page = await _page(client, "dataset", dataset_id)
    refused = await other_editor_client.post(f"{_P}/{page['id']}/archive", json={})
    assert refused.status_code == 403

    archived = (await client.post(f"{_P}/{page['id']}/archive", json={})).json()
    assert (archived["archived"], archived["can_edit"], archived["version"]) == (True, False, 1)
    locked = await client.patch(
        f"{_P}/{page['id']}", json={"body": _doc("x"), "expected_version": 1}
    )
    assert locked.status_code == 423
    params = {"owner_kind": "dataset", "owner_id": dataset_id}
    assert (await client.get(_P, params=params)).json() == []
    listed = await client.get(_P, params={**params, "archived": "true"})
    assert [p["id"] for p in listed.json()] == [page["id"]]

    restored = await client.post(f"{_P}/{page['id']}/archive", json={"restore": True})
    assert restored.json()["archived"] is False
    await _revise(client, restored.json(), "back")


async def test_who_may_see_and_change_a_page(
    client, viewer_client, other_editor_client, other_workspace_client, csv_upload
):
    dataset_id = await _create_dataset(client, csv_upload)
    run_id, protocol_id = await _trained(client, dataset_id)
    page = await _page(client, "dataset", dataset_id)

    body = {"title": "x", "owner_kind": "dataset", "owner_id": dataset_id}
    assert (await viewer_client.post(_P, json=body)).status_code == 403
    seen = (await viewer_client.get(f"{_P}/{page['id']}")).json()
    assert (seen["can_edit"], seen["can_delete"]) == (False, False)
    assert (await other_workspace_client.get(f"{_P}/{page['id']}")).status_code == 404
    assert (await other_editor_client.delete(f"{_P}/{page['id']}")).status_code == 403

    # A colleague's draft protocol, and the run that made it, keep their pages to themselves.
    for kind, owner_id in (("protocol", protocol_id), ("run", run_id)):
        hidden = await _page(client, kind, owner_id)
        assert (await other_editor_client.get(f"{_P}/{hidden['id']}")).status_code == 404
        params = {"owner_kind": kind, "owner_id": owner_id}
        assert (await other_editor_client.get(_P, params=params)).status_code == 404
        body = {"title": "x", "owner_kind": kind, "owner_id": owner_id}
        assert (await other_editor_client.post(_P, json=body)).status_code == 404

    assert (await client.delete(f"{_P}/{page['id']}")).status_code == 204
    assert (await client.get(f"{_P}/{page['id']}")).status_code == 404


async def test_images_are_stored_once_and_only_as_what_they_are(client, other_workspace_client):
    upload = {"file": ("plot.png", _PNG, "image/png")}
    stored = await client.post(f"{_P}/blobs", files=upload)
    assert stored.status_code == 201, stored.text
    blob = stored.json()
    assert (blob["mime"], blob["size"]) == ("image/png", len(_PNG))
    assert (await client.post(f"{_P}/blobs", files=upload)).json() == blob

    served = await client.get(f"{_P}/blobs/{blob['sha256']}")
    assert served.content == _PNG
    assert served.headers["content-type"] == "image/png"
    assert served.headers["content-disposition"] == "attachment"
    assert served.headers["x-content-type-options"] == "nosniff"
    assert "sandbox" in served.headers["content-security-policy"]
    assert (await other_workspace_client.get(f"{_P}/blobs/{blob['sha256']}")).status_code == 404

    html = {"file": ("x.html", b"<script>alert(1)</script>", "text/html")}
    assert (await client.post(f"{_P}/blobs", files=html)).status_code == 422
    disguised = {"file": ("x.png", b"<html><script>alert(1)</script>", "image/png")}
    refused = await client.post(f"{_P}/blobs", files=disguised)
    assert refused.status_code == 422
    assert refused.json()["message"] == "Unsupported file type"


async def test_deleting_an_owner_deletes_its_pages(client, csv_upload, session_factory):
    dataset_id = await _create_dataset(client, csv_upload)
    page = await _page(client, "dataset", dataset_id)
    await _revise(client, page, "kept nowhere")
    assert (await client.delete(f"/api/v1/datasets/{dataset_id}")).status_code == 204
    assert await _rows(session_factory, page["id"]) == 0

    dataset_id = await _create_dataset(client, csv_upload)
    run_id, protocol_id = await _trained(client, dataset_id)
    on_protocol = await _page(client, "protocol", protocol_id)
    on_run = await _page(client, "run", run_id)
    assert (await client.delete(f"/api/v1/protocols/{protocol_id}")).status_code == 204
    assert await _rows(session_factory, on_protocol["id"]) == 0
    assert await _rows(session_factory, on_run["id"]) == 0


async def test_of_two_saves_from_one_version_the_second_loses(
    client, csv_upload, session_factory, workspace_id
):
    """The use case checks the version it read; the UPDATE checks it again, so a save
    that raced past the first check still cannot land on a page that has moved."""
    dataset_id = await _create_dataset(client, csv_upload)
    page = await _page(client, "dataset", dataset_id)
    pages = SqlAlchemyPageRepository(session_factory)
    author = uuid.UUID(page["author_id"])
    page_id = uuid.UUID(page["id"])
    first = await pages.revise(
        workspace_id, page_id, expected_version=0, body=page_body(_doc("a")), author_id=author
    )
    assert first.revision_count == 1
    with pytest.raises(ConflictError):
        await pages.revise(
            workspace_id, page_id, expected_version=0, body=page_body(_doc("b")), author_id=author
        )
    assert [r.revision_no for r in await pages.revisions(page_id)] == [1]
