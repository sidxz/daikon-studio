"""Studies over HTTP: start one from a plan and a file, advance it, read what it made."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from tests.fakes.tool_executor import InProcessToolExecutor

from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.tool_executor import ToolExecutor
from daikonstudio.application.studies import advance_study
from daikonstudio.infrastructure.tools.impl import chem

CSV = (
    b"ID,SMILES,pIC50\n"
    b"a,O=C(Nc1ccccc1)c1ccccc1,6.5\n"
    b"b,CC(=O)[O-].[Na+],5.0\n"
    b"c,O=C1C=CC(=O)C=C1,7.2\n"
    b"d,notasmiles,\n"
)

PLAN: dict[str, Any] = {
    "goal": "Profile these compounds.",
    "restated_goal": "Physicochemical properties of four compounds.",
    "steps": [
        {"tool_id": "standardize", "purpose": "Clean.", "gated": False, "always": True},
        {"tool_id": "property_profile", "purpose": "Profile.", "gated": False},
    ],
}


@pytest.fixture(autouse=True)
def _chem_tools(app: FastAPI) -> None:
    """The real chemistry tools, in this process rather than a spawned child."""
    app.state.container.define(  # type: ignore[type-abstract]
        ToolExecutor, lambda c: InProcessToolExecutor(c[BlobStore], chem.IMPLEMENTATIONS)
    )


async def _start(
    client: httpx.AsyncClient,
    plan: dict[str, Any] = PLAN,
    data: bytes | None = CSV,
    filename: str = "compounds.csv",
) -> httpx.Response:
    files = {"file": (filename, data, "text/csv")} if data is not None else None
    return await client.post("/api/v1/studies", data={"plan": json.dumps(plan)}, files=files)


async def test_start_brings_the_compounds(client: httpx.AsyncClient) -> None:
    response = await _start(client)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["brought"][0]["kind"] == "compound_table"
    assert body["brought"][0]["summary"]["rows"] == 4
    assert [s["status"] for s in body["steps"]] == ["pending", "pending"]
    assert body["running"] is False


async def test_start_reads_a_tsv_with_a_bom(client: httpx.AsyncClient) -> None:
    tsv = b"\xef\xbb\xbf" + CSV.replace(b",", b"\t")
    response = await _start(client, data=tsv, filename="compounds.tsv")
    assert response.status_code == 201, response.text
    summary = response.json()["brought"][0]["summary"]
    assert summary["rows"] == 4
    assert summary["columns"] == ["ID", "SMILES", "pIC50"]


async def test_start_refuses_an_unknown_tool(client: httpx.AsyncClient) -> None:
    plan = {**PLAN, "steps": [{"tool_id": "summon_answer", "purpose": "x", "gated": False}]}
    response = await _start(client, plan)
    assert response.status_code == 422
    assert "summon_answer" in response.json()["message"]


async def test_start_refuses_missing_settings(client: httpx.AsyncClient) -> None:
    plan = {**PLAN, "steps": [{"tool_id": "fetch_pdb", "purpose": "x", "gated": False}]}
    response = await _start(client, plan, data=None)
    assert response.status_code == 422
    assert "pdb_id" in response.json()["message"]


async def test_start_asks_for_the_compounds_file_a_plan_needs(client: httpx.AsyncClient) -> None:
    response = await _start(client, data=None)
    assert response.status_code == 422
    assert "Attach the compounds file" in response.json()["message"]


async def test_start_refuses_a_malformed_plan(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v1/studies", data={"plan": "{not json"})
    assert response.status_code == 422


async def test_advance_runs_and_the_value_reads_back(client: httpx.AsyncClient) -> None:
    study_id = (await _start(client)).json()["id"]

    advanced = await client.post(f"/api/v1/studies/{study_id}/advance")
    assert advanced.status_code == 202, advanced.text
    assert advanced.json()["running"] is True
    # Awaited, not polled: the test database is one connection (test_dataset_builds.py).
    await asyncio.gather(*advance_study._running)

    study = (await client.get(f"/api/v1/studies/{study_id}")).json()
    step = study["steps"][0]
    assert step["status"] == "done"
    assert len(step["outputs"]) == 1
    assert study["status"] == "done"
    assert study["running"] is False

    value = await client.get(f"/api/v1/studies/{study_id}/values/{step['outputs'][0]['id']}")
    assert value.status_code == 200, value.text
    rows = value.json()["data"]
    assert len(rows) == 4
    assert all("canonical_smiles" in row for row in rows)

    listed = (await client.get("/api/v1/studies")).json()
    assert [s["id"] for s in listed] == [study_id]


async def test_another_workspace_cannot_see_or_advance(
    client: httpx.AsyncClient, other_workspace_client: httpx.AsyncClient
) -> None:
    study_id = (await _start(client)).json()["id"]
    assert (await other_workspace_client.get(f"/api/v1/studies/{study_id}")).status_code == 404
    advanced = await other_workspace_client.post(f"/api/v1/studies/{study_id}/advance")
    assert advanced.status_code == 404


async def test_start_keeps_the_answers(client: httpx.AsyncClient) -> None:
    answers = [{"ask": "Which column?", "answer": "pIC50"}]
    response = await _start(client, {**PLAN, "answers": answers})
    assert response.status_code == 201, response.text
    assert response.json()["answers"] == answers
    study_id = response.json()["id"]
    assert (await client.get(f"/api/v1/studies/{study_id}")).json()["answers"] == answers


async def test_start_keeps_the_planning(client: httpx.AsyncClient) -> None:
    planning = {
        "asked": {"questions": [{"ask": "Which column?", "options": ["pIC50"], "default": None}]},
        "plan": {"restated_goal": "rg", "skipped": [{"tool_id": "x", "why": "y"}]},
    }
    response = await _start(client, {**PLAN, "planning": planning})
    assert response.status_code == 201, response.text
    study_id = response.json()["id"]
    assert (await client.get(f"/api/v1/studies/{study_id}")).json()["planning"] == planning
    assert (await _start(client)).json()["planning"] == {}


async def test_run_one_step_leaves_the_next_pending(client: httpx.AsyncClient) -> None:
    study_id = (await _start(client)).json()["id"]

    ran = await client.post(f"/api/v1/studies/{study_id}/steps/0/run")
    assert ran.status_code == 202, ran.text
    await asyncio.gather(*advance_study._running)

    study = (await client.get(f"/api/v1/studies/{study_id}")).json()
    assert [s["status"] for s in study["steps"]] == ["done", "pending"]
    assert study["running"] is False


async def test_run_a_step_that_does_not_exist(client: httpx.AsyncClient) -> None:
    study_id = (await _start(client)).json()["id"]
    response = await client.post(f"/api/v1/studies/{study_id}/steps/9/run")
    assert response.status_code == 422
    assert "No step 9" in response.json()["message"]
