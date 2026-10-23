"""Studies over HTTP, from the first message to an approved candidate.

The model is scripted: it asks one question, writes the plan once that is answered, then says
whatever a test needs, in order. Runs are awaited, not polled: the test database is one
connection (test_dataset_builds.py), so a test waits for the run before it reads again.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from lagom import Singleton
from pydantic import BaseModel
from tests.api.conftest import auth_headers
from tests.fakes.llm import ScriptedLanguageModel, ScriptedLanguageModels
from tests.fakes.tool_executor import InProcessToolExecutor

from daikonstudio.application.planner.plan import (
    DraftPlan,
    Interrogation,
    Question,
    Setting,
)
from daikonstudio.application.planner.plan import Step as PlanStep
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.llm import LanguageModels
from daikonstudio.application.ports.tool_executor import ToolExecutor
from daikonstudio.application.studies import advance_study
from daikonstudio.application.studies.advance_study import RUNNING_MESSAGE
from daikonstudio.application.studies.schemas import ObserveTurn, PlanningTurn, Proposal, ReplyTurn
from daikonstudio.infrastructure.tools.impl import chem
from daikonstudio.infrastructure.tools.runtime import ToolImpl

CSV = (
    b"ID,SMILES,pIC50\n"
    b"a,O=C(Nc1ccccc1)c1ccccc1,6.5\n"
    b"b,CC(=O)[O-].[Na+],5.0\n"
    b"c,O=C1C=CC(=O)C=C1,7.2\n"
    b"d,notasmiles,\n"
)
COMPOUNDS_FILE = ("compounds.csv", CSV)
GOAL = "Work out what drives potency in this series."
ASK = "Which column is the readout?"
ASKING = Interrogation(
    restated_goal="What drives potency in the compounds sent.",
    questions=[Question(ask=ASK, why="Every effect is read against one.", options=["pIC50"])],
)
COMPOUNDS = PlanningTurn(
    say="A property profile first.",
    plan=DraftPlan(
        restated_goal="Physicochemical properties of four compounds.",
        steps=[
            PlanStep(tool_id="standardize", purpose="Clean the structures.", gated=False),
            PlanStep(
                tool_id="property_profile",
                purpose="Profile them against pIC50.",
                gated=False,
                settings=[Setting(name="activity_column", value="pIC50")],
            ),
        ],
        checks=["An effect counts only over at least four compounds."],
    ),
)
PROTEIN = PlanningTurn(
    say="Fetch the structure, then look for pockets.",
    plan=DraftPlan(
        restated_goal="Where a drug could bind carbonic anhydrase II.",
        steps=[
            PlanStep(
                tool_id="fetch_pdb",
                purpose="Fetch 1CA2.",
                gated=False,
                settings=[Setting(name="pdb_id", value="1CA2")],
            ),
            PlanStep(tool_id="find_pockets", purpose="Find the candidate sites.", gated=True),
        ],
    ),
)
QUIET = ObserveTurn(say="", summary="Nothing to add.")
STRUCTURE = {"format": "pdb", "text": "ATOM", "meta": {"chains": ["A"]}}
POCKETS = {
    "meta": {"count": 1},
    "pockets": [
        {
            "rank": 1,
            "residues": [{"chain": "A", "resname": "HIS", "resseq": n} for n in (94, 96, 119)],
            "ligand_overlap": [],
            "annotated": [],
            "interface": False,
        }
    ],
}


@pytest.fixture(autouse=True)
def tools(app: FastAPI) -> dict[str, ToolImpl]:
    """The real chemistry tools, in this process rather than a spawned child. A test puts a
    stand-in here for a tool this machine does not have."""
    impls = dict(chem.IMPLEMENTATIONS)
    app.state.container.define(  # type: ignore[type-abstract]
        ToolExecutor, lambda c: InProcessToolExecutor(c[BlobStore], impls)
    )
    return impls


def _script(app: FastAPI, plan: PlanningTurn, *then: BaseModel) -> ScriptedLanguageModel:
    """A deployment whose model asks one question, writes `plan` once it is answered, then says
    `then` in order, the last of them for every turn after.

    Defined on the app's own child container, which lagom lets a test define once.
    """
    model = ScriptedLanguageModel(ASKING, plan, *then)
    app.state.container.define(  # type: ignore[type-abstract]
        LanguageModels, Singleton(lambda: ScriptedLanguageModels(model))
    )
    return model


async def _open(client: httpx.AsyncClient, *files: tuple[str, bytes]) -> httpx.Response:
    return await client.post(
        "/api/v1/studies", data={"message": GOAL}, files=[("files", file) for file in files]
    )


async def _drafted(client: httpx.AsyncClient, *files: tuple[str, bytes]) -> dict[str, Any]:
    """A draft whose question has been answered, so it has a plan: the study as it reads."""
    opened = await _open(client, *files)
    assert opened.status_code == 201, opened.text
    study_id = opened.json()["id"]
    said = await client.post(
        f"/api/v1/studies/{study_id}/chat", json={"answers": [{"ask": ASK, "answer": "pIC50"}]}
    )
    assert said.status_code == 200, said.text
    return (await client.get(f"/api/v1/studies/{study_id}")).json()


async def _started(client: httpx.AsyncClient, *files: tuple[str, bytes]) -> dict[str, Any]:
    study = await _drafted(client, *files)
    started = await client.post(f"/api/v1/studies/{study['id']}/start")
    assert started.status_code == 200, started.text
    return started.json()


async def _thread(client: httpx.AsyncClient, study_id: str) -> list[dict[str, Any]]:
    response = await client.get(f"/api/v1/studies/{study_id}/chat")
    assert response.status_code == 200, response.text
    turns: list[dict[str, Any]] = response.json()["turns"]
    return turns


async def test_the_first_message_opens_a_draft_with_questions(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    _script(app, COMPOUNDS)

    opened = await _open(client, COMPOUNDS_FILE)

    assert opened.status_code == 201, opened.text
    study = opened.json()
    assert (study["status"], study["started_at"], study["steps"]) == ("draft", None, [])
    assert study["restated_goal"] == ASKING.restated_goal
    assert study["brought"][0]["kind"] == "compound_table"
    assert study["brought"][0]["summary"]["rows"] == 4
    (asking,) = await _thread(client, study["id"])
    assert (asking["role"], asking["kind"]) == ("agent", "asking")
    assert [(q["ask"], q["options"]) for q in asking["questions"]] == [(ASK, ["pIC50"])]
    listed = (await client.get("/api/v1/studies")).json()
    assert [s["id"] for s in listed] == [study["id"]]


async def test_answers_become_a_plan_with_steps_that_carry_ids_and_alternatives(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    _script(app, PROTEIN)
    study_id = (await _open(client)).json()["id"]

    said = await client.post(
        f"/api/v1/studies/{study_id}/chat", json={"answers": [{"ask": ASK, "answer": "pIC50"}]}
    )

    assert said.status_code == 200, said.text
    assert [(t["role"], t["kind"]) for t in said.json()["turns"]] == [
        ("agent", "asking"),
        ("scientist", ""),
        ("agent", "planning"),
    ]
    study = (await client.get(f"/api/v1/studies/{study_id}")).json()
    assert study["answers"] == [{"ask": ASK, "answer": "pIC50"}]
    steps = study["steps"]
    assert [s["tool_id"] for s in steps] == ["fetch_pdb", "prepare_structure", "find_pockets"]
    assert len({s["id"] for s in steps}) == 3
    assert [[a["tool_id"] for a in s["alternatives"]] for s in steps] == [
        ["fetch_alphafold"],
        [],
        [],
    ]
    assert [s["judgment"] for s in steps] == [False, False, True]
    assert [s["problem"] for s in steps] == [None, None, None]


async def test_start_then_run_one_step_and_the_observation_appears_in_the_thread(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    noted = ObserveTurn(say="The structures are ready to profile.", summary="Standardized.")
    _script(app, COMPOUNDS, noted)
    study = await _started(client, COMPOUNDS_FILE)
    assert study["status"] == "open" and study["started_at"] is not None
    first = study["steps"][0]["id"]

    ran = await client.post(f"/api/v1/studies/{study['id']}/steps/{first}/run")

    assert ran.status_code == 202, ran.text
    await asyncio.gather(*advance_study._running)
    study = (await client.get(f"/api/v1/studies/{study['id']}")).json()
    assert [s["status"] for s in study["steps"]] == ["done", "pending"]
    assert study["running"] is False
    (output,) = study["steps"][0]["outputs"]
    assert output["produced_by"] == first
    value = await client.get(f"/api/v1/studies/{study['id']}/values/{output['id']}")
    assert value.status_code == 200, value.text
    assert all("canonical_smiles" in row for row in value.json()["data"])
    turns = await _thread(client, study["id"])
    assert [(t["kind"], t["about"]) for t in turns[-2:]] == [("ran", first), ("observe", first)]
    assert turns[-1]["say"] == noted.say


async def test_a_message_during_a_run_answers_409(
    client: httpx.AsyncClient, app: FastAPI, tools: dict[str, ToolImpl]
) -> None:
    _script(app, COMPOUNDS, QUIET)
    entered, release = threading.Event(), threading.Event()
    standardize = tools["standardize"]

    def held(params: Any, inputs: Any) -> Any:
        entered.set()
        release.wait(10)
        return standardize(params, inputs)

    tools["standardize"] = held
    study = await _started(client, COMPOUNDS_FILE)
    first = study["steps"][0]["id"]
    ran = await client.post(f"/api/v1/studies/{study['id']}/steps/{first}/run")
    assert ran.status_code == 202, ran.text
    try:
        await asyncio.to_thread(entered.wait, 10)
        refused = await client.post(
            f"/api/v1/studies/{study['id']}/chat", json={"message": "Gate the profile."}
        )
    finally:
        release.set()
        await asyncio.gather(*advance_study._running)

    assert refused.status_code == 409, refused.text
    assert refused.json()["message"] == RUNNING_MESSAGE
    study = (await client.get(f"/api/v1/studies/{study['id']}")).json()
    assert study["steps"][0]["status"] == "done"
    assert all(t["say"] != "Gate the profile." for t in await _thread(client, study["id"]))


async def test_a_viewer_can_read_the_study_and_the_thread_but_not_say_or_run(
    client: httpx.AsyncClient,
    app: FastAPI,
    signing_key: tuple[str, str],
    workspace_id: uuid.UUID,
    client_user_id: uuid.UUID,
) -> None:
    _script(app, COMPOUNDS)
    study = await _started(client, COMPOUNDS_FILE)
    path = f"/api/v1/studies/{study['id']}"
    # The author, with only the viewer role: a study is its author's to read at all.
    viewer = auth_headers(
        signing_key[0], workspace_id=workspace_id, role="viewer", user_id=client_user_id
    )

    assert (await client.get(path, headers=viewer)).status_code == 200
    thread = await client.get(f"{path}/chat", headers=viewer)
    assert thread.status_code == 200 and thread.json()["turns"]
    said = await client.post(f"{path}/chat", json={"message": "Run it."}, headers=viewer)
    assert said.status_code == 403
    assert (await client.post(f"{path}/run", json={}, headers=viewer)).status_code == 403
    assert not advance_study._running
    assert len(await _thread(client, study["id"])) == len(thread.json()["turns"])


async def test_swap_tool_records_the_event_and_the_problem(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    _script(app, PROTEIN)
    study = await _drafted(client)
    fetch = study["steps"][0]["id"]

    swapped = await client.post(
        f"/api/v1/studies/{study['id']}/steps/{fetch}/tool", json={"tool_id": "fetch_alphafold"}
    )

    assert swapped.status_code == 200, swapped.text
    step = swapped.json()["steps"][0]
    assert (step["id"], step["tool_id"], step["params"]) == (fetch, "fetch_alphafold", {})
    assert "accession" in step["problem"]
    assert [a["tool_id"] for a in step["alternatives"]] == ["fetch_pdb"]
    last = (await _thread(client, study["id"]))[-1]
    assert (last["role"], last["kind"], last["about"]) == ("event", "swapped", fetch)
    assert last["say"] == (
        "Changed step 1 from Fetch a deposited structure to Fetch a predicted structure."
    )


async def test_approve_writes_an_asserted_pocket(
    client: httpx.AsyncClient,
    app: FastAPI,
    tools: dict[str, ToolImpl],
    client_user_id: uuid.UUID,
) -> None:
    _script(app, PROTEIN, QUIET)
    tools["fetch_pdb"] = lambda _params, _inputs: {"structure": STRUCTURE}
    tools["find_pockets"] = lambda _params, _inputs: {"pockets": POCKETS}
    study = await _started(client)
    assert (await client.post(f"/api/v1/studies/{study['id']}/run", json={})).status_code == 202
    await asyncio.gather(*advance_study._running)
    pockets = study["steps"][2]["id"]

    approved = await client.post(
        f"/api/v1/studies/{study['id']}/steps/{pockets}/approve", json={"candidate": 1}
    )

    assert approved.status_code == 200, approved.text
    step = approved.json()["steps"][2]
    assert step["status"] == "done"
    (candidates,) = (v for v in step["outputs"] if v["kind"] == "pocket_candidates")
    (pocket,) = (v for v in step["outputs"] if v["kind"] == "pocket")
    assert pocket["asserted_by"] == str(client_user_id)
    assert (pocket["produced_by"], pocket["summary"]) == (
        pockets,
        {"rank": 1, "from": candidates["id"]},
    )
    last = (await _thread(client, study["id"]))[-1]
    assert (last["kind"], last["about"]) == ("approved", pockets)


async def test_apply_proposal_twice_answers_409(client: httpx.AsyncClient, app: FastAPI) -> None:
    offer = ReplyTurn(
        say="The profile could stop for you to read it.",
        proposals=[Proposal(does="gate", step=2, gated=True, why="So you read it first.")],
        summary="Planned; nothing has run.",
    )
    _script(app, COMPOUNDS, offer)
    study = await _started(client, COMPOUNDS_FILE)
    said = await client.post(
        f"/api/v1/studies/{study['id']}/chat", json={"message": "Should anything stop?"}
    )
    (proposal,) = said.json()["turns"][-1]["proposals"]
    apply = f"/api/v1/studies/{study['id']}/proposals/{proposal['id']}/apply"

    first = await client.post(apply)
    second = await client.post(apply)

    assert first.status_code == 200, first.text
    assert first.json()["steps"][1]["gated"] is True
    assert second.status_code == 409, second.text
    assert second.json()["message"] == "This change has already been made."
    turns = await _thread(client, study["id"])
    assert [t["kind"] for t in turns].count("applied") == 1


async def test_without_a_key_it_says_which_variable_to_set(client: httpx.AsyncClient) -> None:
    response = await _open(client, COMPOUNDS_FILE)

    assert response.status_code == 503
    assert "STUDIO_ANTHROPIC_API_KEY" in response.json()["detail"]
    # Refused before anything was kept: no draft waits on a model that cannot answer.
    assert (await client.get("/api/v1/studies")).json() == []


async def test_providers_are_empty_when_none_are_configured(client: httpx.AsyncClient) -> None:
    """A provider with no key is left out rather than offered and broken."""
    response = await client.get("/api/v1/studies/providers")
    assert response.status_code == 200
    assert response.json() == []
