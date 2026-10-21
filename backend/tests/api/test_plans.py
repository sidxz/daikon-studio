"""Drafting a plan over HTTP.

A deployment with no model key is the default state, including in this suite, so
the first thing these cover is that the refusal says which variable to set rather
than failing as though the caller lacked permission.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest
from fastapi import FastAPI
from lagom import Singleton
from tests.fakes.llm import ScriptedLanguageModel, ScriptedLanguageModels

from daikonstudio.application.planner.plan import (
    DraftPlan,
    Interrogation,
    Question,
    Step,
)
from daikonstudio.application.ports.llm import LanguageModels


@pytest.fixture
def planning(app: FastAPI) -> Iterator[ScriptedLanguageModel]:
    """A deployment whose model answers with one fixed, coherent plan.

    Defined on the app's own child container: the parent already binds the real
    providers, and lagom refuses a second define on the same container, so this
    overrides rather than reassigns.
    """
    model = ScriptedLanguageModel(
        DraftPlan(
            restated_goal="Understand what drives activity in this series.",
            questions=[],
            steps=[
                Step(tool_id="standardize", purpose="Clean the structures.", gated=True),
                Step(tool_id="find_cores", purpose="Agree a scaffold.", gated=True),
                Step(tool_id="build_sar_table", purpose="Lay out the SAR.", gated=False),
            ],
            checks=["An effect counts only over at least 4 pairs."],
            stop_conditions=["Stop if decomposition covers under 80%."],
            skipped=[],
        )
    )
    app.state.container.define(  # type: ignore[type-abstract]
        LanguageModels, Singleton(lambda: ScriptedLanguageModels(model))
    )
    yield model


async def test_without_a_key_it_says_which_variable_to_set(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v1/plans", data={"goal": "Find the SAR."})
    assert response.status_code == 503
    body = response.json()
    assert "STUDIO_ANTHROPIC_API_KEY" in body["detail"]


async def test_providers_are_empty_when_none_are_configured(client: httpx.AsyncClient) -> None:
    """A provider with no key is left out rather than offered and broken."""
    response = await client.get("/api/v1/plans/providers")
    assert response.status_code == 200
    assert response.json() == []


async def test_an_empty_goal_is_refused_before_any_model_is_asked(
    client: httpx.AsyncClient, planning: ScriptedLanguageModel
) -> None:
    response = await client.post("/api/v1/plans", data={"goal": "   "})
    assert response.status_code == 422
    assert planning.calls == []


async def test_a_plan_expands_composites_and_records_who_wrote_it(
    client: httpx.AsyncClient, planning: ScriptedLanguageModel
) -> None:
    response = await client.post("/api/v1/plans", data={"goal": "Work out what drives activity."})
    assert response.status_code == 200
    plan = response.json()

    # Three steps were asked for; the composite became its two parts, each saying
    # where it came from, so the record keeps every step that will actually run.
    # `standardize` appears once even though the model also listed it: it is
    # prepended because it always runs, not because anyone chose it.
    assert [step["tool_id"] for step in plan["steps"]] == [
        "standardize",
        "find_cores",
        "decompose_rgroups",
        "name_substituents",
    ]
    assert plan["steps"][0]["always"] is True
    assert plan["steps"][1]["always"] is False
    assert plan["steps"][2]["expanded_from"] == "build_sar_table"
    assert plan["steps"][0]["expanded_from"] is None
    assert plan["problems"] == []
    assert plan["written_by"] == "anthropic/scripted"
    assert plan["checks"] and plan["stop_conditions"]


async def test_a_step_that_asks_a_chemist_gates_whatever_the_planner_said(
    client: httpx.AsyncClient, planning: ScriptedLanguageModel
) -> None:
    response = await client.post("/api/v1/plans", data={"goal": "Find the SAR."})
    gated = {s["tool_id"] for s in response.json()["steps"] if s["gated"]}
    assert "find_cores" in gated


async def test_the_catalogue_reaches_the_model(
    client: httpx.AsyncClient, planning: ScriptedLanguageModel
) -> None:
    """The planner may only choose from tools this server has. If the catalogue
    stops reaching the prompt, the model invents tools and the plan is rejected
    downstream for a reason that looks like a model failure."""
    await client.post("/api/v1/plans", data={"goal": "Find the SAR."})
    _system, prompt = planning.calls[0]
    assert "decompose_rgroups" in prompt
    assert "build_sar_table" in prompt
    # Composites are listed first so a model reading top-down prefers the whole
    # sequence over its parts.
    assert prompt.index("build_sar_table") < prompt.index("- standardize")


async def test_a_plan_that_forgot_to_prepare_the_data_gets_it_anyway(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    """Preparing structures is not a judgement call, so it is not the planner's
    to forget. A plan missing it would be silently wrong rather than visibly
    broken: nothing downstream requires clean input in slot terms."""
    model = ScriptedLanguageModel(
        DraftPlan(
            restated_goal="r",
            steps=[Step(tool_id="find_cores", purpose="Agree a scaffold.", gated=True)],
        )
    )
    app.state.container.define(  # type: ignore[type-abstract]
        LanguageModels, Singleton(lambda: ScriptedLanguageModels(model))
    )

    response = await client.post("/api/v1/plans", data={"goal": "Find the SAR."})
    steps = response.json()["steps"]
    assert steps[0]["tool_id"] == "standardize"
    assert steps[0]["always"] is True


async def test_the_shape_of_an_uploaded_file_reaches_the_model(
    client: httpx.AsyncClient, planning: ScriptedLanguageModel
) -> None:
    """Without the shape, the planner asks what it could have read. The counts
    travel; the structures do not."""
    csv = b"id,smiles,IC50_uM\n1,CCO,1.2\n2,c1ccccc1,>50\n3,CCN,0.4\n"
    response = await client.post(
        "/api/v1/plans",
        data={"goal": "Find the SAR."},
        files={"file": ("messy.csv", csv, "text/csv")},
    )
    assert response.status_code == 200

    _system, prompt = planning.calls[0]
    assert "The uploaded file has 3 rows" in prompt
    assert "IC50_uM" in prompt
    # The odd value survives, because that is the whole point of reporting it.
    assert "'>50'" in prompt
    # The structures do not.
    assert "values not shown" in prompt
    assert "c1ccccc1" not in prompt


async def test_a_file_that_is_not_a_table_is_refused(
    client: httpx.AsyncClient, planning: ScriptedLanguageModel
) -> None:
    response = await client.post(
        "/api/v1/plans",
        data={"goal": "Find the SAR."},
        files={"file": ("notes.bin", b"", "application/octet-stream")},
    )
    assert response.status_code == 422
    assert planning.calls == []


@pytest.fixture
def asking(app: FastAPI) -> ScriptedLanguageModel:
    """A deployment whose model answers with one fixed set of questions."""
    model = ScriptedLanguageModel(
        Interrogation(
            restated_goal="Find what drives activity in this series.",
            questions=[
                Question(
                    ask="Which column is the readout that matters?",
                    why="Every effect size is computed against one readout.",
                    options=["IC50_uM", "Ki_nM", "pct_inhib_10uM"],
                ),
                Question(ask="What is the target?", why="ChEMBL needs a name.", options=[]),
            ],
        )
    )
    app.state.container.define(  # type: ignore[type-abstract]
        LanguageModels, Singleton(lambda: ScriptedLanguageModels(model))
    )
    return model


async def test_it_asks_before_it_plans(
    client: httpx.AsyncClient, asking: ScriptedLanguageModel
) -> None:
    """The first half returns questions and nothing else. A model asked to do
    both at once commits to an answer inside the plan before anyone has read the
    question it is asking about."""
    response = await client.post(
        "/api/v1/plans/questions", data={"goal": "Work out what drives activity."}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["restated_goal"]
    assert [q["ask"] for q in body["questions"]] == [
        "Which column is the readout that matters?",
        "What is the target?",
    ]
    assert "steps" not in body


async def test_a_question_carries_its_options_when_the_answers_enumerate(
    client: httpx.AsyncClient, asking: ScriptedLanguageModel
) -> None:
    """Three columns is a choice; a target name is not. A client offers the
    first as options and the second as a text field."""
    response = await client.post(
        "/api/v1/plans/questions", data={"goal": "Work out what drives activity."}
    )
    questions = response.json()["questions"]
    assert questions[0]["options"] == ["IC50_uM", "Ki_nM", "pct_inhib_10uM"]
    assert questions[1]["options"] == []


async def test_the_answers_reach_the_model_including_what_was_not_known(
    client: httpx.AsyncClient, planning: ScriptedLanguageModel
) -> None:
    """An unanswered question, a "not sure" and an answer are three different
    things, and the plan has to treat them differently. All three travel."""
    answers = json.dumps(
        [
            {"ask": "Which column is the readout?", "answer": "IC50_uM"},
            {"ask": "Was the protocol the same?", "answer": "Not sure"},
            {"ask": "What is the target?", "answer": ""},
        ]
    )
    response = await client.post(
        "/api/v1/plans", data={"goal": "Find the SAR.", "answers": answers}
    )
    assert response.status_code == 200

    _system, prompt = planning.calls[0]
    assert "IC50_uM" in prompt
    assert "Not sure" in prompt
    assert "(no answer)" in prompt


async def test_malformed_answers_are_refused_rather_than_ignored(
    client: httpx.AsyncClient, planning: ScriptedLanguageModel
) -> None:
    """Silently dropping them would produce a plan built on assumptions while
    the scientist believes it was built on their answers."""
    response = await client.post(
        "/api/v1/plans", data={"goal": "Find the SAR.", "answers": "not json"}
    )
    assert response.status_code == 422
    assert planning.calls == []


async def test_a_plan_can_still_be_drafted_with_no_answers_at_all(
    client: httpx.AsyncClient, planning: ScriptedLanguageModel
) -> None:
    """Answering nothing is allowed. The plan then has to say what it assumed."""
    response = await client.post("/api/v1/plans", data={"goal": "Find the SAR."})
    assert response.status_code == 200
