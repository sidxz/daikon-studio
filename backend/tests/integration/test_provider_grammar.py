"""The provider compiles every schema a turn sends, with the tools it sends beside it.

The only proof that a schema fits Anthropic's grammar ceiling is a request: the documented
limits are a proxy, and a schema passed them while the API refused it ("The compiled grammar
is too large"), which took every message on a started study down. One request per phase,
as the adapter really sends it, with the real system prompt and the real tools. Paid, so
marked `live` and run on purpose: `pytest -m live`. Skipped without a key in `backend/.env`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import dotenv_values

from daikonstudio.application.planner.catalogue import INDEX_HEADING, index
from daikonstudio.application.studies.prompts import system
from daikonstudio.application.studies.schemas import (
    ACT_TOOLS,
    AskingTurn,
    ObserveTurn,
    ReplyTurn,
    planning_turn_schema,
)
from daikonstudio.infrastructure.llm.anthropic import AnthropicLanguageModel
from daikonstudio.infrastructure.tools.catalog import default_registry
from daikonstudio.settings import Settings

pytestmark = pytest.mark.live

_ENV = dotenv_values(Path(__file__).parents[2] / ".env")
_KEY = _ENV.get("STUDIO_ANTHROPIC_API_KEY") or ""
_PROMPT = "This is a compile check. Answer with the shortest valid answer and do nothing."


@pytest.mark.skipif(not _KEY, reason="no STUDIO_ANTHROPIC_API_KEY in backend/.env")
async def test_every_phase_compiles_at_the_provider() -> None:
    registry = default_registry()
    model = AnthropicLanguageModel(
        api_key=_KEY,
        model=_ENV.get("STUDIO_ANTHROPIC_MODEL")
        or Settings.model_fields["anthropic_model"].default,
    )
    stable = f"{INDEX_HEADING}\n{index(registry.manifests())}"
    asked = await model.parse(system=system("asking", registry), prompt=_PROMPT, schema=AskingTurn)
    assert isinstance(asked, AskingTurn)
    planned = await model.parse(
        system=system("planning", registry),
        prompt=_PROMPT,
        schema=planning_turn_schema(registry),
        stable=stable,
    )
    assert planned.plan.steps == [] or planned.plan.steps
    observed = await model.act(
        system=system("running", registry), prompt=_PROMPT, schema=ObserveTurn, tools=()
    )
    assert isinstance(observed.parsed, ObserveTurn)
    replied = await model.act(
        system=system("running", registry),
        prompt=_PROMPT,
        schema=ReplyTurn,
        tools=ACT_TOOLS,
        stable=stable,
    )
    assert isinstance(replied.parsed, ReplyTurn)
