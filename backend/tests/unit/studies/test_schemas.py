"""What the phase schemas leave out is what the agent cannot do.

The line between observing and acting is a shape, not a promise in a prompt: the schema the
model fills on an event has no field that acts and the turn is offered no tool, and the one
it fills before it plans has no field for a step. A model cannot be talked into a field that
is not there, or call a tool it was not given.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from daikonstudio.application.planner.plan import Interrogation
from daikonstudio.application.studies.schemas import (
    ACT_TOOLS,
    Action,
    AskingTurn,
    Claim,
    Look,
    ObserveTurn,
    PlanningTurn,
    Proposal,
    ReplyTurn,
    act,
    as_action,
    planning_turn_schema,
)
from daikonstudio.infrastructure.tools.catalog import default_registry


def test_the_asking_schema_cannot_carry_steps() -> None:
    assert "steps" not in AskingTurn.model_fields


def test_an_event_turn_has_no_field_to_act_with() -> None:
    assert not {"actions", "run"} & set(ObserveTurn.model_fields)


def test_a_reply_acts_only_through_its_tools() -> None:
    """Every act is a tool call that names the words it carries out; an act in the answer
    itself would be one the grammar pays for and the quote rule could not reach."""
    assert not {"actions", "proposals"} & set(ReplyTurn.model_fields)
    assert [t.name for t in ACT_TOOLS] == [
        "insert_step",
        "remove_step",
        "swap_tool",
        "change_settings",
        "set_gate",
        "replace_terms",
        "approve_candidate",
        "run_steps",
    ]
    for tool in ACT_TOOLS:
        assert {"quoting", "why"} <= set(tool.schema.model_fields), tool.name
    assert as_action(act("run", step=3, until=3, quoting="run it")) == Action(
        does="run", step=3, until=3, quoting="run it"
    )


def test_a_reply_turn_cannot_pause() -> None:
    assert "pause" not in ReplyTurn.model_fields


def test_only_turns_that_can_choose_a_tool_may_ask_to_read_one() -> None:
    assert "wanted" in PlanningTurn.model_fields and "wanted" in ReplyTurn.model_fields
    assert "wanted" not in ObserveTurn.model_fields
    assert "target_kinds" in Interrogation.model_fields
    assert "anyOf" not in json.dumps(Interrogation.model_json_schema())


def test_the_planning_schema_only_admits_the_registry_tool_ids() -> None:
    schema_cls = planning_turn_schema(default_registry())
    defs = schema_cls.model_json_schema()["$defs"]
    step = defs[defs["DraftPlan"]["properties"]["steps"]["items"]["$ref"].rsplit("/", 1)[-1]]
    assert set(step["properties"]["tool_id"]["enum"]) == {
        m.id for m in default_registry().manifests()
    }
    assert planning_turn_schema(default_registry()) is schema_cls  # cached, not rebuilt per call
    with pytest.raises(ValidationError):
        schema_cls.model_validate(
            {
                "say": "",
                "plan": {
                    "restated_goal": "r",
                    "steps": [{"tool_id": "no_such_tool", "purpose": "p", "gated": False}],
                },
            }
        )


def _types(node: Any) -> Iterator[Any]:
    """Every `type` a JSON schema declares, nested models included."""
    if isinstance(node, dict):
        yield node.get("type")
        for value in node.values():
            yield from _types(value)
    elif isinstance(node, list):
        for value in node:
            yield from _types(value)


def test_no_agent_schema_has_a_float_field() -> None:
    """A number the agent writes is a number nobody computed. Whole numbers that point
    at a step, a rank or a row count are references and stay."""
    for model in (ObserveTurn, ReplyTurn, PlanningTurn, Action, Proposal, Claim, Look):
        assert "number" not in list(_types(model.model_json_schema())), model.__name__


def _limits(node: Mapping[str, Any], defs: Mapping[str, Any]) -> tuple[int, int]:
    """The properties a model may leave out, and the unions, with every reference counted
    where it is used: the most a provider could count against its limits."""
    if "$ref" in node:
        return _limits(defs[node["$ref"].rsplit("/", 1)[-1]], defs)
    properties = node.get("properties", {})
    optional = len(properties.keys() - set(node.get("required", ())))
    unions = int("anyOf" in node or "oneOf" in node or isinstance(node.get("type"), list))
    nested = [*properties.values(), *node.get("anyOf", ()), *node.get("oneOf", ())]
    nested += [*node.get("allOf", ()), *([node["items"]] if "items" in node else [])]
    for child in nested:
        more, other = _limits(child, defs)
        optional, unions = optional + more, unions + other
    return optional, unions


def test_every_turn_schema_fits_the_providers_published_limits() -> None:
    """Anthropic documents 24 optional properties and 16 unions per request, and refuses a
    schema it cannot compile ("The compiled grammar is too large") on an internal limit it
    does not publish. This test holds the published ones; the compiled size is proved only by
    a real request, which `tests/integration/test_provider_grammar.py` makes with the key in
    `backend/.env`. "None" is a value, never a union, so the union count stays at zero."""
    for model in (
        PlanningTurn,
        planning_turn_schema(default_registry()),
        ObserveTurn,
        ReplyTurn,
        Interrogation,
    ):
        schema = model.model_json_schema()
        optional, unions = _limits(schema, schema.get("$defs", {}))
        assert unions == 0, model.__name__
        assert optional <= 24, (model.__name__, optional)


def test_a_proposal_cannot_change_the_terms() -> None:
    """A criterion rewritten after the data is not a criterion, and a button must not be
    able to rewrite one."""
    with pytest.raises(ValidationError):
        Proposal.model_validate({"does": "terms", "why": "x"})
