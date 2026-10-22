"""What the phase schemas leave out is what the agent cannot do.

The line between observing and acting is a shape, not a promise in a prompt: the schema the
model fills on an event has no field that acts, and the one it fills before it plans has no
field for a step. A model cannot be talked into a field that is not there.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import ValidationError

from daikonstudio.application.studies.schemas import (
    Action,
    AskingTurn,
    Claim,
    Look,
    ObserveTurn,
    PlanningTurn,
    Proposal,
    ReplyTurn,
)


def test_the_asking_schema_cannot_carry_steps() -> None:
    assert "steps" not in AskingTurn.model_fields


def test_an_event_turn_has_no_field_to_act_with() -> None:
    assert not {"actions", "run"} & set(ObserveTurn.model_fields)


def test_a_reply_turn_cannot_pause() -> None:
    assert "pause" not in ReplyTurn.model_fields


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


def test_a_proposal_cannot_change_the_terms() -> None:
    """A criterion rewritten after the data is not a criterion, and a button must not be
    able to rewrite one."""
    with pytest.raises(ValidationError):
        Proposal.model_validate({"does": "terms", "why": "x"})
