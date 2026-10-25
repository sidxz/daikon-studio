"""What the system prompts may not say, and what each of them says once."""

from __future__ import annotations

from daikonstudio.application.planner import catalogue
from daikonstudio.application.studies import prompts
from daikonstudio.application.studies.prompts import system
from daikonstudio.infrastructure.tools.catalog import default_registry


def test_no_prompt_hardcodes_a_tool_or_an_assay_convention() -> None:
    """A prompt naming a specific tool, column or assay convention is this
    one use case written into the plumbing: it reads fine today and misleads
    the moment the catalogue grows past chemistry."""
    templates = (
        prompts.ASKING,
        prompts.PLANNING,
        prompts.RUNNING,
        catalogue.READING_THE_DATA,
    )
    banned = ("IC50", ">50", "censored", "SMILES", "scaffold", "standardize", "chemist")
    for prompt in templates:
        for word in banned:
            assert word.lower() not in prompt.lower(), f"{word!r} is hardcoded in a prompt"


def test_every_template_carries_the_reading_rules_once() -> None:
    """The prompts once carried their own copy of how to read a file's shape, and the
    copies had drifted. One shared text, interpolated into every phase, is the only
    arrangement that cannot. Rendered rather than searched for, so a template that
    cannot be filled fails here too."""
    registry = default_registry()
    for phase in ("asking", "planning", "running"):
        rendered = system(phase, registry)
        assert rendered.count(catalogue.READING_THE_DATA) == 1, phase


def test_the_planning_prompt_no_longer_carries_a_number_that_changes_per_turn() -> None:
    assert "%(left)" not in prompts.PLANNING and "questions you have left" in prompts.PLANNING
