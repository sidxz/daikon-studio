"""A real-world tool registry for the planner's size tests.

The vendored catalogue is the 177 Scientific Agent Skills by K-Dense (MIT, see the
provenance block in `data/scientific_agent_skills.json`), each turned into a
`ToolManifest` so the planner can be tried against hundreds of tools nobody here
wrote. Names and descriptions are the skills' own; the slot signatures are ours,
one row per catalogue category, because the skills declare none.

Not a catalogue to run: nothing here has an implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from daikonstudio.application.tools.manifest import (
    Slot,
    SlotKind,
    ToolArea,
    ToolManifest,
    ToolSurface,
)
from daikonstudio.application.tools.registry import ToolRegistry

_DATA = Path(__file__).parent / "data" / "scientific_agent_skills.json"

_Signature = tuple[tuple[SlotKind, ...], tuple[SlotKind, ...], ToolArea]

_DEFAULT: _Signature = ((SlotKind.COMPOUND_TABLE,), (SlotKind.PROPERTY_TABLE,), ToolArea.DATA)
_CITING: _Signature = ((SlotKind.CITATIONS,), (SlotKind.CITATIONS,), ToolArea.CONTEXT)
_ANALYSIS: _Signature = (
    (SlotKind.PROPERTY_TABLE,),
    (SlotKind.MODEL_FIT, SlotKind.CORRELATION),
    ToolArea.DATA,
)

# Keyed by the catalogue category heading; every other category takes `_DEFAULT`.
SIGNATURES: dict[str, _Signature] = {
    "Cheminformatics & Drug Discovery": (
        (SlotKind.COMPOUND_TABLE,),
        (SlotKind.PROPERTY_TABLE, SlotKind.ADMET_TABLE),
        ToolArea.SAR,
    ),
    "Protein Engineering & Design": (
        (SlotKind.STRUCTURE,),
        (SlotKind.SITE_ANNOTATIONS,),
        ToolArea.STRUCTURE,
    ),
    "Scientific Databases & Data Access": (
        (),
        (SlotKind.COMPOUND_TABLE, SlotKind.CITATIONS),
        ToolArea.CONTEXT,
    ),
    "Scientific Communication & Publishing": _CITING,
    "Web Search & Information Retrieval": _CITING,
    "Research Methodology & Proposal Writing": _CITING,
    "Data Analysis & Visualization": _ANALYSIS,
    "Machine Learning & Deep Learning": _ANALYSIS,
    "Analysis & Methodology": _ANALYSIS,
}


@dataclass(frozen=True, kw_only=True)
class Example:
    """A worked example from the catalogue's own docs: what a scientist set out to do."""

    title: str
    objective: str
    skills: tuple[str, ...]


@cache
def _fixture() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(_DATA.read_text(encoding="utf-8"))
    return data


def _slots(kinds: tuple[SlotKind, ...]) -> tuple[Slot, ...]:
    return tuple(Slot(key=k.value, kind=k, label=k.value.replace("_", " ")) for k in kinds)


def skills_registry(*, replicate: int = 1) -> ToolRegistry:
    """One manifest per vendored skill.

    `replicate` above 1 appends that many copies of the whole set, ids suffixed
    `-2`, `-3`, ... and names marked as variants, with the same descriptions. For
    size tests only: the copies are as hard to tell apart as real near-duplicates.
    """
    fixture = _fixture()
    version = fixture["source"]["version"]
    manifests: list[ToolManifest] = []
    for n in range(1, replicate + 1):
        for skill in fixture["skills"]:
            inputs, outputs, area = SIGNATURES.get(skill["category"], _DEFAULT)
            manifests.append(
                ToolManifest(
                    id=skill["id"] if n == 1 else f"{skill['id']}-{n}",
                    version=version,
                    name=skill["name"] if n == 1 else f"{skill['name']} (variant {n})",
                    description=skill["description"],
                    area=area,
                    inputs=_slots(inputs),
                    outputs=_slots(outputs),
                    library=skill["id"],
                    surface=ToolSurface.NONE,
                )
            )
    return ToolRegistry(tuple(manifests))


def load_examples() -> list[Example]:
    return [
        Example(title=e["title"], objective=e["objective"], skills=tuple(e["skills"]))
        for e in _fixture()["examples"]
    ]
