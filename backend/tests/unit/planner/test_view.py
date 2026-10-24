"""The view a planning turn reads in full: a typed reach toward the goal, and what was asked for.

The index names every tool; the view is the handful the model reads in full. These pin the
reach's semantics on a hand-built registry, and that the view holds its size and speed at
about a thousand real-world tools.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from daikonstudio.application.planner.catalogue import index
from daikonstudio.application.planner.view import VIEW_CAP, outside, reach, select, widen
from daikonstudio.application.tools.manifest import (
    Slot,
    SlotKind,
    ToolArea,
    ToolManifest,
    ToolSurface,
)
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.infrastructure.tools.catalog import default_registry
from tests.fakes.registry import load_examples, skills_registry


def _tool(
    id: str, takes: tuple[SlotKind, ...], gives: tuple[SlotKind, ...], **kw: Any
) -> ToolManifest:
    return ToolManifest(
        id=id,
        version="1",
        name=id,
        description=f"{id}.",
        area=ToolArea.DATA,
        library="t",
        inputs=tuple(Slot(key=k.value, kind=k, label=k.value) for k in takes),
        outputs=tuple(Slot(key=k.value, kind=k, label=k.value) for k in gives),
        **kw,
    )


MINI = ToolRegistry(
    (
        _tool("get", (), (SlotKind.COMPOUND_TABLE,)),
        _tool("clean", (SlotKind.COMPOUND_TABLE,), (SlotKind.COMPOUND_TABLE,), automatic=True),
        _tool(
            "cores",
            (SlotKind.COMPOUND_TABLE,),
            (SlotKind.CORE_CANDIDATES,),
            judgment=True,
            resolves_to=SlotKind.CORE,
            surface=ToolSurface.EDITOR,
        ),
        _tool("rgroups", (SlotKind.COMPOUND_TABLE, SlotKind.CORE), (SlotKind.RGROUP_TABLE,)),
        _tool("pairs", (SlotKind.RGROUP_TABLE,), (SlotKind.PAIRS,)),
        _tool("profile", (SlotKind.COMPOUND_TABLE,), (SlotKind.PROPERTY_TABLE,)),
        _tool(
            "pockets",
            (SlotKind.STRUCTURE,),
            (SlotKind.POCKET_CANDIDATES,),
            judgment=True,
            resolves_to=SlotKind.POCKET,
            surface=ToolSurface.VIEW,
        ),
        _tool(
            "sar",
            (SlotKind.COMPOUND_TABLE,),
            (SlotKind.PAIRS,),
            expands_to=("cores", "rgroups", "pairs"),
        ),
    )
)


def test_reach_follows_the_typed_path_to_the_target_and_nothing_else() -> None:
    ids = {
        m.id: d
        for m, d in reach(MINI, in_hand={SlotKind.COMPOUND_TABLE}, targets={SlotKind.PAIRS})
    }
    assert set(ids) == {"cores", "rgroups", "pairs", "sar"}  # resolves_to counts as production
    assert ids["pairs"] == 0 and ids["rgroups"] == 1 and ids["cores"] == 2
    assert "profile" not in ids and "pockets" not in ids


def test_reach_is_empty_without_a_target_or_without_a_path() -> None:
    assert reach(MINI, in_hand={SlotKind.COMPOUND_TABLE}, targets=set()) == []
    assert (
        reach(MINI, in_hand=set(), targets={SlotKind.POCKET_CANDIDATES}) == []
    )  # nothing makes a structure


def test_select_falls_back_to_wanted_wiring_and_automatic_when_nothing_reaches() -> None:
    view = {
        m.id
        for m in select(
            MINI,
            wanted=["profile", "no_such_tool"],
            in_plan=[],
            in_hand=set(),
            targets={SlotKind.POCKET_CANDIDATES},
        )
    }
    assert view == {"profile", "get", "clean"}


def test_select_never_cuts_automatic_tools_and_prefers_a_composite_with_its_parts() -> None:
    view = [
        m.id
        for m in select(
            MINI,
            wanted=["rgroups", "pairs"],
            in_plan=[],
            in_hand={SlotKind.COMPOUND_TABLE, SlotKind.CORE},
            targets=set(),
            cap=2,
        )
    ]
    assert "clean" in view and len(view) <= 3
    full = {
        m.id
        for m in select(
            MINI,
            wanted=["cores", "rgroups", "pairs"],
            in_plan=[],
            in_hand={SlotKind.COMPOUND_TABLE},
            targets=set(),
        )
    }
    assert {"sar", "cores", "rgroups", "pairs"} <= full


def test_outside_and_widen() -> None:
    view = select(
        MINI, wanted=["profile"], in_plan=[], in_hand={SlotKind.COMPOUND_TABLE}, targets=set()
    )
    assert outside(MINI, view, ["pairs", "no_such_tool", "profile"]) == ["pairs"]
    assert {"pairs", "rgroups"} <= {
        m.id for m in widen(MINI, view, ["pairs"], in_hand={SlotKind.COMPOUND_TABLE})
    }


def test_the_index_and_the_view_hold_at_a_thousand_real_world_tools() -> None:
    five = ToolRegistry(
        (*default_registry().manifests(), *skills_registry(replicate=5).manifests())
    )  # about 900
    three = ToolRegistry(
        (*default_registry().manifests(), *skills_registry(replicate=3).manifests())
    )  # about 560
    assert (
        len(index(three.manifests())) // 4 < 35_000
    )  # about 60 tokens a line on real descriptions
    assert len(index(five.manifests())) // 4 < 60_000
    example = load_examples()[0]
    started = perf_counter()
    view = select(
        five,
        wanted=list(example.skills),
        in_plan=[],
        in_hand={SlotKind.COMPOUND_TABLE},
        targets={SlotKind.PROPERTY_TABLE},
    )
    assert perf_counter() - started < 1.0
    assert set(example.skills) <= {m.id for m in view}
    assert len(view) <= VIEW_CAP + sum(1 for m in five.manifests() if m.automatic)
