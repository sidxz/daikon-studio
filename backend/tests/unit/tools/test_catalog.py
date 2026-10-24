"""What the tool catalogue must never get wrong.

The catalogue is read by a planner that will compose these tools into a plan, so
the invariants here are about composability, not about any one tool being right:
a tool whose input nothing can produce is a dead end a planner would route into
and then fail inside, which is much harder to diagnose than a catalogue that
refuses to be built.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from itertools import combinations

import pytest

from daikonstudio.application.tools.manifest import (
    Slot,
    SlotKind,
    ToolArea,
    ToolManifest,
    ToolSurface,
)
from daikonstudio.application.tools.registry import ToolRegistry, UnknownToolError
from daikonstudio.domain.shared.errors import NotFoundError
from daikonstudio.infrastructure.tools.catalog import default_registry


def test_every_input_kind_is_reachable() -> None:
    """A tool that needs something nothing in the catalogue can make is a trap:
    the planner can reach it and cannot feed it. Compound tables are exempt --
    they are what a scientist brings.

    `resolves_to` counts as reachable. A core is never computed; it exists because
    a chemist approved one of the proposed candidates, and leaving that out of the
    reachability check is what first caught this distinction.
    """
    registry = default_registry()
    produced = {slot.kind for m in registry.manifests() for slot in m.outputs}
    produced |= {m.resolves_to for m in registry.manifests() if m.resolves_to is not None}
    for manifest in registry.manifests():
        for slot in manifest.inputs:
            if slot.kind is SlotKind.COMPOUND_TABLE:
                continue
            assert slot.kind in produced, (
                f"{manifest.id} needs {slot.kind.value}, which no tool produces."
            )


def test_at_least_one_tool_is_a_source() -> None:
    """Something has to be reachable from an empty plan, or nothing starts."""
    registry = default_registry()
    sources = [m for m in registry.manifests() if not m.inputs]
    starters = [
        m
        for m in registry.manifests()
        if all(s.kind is SlotKind.COMPOUND_TABLE or s.optional for s in m.inputs)
    ]
    assert sources or starters


def test_ids_are_unique() -> None:
    manifest = default_registry().manifests()[0]
    with pytest.raises(ValueError, match="unique"):
        ToolRegistry((manifest, manifest))


def test_unknown_tool_is_a_not_found_error() -> None:
    """Not a bare KeyError: the registry is handed to the route directly, so an
    unmapped error type here surfaces as a 500 rather than a 404."""
    with pytest.raises(UnknownToolError) as caught:
        default_registry().get("no-such-tool")
    assert isinstance(caught.value, NotFoundError)
    assert "no-such-tool" in str(caught.value)


def test_a_tool_that_asks_for_a_decision_must_offer_somewhere_to_make_it() -> None:
    with pytest.raises(ValueError, match="no surface"):
        ToolManifest(
            id="x",
            version="1",
            name="X",
            description="d",
            area=ToolArea.SAR,
            inputs=(),
            outputs=(Slot(key="o", kind=SlotKind.CORE, label="Core"),),
            library="l",
            judgment=True,
            surface=ToolSurface.NONE,
        )


def test_descriptions_say_something() -> None:
    """The planner chooses tools from these sentences, so a thin description is a
    functional defect rather than a documentation one."""
    for manifest in default_registry().manifests():
        assert len(manifest.description) > 120, f"{manifest.id} is described too thinly."
        assert manifest.library, f"{manifest.id} does not say what does the work."


def test_chaining_helpers_find_the_sar_table() -> None:
    registry = default_registry()
    makers = {m.id for m in registry.producing(SlotKind.RGROUP_TABLE)}
    users = {m.id for m in registry.requiring(SlotKind.RGROUP_TABLE)}
    assert "decompose_rgroups" in makers
    assert "matched_pairs" in users


def test_only_a_judgment_tool_resolves_to_anything() -> None:
    """`resolves_to` marks the seam where a computed value becomes an asserted
    one. A tool that is not asking anyone anything has no such seam."""
    with pytest.raises(ValueError, match="does not return candidates"):
        ToolManifest(
            id="x",
            version="1",
            name="X",
            description="d",
            area=ToolArea.SAR,
            inputs=(),
            outputs=(Slot(key="o", kind=SlotKind.CORE_CANDIDATES, label="Candidates"),),
            library="l",
            resolves_to=SlotKind.CORE,
        )


def test_a_core_is_approved_rather_than_computed() -> None:
    """No tool outputs a core. The only route to one is a chemist choosing a
    candidate, which is the invariant the rest of an SAR study rests on."""
    registry = default_registry()
    assert not registry.producing(SlotKind.CORE)
    approvers = {m.id for m in registry.manifests() if m.resolves_to is SlotKind.CORE}
    assert approvers == {"find_cores", "score_core"}


def test_a_structure_comes_from_a_file_or_an_archive() -> None:
    """Two sources make a structure from an id in the goal; everything else that
    emits one was given one and cleaned it. Curated sites from UniProt feed the
    pocket finder, which is what makes that entry worth fetching."""
    registry = default_registry()
    sources = {
        m.id
        for m in registry.producing(SlotKind.STRUCTURE)
        if SlotKind.STRUCTURE not in {s.kind for s in m.inputs}
    }
    assert sources
    assert all(not registry.get(t).inputs and registry.get(t).external for t in sources)


def test_settings_are_rendered_for_tools_and_composites() -> None:
    from daikonstudio.application.planner.catalogue import catalogue

    text = catalogue(default_registry().manifests())
    assert "settings: pdb_id (required): Four-character PDB id" in text
    assert "check_properties" in text
    assert "activity_column (required)" in text


def test_sar_settings_reach_the_planner_through_the_composites() -> None:
    from daikonstudio.application.planner.catalogue import catalogue

    text = catalogue(default_registry().manifests())
    assert "higher_is_better (required)" in text
    assert "min_delta (required)" in text
    assert "core (required): The core as the scientist typed it" in text


def test_only_cores_and_pockets_resolve() -> None:
    kinds = {m.resolves_to for m in default_registry().manifests() if m.resolves_to}
    assert kinds == {SlotKind.CORE, SlotKind.POCKET}


THRESHOLD = 0.75  # highest same-signature ratio today is 0.05 (find_pockets, predict_pockets)


def _signature(m: ToolManifest) -> tuple[frozenset[SlotKind], frozenset[SlotKind]]:
    takes = frozenset(s.kind for s in m.inputs if not s.optional)
    gives = frozenset(s.kind for s in m.outputs) | (
        {m.resolves_to} if m.resolves_to else frozenset()
    )
    return takes, gives


def test_no_two_tools_with_the_same_signature_read_alike() -> None:
    for a, b in combinations(default_registry().manifests(), 2):
        if a.id in b.expands_to or b.id in a.expands_to or _signature(a) != _signature(b):
            continue
        ratio = SequenceMatcher(None, a.description.lower(), b.description.lower()).ratio()
        assert ratio < THRESHOLD, (
            f"{a.id} and {b.id} take and give the same kinds and read alike ({ratio:.2f}); "
            f"say in each description when it, and not the other, is the one to use."
        )
