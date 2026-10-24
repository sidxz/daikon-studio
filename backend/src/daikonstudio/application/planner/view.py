"""The tools a turn reads in full, chosen by the host from the registry.

Past a few dozen tools the planner reads a one-line index of everything and the full
description of only a few: the view. What goes in it is decided here, deterministically,
from what the model asked for, what the plan already uses, and a typed reach from what the
scientist brought toward what the study should end with. Pure: registry in, manifests out.

The reach is an intersection on purpose. Forward reachability alone (every tool whose inputs
can be made) is nearly the whole catalogue once many tools take a compound table, and scored
worse than no filter; the backward walk from the target kinds is what makes it goal-directed.
`resolves_to` counts as production on both sides: nothing outputs a core or a pocket, a
judgment step resolves to one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from daikonstudio.application.tools.manifest import SlotKind, ToolManifest
from daikonstudio.application.tools.registry import ToolRegistry

# Non-automatic entries in a view. Automatic tools ride on top: they are never cut, so the
# names `always_runs()` gives the model always resolve.
VIEW_CAP = 40
# How many tool ids a turn may name in `wanted`.
WANTED_CAP = 30


def _gives(m: ToolManifest) -> set[SlotKind]:
    kinds = {s.kind for s in m.outputs}
    if m.resolves_to is not None:
        kinds.add(m.resolves_to)
    return kinds


def _needs(m: ToolManifest) -> set[SlotKind]:
    return {s.kind for s in m.inputs if not s.optional}


def _forward(manifests: list[ToolManifest], in_hand: set[SlotKind]) -> set[str]:
    """Ids of every tool whose required inputs can all be made from what is in hand."""
    available = set(in_hand)
    reached: set[str] = set()
    grew = True
    while grew:
        grew = False
        for m in manifests:
            if m.id not in reached and _needs(m) <= available:
                reached.add(m.id)
                available |= _gives(m)
                grew = True
    return reached


def _reach(
    manifests: list[ToolManifest],
    forward: set[str],
    in_hand: set[SlotKind],
    targets: set[SlotKind],
) -> list[tuple[ToolManifest, int]]:
    needed = set(targets)
    distance: dict[str, int] = {}
    step = 0
    while joined := [m for m in manifests if m.id not in distance and _gives(m) & needed]:
        for m in joined:
            distance[m.id] = step
            # What is in hand needs no producer.
            needed |= _needs(m) - in_hand
        step += 1
    on_path = [(m, distance[m.id]) for m in manifests if m.id in distance and m.id in forward]
    return sorted(on_path, key=lambda pair: pair[1])  # stable: registry order within a distance


def _wiring(
    manifests: list[ToolManifest], forward: set[str], selected: Iterable[ToolManifest]
) -> list[ToolManifest]:
    """The forward-reachable producers of what `selected` requires, so the model can connect it."""
    kinds: set[SlotKind] = set()
    for m in selected:
        kinds |= _needs(m)
    return [m for m in manifests if m.id in forward and _gives(m) & kinds]


def reach(
    registry: ToolRegistry, *, in_hand: set[SlotKind], targets: set[SlotKind]
) -> list[tuple[ToolManifest, int]]:
    """Tools both forward-reachable from `in_hand` and on a backward path to a target kind.

    Each comes with its backward distance: producers of a target are 0, producers of what
    they need are 1, and so on. Ordered by distance, registry order within one.
    """
    manifests = registry.manifests()
    return _reach(manifests, _forward(manifests, in_hand), in_hand, targets)


def select(
    registry: ToolRegistry,
    *,
    wanted: Sequence[str],
    in_plan: Sequence[str],
    in_hand: set[SlotKind],
    targets: set[SlotKind],
    cap: int = VIEW_CAP,
) -> list[ToolManifest]:
    """The view for a planning turn.

    What the model named, what the plan uses, the typed reach, the producers that wire those
    together, composites whose parts are all here and the parts of any composite that is, and
    every automatic tool. Over `cap` the rest is cut in that priority -- wanted, in plan,
    composites and parts, wiring, then reach by distance -- and the index still names it.
    Unknown ids are ignored.
    """
    manifests = registry.manifests()
    forward = _forward(manifests, in_hand)
    named, used = set(wanted), set(in_plan)
    asked = [m for m in manifests if m.id in named]
    planned = [m for m in manifests if m.id in used]
    reached = [m for m, _ in _reach(manifests, forward, in_hand, targets)]
    wiring = _wiring(manifests, forward, [*asked, *planned, *reached])
    chosen = {m.id for m in [*asked, *planned, *reached, *wiring]}
    composites = {
        m.id: m.expands_to
        for m in manifests
        if m.expands_to and (m.id in chosen or set(m.expands_to) <= chosen)
    }
    parts = {p for expands_to in composites.values() for p in expands_to}
    whole = [m for m in manifests if m.id in composites or m.id in parts]
    automatic = [m for m in manifests if m.automatic]

    view: list[ToolManifest] = []
    seen: set[str] = set()
    room = cap
    for m in [*asked, *planned, *whole, *wiring, *reached, *automatic]:
        if m.id in seen:
            continue
        seen.add(m.id)
        if m.automatic:
            view.append(m)
        elif room > 0:
            view.append(m)
            room -= 1
    return view


def outside(registry: ToolRegistry, view: Sequence[ToolManifest], ids: Iterable[str]) -> list[str]:
    """The ids that name a tool in the registry but not in the view, in first-seen order."""
    known = {m.id for m in registry.manifests()}
    shown = {m.id for m in view}
    return list(dict.fromkeys(i for i in ids if i in known and i not in shown))


def widen(
    registry: ToolRegistry,
    view: Sequence[ToolManifest],
    ids: Iterable[str],
    *,
    in_hand: set[SlotKind],
) -> list[ToolManifest]:
    """The view plus the named tools it lacked, and their wiring. Not capped: this is the
    one second pass, and what it adds was asked for by name."""
    manifests = registry.manifests()
    extra = [registry.get(i) for i in outside(registry, view, ids)]
    widened = list(view)
    seen = {m.id for m in view}
    for m in [*extra, *_wiring(manifests, _forward(manifests, in_hand), extra)]:
        if m.id not in seen:
            seen.add(m.id)
            widened.append(m)
    return widened
