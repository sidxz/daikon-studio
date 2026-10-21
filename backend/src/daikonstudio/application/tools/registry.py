from __future__ import annotations

from daikonstudio.application.tools.manifest import SlotKind, ToolManifest
from daikonstudio.domain.shared.errors import DomainError, NotFoundError


class UnknownToolError(NotFoundError):
    """Raised by `.get()` for an id this server does not offer.

    Subclasses `NotFoundError` for the same reason `UnknownEngineError` does: the
    registry is handed to the interface layer directly (`routes/tools.py` depends
    on it, not on a Result-returning use case), so an unmapped error type here
    would surface as a bare 500 instead of something a client can act on.

    Calls `DomainError.__init__` rather than `NotFoundError.__init__` because the
    message names the missing tool id rather than filling that class's
    `(entity_type, entity_id)` sentence.
    """

    def __init__(self, message: str) -> None:
        DomainError.__init__(self, message)


class ToolRegistry:
    """The tools this server offers, in the order a catalogue should show them.

    A dict rather than a scan: a curated handful of tools is not a discovery
    problem, and a plugin loader would make the catalogue depend on what happens
    to be installed -- which is the one thing a planner must be able to trust.
    """

    def __init__(self, manifests: tuple[ToolManifest, ...]) -> None:
        duplicates = sorted({m.id for m in manifests if sum(x.id == m.id for x in manifests) > 1})
        if duplicates:
            raise ValueError(f"Tool ids must be unique; repeated: {', '.join(duplicates)}.")
        self._manifests = manifests

    def get(self, tool_id: str) -> ToolManifest:
        for manifest in self._manifests:
            if manifest.id == tool_id:
                return manifest
        raise UnknownToolError(f"Tool '{tool_id}' is not available on this server.")

    def manifests(self) -> list[ToolManifest]:
        return list(self._manifests)

    def producing(self, kind: SlotKind) -> list[ToolManifest]:
        """Every tool that can produce `kind`.

        The planner's chaining primitive, and the reason slots are typed at all:
        asked for an SAR table it can find which tools emit one, rather than
        pattern-matching on names.
        """
        return [m for m in self._manifests if any(s.kind is kind for s in m.outputs)]

    def requiring(self, kind: SlotKind) -> list[ToolManifest]:
        """Every tool that needs `kind`, optional slots included.

        The other half of chaining: what a step unlocks. Optional inputs count,
        because a tool that can use a property table is still a reason to compute
        one.
        """
        return [m for m in self._manifests if any(s.kind is kind for s in m.inputs)]

    def expand(self, tool_id: str) -> list[str]:
        """The tools a step actually runs, in order.

        A composite expands into its parts; everything else is itself. One level
        deep by construction -- `ToolManifest.expands_to` is not allowed to name
        another composite, so this needs no recursion and cannot loop.
        """
        manifest = self.get(tool_id)
        return list(manifest.expands_to) if manifest.expands_to else [tool_id]
