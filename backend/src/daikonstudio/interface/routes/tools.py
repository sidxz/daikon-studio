"""Tools endpoint: the self-describing catalogue of analysis tools.

The same pattern as `engines.py` and for the same reason: the catalogue page and,
later, the planner both read what a tool does and what it consumes and produces
from this response rather than from anything hardcoded on the client. A tool added
to `infrastructure/tools/catalog.py` appears in both without a frontend change.

Read-only. Nothing here runs a tool -- a tool runs inside a plan, which does not
exist yet, and the catalogue is useful before it does.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from daikonstudio.application.tools.manifest import Slot, ToolManifest
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies._core import AuthDep

router = APIRouter(prefix="/api/v1/tools", tags=["tools"])

ToolRegistryDep = Annotated[ToolRegistry, Depends(use_case(ToolRegistry))]


class SlotResponse(BaseModel):
    """One thing a tool consumes or produces. `kind` is the vocabulary a plan is
    written in: a planner chains two tools by matching one's output kind to the
    other's input kind."""

    key: str
    kind: str
    label: str
    optional: bool

    @classmethod
    def from_slot(cls, slot: Slot) -> SlotResponse:
        return cls(key=slot.key, kind=slot.kind.value, label=slot.label, optional=slot.optional)


class ToolManifestResponse(BaseModel):
    """A tool's manifest: what it does, what it needs, what it gives back."""

    id: str
    version: str
    name: str
    description: str
    # Which kind of work this belongs to: "data", "sar", "structure" or
    # "context". A grouping for the catalogue, never a restriction on a plan.
    area: str
    inputs: list[SlotResponse]
    outputs: list[SlotResponse]
    # What actually does the work, so a study can cite it.
    library: str
    # Returns ranked candidates for a person to choose between rather than an
    # answer. A client should not present the first one as decided.
    judgment: bool
    # What a chosen candidate becomes. Null unless this tool asks for a decision.
    # A core, for instance, is only ever this: approved, never computed.
    resolves_to: str | None
    # How much of a person the result needs: "none", "view" or "editor".
    surface: str
    # Runs outside this process, so it can be unavailable when the rest is not.
    external: bool
    # The tools this one runs, in order. A composite: one catalogue entry
    # standing for a sequence that is always performed together. Empty for
    # every ordinary tool.
    expands_to: list[str]
    # Which runner lane can serve it. "default" needs nothing beyond what the
    # API already has; anything else names a dependency set that lives in its
    # own image.
    lane: str

    @classmethod
    def from_manifest(cls, manifest: ToolManifest) -> ToolManifestResponse:
        return cls(
            id=manifest.id,
            version=manifest.version,
            name=manifest.name,
            description=manifest.description,
            area=manifest.area.value,
            inputs=[SlotResponse.from_slot(s) for s in manifest.inputs],
            outputs=[SlotResponse.from_slot(s) for s in manifest.outputs],
            library=manifest.library,
            judgment=manifest.judgment,
            resolves_to=manifest.resolves_to.value if manifest.resolves_to else None,
            surface=manifest.surface.value,
            external=manifest.external,
            expands_to=list(manifest.expands_to),
            lane=manifest.lane,
        )


@router.get("", response_model=list[ToolManifestResponse])
async def list_tools(auth: AuthDep, registry: ToolRegistryDep) -> list[ToolManifestResponse]:
    """Every tool this server offers, in the order a chemist would perform them."""
    return [ToolManifestResponse.from_manifest(m) for m in registry.manifests()]


@router.get("/{tool_id}", response_model=ToolManifestResponse)
async def get_tool(tool_id: str, auth: AuthDep, registry: ToolRegistryDep) -> ToolManifestResponse:
    """One tool by id. An unknown id is a 404, through `UnknownToolError`."""
    return ToolManifestResponse.from_manifest(registry.get(tool_id))
