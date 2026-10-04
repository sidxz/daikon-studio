"""Engines endpoint: the self-describing catalog of available engines and their parameters.

The frontend renders the engine picker and the per-engine condition form directly from
this response -- no hardcoded form definitions, no client-side knowledge of what an engine
takes. This is the same pattern as the sibling project's self-describing algorithms endpoint.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from daikonstudio.application.engines.manifest import ConditionSpec, EngineManifest
from daikonstudio.application.engines.registry import EngineRegistry
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies._core import AuthDep

router = APIRouter(prefix="/api/v1/engines", tags=["engines"])

EngineRegistryDep = Annotated[EngineRegistry, Depends(use_case(EngineRegistry))]


class ConditionResponse(BaseModel):
    """A single engine condition parameter specification, carrying everything
    a form needs to render without any hardcoded knowledge of the engine."""

    key: str
    label: str
    type: str
    required: bool
    default: object | None
    minimum: float | None
    maximum: float | None
    options: tuple[str, ...]
    help: str | None

    @classmethod
    def from_spec(cls, spec: ConditionSpec) -> ConditionResponse:
        return cls(
            key=spec.key,
            label=spec.label,
            type=spec.type.value,
            required=spec.required,
            default=spec.default,
            minimum=spec.minimum,
            maximum=spec.maximum,
            options=spec.options,
            help=spec.help,
        )


class EngineManifestResponse(BaseModel):
    """An engine's manifest: its identity, capabilities, and parameter specifications."""

    id: str
    version: str
    name: str
    description: str
    tasks: list[str]
    conditions: list[ConditionResponse]
    is_baseline: bool
    supports_multitask: bool
    # Which runner lane serves this engine ("default" or "gpu"). A client that
    # knows this can say "waiting for a gpu runner" instead of just "Queued".
    lane: str

    @classmethod
    def from_manifest(cls, manifest: EngineManifest) -> EngineManifestResponse:
        return cls(
            id=manifest.id,
            version=manifest.version,
            name=manifest.name,
            description=manifest.description,
            tasks=[task.value for task in manifest.tasks],
            conditions=[ConditionResponse.from_spec(spec) for spec in manifest.conditions],
            is_baseline=manifest.is_baseline,
            supports_multitask=manifest.supports_multitask,
            lane=manifest.lane,
        )


@router.get("", response_model=list[EngineManifestResponse])
async def list_engines(
    auth: AuthDep,
    registry: EngineRegistryDep,
) -> list[EngineManifestResponse]:
    """List all available engines with their condition specifications.

    The response carries everything a form needs to render without any hardcoded
    engine knowledge: for each engine, its id, name, description, supported tasks,
    baseline flag, and all condition parameters (key, label, type, required, default,
    bounds, options, help text).
    """
    return [EngineManifestResponse.from_manifest(m) for m in registry.manifests()]
