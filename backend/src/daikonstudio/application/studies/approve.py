"""Writing down the scientist's choice of a candidate a step reported.

The seam where a computed result becomes an asserted one. No tool produces a pocket or a
core: a step reports candidates, the scientist chooses one by the rank the step gave it,
and what is written is that candidate's own payload under the kind the tool resolves to,
naming who chose it and which result it was chosen from.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.studies.keys import value_key
from daikonstudio.application.tools import codecs
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.domain.studies.study import SlotValue, StepStatus, Study


def _entry(kind: str, data: object, candidate: int) -> dict[str, Any] | None:
    """The candidate a result reported as `candidate`: a pocket by its rank, a core by its
    place in the list, counted from 1."""
    if kind == "pocket_candidates" and isinstance(data, dict):
        return next((p for p in data.get("pockets") or [] if p.get("rank") == candidate), None)
    if kind == "core_candidates" and isinstance(data, list) and 1 <= candidate <= len(data):
        entry = data[candidate - 1]
        return entry if isinstance(entry, dict) else None
    return None


async def approve_candidate(
    study: Study,
    step_id: uuid.UUID,
    candidate: int,
    *,
    actor: uuid.UUID,
    workspace_id: uuid.UUID,
    registry: ToolRegistry,
    store: BlobStore,
) -> SlotValue:
    """Write the chosen candidate as the value it resolves to, from the step's newest result.

    The blob is written here; the caller records the value with `Study.approve`. Raises
    `ValidationError` when the step has no finished result to choose from, or reported no
    candidate by that rank.
    """
    step = study.step(step_id)
    manifest = registry.get(step.tool_id)
    label = f"Step {step.position + 1} ({step.name})"
    if manifest.resolves_to is None:
        raise ValidationError(f"{label} reports no candidates to choose from.")
    newest = max(study.outputs_of(step_id), key=lambda v: v.created_at, default=None)
    if step.status is not StepStatus.DONE or newest is None:
        raise ValidationError(f"{label} has not finished, so there is nothing to choose yet.")
    data = codecs.decode(newest.kind, await asyncio.to_thread(store.get_bytes, newest.blob_key))
    entry = _entry(newest.kind, data, candidate)
    if entry is None:
        raise ValidationError(f"{label} reported no candidate {candidate}.")
    kind = manifest.resolves_to.value
    blob, summary = codecs.encode(
        kind, {"meta": {"rank": candidate, "from": str(newest.id)}, **entry}
    )
    value_id = uuid.uuid4()
    key = value_key(workspace_id, study.id, value_id, codecs.extension(kind))
    await asyncio.to_thread(store.put_bytes, key, blob)
    return SlotValue(
        id=value_id,
        key=kind,
        kind=kind,
        blob_key=key,
        summary=summary,
        produced_by=step_id,
        asserted_by=actor,
        created_at=datetime.now(UTC),
    )
