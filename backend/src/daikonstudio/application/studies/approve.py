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
from daikonstudio.application.tools.manifest import ToolManifest
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.domain.studies.study import SlotValue, Step, StepStatus, Study

# The kinds a candidate can be read out of below. Approval is offered on no other.
_READABLE = frozenset({"pocket_candidates", "core_candidates"})


def _entry(kind: str, data: object, candidate: int) -> dict[str, Any] | None:
    """The candidate a result reported as `candidate`: a pocket by its rank, a core by its
    place in the list, counted from 1."""
    if kind == "pocket_candidates" and isinstance(data, dict):
        return next((p for p in data.get("pockets") or [] if p.get("rank") == candidate), None)
    if kind == "core_candidates" and isinstance(data, list) and 1 <= candidate <= len(data):
        entry = data[candidate - 1]
        return entry if isinstance(entry, dict) else None
    return None


def reported_candidates(study: Study, step: Step, manifest: ToolManifest) -> SlotValue | None:
    """The newest finished result the step's tool reported candidates in, if there is one.

    Only what the tool itself put out: an approval is stored against the step as well, and
    choosing again has to read the candidates, not the last choice. Only kinds a candidate can
    be read from count, so a choice is never offered that could not be written.
    """
    if step.status is not StepStatus.DONE or manifest.resolves_to is None:
        return None
    return max(
        (v for v in study.outputs_of(step.id) if v.asserted_by is None and v.kind in _READABLE),
        key=lambda v: v.created_at,
        default=None,
    )


async def approve_candidate(
    study: Study,
    step_id: uuid.UUID,
    candidate: int,
    *,
    actor: uuid.UUID,
    workspace_id: uuid.UUID,
    registry: ToolRegistry,
    store: BlobStore,
    offered_on: str | None = None,
) -> SlotValue:
    """Write the chosen candidate as the value it resolves to, from the candidates the step
    reported last.

    The blob is written here; the caller records the value with `Study.approve`. Raises
    `ValidationError` when the step has no finished result to choose from, reported no
    candidate by that rank, or has reported a newer result than `offered_on`, the id of the
    one a proposed choice was offered on: the same rank there may be another candidate.
    """
    step = study.step(step_id)
    manifest = registry.get(step.tool_id)
    label = f"Step {step.position + 1} ({step.name})"
    reported = reported_candidates(study, step, manifest)
    if reported is None or manifest.resolves_to is None:
        raise ValidationError(f"{label} has no finished candidates to choose from.")
    if offered_on is not None and offered_on != str(reported.id):
        raise ValidationError("This result has changed since the candidate was offered.")
    stored = await asyncio.to_thread(store.get_bytes, reported.blob_key)
    entry = _entry(reported.kind, codecs.decode(reported.kind, stored), candidate)
    if entry is None:
        raise ValidationError(f"{label} reported no candidate {candidate}.")
    kind = manifest.resolves_to.value
    blob, summary = codecs.encode(
        kind, {"meta": {"rank": candidate, "from": str(reported.id)}, **entry}
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
