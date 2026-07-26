"""The Collection aggregate -- a scientist's saved triage decision.

A Collection is derived from a prediction Run, not uploaded fresh: it exists
to record *which* rows of a Run's results a scientist chose to act on, and
the lineage that produced them (`derived_from_run_id`). Only `derived_from_run_id`
(a bare UUID) crosses the boundary, never a `Run` object -- the same reason
`Dataset` never imports `domain.catalog`, this module may not import
`domain.execution` (the "Bounded context independence" contract).

`snapshot_uri` points at the Collection's *own* copy of the selected rows,
not a pointer back into the Run's results blob. The Run's results are
immutable, which argues for referencing -- but a Collection is meant to
outlive the Run that produced it: nothing today deletes a Run's blob, but
nothing guarantees a future retention policy won't reclaim an old
prediction's raw Parquet to save storage, the way a cache is swept. A
Collection that only pointed at that blob would silently go dark the day
that happens, and "the compounds I saved for synthesis" is exactly the kind
of artifact that must not depend on a cache surviving. Copying the selected
rows at creation time (`application/data/create_collection.py`) is the
one-time cost that buys that independence.

`provenance` stamps every Collection `AI_PREDICTED` (see
`domain/shared/provenance.py`) -- a Collection's members were never measured,
only predicted, and that must be legible wherever the Collection's data is
read or exported, not just inferable from which endpoint produced it.
"""

from __future__ import annotations

import uuid
from datetime import date
from datetime import datetime as datetime_
from typing import Any

from daikonstudio.domain.shared.entity import AggregateRoot
from daikonstudio.domain.shared.provenance import (
    Citation,
    GenerationMethod,
    Provenance,
    ProvenanceSourceType,
)


class Collection(AggregateRoot):
    def __init__(
        self,
        *,
        workspace_id: uuid.UUID,
        name: str,
        derived_from_run_id: uuid.UUID,
        member_count: int,
        snapshot_uri: str,
        provenance: Provenance,
        id: uuid.UUID | None = None,
        created_at: datetime_ | None = None,
        updated_at: datetime_ | None = None,
        version: int = 1,
    ) -> None:
        super().__init__(id=id, created_at=created_at, updated_at=updated_at, version=version)
        self.workspace_id = workspace_id
        self.name = name
        self.derived_from_run_id = derived_from_run_id
        self.member_count = member_count
        self.snapshot_uri = snapshot_uri
        self.provenance = provenance


def provenance_to_dict(provenance: Provenance) -> dict[str, Any]:
    """Plain JSON for the `collections.provenance` JSONB column.

    Colocated here rather than in `domain/shared/provenance.py`: that module
    is a verbatim copy shared with a sibling project (see its own docstring),
    kept identical on purpose so it stays a drop-in match. This bounded
    context is the one that needs to persist a `Provenance` today, so its own
    (de)serialization lives with the aggregate that uses it -- the same
    reasoning `readout.py` gives for flattening `unit`/`direction` to plain
    strings at its own JSONB boundary rather than upstream.
    """
    return {
        "source_type": provenance.source_type.value,
        "generation_method": provenance.generation_method.value,
        "citations": [
            {"pmid": c.pmid, "doi": c.doi, "url": c.url, "label": c.label}
            for c in provenance.citations
        ],
        "contributor_researcher": provenance.contributor_researcher,
        "contributor_organization_id": (
            str(provenance.contributor_organization_id)
            if provenance.contributor_organization_id
            else None
        ),
        "observed_on": provenance.observed_on.isoformat() if provenance.observed_on else None,
        "note": provenance.note,
    }


def provenance_from_dict(data: dict[str, Any]) -> Provenance:
    observed_on = data.get("observed_on")
    organization_id = data.get("contributor_organization_id")
    return Provenance(
        source_type=ProvenanceSourceType(data["source_type"]),
        generation_method=GenerationMethod(data["generation_method"]),
        citations=tuple(
            Citation(pmid=c.get("pmid"), doi=c.get("doi"), url=c.get("url"), label=c.get("label"))
            for c in data.get("citations", [])
        ),
        contributor_researcher=data.get("contributor_researcher"),
        contributor_organization_id=uuid.UUID(organization_id) if organization_id else None,
        observed_on=date.fromisoformat(observed_on) if observed_on else None,
        note=data.get("note"),
    )
