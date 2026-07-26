"""Collection endpoints: save a triage selection, read it back, export it.

There is no PATCH, no DELETE and no listing endpoint here -- not scoped for
this task. A Collection is created once, from a `ready` Run's results, and
read back either as JSON (to confirm what was saved) or as a file (to act on
it). Every request body here is `extra="forbid"`, the same reason every
other route module in this app gives: `workspace_id` is refused, not
silently dropped.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, ConfigDict

from daikonstudio.application.data.create_collection import (
    CreateCollection,
    CreateCollectionCommand,
    GetCollection,
    GetCollectionQuery,
)
from daikonstudio.application.data.export_collection import (
    ExportCollection,
    ExportCollectionQuery,
    ExportFormat,
)
from daikonstudio.domain.data.collection import Collection
from daikonstudio.domain.shared.provenance import Citation, Provenance
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies._core import AuthDep
from daikonstudio.interface.error_handlers import result_to_response

router = APIRouter(prefix="/api/v1/collections", tags=["collections"])

CreateCollectionDep = Annotated[CreateCollection, Depends(use_case(CreateCollection))]
GetCollectionDep = Annotated[GetCollection, Depends(use_case(GetCollection))]
ExportCollectionDep = Annotated[ExportCollection, Depends(use_case(ExportCollection))]


class CreateCollectionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    run_id: uuid.UUID
    row_ids: list[int]


class CitationResponse(BaseModel):
    pmid: str | None
    doi: str | None
    url: str | None
    label: str | None

    @classmethod
    def from_domain(cls, citation: Citation) -> CitationResponse:
        return cls(pmid=citation.pmid, doi=citation.doi, url=citation.url, label=citation.label)


class ProvenanceResponse(BaseModel):
    source_type: str
    generation_method: str
    citations: list[CitationResponse]
    contributor_researcher: str | None
    contributor_organization_id: uuid.UUID | None
    observed_on: date | None
    note: str | None

    @classmethod
    def from_domain(cls, provenance: Provenance) -> ProvenanceResponse:
        return cls(
            source_type=provenance.source_type.value,
            generation_method=provenance.generation_method.value,
            citations=[CitationResponse.from_domain(c) for c in provenance.citations],
            contributor_researcher=provenance.contributor_researcher,
            contributor_organization_id=provenance.contributor_organization_id,
            observed_on=provenance.observed_on,
            note=provenance.note,
        )


class CollectionResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    derived_from_run_id: uuid.UUID
    member_count: int
    snapshot_uri: str
    provenance: ProvenanceResponse
    created_at: datetime

    @classmethod
    def from_domain(cls, collection: Collection) -> CollectionResponse:
        return cls(
            id=collection.id,
            workspace_id=collection.workspace_id,
            name=collection.name,
            derived_from_run_id=collection.derived_from_run_id,
            member_count=collection.member_count,
            snapshot_uri=collection.snapshot_uri,
            provenance=ProvenanceResponse.from_domain(collection.provenance),
            created_at=collection.created_at,
        )


@router.post("", response_model=CollectionResponse, status_code=201)
async def create_collection(
    body: CreateCollectionBody, auth: AuthDep, service: CreateCollectionDep
) -> CollectionResponse:
    command = CreateCollectionCommand(
        name=body.name, run_id=body.run_id, row_ids=tuple(body.row_ids)
    )
    return CollectionResponse.from_domain(result_to_response(await service(command, auth=auth)))


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(
    collection_id: uuid.UUID, auth: AuthDep, service: GetCollectionDep
) -> CollectionResponse:
    collection = result_to_response(
        await service(GetCollectionQuery(collection_id=collection_id), auth=auth)
    )
    return CollectionResponse.from_domain(collection)


@router.get("/{collection_id}/export")
async def export_collection(
    collection_id: uuid.UUID,
    auth: AuthDep,
    service: ExportCollectionDep,
    format: ExportFormat,
) -> Response:
    export = result_to_response(
        await service(ExportCollectionQuery(collection_id=collection_id, format=format), auth=auth)
    )
    return Response(
        content=export.content,
        media_type=export.media_type,
        headers={"Content-Disposition": f'attachment; filename="{export.filename}"'},
    )
