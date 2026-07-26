"""Dataset endpoints.

Two-phase creation: `POST /uploads` parks the raw file and hands back an opaque
`upload_ref`, then `POST ""` references it. That keeps the expensive, slow part
(chemistry validation, splitting, freezing) off the multipart request, and means
a failed create can be retried against the same bytes.

There is no PATCH and no DELETE. A Dataset is immutable and cited by id.

Every request body here is `extra="forbid"`. That is what makes
`workspace_id` unspoofable in the honest sense: not silently dropped, but
refused, so no client ever comes to believe it had an effect.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, UploadFile
from pydantic import BaseModel, ConfigDict

from daikonstudio.application.data.create_dataset import (
    CreateDataset,
    CreateDatasetCommand,
    StoreUpload,
)
from daikonstudio.application.data.get_dataset import GetDataset, GetDatasetQuery
from daikonstudio.application.data.list_datasets import ListDatasets, ListDatasetsQuery
from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy, split_to_dict
from daikonstudio.domain.data.target import Direction, TargetKind, TargetSpec, target_to_dict
from daikonstudio.domain.data.validation import report_to_dict
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies._core import AuthDep
from daikonstudio.interface.error_handlers import result_to_response
from daikonstudio.interface.pagination import PaginatedResponse

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

# ponytail: a crude ceiling so one request cannot spool an unbounded file. The
# real enforcement belongs at the reverse proxy (client_max_body_size); this is
# the backstop for when the app is reached directly.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

StoreUploadDep = Annotated[StoreUpload, Depends(use_case(StoreUpload))]
CreateDatasetDep = Annotated[CreateDataset, Depends(use_case(CreateDataset))]
GetDatasetDep = Annotated[GetDataset, Depends(use_case(GetDataset))]
ListDatasetsDep = Annotated[ListDatasets, Depends(use_case(ListDatasets))]


class TargetBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    kind: TargetKind
    unit: str | None = None
    direction: Direction | None = None


class SplitBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: SplitStrategy
    seed: int
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)


class CreateDatasetBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    upload_ref: str
    structure_column: str
    target: TargetBody
    split: SplitBody


class UploadResponse(BaseModel):
    upload_ref: uuid.UUID


class DatasetResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    structure_column: str
    target: dict[str, Any]
    split: dict[str, Any]
    content_hash: str
    snapshot_uri: str
    row_count: int
    validation_report: dict[str, Any]
    version: int
    created_at: datetime

    @classmethod
    def from_domain(cls, dataset: Dataset) -> DatasetResponse:
        return cls(
            id=dataset.id,
            workspace_id=dataset.workspace_id,
            name=dataset.name,
            structure_column=dataset.structure_column,
            target=target_to_dict(dataset.target),
            split=split_to_dict(dataset.split),
            content_hash=dataset.content_hash,
            snapshot_uri=dataset.snapshot_uri,
            row_count=dataset.row_count,
            validation_report=report_to_dict(dataset.validation_report),
            version=dataset.version,
            created_at=dataset.created_at,
        )


@router.post("/uploads", response_model=UploadResponse, status_code=201)
async def upload_dataset_file(
    auth: AuthDep, service: StoreUploadDep, file: UploadFile
) -> UploadResponse:
    # Checked before .read(), which is what pulls the whole (possibly spooled)
    # body into memory -- after that the damage is already done.
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise ValidationError(
            f"The uploaded file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
        )
    upload_ref = result_to_response(await service(await file.read(), auth=auth))
    return UploadResponse(upload_ref=upload_ref)


@router.post("", response_model=DatasetResponse, status_code=201)
async def create_dataset(
    body: CreateDatasetBody, auth: AuthDep, service: CreateDatasetDep
) -> DatasetResponse:
    try:
        split = SplitSpec(
            strategy=body.split.strategy, seed=body.split.seed, fractions=body.split.fractions
        )
    except ValueError as error:
        # SplitSpec enforces its own invariants (fractions sum to 1, none negative).
        raise ValidationError(str(error)) from error

    command = CreateDatasetCommand(
        name=body.name,
        upload_ref=body.upload_ref,
        structure_column=body.structure_column,
        target=TargetSpec(
            column=body.target.column,
            kind=body.target.kind,
            unit=body.target.unit,
            direction=body.target.direction,
        ),
        split=split,
    )
    return DatasetResponse.from_domain(result_to_response(await service(command, auth=auth)))


@router.get("", response_model=PaginatedResponse[DatasetResponse])
async def list_datasets(
    auth: AuthDep,
    service: ListDatasetsDep,
    cursor: str | None = None,
    limit: int | None = None,
) -> PaginatedResponse[DatasetResponse]:
    # `limit` is clamped inside the use case, not here: a worker calling it
    # directly must get the same ceiling as an HTTP caller.
    page = result_to_response(
        await service(ListDatasetsQuery(cursor=cursor, limit=limit), auth=auth)
    )
    return PaginatedResponse(
        items=[DatasetResponse.from_domain(dataset) for dataset in page.items],
        next_cursor=page.next_cursor,
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: uuid.UUID, auth: AuthDep, service: GetDatasetDep
) -> DatasetResponse:
    dataset = result_to_response(await service(GetDatasetQuery(dataset_id=dataset_id), auth=auth))
    return DatasetResponse.from_domain(dataset)
