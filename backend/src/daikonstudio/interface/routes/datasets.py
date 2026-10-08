"""Dataset endpoints.

Two-phase creation: `POST /uploads` parks the raw file and hands back an opaque
`upload_ref`, then `POST ""` references it. That keeps the expensive, slow part
(chemistry validation, splitting, freezing) off the multipart request, and means
a failed create can be retried against the same bytes.

There is no PATCH: a Dataset is immutable and cited by id. The one setting that
is not frozen, its identifier column, has its own `PUT .../id-column`. DELETE removes one
only when nothing depends on it; see `application/data/delete_dataset.py`.

Every request body here is `extra="forbid"`. That is what makes
`workspace_id` unspoofable in the honest sense: not silently dropped, but
refused, so no client ever comes to believe it had an effect.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from daikonstudio.application.auth import AuthContext, is_editor, may_delete
from daikonstudio.application.data.build_dataset import GetDatasetBuild, StartDatasetBuild
from daikonstudio.application.data.create_dataset import (
    CreateDataset,
    CreateDatasetCommand,
    StoreUpload,
)
from daikonstudio.application.data.delete_dataset import DeleteDataset, DeleteDatasetCommand
from daikonstudio.application.data.get_dataset import GetDataset, GetDatasetQuery
from daikonstudio.application.data.get_dataset_compounds import (
    Compound,
    CompoundPage,
    GetDatasetCompounds,
    GetDatasetCompoundsQuery,
)
from daikonstudio.application.data.get_dataset_profile import (
    GetDatasetProfile,
    GetDatasetProfileQuery,
    ProfileComputing,
)
from daikonstudio.application.data.list_datasets import ListDatasets, ListDatasetsQuery
from daikonstudio.application.data.preview_dataset import (
    DatasetPreview,
    DatasetReadiness,
    FreezeDatasetPreview,
    GetDatasetPreview,
    GetDatasetReadiness,
    StartDatasetPreview,
)
from daikonstudio.application.data.set_dataset_id_column import (
    GetDatasetColumns,
    GetDatasetColumnsQuery,
    SetDatasetIdColumn,
    SetDatasetIdColumnCommand,
)
from daikonstudio.application.folders.manage import FileDataset, FileItemCommand
from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.data.dataset_build import DatasetBuild
from daikonstudio.domain.data.profile import DatasetProfile, profile_to_dict
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy, split_to_dict
from daikonstudio.domain.data.structure_kind import StructureKind
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
StartDatasetBuildDep = Annotated[StartDatasetBuild, Depends(use_case(StartDatasetBuild))]
GetDatasetBuildDep = Annotated[GetDatasetBuild, Depends(use_case(GetDatasetBuild))]
GetDatasetDep = Annotated[GetDataset, Depends(use_case(GetDataset))]
DeleteDatasetDep = Annotated[DeleteDataset, Depends(use_case(DeleteDataset))]
SetDatasetIdColumnDep = Annotated[SetDatasetIdColumn, Depends(use_case(SetDatasetIdColumn))]
GetDatasetColumnsDep = Annotated[GetDatasetColumns, Depends(use_case(GetDatasetColumns))]
FileDatasetDep = Annotated[FileDataset, Depends(use_case(FileDataset))]
ListDatasetsDep = Annotated[ListDatasets, Depends(use_case(ListDatasets))]
GetDatasetProfileDep = Annotated[GetDatasetProfile, Depends(use_case(GetDatasetProfile))]
GetDatasetCompoundsDep = Annotated[GetDatasetCompounds, Depends(use_case(GetDatasetCompounds))]
StartDatasetPreviewDep = Annotated[StartDatasetPreview, Depends(use_case(StartDatasetPreview))]
GetDatasetPreviewDep = Annotated[GetDatasetPreview, Depends(use_case(GetDatasetPreview))]
FreezeDatasetPreviewDep = Annotated[FreezeDatasetPreview, Depends(use_case(FreezeDatasetPreview))]
GetDatasetReadinessDep = Annotated[GetDatasetReadiness, Depends(use_case(GetDatasetReadiness))]


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

    # `max_length` matches the `String(256)`/`String(128)` columns these values
    # are eventually written into (`DatasetModel.name`/`.structure_column`) --
    # without it, an over-long value reaches asyncpg and comes back as an
    # unmapped `StringDataRightTruncationError` (500), rather than a 422 that
    # names the field (whole-branch review, Important 3).
    name: str = Field(max_length=256)
    upload_ref: str
    structure_column: str = Field(max_length=128)
    # ponytail: no cap on the target count. A very wide dataset (ToxCast, ~600
    # targets) fans out one fit per target per leg, and `deadline_scale` grows with
    # it. Add a `max_length` here if that ever happens.
    targets: list[TargetBody] = Field(min_length=1)
    split: SplitBody
    id_column: str | None = Field(default=None, max_length=128)
    file_name: str | None = Field(default=None, max_length=256)


class UploadResponse(BaseModel):
    upload_ref: uuid.UUID


class DatasetBuildResponse(BaseModel):
    """A dataset being built. `done` of `total` rows through `stage`; `total` is 0 for a
    stage with no row count. On `failed`, `error` is the body `POST /datasets` would
    have answered with (a 422's `detail` is the whole validation report)."""

    id: uuid.UUID
    name: str
    status: Literal["running", "succeeded", "failed"]
    stage: str
    done: int
    total: int
    dataset_id: uuid.UUID | None
    error: dict[str, object] | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, build: DatasetBuild) -> DatasetBuildResponse:
        return cls(
            id=build.id,
            name=build.name,
            status=build.status.value,
            stage=build.stage,
            done=build.done,
            total=build.total,
            dataset_id=build.dataset_id,
            error=build.error,
            created_at=build.created_at,
            updated_at=build.updated_at,
        )


class InvalidRowResponse(BaseModel):
    row_number: int
    value: str
    reason: str


class ConflictRowResponse(BaseModel):
    structure: str
    # A compound can conflict in one binary target and agree in another, and the
    # file is fixed in that column, so a conflict has to say which one it is in.
    column: str
    values: list[int]
    row_numbers: list[int]


class ValidationReportResponse(BaseModel):
    """The report, typed rather than a bare `dict`.

    It reaches a client by two routes -- on an accepted `DatasetResponse`, and
    as the `detail` of the 422 that `InvalidDatasetError` raises when a file is
    refused at the door. The rejection is the case that matters: which rows
    failed and why, how many duplicates collapsed, how wide the assay spread
    was, is the whole value of the validation pass, and a UI cannot render any
    of it from `dict[str, Any]`. The error path serialises the same dataclasses
    through `report_to_dict`, so both bodies share this shape even though only
    this one appears in the OpenAPI contract.
    """

    total_rows: int
    valid_rows: int
    invalid: list[InvalidRowResponse]
    conflicting: list[ConflictRowResponse]
    duplicates_collapsed: int
    salts_flagged: int
    # Keyed by target column; numeric targets with replicates only.
    duplicate_spread: dict[str, float]
    # `"molecule"` or `"sequence"`: what the structure column was found to hold. The
    # browser needs it to avoid saying SMILES about a protein, or drawing a 2D
    # depiction of one -- there is no other signal for it on a Dataset, since the
    # column role is deliberately one modality-agnostic "structure".
    structure_kind: str = StructureKind.MOLECULE.value


class DatasetResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    structure_column: str
    # `TargetBody`/`SplitBody` are reused verbatim rather than mirrored into
    # `*Response` twins: the spec a scientist submits *is* the spec that gets
    # frozen and read back, so one model for both directions is what keeps the
    # two from ever drifting apart.
    targets: list[TargetBody]
    split: SplitBody
    content_hash: str
    snapshot_uri: str
    row_count: int
    validation_report: ValidationReportResponse
    version: int
    created_at: datetime
    # Whether this viewer may delete it, by role and creator. Dependents (protocols
    # trained on it, runs in progress) are checked only when DELETE is requested.
    can_delete: bool
    # Which snapshot column holds the compounds' own IDs, if any.
    id_column: str | None
    # Whether this viewer may change its settings (the identifier column).
    can_edit: bool
    created_by: uuid.UUID | None
    # The shared folder it is filed in, if any.
    folder_id: uuid.UUID | None

    @classmethod
    def from_domain(cls, dataset: Dataset, *, auth: AuthContext | None) -> DatasetResponse:
        return cls(
            id=dataset.id,
            workspace_id=dataset.workspace_id,
            name=dataset.name,
            structure_column=dataset.structure_column,
            targets=[TargetBody.model_validate(target_to_dict(t)) for t in dataset.targets],
            split=SplitBody.model_validate(split_to_dict(dataset.split)),
            content_hash=dataset.content_hash,
            snapshot_uri=dataset.snapshot_uri,
            row_count=dataset.row_count,
            validation_report=ValidationReportResponse.model_validate(
                report_to_dict(dataset.validation_report)
            ),
            version=dataset.version,
            created_at=dataset.created_at,
            can_delete=may_delete(auth, dataset.created_by),
            id_column=dataset.id_column,
            can_edit=is_editor(auth),
            created_by=dataset.created_by,
            folder_id=dataset.folder_id,
        )


class HistogramResponse(BaseModel):
    """`edges` is one longer than `counts`; bin *i* spans `edges[i]`..`edges[i+1]`."""

    edges: list[float]
    counts: list[int]


class SplitHistogramResponse(BaseModel):
    """Train and test counts over shared edges, so the two can be overlaid."""

    edges: list[float]
    train: list[int]
    test: list[int]


class NumericSummaryResponse(BaseModel):
    minimum: float
    maximum: float
    mean: float
    median: float
    std: float


class TargetDistributionResponse(BaseModel):
    histogram: SplitHistogramResponse
    train: NumericSummaryResponse
    test: NumericSummaryResponse


class ClassBalanceResponse(BaseModel):
    split: str
    positive: int
    negative: int


class TargetClassBalanceResponse(ClassBalanceResponse):
    column: str


class DatasetReadinessResponse(BaseModel):
    row_count: int
    partition_counts: dict[str, int]
    class_balance: list[TargetClassBalanceResponse]
    warnings: list[str]

    @classmethod
    def from_domain(cls, readiness: DatasetReadiness) -> DatasetReadinessResponse:
        from dataclasses import asdict

        return cls.model_validate(asdict(readiness))


class DatasetPreparationResponse(BaseModel):
    name: str
    file_name: str | None = None
    structure_column: str
    targets: list[TargetBody]
    split: SplitBody
    id_column: str | None
    validation_report: ValidationReportResponse
    readiness: DatasetReadinessResponse
    expires_at: datetime


class DatasetPreviewResponse(DatasetBuildResponse):
    preparation: DatasetPreparationResponse | None

    @classmethod
    def from_preview(cls, preview: DatasetPreview) -> DatasetPreviewResponse:
        return cls(
            **DatasetBuildResponse.from_domain(preview.build).model_dump(),
            preparation=DatasetPreparationResponse.model_validate(preview.preparation)
            if preview.preparation is not None
            else None,
        )


class FreezeDatasetBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=256)


class SimilarityProfileResponse(BaseModel):
    histogram: HistogramResponse
    median: float
    within_domain: float
    within_domain_threshold: float
    near_duplicates: int
    near_duplicate_threshold: float


class ScaffoldEntryResponse(BaseModel):
    smiles: str
    count: int


class ScaffoldProfileResponse(BaseModel):
    unique_count: int
    singleton_count: int
    largest_fraction: float
    cumulative_coverage: list[float]
    top: list[ScaffoldEntryResponse]
    cross_split_scaffolds: int
    cross_split_compounds: int


class DescriptorProfileResponse(BaseModel):
    name: str
    histogram: SplitHistogramResponse
    median: float
    target_correlation: float | None


class ActivityCliffResponse(BaseModel):
    left_structure: str
    right_structure: str
    left_value: float
    right_value: float
    similarity: float
    delta: float


class DatasetProfileResponse(BaseModel):
    """What the Dataset is made of -- see `domain/data/profile.py` for what each
    section means and why it is here.

    `target_distribution` and `class_balance` are mutually exclusive: a numeric
    target populates the first and a binary one the second, so a consumer must
    branch on `target_kind` rather than render whichever is non-empty.

    `similarity` is `null` only when there was no train/test pair to compare,
    never an all-zero histogram. `cliffs_sampled_from` is non-null when the
    pair scan ran on a subsample -- "no cliffs found among 3000 of 12000
    compounds" is a different claim from "no cliffs", and this is which one
    was made.

    On a dataset whose structure column holds amino-acid sequences, every
    chemistry section is absent: `scaffolds` and `similarity` are `null` and
    `descriptors` and `activity_cliffs` are empty. A sequence has no ring
    system and no Tanimoto neighbour, so a consumer must omit those sections
    rather than render a zero -- which would read as a measurement.
    """

    compounds: int
    partition_counts: dict[str, int]
    target_kind: str
    target_distribution: TargetDistributionResponse | None
    class_balance: list[ClassBalanceResponse]
    similarity: SimilarityProfileResponse | None
    scaffolds: ScaffoldProfileResponse | None
    descriptors: list[DescriptorProfileResponse]
    best_descriptor: str | None
    activity_cliffs: list[ActivityCliffResponse]
    cliffs_sampled_from: int | None

    @classmethod
    def from_domain(cls, profile: DatasetProfile) -> DatasetProfileResponse:
        # Through the same `profile_to_dict` the cache blob is written with, so
        # the wire shape and the stored shape cannot drift apart -- a field
        # renamed in one is renamed in both or fails validation here.
        return cls.model_validate(profile_to_dict(profile))


class CompoundResponse(BaseModel):
    structure: str
    targets: dict[str, float | None]
    split: str
    compound_id: str | None

    @classmethod
    def from_domain(cls, compound: Compound) -> CompoundResponse:
        return cls(
            structure=compound.structure,
            targets=compound.targets,
            split=compound.split,
            compound_id=compound.compound_id,
        )


class CompoundPageResponse(BaseModel):
    """`total` is the count *after* the split filter, which is what a pager
    needs; the Dataset's own `row_count` answers a different question."""

    items: list[CompoundResponse]
    total: int

    @classmethod
    def from_domain(cls, page: CompoundPage) -> CompoundPageResponse:
        return cls(
            items=[CompoundResponse.from_domain(item) for item in page.items],
            total=page.total,
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


def _create_command(body: CreateDatasetBody) -> CreateDatasetCommand:
    try:
        split = SplitSpec(
            strategy=body.split.strategy, seed=body.split.seed, fractions=body.split.fractions
        )
    except ValueError as error:
        # SplitSpec enforces its own invariants (fractions sum to 1, none negative).
        raise ValidationError(str(error)) from error

    return CreateDatasetCommand(
        name=body.name,
        upload_ref=body.upload_ref,
        structure_column=body.structure_column,
        targets=tuple(
            TargetSpec(column=t.column, kind=t.kind, unit=t.unit, direction=t.direction)
            for t in body.targets
        ),
        split=split,
        id_column=body.id_column,
        file_name=body.file_name,
    )


@router.post("", response_model=DatasetResponse, status_code=201)
async def create_dataset(
    body: CreateDatasetBody, auth: AuthDep, service: CreateDatasetDep
) -> DatasetResponse:
    """Build the dataset within the request, for scripts. A large file takes minutes;
    the wizard uses `POST /datasets/builds` instead, which reports progress."""
    return DatasetResponse.from_domain(
        result_to_response(await service(_create_command(body), auth=auth)), auth=auth
    )


@router.post("/builds", response_model=DatasetBuildResponse, status_code=202)
async def start_dataset_build(
    body: CreateDatasetBody, auth: AuthDep, service: StartDatasetBuildDep
) -> DatasetBuildResponse:
    """Start building the dataset in the background; poll `GET /datasets/builds/{id}`."""
    return DatasetBuildResponse.from_domain(
        result_to_response(await service(_create_command(body), auth=auth))
    )


@router.get("/builds/{build_id}", response_model=DatasetBuildResponse)
async def get_dataset_build(
    build_id: uuid.UUID, auth: AuthDep, service: GetDatasetBuildDep
) -> DatasetBuildResponse:
    return DatasetBuildResponse.from_domain(result_to_response(await service(build_id, auth=auth)))


@router.post("/previews", response_model=DatasetPreviewResponse, status_code=202)
async def start_dataset_preview(
    body: CreateDatasetBody, auth: AuthDep, service: StartDatasetPreviewDep
) -> DatasetPreviewResponse:
    """Validate and split in the background without creating a dataset."""
    return DatasetPreviewResponse.from_preview(
        result_to_response(await service(_create_command(body), auth=auth))
    )


@router.get("/previews/{build_id}", response_model=DatasetPreviewResponse)
async def get_dataset_preview(
    build_id: uuid.UUID, auth: AuthDep, service: GetDatasetPreviewDep
) -> DatasetPreviewResponse:
    return DatasetPreviewResponse.from_preview(
        result_to_response(await service(build_id, auth=auth))
    )


@router.post("/previews/{build_id}/freeze", response_model=DatasetResponse, status_code=201)
async def freeze_dataset_preview(
    build_id: uuid.UUID, body: FreezeDatasetBody, auth: AuthDep, service: FreezeDatasetPreviewDep
) -> DatasetResponse:
    """Create a dataset from the exact preparation shown in its review."""
    return DatasetResponse.from_domain(
        result_to_response(await service(build_id, body.name, auth=auth)), auth=auth
    )


@router.get("", response_model=PaginatedResponse[DatasetResponse])
async def list_datasets(
    auth: AuthDep,
    service: ListDatasetsDep,
    cursor: str | None = None,
    limit: int | None = None,
    folder_id: uuid.UUID | None = None,
    q: Annotated[str | None, Query(max_length=256)] = None,
    target_kind: TargetKind | None = None,
    split_strategy: SplitStrategy | None = None,
) -> PaginatedResponse[DatasetResponse]:
    # `limit` is clamped inside the use case, not here: a worker calling it
    # directly must get the same ceiling as an HTTP caller.
    page = result_to_response(
        await service(
            ListDatasetsQuery(
                cursor=cursor,
                limit=limit,
                folder_id=folder_id,
                q=q,
                target_kind=target_kind,
                split_strategy=split_strategy,
            ),
            auth=auth,
        )
    )
    return PaginatedResponse(
        items=[DatasetResponse.from_domain(dataset, auth=auth) for dataset in page.items],
        next_cursor=page.next_cursor,
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: uuid.UUID, auth: AuthDep, service: GetDatasetDep
) -> DatasetResponse:
    dataset = result_to_response(await service(GetDatasetQuery(dataset_id=dataset_id), auth=auth))
    return DatasetResponse.from_domain(dataset, auth=auth)


@router.get("/{dataset_id}/readiness", response_model=DatasetReadinessResponse)
async def get_dataset_readiness(
    dataset_id: uuid.UUID, auth: AuthDep, service: GetDatasetReadinessDep
) -> DatasetReadinessResponse:
    """Lightweight, exact partition counts and target checks for training setup."""
    return DatasetReadinessResponse.from_domain(
        result_to_response(await service(dataset_id, auth=auth))
    )


class SetIdColumnBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id_column: str | None = Field(max_length=128)


class FolderBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder_id: uuid.UUID | None


@router.put("/{dataset_id}/folder", response_model=DatasetResponse)
async def file_dataset(
    dataset_id: uuid.UUID, body: FolderBody, auth: AuthDep, service: FileDatasetDep
) -> DatasetResponse:
    """Any editor; `null` unfiles it. Organization only: nothing frozen changes."""
    dataset = result_to_response(
        await service(FileItemCommand(item_id=dataset_id, folder_id=body.folder_id), auth=auth)
    )
    return DatasetResponse.from_domain(dataset, auth=auth)


class DatasetColumnsResponse(BaseModel):
    """Snapshot columns that may be named as the identifier."""

    columns: list[str]


@router.put("/{dataset_id}/id-column", response_model=DatasetResponse)
async def set_dataset_id_column(
    dataset_id: uuid.UUID, body: SetIdColumnBody, auth: AuthDep, service: SetDatasetIdColumnDep
) -> DatasetResponse:
    """Any editor; `null` clears it. Display metadata: nothing frozen changes."""
    dataset = result_to_response(
        await service(
            SetDatasetIdColumnCommand(dataset_id=dataset_id, id_column=body.id_column), auth=auth
        )
    )
    return DatasetResponse.from_domain(dataset, auth=auth)


@router.get("/{dataset_id}/columns", response_model=DatasetColumnsResponse)
async def get_dataset_columns(
    dataset_id: uuid.UUID, auth: AuthDep, service: GetDatasetColumnsDep
) -> DatasetColumnsResponse:
    columns = result_to_response(
        await service(GetDatasetColumnsQuery(dataset_id=dataset_id), auth=auth)
    )
    return DatasetColumnsResponse(columns=columns)


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: uuid.UUID, auth: AuthDep, service: DeleteDatasetDep
) -> Response:
    """By an admin or its creator, and only when no protocol was trained on it and no
    training run on it is in progress. See `application/data/delete_dataset.py`."""
    result_to_response(await service(DeleteDatasetCommand(dataset_id=dataset_id), auth=auth))
    return Response(status_code=204)


class ProfileComputingResponse(BaseModel):
    """202 while the profile is computed in the background. Poll the same URL;
    every request joins the one computation, so polling never starts another."""

    status: Literal["computing"] = "computing"
    started_at: datetime
    compounds: int


@router.get(
    "/{dataset_id}/profile",
    response_model=DatasetProfileResponse,
    responses={202: {"model": ProfileComputingResponse, "description": "Being computed"}},
)
async def get_dataset_profile(
    dataset_id: uuid.UUID,
    auth: AuthDep,
    service: GetDatasetProfileDep,
    target: Annotated[int, Query(ge=0)] = 0,
) -> DatasetProfileResponse | JSONResponse:
    """Computed once per Dataset, in the background, and cached beside its
    snapshot. Until it is saved this answers 202 with when the computation
    started; afterwards, 200 with the profile. See
    `application/data/get_dataset_profile.py`."""
    profile = result_to_response(
        await service(GetDatasetProfileQuery(dataset_id=dataset_id, target=target), auth=auth)
    )
    if isinstance(profile, ProfileComputing):
        body = ProfileComputingResponse(started_at=profile.started_at, compounds=profile.compounds)
        return JSONResponse(status_code=202, content=body.model_dump(mode="json"))
    return DatasetProfileResponse.from_domain(profile)


@router.get("/{dataset_id}/compounds", response_model=CompoundPageResponse)
async def get_dataset_compounds(
    dataset_id: uuid.UUID,
    auth: AuthDep,
    service: GetDatasetCompoundsDep,
    offset: int = 0,
    limit: int = 50,
    sort: Literal["target", "split"] | None = None,
    target: Annotated[int, Query(ge=0)] = 0,
    sort_dir: Literal["asc", "desc"] = "asc",
    split: Literal["train", "validation", "test"] | None = None,
    q: Annotated[str | None, Query(max_length=128)] = None,
) -> CompoundPageResponse:
    """`sort` and `split` are Literals, so FastAPI rejects anything else itself
    -- there is no column name here a client could reach the frame with."""
    page = result_to_response(
        await service(
            GetDatasetCompoundsQuery(
                dataset_id=dataset_id,
                offset=offset,
                limit=limit,
                sort=sort,
                target=target,
                descending=sort_dir == "desc",
                split=split,
                q=q,
            ),
            auth=auth,
        )
    )
    return CompoundPageResponse.from_domain(page)
