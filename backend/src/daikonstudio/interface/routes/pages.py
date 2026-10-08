"""Lab-notebook page endpoints: pages on a dataset, protocol or run, their history, and
the images embedded in them.

Literal paths (`/blobs`) are registered before `/{page_id}`. Every body is
`extra="forbid"`.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response, UploadFile
from pydantic import BaseModel, ConfigDict

from daikonstudio.application.auth import AuthContext, is_editor, may_delete
from daikonstudio.application.pages.manage import (
    ArchivePage,
    ArchivePageCommand,
    CreatePage,
    CreatePageCommand,
    DeletePage,
    GetPage,
    GetPageBlob,
    GetPageContent,
    GetPageContentQuery,
    ListPageRevisions,
    ListPages,
    ListPagesQuery,
    RetitlePage,
    RetitlePageCommand,
    RevisePage,
    RevisePageCommand,
    UploadPageBlob,
    UploadPageBlobCommand,
)
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.domain.shared.page import Page, PageBlob, PageOwnerKind, PageRevision
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies._core import AuthDep
from daikonstudio.interface.error_handlers import result_to_response

router = APIRouter(prefix="/api/v1/pages", tags=["pages"])

MAX_BLOB_BYTES = 25 * 1024 * 1024

ListPagesDep = Annotated[ListPages, Depends(use_case(ListPages))]
CreatePageDep = Annotated[CreatePage, Depends(use_case(CreatePage))]
GetPageDep = Annotated[GetPage, Depends(use_case(GetPage))]
RevisePageDep = Annotated[RevisePage, Depends(use_case(RevisePage))]
RetitlePageDep = Annotated[RetitlePage, Depends(use_case(RetitlePage))]
ArchivePageDep = Annotated[ArchivePage, Depends(use_case(ArchivePage))]
DeletePageDep = Annotated[DeletePage, Depends(use_case(DeletePage))]
ListPageRevisionsDep = Annotated[ListPageRevisions, Depends(use_case(ListPageRevisions))]
GetPageContentDep = Annotated[GetPageContent, Depends(use_case(GetPageContent))]
UploadPageBlobDep = Annotated[UploadPageBlob, Depends(use_case(UploadPageBlob))]
GetPageBlobDep = Annotated[GetPageBlob, Depends(use_case(GetPageBlob))]


class PageResponse(BaseModel):
    id: uuid.UUID
    owner_kind: PageOwnerKind
    owner_id: uuid.UUID
    title: str
    # sha256 of the current body; null until the first save.
    head_sha256: str | None
    author_id: str
    archived: bool
    revision_count: int
    # Send it back as `expected_version` when saving.
    version: int
    created_at: datetime
    updated_at: datetime
    last_edited_by: str | None
    can_edit: bool
    can_archive: bool
    can_delete: bool

    @classmethod
    def from_domain(cls, page: Page, auth: AuthContext) -> PageResponse:
        # Archiving is for the same people as deleting: the author or an admin.
        mine = may_delete(auth, page.created_by)
        return cls(
            id=page.id,
            owner_kind=page.owner_kind,
            owner_id=page.owner_id,
            title=page.title,
            head_sha256=page.head_sha256,
            author_id=str(page.created_by),
            archived=page.archived,
            revision_count=page.revision_count,
            version=page.version,
            created_at=page.created_at,
            updated_at=page.updated_at,
            last_edited_by=str(page.last_edited_by) if page.last_edited_by else None,
            can_edit=is_editor(auth) and not page.archived,
            can_archive=mine,
            can_delete=mine,
        )


class RevisionResponse(BaseModel):
    revision_no: int
    sha256: str
    author_id: str
    created_at: datetime

    @classmethod
    def from_domain(cls, revision: PageRevision) -> RevisionResponse:
        return cls(
            revision_no=revision.revision_no,
            sha256=revision.sha256,
            author_id=str(revision.author_id),
            created_at=revision.created_at,
        )


class PageBlobResponse(BaseModel):
    sha256: str
    mime: str
    size: int

    @classmethod
    def from_domain(cls, blob: PageBlob) -> PageBlobResponse:
        return cls(sha256=blob.sha256, mime=blob.mime, size=blob.size_bytes)


class CreatePageBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Length is checked by the domain, which answers in the product's own words.
    title: str
    owner_kind: PageOwnerKind
    owner_id: uuid.UUID


class RevisePageBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # A ProseMirror document; the domain checks its shape and size.
    body: dict[str, Any]
    expected_version: int


class RetitlePageBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str


class ArchivePageBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    restore: bool = False


@router.get("", response_model=list[PageResponse])
async def list_pages(
    owner_kind: PageOwnerKind,
    owner_id: uuid.UUID,
    auth: AuthDep,
    service: ListPagesDep,
    archived: bool = False,
) -> list[PageResponse]:
    """The owner's pages, newest-updated first. `archived=true` lists only archived ones."""
    pages = result_to_response(
        await service(
            ListPagesQuery(owner_kind=owner_kind, owner_id=owner_id, archived=archived), auth=auth
        )
    )
    return [PageResponse.from_domain(page, auth) for page in pages]


@router.post("", response_model=PageResponse, status_code=201)
async def create_page(body: CreatePageBody, auth: AuthDep, service: CreatePageDep) -> PageResponse:
    page = result_to_response(
        await service(
            CreatePageCommand(
                title=body.title, owner_kind=body.owner_kind, owner_id=body.owner_id
            ),
            auth=auth,
        )
    )
    return PageResponse.from_domain(page, auth)


@router.post("/blobs", response_model=PageBlobResponse, status_code=201)
async def upload_page_blob(
    auth: AuthDep, service: UploadPageBlobDep, file: UploadFile
) -> PageBlobResponse:
    """An image to embed in a page: PNG, JPEG, GIF, WebP or SVG."""
    # Checked before .read(), as for dataset uploads.
    if file.size is not None and file.size > MAX_BLOB_BYTES:
        raise ValidationError(
            f"The uploaded file exceeds the {MAX_BLOB_BYTES // (1024 * 1024)} MB limit"
        )
    blob = result_to_response(
        await service(
            UploadPageBlobCommand(data=await file.read(), mime=file.content_type), auth=auth
        )
    )
    return PageBlobResponse.from_domain(blob)


@router.get("/blobs/{sha256}")
async def get_page_blob(sha256: str, auth: AuthDep, service: GetPageBlobDep) -> Response:
    """Served as a download in a sandbox, so a file that slipped through as something
    else cannot run in the studio's origin when opened directly."""
    blob, data = result_to_response(await service(sha256, auth=auth))
    return Response(
        content=data,
        media_type=blob.mime,
        headers={
            "Content-Disposition": "attachment",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "sandbox; default-src 'none'",
        },
    )


@router.get("/{page_id}", response_model=PageResponse)
async def get_page(page_id: uuid.UUID, auth: AuthDep, service: GetPageDep) -> PageResponse:
    return PageResponse.from_domain(result_to_response(await service(page_id, auth=auth)), auth)


@router.patch("/{page_id}", response_model=PageResponse)
async def revise_page(
    page_id: uuid.UUID, body: RevisePageBody, auth: AuthDep, service: RevisePageDep
) -> PageResponse:
    """Save a new body. 409 if the page moved past `expected_version`; saving the body it
    already has changes nothing."""
    page = result_to_response(
        await service(
            RevisePageCommand(
                page_id=page_id, body=body.body, expected_version=body.expected_version
            ),
            auth=auth,
        )
    )
    return PageResponse.from_domain(page, auth)


@router.put("/{page_id}/title", response_model=PageResponse)
async def retitle_page(
    page_id: uuid.UUID, body: RetitlePageBody, auth: AuthDep, service: RetitlePageDep
) -> PageResponse:
    page = result_to_response(
        await service(RetitlePageCommand(page_id=page_id, title=body.title), auth=auth)
    )
    return PageResponse.from_domain(page, auth)


@router.post("/{page_id}/archive", response_model=PageResponse)
async def archive_page(
    page_id: uuid.UUID, body: ArchivePageBody, auth: AuthDep, service: ArchivePageDep
) -> PageResponse:
    page = result_to_response(
        await service(ArchivePageCommand(page_id=page_id, restore=body.restore), auth=auth)
    )
    return PageResponse.from_domain(page, auth)


@router.delete("/{page_id}", status_code=204)
async def delete_page(page_id: uuid.UUID, auth: AuthDep, service: DeletePageDep) -> Response:
    result_to_response(await service(page_id, auth=auth))
    return Response(status_code=204)


@router.get("/{page_id}/revisions", response_model=list[RevisionResponse])
async def list_page_revisions(
    page_id: uuid.UUID, auth: AuthDep, service: ListPageRevisionsDep
) -> list[RevisionResponse]:
    """Oldest first."""
    revisions = result_to_response(await service(page_id, auth=auth))
    return [RevisionResponse.from_domain(revision) for revision in revisions]


@router.get("/{page_id}/content/{sha256}")
async def get_page_content(
    page_id: uuid.UUID, sha256: str, auth: AuthDep, service: GetPageContentDep
) -> dict[str, Any]:
    """One revision's document, by the sha256 the revision list names."""
    return result_to_response(
        await service(GetPageContentQuery(page_id=page_id, sha256=sha256), auth=auth)
    )
