"""Folder endpoints: list (with counts), create, rename, delete.

Filing an item lives beside the item (`PUT /datasets/{id}/folder`,
`PUT /protocols/{id}/folder`). Every body is `extra="forbid"`.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, ConfigDict

from daikonstudio.application.folders.manage import (
    CreateFolder,
    CreateFolderCommand,
    DeleteFolder,
    ListFolders,
    RenameFolder,
    RenameFolderCommand,
)
from daikonstudio.domain.shared.folder import Folder, FolderKind
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies._core import AuthDep
from daikonstudio.interface.error_handlers import result_to_response

router = APIRouter(prefix="/api/v1/folders", tags=["folders"])

ListFoldersDep = Annotated[ListFolders, Depends(use_case(ListFolders))]
CreateFolderDep = Annotated[CreateFolder, Depends(use_case(CreateFolder))]
RenameFolderDep = Annotated[RenameFolder, Depends(use_case(RenameFolder))]
DeleteFolderDep = Annotated[DeleteFolder, Depends(use_case(DeleteFolder))]


class FolderResponse(BaseModel):
    id: uuid.UUID
    kind: FolderKind
    name: str
    # Items filed here that the viewer can see (a colleague's draft protocol is not counted).
    item_count: int
    created_by: uuid.UUID

    @classmethod
    def from_domain(cls, folder: Folder, item_count: int = 0) -> FolderResponse:
        return cls(
            id=folder.id,
            kind=folder.kind,
            name=folder.name,
            item_count=item_count,
            created_by=folder.created_by,
        )


class FolderListResponse(BaseModel):
    items: list[FolderResponse]
    can_edit: bool


class CreateFolderBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: FolderKind
    # Length is checked by the domain, which answers in the product's own words.
    name: str


class RenameFolderBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str


@router.get("", response_model=FolderListResponse)
async def list_folders(
    kind: FolderKind, auth: AuthDep, service: ListFoldersDep
) -> FolderListResponse:
    listing = result_to_response(await service(kind, auth=auth))
    return FolderListResponse(
        items=[FolderResponse.from_domain(f, n) for f, n in listing.items],
        can_edit=listing.can_edit,
    )


@router.post("", response_model=FolderResponse, status_code=201)
async def create_folder(
    body: CreateFolderBody, auth: AuthDep, service: CreateFolderDep
) -> FolderResponse:
    folder = result_to_response(
        await service(CreateFolderCommand(kind=body.kind, name=body.name), auth=auth)
    )
    return FolderResponse.from_domain(folder)


@router.patch("/{folder_id}", response_model=FolderResponse)
async def rename_folder(
    folder_id: uuid.UUID, body: RenameFolderBody, auth: AuthDep, service: RenameFolderDep
) -> FolderResponse:
    folder = result_to_response(
        await service(RenameFolderCommand(folder_id=folder_id, name=body.name), auth=auth)
    )
    return FolderResponse.from_domain(folder)


@router.delete("/{folder_id}", status_code=204)
async def delete_folder(folder_id: uuid.UUID, auth: AuthDep, service: DeleteFolderDep) -> Response:
    result_to_response(await service(folder_id, auth=auth))
    return Response(status_code=204)
