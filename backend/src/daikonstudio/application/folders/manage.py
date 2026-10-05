"""Shared folders: list with counts, create, rename, delete, and file an item.

Folders are workspace-wide and flat. Reads are open to any member; changes need an
editor. A protocol folder's count and a filing both go through the caller's view of
protocols, so a colleague's draft is neither counted for you nor fileable by you.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import (
    AuthContext,
    is_editor,
    require_authenticated,
    require_editor,
)
from daikonstudio.application.catalog.visibility import visible_protocol
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.folder_repository import FolderRepository
from daikonstudio.application.ports.protocol_access import ProtocolAccess
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.shared.errors import DomainError, NotFoundError, ValidationError
from daikonstudio.domain.shared.folder import Folder, FolderKind, clean_folder_name


@dataclass(frozen=True, kw_only=True)
class FolderList:
    items: list[tuple[Folder, int]]
    can_edit: bool


class ListFolders:
    def __init__(
        self,
        folders: FolderRepository,
        datasets: DatasetRepository,
        protocols: ProtocolRepository,
        access: ProtocolAccess,
    ) -> None:
        self._folders = folders
        self._datasets = datasets
        self._protocols = protocols
        self._access = access

    async def __call__(
        self, kind: FolderKind, auth: AuthContext | None = None
    ) -> Result[FolderList, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        folders = await self._folders.list(auth.workspace_id, kind)
        if kind is FolderKind.DATASET:
            counts = await self._datasets.count_by_folder(auth.workspace_id)
        else:
            visible = await self._access.visible_ids(auth)
            counts = await self._protocols.count_by_folder(auth.workspace_id, visible)
        return Success(
            FolderList(
                items=[(folder, counts.get(folder.id, 0)) for folder in folders],
                can_edit=is_editor(auth),
            )
        )


@dataclass(frozen=True, kw_only=True)
class CreateFolderCommand:
    kind: FolderKind
    name: str


class CreateFolder:
    def __init__(self, folders: FolderRepository) -> None:
        self._folders = folders

    async def __call__(
        self, command: CreateFolderCommand, auth: AuthContext | None = None
    ) -> Result[Folder, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        try:
            name = clean_folder_name(command.name)
        except ValidationError as error:
            return Failure(error)
        folder = Folder(
            workspace_id=auth.workspace_id,
            kind=command.kind,
            name=name,
            created_by=auth.user_id,
        )
        await self._folders.add(folder)
        return Success(folder)


@dataclass(frozen=True, kw_only=True)
class RenameFolderCommand:
    folder_id: uuid.UUID
    name: str


class RenameFolder:
    def __init__(self, folders: FolderRepository) -> None:
        self._folders = folders

    async def __call__(
        self, command: RenameFolderCommand, auth: AuthContext | None = None
    ) -> Result[Folder, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        folder = await self._folders.get(auth.workspace_id, command.folder_id)
        if folder is None:
            return Failure(NotFoundError("Folder", str(command.folder_id)))
        try:
            name = clean_folder_name(command.name)
        except ValidationError as error:
            return Failure(error)
        await self._folders.rename(auth.workspace_id, folder.id, name)
        renamed = await self._folders.get(auth.workspace_id, folder.id)
        return Success(renamed or folder)


class DeleteFolder:
    """Deleting a folder unfiles what was in it; nothing else is touched."""

    def __init__(self, folders: FolderRepository) -> None:
        self._folders = folders

    async def __call__(
        self, folder_id: uuid.UUID, auth: AuthContext | None = None
    ) -> Result[None, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        if await self._folders.get(auth.workspace_id, folder_id) is None:
            return Failure(NotFoundError("Folder", str(folder_id)))
        await self._folders.delete(auth.workspace_id, folder_id)
        return Success(None)


@dataclass(frozen=True, kw_only=True)
class FileItemCommand:
    item_id: uuid.UUID
    folder_id: uuid.UUID | None


async def _check_folder(
    folders: FolderRepository, auth: AuthContext, folder_id: uuid.UUID | None, kind: FolderKind
) -> ValidationError | None:
    if folder_id is None:
        return None
    folder = await folders.get(auth.workspace_id, folder_id)
    if folder is None or folder.kind is not kind:
        return ValidationError("Choose a folder for this kind of item.")
    return None


class FileDataset:
    def __init__(self, datasets: DatasetRepository, folders: FolderRepository) -> None:
        self._datasets = datasets
        self._folders = folders

    async def __call__(
        self, command: FileItemCommand, auth: AuthContext | None = None
    ) -> Result[Dataset, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        dataset = await self._datasets.get(auth.workspace_id, command.item_id)
        if dataset is None:
            return Failure(NotFoundError("Dataset", str(command.item_id)))
        error = await _check_folder(self._folders, auth, command.folder_id, FolderKind.DATASET)
        if error is not None:
            return Failure(error)
        await self._datasets.set_folder(auth.workspace_id, dataset.id, command.folder_id)
        dataset.folder_id = command.folder_id
        return Success(dataset)


class FileProtocol:
    def __init__(
        self, protocols: ProtocolRepository, folders: FolderRepository, access: ProtocolAccess
    ) -> None:
        self._protocols = protocols
        self._folders = folders
        self._access = access

    async def __call__(
        self, command: FileItemCommand, auth: AuthContext | None = None
    ) -> Result[InSilicoProtocol, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        protocol = await visible_protocol(
            self._protocols, self._access, auth, auth.workspace_id, command.item_id
        )
        if protocol is None:
            return Failure(NotFoundError("Protocol", str(command.item_id)))
        error = await _check_folder(self._folders, auth, command.folder_id, FolderKind.PROTOCOL)
        if error is not None:
            return Failure(error)
        await self._protocols.set_folder(auth.workspace_id, protocol.id, command.folder_id)
        protocol.folder_id = command.folder_id
        return Success(protocol)
