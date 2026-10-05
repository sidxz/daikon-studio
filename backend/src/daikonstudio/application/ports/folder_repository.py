"""Persistence port for folders. Workspace-scoped in the SQL, like every other port.
`add` and `rename` raise `ConflictError` when the name is taken (case-insensitively)
within the workspace and kind. `delete` unfiles whatever was in the folder."""

import uuid
from typing import Protocol

from daikonstudio.domain.shared.folder import Folder, FolderKind


class FolderRepository(Protocol):
    async def add(self, folder: Folder) -> None: ...

    async def get(self, workspace_id: uuid.UUID, folder_id: uuid.UUID) -> Folder | None: ...

    async def list(self, workspace_id: uuid.UUID, kind: FolderKind) -> list[Folder]: ...

    async def rename(self, workspace_id: uuid.UUID, folder_id: uuid.UUID, name: str) -> None: ...

    async def delete(self, workspace_id: uuid.UUID, folder_id: uuid.UUID) -> None: ...
