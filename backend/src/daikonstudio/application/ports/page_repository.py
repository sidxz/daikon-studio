"""Persistence port for pages, their revisions and their embedded files. Workspace-scoped
in the SQL, like every other port.

`revise` is the one write that races: it raises `ConflictError` unless the page is still
at `expected_version` when the row is updated, so of two saves from the same version one
loses. `content` answers only for a sha256 that is one of that page's own revisions."""

import builtins
import uuid
from collections.abc import Sequence
from typing import Any, Protocol

from daikonstudio.domain.shared.page import (
    Page,
    PageBlob,
    PageBody,
    PageOwnerKind,
    PageRevision,
)


class PageRepository(Protocol):
    async def add(self, page: Page) -> None: ...

    async def get(self, workspace_id: uuid.UUID, page_id: uuid.UUID) -> Page | None: ...

    async def list(
        self,
        workspace_id: uuid.UUID,
        owner_kind: PageOwnerKind,
        owner_id: uuid.UUID,
        *,
        archived: bool,
        limit: int,
    ) -> builtins.list[Page]:
        """Newest-updated first."""
        ...

    async def revise(
        self,
        workspace_id: uuid.UUID,
        page_id: uuid.UUID,
        *,
        expected_version: int,
        body: PageBody,
        author_id: uuid.UUID,
    ) -> Page: ...

    async def retitle(self, workspace_id: uuid.UUID, page_id: uuid.UUID, title: str) -> Page: ...

    async def set_archived(
        self, workspace_id: uuid.UUID, page_id: uuid.UUID, archived: bool
    ) -> Page: ...

    async def delete(self, workspace_id: uuid.UUID, page_id: uuid.UUID) -> None: ...

    async def delete_for_owners(
        self, workspace_id: uuid.UUID, owner_kind: PageOwnerKind, owner_ids: Sequence[uuid.UUID]
    ) -> None:
        """Every page on these owners, for the use cases that delete the owners."""
        ...

    async def revisions(self, page_id: uuid.UUID) -> builtins.list[PageRevision]:
        """Oldest first. Callers have already scoped the page to its workspace."""
        ...

    async def content(self, page_id: uuid.UUID, sha256: str) -> dict[str, Any] | None: ...

    async def add_blob(self, blob: PageBlob) -> None:
        """A no-op when the workspace already holds this sha256."""
        ...

    async def get_blob(self, workspace_id: uuid.UUID, sha256: str) -> PageBlob | None: ...
