"""Lab-notebook pages: list, create, read, revise, retitle, archive, delete, history, and
the files embedded in them.

A page is exactly as visible as the dataset, protocol or run it sits on, so a hidden or
missing owner is a 404 and never a 403: a colleague's draft protocol keeps its pages to
itself. Writing needs an editor; archiving needs the page's author or an admin, the same
people who may delete it.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass
from typing import Any

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import (
    AuthContext,
    may_delete,
    require_authenticated,
    require_editor,
    require_may_delete,
)
from daikonstudio.application.catalog.visibility import visible_protocol
from daikonstudio.application.execution.visibility import run_visible
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.page_repository import PageRepository
from daikonstudio.application.ports.protocol_access import ProtocolAccess
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.domain.shared.errors import (
    AuthorizationError,
    ConflictError,
    DataLockedError,
    DomainError,
    NotFoundError,
    ValidationError,
)
from daikonstudio.domain.shared.page import (
    Page,
    PageBlob,
    PageOwnerKind,
    PageRevision,
    clean_page_title,
    page_blob_mime,
    page_body,
)

MAX_LISTED_PAGES = 200

STALE = "This page was changed since you opened it. Reload to see the latest version."
ARCHIVED = "This page is archived. Restore it to edit it."


def page_blob_key(workspace_id: uuid.UUID, sha256: str) -> str:
    return f"{workspace_id}/page-blobs/{sha256}"


class PageOwners:
    """Whether the caller may see a page's owner, by the owner's own read rules."""

    def __init__(
        self,
        datasets: DatasetRepository,
        protocols: ProtocolRepository,
        runs: RunRepository,
        access: ProtocolAccess,
    ) -> None:
        self._datasets = datasets
        self._protocols = protocols
        self._runs = runs
        self._access = access

    async def visible(self, auth: AuthContext, kind: PageOwnerKind, owner_id: uuid.UUID) -> bool:
        if kind is PageOwnerKind.DATASET:
            return await self._datasets.get(auth.workspace_id, owner_id) is not None
        if kind is PageOwnerKind.PROTOCOL:
            protocol = await visible_protocol(
                self._protocols, self._access, auth, auth.workspace_id, owner_id
            )
            return protocol is not None
        run = await self._runs.get(auth.workspace_id, owner_id)
        return run is not None and await run_visible(run, auth, self._access)


async def _visible_page(
    pages: PageRepository, owners: PageOwners, auth: AuthContext, page_id: uuid.UUID
) -> Page | None:
    page = await pages.get(auth.workspace_id, page_id)
    if page is None or not await owners.visible(auth, page.owner_kind, page.owner_id):
        return None
    return page


def _not_found(page_id: uuid.UUID) -> Failure[NotFoundError]:
    return Failure(NotFoundError("Page", str(page_id)))


@dataclass(frozen=True, kw_only=True)
class ListPagesQuery:
    owner_kind: PageOwnerKind
    owner_id: uuid.UUID
    archived: bool = False


class ListPages:
    def __init__(self, pages: PageRepository, owners: PageOwners) -> None:
        self._pages = pages
        self._owners = owners

    async def __call__(
        self, query: ListPagesQuery, auth: AuthContext | None = None
    ) -> Result[list[Page], DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        if not await self._owners.visible(auth, query.owner_kind, query.owner_id):
            return Failure(NotFoundError(query.owner_kind.value.title(), str(query.owner_id)))
        return Success(
            await self._pages.list(
                auth.workspace_id,
                query.owner_kind,
                query.owner_id,
                archived=query.archived,
                limit=MAX_LISTED_PAGES,
            )
        )


@dataclass(frozen=True, kw_only=True)
class CreatePageCommand:
    title: str
    owner_kind: PageOwnerKind
    owner_id: uuid.UUID


class CreatePage:
    def __init__(self, pages: PageRepository, owners: PageOwners) -> None:
        self._pages = pages
        self._owners = owners

    async def __call__(
        self, command: CreatePageCommand, auth: AuthContext | None = None
    ) -> Result[Page, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        if not await self._owners.visible(auth, command.owner_kind, command.owner_id):
            return Failure(NotFoundError(command.owner_kind.value.title(), str(command.owner_id)))
        try:
            title = clean_page_title(command.title)
        except ValidationError as error:
            return Failure(error)
        page = Page(
            workspace_id=auth.workspace_id,
            owner_kind=command.owner_kind,
            owner_id=command.owner_id,
            title=title,
            created_by=auth.user_id,
        )
        await self._pages.add(page)
        return Success(page)


class GetPage:
    def __init__(self, pages: PageRepository, owners: PageOwners) -> None:
        self._pages = pages
        self._owners = owners

    async def __call__(
        self, page_id: uuid.UUID, auth: AuthContext | None = None
    ) -> Result[Page, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        page = await _visible_page(self._pages, self._owners, auth, page_id)
        return _not_found(page_id) if page is None else Success(page)


@dataclass(frozen=True, kw_only=True)
class RevisePageCommand:
    page_id: uuid.UUID
    body: dict[str, Any]
    expected_version: int


class RevisePage:
    """Save a new body. Saving the body the page already shows changes nothing."""

    def __init__(self, pages: PageRepository, owners: PageOwners) -> None:
        self._pages = pages
        self._owners = owners

    async def __call__(
        self, command: RevisePageCommand, auth: AuthContext | None = None
    ) -> Result[Page, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        page = await _visible_page(self._pages, self._owners, auth, command.page_id)
        if page is None:
            return _not_found(command.page_id)
        try:
            body = page_body(command.body)
        except ValidationError as error:
            return Failure(error)
        if command.expected_version != page.version:
            return Failure(ConflictError(STALE))
        if page.archived:
            return Failure(DataLockedError(ARCHIVED))
        if body.sha256 == page.head_sha256:
            return Success(page)
        try:
            revised = await self._pages.revise(
                auth.workspace_id,
                page.id,
                expected_version=command.expected_version,
                body=body,
                author_id=auth.user_id,
            )
        except ConflictError as error:
            return Failure(error)
        return Success(revised)


@dataclass(frozen=True, kw_only=True)
class RetitlePageCommand:
    page_id: uuid.UUID
    title: str


class RetitlePage:
    def __init__(self, pages: PageRepository, owners: PageOwners) -> None:
        self._pages = pages
        self._owners = owners

    async def __call__(
        self, command: RetitlePageCommand, auth: AuthContext | None = None
    ) -> Result[Page, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        page = await _visible_page(self._pages, self._owners, auth, command.page_id)
        if page is None:
            return _not_found(command.page_id)
        if page.archived:
            return Failure(DataLockedError(ARCHIVED))
        try:
            title = clean_page_title(command.title)
        except ValidationError as error:
            return Failure(error)
        return Success(await self._pages.retitle(auth.workspace_id, page.id, title))


@dataclass(frozen=True, kw_only=True)
class ArchivePageCommand:
    page_id: uuid.UUID
    restore: bool = False


class ArchivePage:
    def __init__(self, pages: PageRepository, owners: PageOwners) -> None:
        self._pages = pages
        self._owners = owners

    async def __call__(
        self, command: ArchivePageCommand, auth: AuthContext | None = None
    ) -> Result[Page, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        page = await _visible_page(self._pages, self._owners, auth, command.page_id)
        if page is None:
            return _not_found(command.page_id)
        if not may_delete(auth, page.created_by):
            raise AuthorizationError("Only an admin or the page's author can archive it.")
        archived = not command.restore
        if page.archived == archived:
            return Success(page)
        return Success(await self._pages.set_archived(auth.workspace_id, page.id, archived))


class DeletePage:
    def __init__(self, pages: PageRepository, owners: PageOwners) -> None:
        self._pages = pages
        self._owners = owners

    async def __call__(
        self, page_id: uuid.UUID, auth: AuthContext | None = None
    ) -> Result[None, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        page = await _visible_page(self._pages, self._owners, auth, page_id)
        if page is None:
            return _not_found(page_id)
        require_may_delete(auth, page.created_by)
        await self._pages.delete(auth.workspace_id, page.id)
        return Success(None)


class ListPageRevisions:
    def __init__(self, pages: PageRepository, owners: PageOwners) -> None:
        self._pages = pages
        self._owners = owners

    async def __call__(
        self, page_id: uuid.UUID, auth: AuthContext | None = None
    ) -> Result[list[PageRevision], DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        page = await _visible_page(self._pages, self._owners, auth, page_id)
        if page is None:
            return _not_found(page_id)
        return Success(await self._pages.revisions(page.id))


@dataclass(frozen=True, kw_only=True)
class GetPageContentQuery:
    page_id: uuid.UUID
    sha256: str


class GetPageContent:
    """One revision's body, by its sha256. Only this page's own: the hash of another
    page's body is not a way into that page."""

    def __init__(self, pages: PageRepository, owners: PageOwners) -> None:
        self._pages = pages
        self._owners = owners

    async def __call__(
        self, query: GetPageContentQuery, auth: AuthContext | None = None
    ) -> Result[dict[str, Any], DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        page = await _visible_page(self._pages, self._owners, auth, query.page_id)
        if page is None:
            return _not_found(query.page_id)
        body = await self._pages.content(page.id, query.sha256)
        if body is None:
            return Failure(NotFoundError("Revision", query.sha256))
        return Success(body)


@dataclass(frozen=True, kw_only=True)
class UploadPageBlobCommand:
    data: bytes
    mime: str | None


class UploadPageBlob:
    """Store an image for a page to embed, once per workspace however often it is added."""

    def __init__(self, pages: PageRepository, store: BlobStore) -> None:
        self._pages = pages
        self._store = store

    async def __call__(
        self, command: UploadPageBlobCommand, auth: AuthContext | None = None
    ) -> Result[PageBlob, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        if not command.data:
            return Failure(ValidationError("The uploaded file is empty"))
        try:
            mime = page_blob_mime(command.mime, command.data)
        except ValidationError as error:
            return Failure(error)
        sha256 = hashlib.sha256(command.data).hexdigest()
        existing = await self._pages.get_blob(auth.workspace_id, sha256)
        if existing is not None:
            return Success(existing)
        # Off the loop: megabytes to network storage would stall every other request.
        await asyncio.to_thread(
            self._store.put_bytes, page_blob_key(auth.workspace_id, sha256), command.data
        )
        blob = PageBlob(
            workspace_id=auth.workspace_id,
            sha256=sha256,
            mime=mime,
            size_bytes=len(command.data),
            created_by=auth.user_id,
        )
        await self._pages.add_blob(blob)
        return Success(blob)


class GetPageBlob:
    def __init__(self, pages: PageRepository, store: BlobStore) -> None:
        self._pages = pages
        self._store = store

    async def __call__(
        self, sha256: str, auth: AuthContext | None = None
    ) -> Result[tuple[PageBlob, bytes], DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        blob = await self._pages.get_blob(auth.workspace_id, sha256)
        if blob is None:
            return Failure(NotFoundError("File", sha256))
        data = await asyncio.to_thread(
            self._store.get_bytes, page_blob_key(auth.workspace_id, sha256)
        )
        return Success((blob, data))
