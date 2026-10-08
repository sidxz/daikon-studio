"""A lab-notebook page: a rich-text document (ProseMirror JSON) mounted on one dataset,
protocol or run. Shared across domain contexts, like `folder.py`, so the owner is a
plain kind and id rather than an import of another context's aggregate.

Every save is a revision. Bodies are content-addressed by the sha256 of their canonical
JSON, so an unchanged save is recognizable and identical bodies are stored once.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from daikonstudio.domain.shared.errors import ValidationError

MAX_PAGE_TITLE = 200
MAX_PAGE_BODY_BYTES = 2 * 1024 * 1024
RASTER_MIMES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})


class PageOwnerKind(StrEnum):
    DATASET = "dataset"
    PROTOCOL = "protocol"
    RUN = "run"


@dataclass(frozen=True, kw_only=True)
class Page:
    workspace_id: uuid.UUID
    owner_kind: PageOwnerKind
    owner_id: uuid.UUID
    title: str
    created_by: uuid.UUID
    head_sha256: str | None = None
    last_edited_by: uuid.UUID | None = None
    archived: bool = False
    revision_count: int = 0
    # Bumped by every change (revise, retitle, archive); a revise names the one it read.
    version: int = 0
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, kw_only=True)
class PageRevision:
    revision_no: int
    sha256: str
    author_id: uuid.UUID
    created_at: datetime


@dataclass(frozen=True, kw_only=True)
class PageBlob:
    """An image or file embedded in a page, stored once per workspace by its sha256."""

    workspace_id: uuid.UUID
    sha256: str
    mime: str
    size_bytes: int
    created_by: uuid.UUID


@dataclass(frozen=True, kw_only=True)
class PageBody:
    body: dict[str, Any]
    sha256: str
    size_bytes: int


def clean_page_title(raw: str) -> str:
    title = raw.strip()
    if not 1 <= len(title) <= MAX_PAGE_TITLE:
        raise ValidationError("A page title must be 1 to 200 characters.")
    return title


def page_body(body: object) -> PageBody:
    """Validate a document and address it by the sha256 of its canonical JSON."""
    if not isinstance(body, dict) or body.get("type") != "doc":
        raise ValidationError("A page body must be a document.")
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    if len(canonical) > MAX_PAGE_BODY_BYTES:
        raise ValidationError("A page can hold at most 2 MB of text and formatting.")
    return PageBody(
        body=body, sha256=hashlib.sha256(canonical).hexdigest(), size_bytes=len(canonical)
    )


def _sniff_raster(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def page_blob_mime(declared: str | None, data: bytes) -> str:
    """The type a page file is stored and served as: an image, and only the image its
    bytes say it is. The browser's declared type is never trusted alone, or an HTML
    page uploaded as `image/png` would be served back to run in the studio's origin.
    An SVG cannot be sniffed the same way; it is accepted because pages only ever show
    files through `<img>`, where an SVG's scripts do not run."""
    mime = (declared or "").split(";")[0].strip().lower()
    if mime == "image/svg+xml" and data.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"<"):
        return mime
    if mime in RASTER_MIMES and _sniff_raster(data) == mime:
        return mime
    raise ValidationError("Unsupported file type")
