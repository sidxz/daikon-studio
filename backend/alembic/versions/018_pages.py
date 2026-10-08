"""pages: lab-notebook pages on datasets, protocols and runs

`pages` holds one row per page, mounted on its owner by kind and id (no foreign key:
the owner may be any of three tables, so deleting an owner deletes its pages in the
use case). Every save appends a `page_revisions` row, which cascades with its page,
naming a body in `page_contents` by sha256; bodies are shared and kept. `page_blobs`
catalogues the files pages embed, once per workspace and sha256; the bytes live in the
blob store. Additive.

Revision ID: 018
Revises: 017
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "018"
down_revision: str | None = "017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
    )


def upgrade() -> None:
    op.create_table(
        "pages",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("owner_kind", sa.String(16), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("head_sha256", sa.String(64), nullable=True),
        sa.Column("created_by", sa.Uuid(), nullable=False),
        sa.Column("last_edited_by", sa.Uuid(), nullable=True),
        sa.Column("archived", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("revision_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("version", sa.Integer(), server_default="0", nullable=False),
        _created_at(),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("owner_kind IN ('dataset','protocol','run')", name="ck_pages_owner_kind"),
    )
    op.create_index("ix_pages_owner", "pages", ["workspace_id", "owner_kind", "owner_id"])
    op.create_table(
        "page_contents",
        sa.Column("sha256", sa.String(64), primary_key=True),
        sa.Column("body", JSONB(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        _created_at(),
    )
    op.create_table(
        "page_revisions",
        sa.Column(
            "page_id",
            sa.Uuid(),
            sa.ForeignKey("pages.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("revision_no", sa.Integer(), primary_key=True),
        sa.Column(
            "sha256", sa.String(64), sa.ForeignKey("page_contents.sha256"), nullable=False
        ),
        sa.Column("author_id", sa.Uuid(), nullable=False),
        _created_at(),
    )
    op.create_table(
        "page_blobs",
        sa.Column("workspace_id", sa.Uuid(), primary_key=True),
        sa.Column("sha256", sa.String(64), primary_key=True),
        sa.Column("mime", sa.String(255), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("created_by", sa.Uuid(), nullable=False),
        _created_at(),
    )


def downgrade() -> None:
    op.drop_table("page_blobs")
    op.drop_table("page_revisions")
    op.drop_table("page_contents")
    op.drop_index("ix_pages_owner", table_name="pages")
    op.drop_table("pages")
