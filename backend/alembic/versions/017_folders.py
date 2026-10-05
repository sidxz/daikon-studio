"""folders: shared dataset and protocol folders

A flat, workspace-wide folder per kind ("dataset" or "protocol"), named uniquely
within its workspace and kind regardless of case. `datasets.folder_id` and
`protocols.folder_id` file an item; deleting a folder unfiles its items
(ON DELETE SET NULL) rather than touching them. Additive.

Revision ID: 017
Revises: 016
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "017"
down_revision: str | None = "016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "folders",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("created_by", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("kind IN ('dataset','protocol')", name="ck_folders_kind"),
    )
    op.create_index("ix_folders_workspace_id", "folders", ["workspace_id"])
    op.create_index(
        "uq_folders_workspace_kind_name",
        "folders",
        ["workspace_id", "kind", sa.text("lower(name)")],
        unique=True,
    )
    for table in ("datasets", "protocols"):
        op.add_column(
            table,
            sa.Column(
                "folder_id",
                sa.Uuid(),
                sa.ForeignKey("folders.id", ondelete="SET NULL"),
                nullable=True,
            ),
        )


def downgrade() -> None:
    for table in ("protocols", "datasets"):
        op.drop_column(table, "folder_id")
    op.drop_index("uq_folders_workspace_kind_name", table_name="folders")
    op.drop_index("ix_folders_workspace_id", table_name="folders")
    op.drop_table("folders")
