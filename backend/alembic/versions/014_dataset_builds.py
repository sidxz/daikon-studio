"""dataset_builds: datasets built in the background, with progress

Creating a dataset from a large file runs RDKit over every row, which takes minutes;
the wizard now starts a build and polls this row for its stage and row count. A new
table, nothing existing changes, so it deploys in any order with the API.

Revision ID: 014
Revises: 013
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "014"
down_revision: str | None = "013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "dataset_builds",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_by", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("stage", sa.String(64), nullable=False),
        sa.Column("done", sa.Integer(), nullable=False),
        sa.Column("total", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Uuid(), nullable=True),
        sa.Column("error", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_dataset_builds_workspace_id", "dataset_builds", ["workspace_id"])


def downgrade() -> None:
    op.drop_index("ix_dataset_builds_workspace_id", table_name="dataset_builds")
    op.drop_table("dataset_builds")
