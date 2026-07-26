"""datasets

The unique index on (workspace_id, content_hash) is the load-bearing one: it is
what makes "two Datasets with the same hash are the same data" an enforced fact
rather than a convention, and it is what turns a re-upload of identical data into
a conflict the application can answer with the id of the Dataset that already
holds it.

Revision ID: 002
Revises: 001
Create Date: 2026-07-25 21:13:26.967250

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "datasets",
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("target", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("split", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("snapshot_uri", sa.Text(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("validation_report", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_datasets_workspace_created_at", "datasets", ["workspace_id", "created_at"]
    )
    op.create_index(op.f("ix_datasets_workspace_id"), "datasets", ["workspace_id"])
    op.create_index(
        "uq_datasets_workspace_content_hash",
        "datasets",
        ["workspace_id", "content_hash"],
        unique=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("uq_datasets_workspace_content_hash", table_name="datasets")
    op.drop_index(op.f("ix_datasets_workspace_id"), table_name="datasets")
    op.drop_index("ix_datasets_workspace_created_at", table_name="datasets")
    op.drop_table("datasets")
