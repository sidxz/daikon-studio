"""protocols

`parent_protocol_id` is a nullable self-reference: the version chain lives on
this one table (parent id + protocol_version) rather than a separate
ProtocolVersion table, mirroring how the sibling assay-protocol project models
lab protocol versions. The status check constraint is the database's half of
`publish()`'s invariant -- only ('draft', 'published') is ever a legal value,
so a stray write from outside the aggregate cannot park a row in limbo.

Revision ID: 003
Revises: 002
Create Date: 2026-07-25 22:40:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "protocols",
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("dataset_id", sa.Uuid(), nullable=False),
        sa.Column("engine_id", sa.String(length=128), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("readouts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("conditions", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("parent_protocol_id", sa.Uuid(), nullable=True),
        sa.Column("protocol_version", sa.Integer(), nullable=False),
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
        sa.CheckConstraint("status IN ('draft','published')", name="ck_protocols_status"),
        sa.ForeignKeyConstraint(["parent_protocol_id"], ["protocols.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_protocols_workspace_created_at", "protocols", ["workspace_id", "created_at"]
    )
    op.create_index(op.f("ix_protocols_workspace_id"), "protocols", ["workspace_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_protocols_workspace_id"), table_name="protocols")
    op.drop_index("ix_protocols_workspace_created_at", table_name="protocols")
    op.drop_table("protocols")
