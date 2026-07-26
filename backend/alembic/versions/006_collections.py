"""collections

A Collection is a scientist's saved triage decision over a `ready` prediction
Run's results -- `derived_from_run_id` records that lineage, but `snapshot_uri`
points at the Collection's *own* copy of the selected rows, not back into the
Run's blob (see `domain/data/collection.py`'s docstring for why: a Run's
results blob is not guaranteed to outlive the Run row forever, and a saved
Collection must). `provenance` is a JSONB envelope, the same shape
`datasets.target`/`datasets.split` already use for a typed value object at
this boundary.

Named 006 rather than 005: the task brief that specified this migration was
written against an earlier state of the migration chain, before 005 was
claimed by `005_training_inputs.py`. This picks up where the chain actually
is.

Revision ID: 006
Revises: 005
Create Date: 2026-07-26 00:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "collections",
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("derived_from_run_id", sa.Uuid(), nullable=False),
        sa.Column("member_count", sa.Integer(), nullable=False),
        sa.Column("snapshot_uri", sa.Text(), nullable=False),
        sa.Column("provenance", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
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
    op.create_index(op.f("ix_collections_workspace_id"), "collections", ["workspace_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_collections_workspace_id"), table_name="collections")
    op.drop_table("collections")
