"""studies: a saved analysis plan, its steps and the values they made

Steps and values are JSONB on the study row. Additive and reversible.

Revision ID: 020
Revises: 019
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "020"
down_revision: str | None = "019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "studies",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_by", sa.Uuid(), nullable=False),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column("restated_goal", sa.Text(), nullable=False),
        sa.Column("checks", JSONB(), nullable=False),
        sa.Column("stop_conditions", JSONB(), nullable=False),
        sa.Column("steps", JSONB(), nullable=False),
        sa.Column("values", JSONB(), nullable=False),
    )
    op.create_index("ix_studies_workspace_id", "studies", ["workspace_id"])
    op.create_index("ix_studies_workspace_created_at", "studies", ["workspace_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_studies_workspace_created_at", table_name="studies")
    op.drop_index("ix_studies_workspace_id", table_name="studies")
    op.drop_table("studies")
