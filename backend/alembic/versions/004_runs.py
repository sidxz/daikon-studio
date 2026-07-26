"""runs

One execution -- training a model or making predictions. `status` is
constrained to the five values the aggregate uses (`Run` in
domain/execution/run.py); `cache_key` is indexed per workspace so Task 17 can
look up an identical prior request before enqueueing a new one instead of
recomputing it.

Revision ID: 004
Revises: 003
Create Date: 2026-07-25 23:10:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "runs",
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("requested_by", sa.Uuid(), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("phase", sa.String(length=256), nullable=True),
        sa.Column("result_uri", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
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
        sa.CheckConstraint(
            "status IN ('pending','running','ready','failed','cancelled')",
            name="ck_runs_status",
        ),
        sa.CheckConstraint("kind IN ('training','prediction')", name="ck_runs_kind"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_runs_workspace_created_at", "runs", ["workspace_id", "created_at"])
    op.create_index("ix_runs_workspace_cache_key", "runs", ["workspace_id", "cache_key"])
    op.create_index(op.f("ix_runs_workspace_id"), "runs", ["workspace_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_runs_workspace_id"), table_name="runs")
    op.drop_index("ix_runs_workspace_cache_key", table_name="runs")
    op.drop_index("ix_runs_workspace_created_at", table_name="runs")
    op.drop_table("runs")
