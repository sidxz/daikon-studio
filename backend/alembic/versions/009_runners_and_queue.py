"""runners + queue columns

The runs table becomes the job queue: a pending run with a lane set is
claimable by a registered runner. Runners are instance-level (no
workspace_id) -- see the 2026-08-04 self-hosted-runners spec.

Revision ID: 009
Revises: 008
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "009"
down_revision: str | None = "008"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "runners",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("lanes", JSONB(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_runners_name"),
        sa.UniqueConstraint("token_hash", name="uq_runners_token_hash"),
    )

    op.add_column("runs", sa.Column("lane", sa.String(length=32), nullable=True))
    op.add_column("runs", sa.Column("claimed_by", sa.Uuid(), nullable=True))
    op.add_column("runs", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "runs", sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False)
    )
    op.create_index(
        "ix_runs_claimable",
        "runs",
        ["lane", "created_at"],
        postgresql_where=sa.text("status = 'pending'"),
    )
    # Cutover backfill. arq's in-flight queue disappears with Valkey, so:
    # pending runs become claimable on the default lane (locally the gpu lane
    # ran on CPU anyway; a wrong-lane pending run at cutover re-runs slower,
    # not wrongly), and running runs get an already-expired lease so the first
    # sweep requeues them instead of leaving them RUNNING forever.
    op.execute("UPDATE runs SET lane = 'default' WHERE status IN ('pending', 'running')")
    op.execute("UPDATE runs SET lease_expires_at = now() WHERE status = 'running'")


def downgrade() -> None:
    op.drop_index("ix_runs_claimable", table_name="runs")
    op.drop_column("runs", "attempts")
    op.drop_column("runs", "lease_expires_at")
    op.drop_column("runs", "claimed_by")
    op.drop_column("runs", "lane")
    op.drop_table("runners")
