"""run_epochs: a training run's finished epochs, for the run page's live charts

Each neural fit (chemprop, MoLFormer) records one row per epoch: training and
validation loss and the validation scores. Append-only, stamped with the run's
attempt; deleted with its run. A new table, nothing existing changes, so it deploys
in any order with the API -- but run it before runners that send epochs, or their
posts fail (harmlessly: a runner drops points it cannot save).

Revision ID: 015
Revises: 014
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "015"
down_revision: str | None = "014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "run_epochs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("fit", sa.String(32), nullable=False),
        sa.Column("target", sa.Text(), nullable=True),
        sa.Column("member", sa.Integer(), nullable=True),
        sa.Column("members", sa.Integer(), nullable=True),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("epochs", sa.Integer(), nullable=False),
        sa.Column("train_loss", sa.Float(), nullable=True),
        sa.Column("val_loss", sa.Float(), nullable=True),
        sa.Column("scores", postgresql.JSONB(), nullable=False),
        sa.Column("device", sa.String(64), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_run_epochs_run_id", "run_epochs", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_run_epochs_run_id", table_name="run_epochs")
    op.drop_table("run_epochs")
