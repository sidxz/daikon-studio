"""run_epochs.kept_epoch and kept_by: the epoch a fit keeps, and the rule that chose it

A classification fit now keeps the epoch its `epoch_selection` names (best validation
PR AUC by default) rather than the lowest validation loss, so the run page can no
longer work the kept epoch out for itself: the fit reports it, and which rule chose
it ("auprc", "auroc" or "loss"). Nullable, and null on every row recorded before,
whose fits all kept the lowest validation loss. Additive, so it deploys before the
runners that send it; runners that do not send it are unaffected.

Revision ID: 016
Revises: 015
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "016"
down_revision: str | None = "015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("run_epochs", sa.Column("kept_epoch", sa.Integer(), nullable=True))
    op.add_column("run_epochs", sa.Column("kept_by", sa.String(16), nullable=True))


def downgrade() -> None:
    op.drop_column("run_epochs", "kept_by")
    op.drop_column("run_epochs", "kept_epoch")
