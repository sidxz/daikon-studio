"""tool runs: a run may be a tool step, queued for a runner

Widens `ck_runs_kind`; downgrade narrows it again (and fails while tool runs exist).

Revision ID: 022
Revises: 021
"""

from collections.abc import Sequence

from alembic import op

revision: str = "022"
down_revision: str | None = "021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("ck_runs_kind", "runs", type_="check")
    op.create_check_constraint("ck_runs_kind", "runs", "kind IN ('training','prediction','tool')")


def downgrade() -> None:
    op.drop_constraint("ck_runs_kind", "runs", type_="check")
    op.create_check_constraint("ck_runs_kind", "runs", "kind IN ('training','prediction')")
