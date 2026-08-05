"""runs.sweep_id and runs.metrics

Fan-out: N training configs submitted as one named group. `sweep_id` is that
group, and there is no `sweeps` table behind it -- a sweep's only state is its
members, so a `GROUP BY` answers everything the list page asks and a second
aggregate inside the execution context would own no invariant the Run does not
already own.

`metrics` denormalises the headline number out of the Scorecard blob. Building
a Scorecard recomputes RDKit Tanimoto similarity over train x test on every
call; a page whose purpose is comparing twenty runs must not pay that twenty
times per visit.

No backfill for either. Every existing run predates sweeps, so `sweep_id` is
correctly NULL; and `metrics` for a historical run is recoverable only by
reading and parsing its Scorecard blob, which is exactly the cost this column
exists to avoid paying. Old runs rank as unmeasured, which is honest -- they
were never part of a sweep.

Revision ID: 010
Revises: 009
Create Date: 2026-08-05 00:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "010"
down_revision: str | None = "009"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("sweep_id", sa.Uuid(), nullable=True))
    op.add_column("runs", sa.Column("metrics", postgresql.JSONB(), nullable=True))
    op.create_index("ix_runs_workspace_sweep_id", "runs", ["workspace_id", "sweep_id"])


def downgrade() -> None:
    op.drop_index("ix_runs_workspace_sweep_id", table_name="runs")
    op.drop_column("runs", "metrics")
    op.drop_column("runs", "sweep_id")
