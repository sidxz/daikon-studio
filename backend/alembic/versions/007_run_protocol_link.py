"""runs.protocol_id

A finished training Run had no way to name the Protocol it produced. The id was
recoverable only by parsing it back out of the blob path in `result_uri`, which
would tie every client to the storage layout, so a client holding a run id could
not reach the Scorecard its own work had just built.

`protocol_id` is nullable by nature rather than by laxity: a training Run has no
Protocol until it finishes. It is a bare indexed UUID, not a ForeignKey --
`catalog` and `execution` are separate bounded contexts, and cross-context
references are plain ids by contract.

The backfill deliberately covers prediction runs only. Their `params` has
carried `protocol_id` since the run was enqueued, so the value is already known
and exact. Training runs have no such record -- inferring one from `result_uri`
is the very coupling this column exists to remove, and an existing training Run
losing its link is a cosmetic gap in dev data, not a correctness problem.

Revision ID: 007
Revises: 006
Create Date: 2026-07-29 00:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "007"
down_revision: str | None = "006"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("protocol_id", sa.Uuid(), nullable=True))
    op.execute(
        """
        UPDATE runs
           SET protocol_id = (params ->> 'protocol_id')::uuid
         WHERE kind = 'prediction'
           AND params ? 'protocol_id'
           AND params ->> 'protocol_id' IS NOT NULL
        """
    )
    # Backs "which runs belong to this Protocol" -- the Protocol detail page's
    # run history, and the only query this column exists to serve.
    op.create_index("ix_runs_workspace_protocol_id", "runs", ["workspace_id", "protocol_id"])


def downgrade() -> None:
    op.drop_index("ix_runs_workspace_protocol_id", table_name="runs")
    op.drop_column("runs", "protocol_id")
