"""training inputs

Two columns the earlier tasks left out, both of which training cannot run
without.

`datasets.structure_column` -- the snapshot Parquet holds the structures under
whatever name the uploaded CSV used, and nothing recorded which name that was.
Every consumer of a Dataset (training, prediction, scaffold analysis) has to
featurize that column, and guessing it from column order would make the answer
depend on how polars happens to order a group_by.

`runs.params` -- arq hands the worker a bare `run_id` and nothing else, so the
Run row is the only place the job's own inputs (which dataset, which engine,
which conditions) can live. Kept on the row rather than in a side blob so
"what did this run actually train?" is answerable in SQL.

Revision ID: 005
Revises: 004
Create Date: 2026-07-25 23:55:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # server_default then drop: both tables may already hold rows on a dev
    # database, and NOT NULL is the point -- a Dataset with no structure column
    # is untrainable, and a Run with no params is unexecutable. The backfill
    # value is deliberately obvious rather than plausible: any pre-existing row
    # genuinely has no answer here, and "smiles" would be a guess wearing a
    # correct-looking face.
    op.add_column(
        "datasets",
        sa.Column(
            "structure_column", sa.String(length=128), nullable=False, server_default="unknown"
        ),
    )
    op.alter_column("datasets", "structure_column", server_default=None)
    op.add_column(
        "runs",
        sa.Column(
            "params",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
    )
    op.alter_column("runs", "params", server_default=None)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("runs", "params")
    op.drop_column("datasets", "structure_column")
