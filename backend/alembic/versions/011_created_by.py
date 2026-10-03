"""datasets.created_by and protocols.created_by

Who may delete a dataset or a draft protocol: an admin, or the person who
created it. Nothing recorded the creator before this revision.

Protocols are backfilled from the training run that produced them, whose
`requested_by` is exactly that person. Datasets cannot be: nothing recorded who
uploaded one, so existing datasets keep NULL and only an admin can delete them.

Revision ID: 011
Revises: 010
Create Date: 2026-10-03 00:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "011"
down_revision: str | None = "010"
branch_labels: str | None = None
depends_on: str | None = None

# The earliest training run linked to each protocol. A protocol has one today;
# DISTINCT ON keeps this correct if that ever changes.
BACKFILL_PROTOCOL_CREATORS = """
UPDATE protocols AS p
SET created_by = r.requested_by
FROM (
    SELECT DISTINCT ON (protocol_id) protocol_id, workspace_id, requested_by
    FROM runs
    WHERE kind = 'training' AND protocol_id IS NOT NULL
    ORDER BY protocol_id, created_at
) AS r
WHERE r.protocol_id = p.id AND r.workspace_id = p.workspace_id AND p.created_by IS NULL
"""


def upgrade() -> None:
    op.add_column("datasets", sa.Column("created_by", sa.Uuid(), nullable=True))
    op.add_column("protocols", sa.Column("created_by", sa.Uuid(), nullable=True))
    op.execute(BACKFILL_PROTOCOL_CREATORS)


def downgrade() -> None:
    op.drop_column("protocols", "created_by")
    op.drop_column("datasets", "created_by")
