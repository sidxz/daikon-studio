"""datasets.id_column

Which snapshot column holds the compounds' own IDs. Every snapshot already keeps
the upload's other columns; nothing recorded which one is the identifier. NULL
for every existing dataset, which can be given one later from its page.

Revision ID: 012
Revises: 011
Create Date: 2026-10-03 00:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "012"
down_revision: str | None = "011"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("id_column", sa.String(128), nullable=True))


def downgrade() -> None:
    op.drop_column("datasets", "id_column")
