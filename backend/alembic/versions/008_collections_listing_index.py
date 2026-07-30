"""collections listing index

`GET /api/v1/collections` pages newest-first by keyset on
`(created_at, id)`, the same shape `datasets` and `runs` already index. Those
two got their index with their table; `collections` did not, because it had no
listing endpoint until now.

Revision ID: 008
Revises: 007
Create Date: 2026-07-29 00:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "008"
down_revision: str | None = "007"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_index(
        "ix_collections_workspace_created_at", "collections", ["workspace_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_collections_workspace_created_at", table_name="collections")
