"""runners: the build each agent reports at start

Three nullable columns -- version, git sha and build date -- so an operator can
see which image a runner is running and spot one left behind by a deploy that
upgraded the API first (RELEASING.md). Nullable with no backfill: a runner that
has not restarted since this migration has reported nothing, and "unknown" is
the honest answer until it does. Additive and reversible.

Revision ID: 019
Revises: 018
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "019"
down_revision: str | None = "018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMNS = ("build_version", "build_git_sha", "build_date")


def upgrade() -> None:
    for name in _COLUMNS:
        op.add_column("runners", sa.Column(name, sa.Text(), nullable=True))


def downgrade() -> None:
    for name in _COLUMNS:
        op.drop_column("runners", name)
