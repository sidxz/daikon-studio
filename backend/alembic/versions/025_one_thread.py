"""one thread: a study keeps one conversation, a start time and a mode

A study held a conversation per stage and the plan it was started from. It now holds
one `thread`, a `started_at` that ends the draft, a `mode` and `amended_after_steps`.

Empties `studies` first, both ways. The table was never released (020-024 are
unreleased on this branch), and a row in the old shape -- an integer `produced_by` on
its values, a conversation per stage -- cannot be read by the new repository, nor a
new-shape row by the old one.

Revision ID: 025
Revises: 024
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "025"
down_revision: str | None = "024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DELETE FROM studies")
    op.add_column("studies", sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("studies", sa.Column("mode", sa.Text(), nullable=False, server_default="run"))
    op.add_column("studies", sa.Column("amended_after_steps", sa.Integer(), nullable=True))
    op.add_column(
        "studies",
        sa.Column("thread", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.drop_column("studies", "planning")
    op.drop_column("studies", "conversations")


def downgrade() -> None:
    op.execute("DELETE FROM studies")
    op.add_column(
        "studies",
        sa.Column("planning", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.add_column(
        "studies",
        sa.Column("conversations", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
    )
    op.drop_column("studies", "thread")
    op.drop_column("studies", "amended_after_steps")
    op.drop_column("studies", "mode")
    op.drop_column("studies", "started_at")
