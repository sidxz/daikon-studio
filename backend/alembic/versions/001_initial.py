"""initial

Revision ID: 001
Revises:
Create Date: 2026-07-25 16:38:02.260254

"""

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # ponytail: empty baseline — no ORM models exist yet (Task 3). Later
    # migrations build on this revision; this one deliberately does nothing.
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
