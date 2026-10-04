"""datasets.targets, and the per-target shapes that come with it

A Dataset carries an ordered list of target specs instead of exactly one; every
existing row becomes a one-element list, in place. Two values inside
`validation_report` change shape in the same step, because each only means
something per target once there can be several:

- `duplicate_spread`, a number or null, becomes an object keyed by target column
  (empty when there was nothing to measure);
- every `conflicting` entry gains the `column` its labels disagree in.

Downgrade restores the single-target shapes, and refuses while any dataset has
more than one target: dropping the others would silently destroy data that
protocols and runs still cite.

Revision ID: 013
Revises: 012
Create Date: 2026-10-03 00:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "013"
down_revision: str | None = "012"
branch_labels: str | None = None
depends_on: str | None = None

TARGETS_FROM_TARGET = "UPDATE datasets SET targets = jsonb_build_array(target)"

KEY_SPREAD_BY_TARGET = """
UPDATE datasets
SET validation_report = jsonb_set(
    validation_report,
    '{duplicate_spread}',
    CASE
        WHEN jsonb_typeof(validation_report -> 'duplicate_spread') = 'number'
        THEN jsonb_build_object(target ->> 'column', validation_report -> 'duplicate_spread')
        ELSE '{}'::jsonb
    END
)
"""

TAG_CONFLICT_COLUMNS = """
UPDATE datasets
SET validation_report = jsonb_set(
    validation_report,
    '{conflicting}',
    (
        SELECT COALESCE(
            jsonb_agg(entry || jsonb_build_object('column', target ->> 'column')), '[]'::jsonb
        )
        FROM jsonb_array_elements(validation_report -> 'conflicting') AS entry
    )
)
WHERE jsonb_typeof(validation_report -> 'conflicting') = 'array'
"""

COUNT_MULTI_TARGET = "SELECT count(*) FROM datasets WHERE jsonb_array_length(targets) > 1"

TARGET_FROM_TARGETS = "UPDATE datasets SET target = targets -> 0"

UNKEY_SPREAD = """
UPDATE datasets
SET validation_report = jsonb_set(
    validation_report,
    '{duplicate_spread}',
    COALESCE(validation_report -> 'duplicate_spread' -> (target ->> 'column'), 'null'::jsonb)
)
"""

UNTAG_CONFLICT_COLUMNS = """
UPDATE datasets
SET validation_report = jsonb_set(
    validation_report,
    '{conflicting}',
    (
        SELECT COALESCE(jsonb_agg(entry - 'column'), '[]'::jsonb)
        FROM jsonb_array_elements(validation_report -> 'conflicting') AS entry
    )
)
WHERE jsonb_typeof(validation_report -> 'conflicting') = 'array'
"""


def upgrade() -> None:
    op.add_column("datasets", sa.Column("targets", postgresql.JSONB(), nullable=True))
    op.execute(TARGETS_FROM_TARGET)
    op.execute(KEY_SPREAD_BY_TARGET)
    op.execute(TAG_CONFLICT_COLUMNS)
    op.alter_column("datasets", "targets", nullable=False)
    op.drop_column("datasets", "target")


def downgrade() -> None:
    if op.get_bind().execute(sa.text(COUNT_MULTI_TARGET)).scalar_one():
        raise RuntimeError(
            "Cannot downgrade below 013: some datasets have more than one target, and the "
            "single-target schema would silently drop the others. Delete those datasets "
            "and their protocols first."
        )
    op.add_column("datasets", sa.Column("target", postgresql.JSONB(), nullable=True))
    op.execute(TARGET_FROM_TARGETS)
    op.execute(UNKEY_SPREAD)
    op.execute(UNTAG_CONFLICT_COLUMNS)
    op.alter_column("datasets", "target", nullable=False)
    op.drop_column("datasets", "targets")
