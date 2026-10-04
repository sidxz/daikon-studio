"""Bring the database to this image's migration head when the API starts.

The API migrates itself, so a deploy is one step: no separate migrate job to order
before it (Swarm ignores compose's `depends_on`, which is what that ordering hung on).
Only the API's start calls this -- the same image is the CPU runner, whose process
never touches the database -- and `alembic upgrade head` stays the manual route.

Two APIs starting together (a second replica, or a rolling update's overlap) are safe:
`alembic/env.py` takes a Postgres advisory lock for the length of the migration
transaction, so the second waits, then finds the database already at head.

The config is built in code, with no alembic.ini. `env.py` only calls `fileConfig`
when there is an ini file, and called in-process that replaces the app's logging
setup -- Duar lost every structured log line that way until 0.20.1.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

import daikonstudio

# <repo>/backend/alembic in a checkout, /app/alembic in the image: beside `src` either way.
SCRIPT_LOCATION = Path(daikonstudio.__file__).resolve().parents[2] / "alembic"


def upgrade_to_head(database_url: str) -> None:
    """Synchronous, and `env.py` runs its own event loop: call it off the running one."""
    config = Config()
    config.set_main_option("script_location", str(SCRIPT_LOCATION))
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")
