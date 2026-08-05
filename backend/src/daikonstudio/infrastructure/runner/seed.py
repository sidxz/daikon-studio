"""Local-dev bootstrap, NOT for production: upserts the two fixed-token
runners `make dev`'s stack expects, so a developer can start
`python -m daikonstudio.infrastructure.runner` without first registering a
runner through the UI. A production runner is always created through
`POST /api/v1/runners` (`CreateRunner`), which mints a random token surfaced
exactly once and never written to a file.

Run as `python -m daikonstudio.infrastructure.runner.seed`.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import create_async_engine

from daikonstudio.infrastructure.persistence.sqlalchemy.runners.models import RunnerModel
from daikonstudio.settings import Settings

# ponytail: fixed plaintext dev tokens; fine while the dev DB only listens on
# 127.0.0.1 -- production runners come from the UI.
_DEV_RUNNERS: list[tuple[str, list[str], str]] = [
    ("dev-local-default", ["default"], "drt_dev_default"),
    ("dev-local-gpu", ["gpu"], "drt_dev_gpu"),
]


def _hash_token(token: str) -> str:
    # Same digest `application.runners.manage._hash_token` computes -- not
    # imported since that name is module-private there.
    return hashlib.sha256(token.encode()).hexdigest()


async def seed() -> list[str]:
    engine = create_async_engine(Settings().database_url)
    try:
        async with engine.begin() as conn:
            for name, lanes, token in _DEV_RUNNERS:
                stmt = pg_insert(RunnerModel).values(
                    id=uuid.uuid4(),
                    name=name,
                    lanes=lanes,
                    token_hash=_hash_token(token),
                    version=1,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["name"],
                    set_={
                        "token_hash": stmt.excluded.token_hash,
                        "lanes": stmt.excluded.lanes,
                        "revoked_at": None,
                    },
                )
                await conn.execute(stmt)
    finally:
        await engine.dispose()
    return [name for name, _, _ in _DEV_RUNNERS]


async def main() -> None:
    for name in await seed():
        print(f"ensured runner {name!r}")


if __name__ == "__main__":
    asyncio.run(main())
