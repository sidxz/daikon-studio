"""Local-dev bootstrap, NOT for production: upserts the two fixed-token
runners `make dev`'s stack expects, so a developer can start
`python -m daikonstudio.infrastructure.runner` without first registering a
runner through the UI. A production runner is always created through
`POST /api/v1/runners` (`CreateRunner`), which mints a random token surfaced
exactly once and never written to a file.

The two tokens below (`drt_dev_default`, `drt_dev_gpu`) are PUBLIC -- they are
committed to this repo in plaintext, so anyone who can read the source knows
them. Refuses to run at all unless `STUDIO_INLINE_JOBS=1` or `STUDIO_DEV_SEED=1`
is set (Important 4, final review): a shared/staging Postgres this points at
must never get these rows, and this refusal is the only thing stopping that
short of every operator remembering not to run it.

Run as `python -m daikonstudio.infrastructure.runner.seed`.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
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


async def seed(settings: Settings | None = None) -> list[str]:
    resolved = settings or Settings()
    if not (resolved.inline_jobs or resolved.dev_seed):
        raise RuntimeError(
            "refusing to seed fixed-token dev runners: neither STUDIO_INLINE_JOBS=1 nor "
            "STUDIO_DEV_SEED=1 is set. These tokens (drt_dev_default, drt_dev_gpu) are "
            "public -- committed to this repo -- so this must never run against a shared "
            "or staging database. Set STUDIO_DEV_SEED=1 to seed a local dev database."
        )
    engine = create_async_engine(resolved.database_url)
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
                # No `revoked_at: None` here -- an operator who deliberately
                # revoked one of these dev runners must stay revoked across a
                # re-seed, not get silently un-revoked (Important 4, final
                # review). token_hash/lanes still refresh either way; a
                # revoked runner's `is_revoked` check still blocks auth
                # regardless of what its token_hash is.
                stmt = stmt.on_conflict_do_update(
                    index_elements=["name"],
                    set_={"token_hash": stmt.excluded.token_hash, "lanes": stmt.excluded.lanes},
                )
                await conn.execute(stmt)
    finally:
        await engine.dispose()
    return [name for name, _, _ in _DEV_RUNNERS]


async def main() -> None:
    try:
        names = await seed()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
    for name in names:
        print(f"ensured runner {name!r}")


if __name__ == "__main__":
    asyncio.run(main())
