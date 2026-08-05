"""Integration coverage for `infrastructure/runner/seed.py`'s dev-only gating
and no-resurrect behaviour (Important 4, final review): the fixed
`drt_dev_default`/`drt_dev_gpu` tokens are public (committed to this repo),
so seeding them must refuse without an explicit opt-in, and must never
un-revoke a runner an operator deliberately revoked.

Runs straight against `_migrated_engine` (see `tests/conftest.py`) rather
than the shared `app` fixture -- `seed()` opens its own engine from a
`Settings` it is handed, the same shape `python -m ...runner.seed` runs in.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from daikonstudio.infrastructure.persistence.sqlalchemy.runners.models import RunnerModel
from daikonstudio.infrastructure.persistence.sqlalchemy.runners.repository import (
    SqlAlchemyRunnerRepository,
)
from daikonstudio.infrastructure.runner import seed as seed_module
from daikonstudio.settings import Settings

_DEV_RUNNER_NAMES = [name for name, _, _ in seed_module._DEV_RUNNERS]


@pytest_asyncio.fixture(autouse=True)
async def _cleanup_dev_runners(_migrated_engine: AsyncEngine) -> AsyncIterator[None]:
    yield
    async with _migrated_engine.begin() as conn:
        await conn.execute(delete(RunnerModel).where(RunnerModel.name.in_(_DEV_RUNNER_NAMES)))


def _database_url(engine: AsyncEngine) -> str:
    # `str(engine.url)` masks the password ("***") -- fine for the other
    # fixtures that only carry this Settings for show (their app.state
    # container is repointed at the real engine separately), but `seed()`
    # actually dials this URL, so the real password has to survive.
    return engine.url.render_as_string(hide_password=False)


async def test_seed_refuses_without_an_explicit_opt_in(_migrated_engine: AsyncEngine) -> None:
    settings = Settings(
        database_url=_database_url(_migrated_engine), inline_jobs=False, dev_seed=False
    )
    with pytest.raises(RuntimeError, match="STUDIO_DEV_SEED"):
        await seed_module.seed(settings)


async def test_seed_runs_when_inline_jobs_is_set(_migrated_engine: AsyncEngine) -> None:
    """`make dev`'s STUDIO_INLINE_JOBS=1 local flow must keep working unchanged."""
    settings = Settings(
        database_url=_database_url(_migrated_engine), inline_jobs=True, dev_seed=False
    )
    names = await seed_module.seed(settings)
    assert set(names) == set(_DEV_RUNNER_NAMES)


async def test_seed_does_not_resurrect_a_revoked_dev_runner(_migrated_engine: AsyncEngine) -> None:
    settings = Settings(database_url=_database_url(_migrated_engine), dev_seed=True)
    await seed_module.seed(settings)

    sessions = async_sessionmaker(bind=_migrated_engine, expire_on_commit=False)
    repository = SqlAlchemyRunnerRepository(sessions)
    seeded = await repository.get_by_token_hash(seed_module._hash_token("drt_dev_default"))
    assert seeded is not None
    await repository.revoke(seeded.id)

    await seed_module.seed(settings)  # a plain re-seed, e.g. from `make up` again

    reloaded = await repository.get(seeded.id)
    assert reloaded is not None
    assert reloaded.is_revoked, "re-seeding must not un-revoke a deliberately revoked dev runner"
