"""Regression coverage for a lagom composition-root bug (Task 16 review,
Important 2): `JobEnqueuer` must not cache the sessionmaker of whichever
container resolves it first.

`ArqEnqueuer` is the only right thing to cache as a `Singleton` -- its Redis
pool depends only on `settings.redis_url`, fixed at container-build time, so
it is genuinely process-wide. `InlineEnqueuer` depends on `c[async_sessionmaker]`,
which a test harness overrides per child container (`tests/api/conftest.py`
builds a fresh child container per test); wrapping the whole `JobEnqueuer`
binding in a `Singleton`, as an earlier version of `container.py` did, caches
whichever sessionmaker resolved it *first* and hands that same `InlineEnqueuer`
to every later child container too -- silently writing through another test's
(rolled-back) session. `JobEnqueuer` must instead be a plain factory, re-run
per resolution against whichever container is actually asking.

No database or network needed: `async_sessionmaker` is overridden with a
plain sentinel object per child, and nothing here ever calls it.
"""

from __future__ import annotations

from lagom import Container, Singleton
from sqlalchemy.ext.asyncio import async_sessionmaker

from daikonstudio.application.execution.enqueue import JobEnqueuer
from daikonstudio.infrastructure.di.container import create_container
from daikonstudio.infrastructure.worker import InlineEnqueuer
from daikonstudio.settings import Settings


def _child_with_sessionmaker(parent: Container, sessions: object) -> Container:
    child = Container(parent)
    child.define(async_sessionmaker, Singleton(lambda: sessions))
    return child


def test_two_child_containers_get_job_enqueuers_bound_to_their_own_sessionmaker() -> None:
    parent = create_container(Settings(inline_jobs=True))
    sessions_a, sessions_b = object(), object()

    # Resolution order matters for reproducing the bug: the old code cached
    # whichever sessionmaker resolved `JobEnqueuer` first, so `enqueuer_b` is
    # exactly where a stale/wrong sessionmaker would surface.
    enqueuer_a = _child_with_sessionmaker(parent, sessions_a)[JobEnqueuer]
    enqueuer_b = _child_with_sessionmaker(parent, sessions_b)[JobEnqueuer]

    assert isinstance(enqueuer_a, InlineEnqueuer)
    assert isinstance(enqueuer_b, InlineEnqueuer)
    assert enqueuer_a._ctx["sessions"] is sessions_a
    assert enqueuer_b._ctx["sessions"] is sessions_b


def test_the_redis_pool_backed_enqueuer_is_still_cached_process_wide() -> None:
    """The fix must not throw the caching out entirely -- `ArqEnqueuer`'s own
    binding is a `Singleton` (see `container.py`), and two containers built
    from the same parent (neither in inline mode) must share one instance."""
    parent = create_container(Settings(inline_jobs=False))
    child_a = Container(parent)
    child_b = Container(parent)

    assert child_a[JobEnqueuer] is child_b[JobEnqueuer]
