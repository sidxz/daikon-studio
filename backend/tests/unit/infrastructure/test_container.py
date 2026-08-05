"""Regression coverage for a lagom composition-root bug (Task 16 review,
Important 2): `JobEnqueuer` must not cache the sessionmaker of whichever
container resolves it first.

Both `JobEnqueuer` branches depend on `c[async_sessionmaker]` -- `InlineEnqueuer`
directly, `DbEnqueuer` through the `RunQueue` it wraps -- and a test harness
overrides that binding per child container (`tests/api/conftest.py` builds a
fresh child container per test). Wrapping `JobEnqueuer` (or `RunQueue`) in a
`Singleton`, as an earlier version of `container.py` did for the arq-backed
enqueuer, caches whichever sessionmaker resolved it *first* and hands that same
instance to every later child container too -- silently writing through another
test's (rolled-back) session. Both bindings must instead be plain factories,
re-run per resolution against whichever container is actually asking.

No database or network needed: `async_sessionmaker` is overridden with a
plain sentinel object per child, and nothing here ever calls it.
"""

from __future__ import annotations

from lagom import Container, Singleton
from sqlalchemy.ext.asyncio import async_sessionmaker

from daikonstudio.application.execution.enqueue import JobEnqueuer
from daikonstudio.infrastructure.di.container import create_container
from daikonstudio.infrastructure.jobs import DbEnqueuer, InlineEnqueuer
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
    assert enqueuer_a._ctx["runs"]._sessions is sessions_a  # type: ignore[attr-defined]
    assert enqueuer_b._ctx["runs"]._sessions is sessions_b  # type: ignore[attr-defined]


def test_the_db_enqueuer_branch_also_binds_to_its_own_containers_sessionmaker() -> None:
    """Same bug, other branch: `DbEnqueuer` wraps a `RunQueue` built from
    `c[async_sessionmaker]`, so it must track a child container's override too,
    not the parent's -- there is no Redis pool left to justify caching either
    branch as a `Singleton` (see `container.py`)."""
    parent = create_container(Settings(inline_jobs=False))
    sessions_a, sessions_b = object(), object()

    enqueuer_a = _child_with_sessionmaker(parent, sessions_a)[JobEnqueuer]
    enqueuer_b = _child_with_sessionmaker(parent, sessions_b)[JobEnqueuer]

    assert isinstance(enqueuer_a, DbEnqueuer)
    assert isinstance(enqueuer_b, DbEnqueuer)
    assert enqueuer_a._queue._sessions is sessions_a  # type: ignore[attr-defined]
    assert enqueuer_b._queue._sessions is sessions_b  # type: ignore[attr-defined]
