"""Runner-token auth dependencies -- the machine-facing counterpart to
Sentinel's human-facing `AuthDep`.

`get_runner` is the entire authentication boundary for `/api/v1/runner/*`:
Sentinel does not run on that prefix at all (see `app.py`'s `exclude_paths`),
so a request that clears this dependency has proven nothing except that it
carries a live, non-revoked runner token -- no workspace, no user, no role.

`claimed_run` layers authorization on top of that: proving the token's own
runner is also the one holding the lease on *this* run. It is used on every
run-scoped route, including reads, because `RunQueue.verify_claim` extends
the lease as a side effect (see its own docstring) -- so every authenticated
call against a claimed run doubles as that runner's heartbeat, with no
separate heartbeat endpoint needed.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable, Coroutine
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request
from lagom import Container

from daikonstudio.application.ports.run_queue import RunQueue
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.application.ports.runner_repository import RunnerRepository
from daikonstudio.domain.execution.run import Run
from daikonstudio.domain.runners.runner import Runner
from daikonstudio.domain.shared.errors import AuthorizationError, NotFoundError
from daikonstudio.interface.dependencies._container import get_container
from daikonstudio.settings import Settings

__all__ = ["ClaimedRunRead", "ClaimedRunWrite", "RunnerDep", "claimed_run", "get_runner"]


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Invalid, missing, or revoked runner token",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_runner(
    request: Request, container: Annotated[Container, Depends(get_container)]
) -> Runner:
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise _unauthorized()

    runners = container[RunnerRepository]  # type: ignore[type-abstract]
    runner = await runners.get_by_token_hash(hashlib.sha256(token.encode()).hexdigest())
    if runner is None or runner.is_revoked:
        raise _unauthorized()

    await runners.touch_last_seen(runner.id)  # the liveness signal
    return runner


RunnerDep = Annotated[Runner, Depends(get_runner)]


def claimed_run(*, require_active: bool) -> Callable[..., Coroutine[Any, Any, Run]]:
    async def _dependency(
        run_id: uuid.UUID,
        runner: RunnerDep,
        container: Annotated[Container, Depends(get_container)],
    ) -> Run:
        # Unknown-id-is-404 must be checked before the claim check, or a run
        # nobody has ever claimed would 403 -- and 403 would confirm it exists.
        run = await container[RunRepository].get_by_id(run_id)  # type: ignore[type-abstract]
        if run is None:
            raise NotFoundError("Run", str(run_id))

        settings = container[Settings]
        held = await container[RunQueue].verify_claim(  # type: ignore[type-abstract]
            run_id,
            runner_id=runner.id,
            lease_seconds=settings.runner_lease_seconds,
            require_active=require_active,
        )
        if not held:
            raise AuthorizationError(f"Run '{run_id}' is not claimed by this runner")
        return run

    return _dependency


ClaimedRunRead = Annotated[Run, Depends(claimed_run(require_active=False))]
ClaimedRunWrite = Annotated[Run, Depends(claimed_run(require_active=True))]
