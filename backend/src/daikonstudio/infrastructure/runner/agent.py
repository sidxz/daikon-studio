"""The self-hosted runner agent process: `python -m daikonstudio.infrastructure.runner`.

Polls `POST /api/v1/runner/claim` in a loop. A 204 means nothing to claim --
sleep `poll_seconds` and try again. A 200 hands back a claimed `Run`; the
agent builds an HTTP-backed ctx for it (`build_http_ctx`, the runner-side
twin of `infrastructure.jobs.build_sqlalchemy_ctx`) and executes it through
the exact same `jobs.run_job` an in-process worker would call. One claim at a
time, one process -- a self-hosted runner is one machine (or one GPU) working
one job at a time, so there is no concurrency knob to set here.

`run_job` re-raises a handler failure after persisting FAILED on the row
(see its own docstring); `poll_once` catches that as a plain `Exception` and
logs it, so one bad job does not end the agent -- the row is already FAILED,
and the loop moves on to the next claim. The claim call itself gets the same
treatment against `httpx.HTTPError` (a non-2xx response, or the studio being
unreachable at all): a studio mid-deploy or mid-restart must not take every
runner attached to it down too, since nothing in this repo supervises or
restarts this process -- a dead agent stays dead. `SystemExit`,
`KeyboardInterrupt`, and `asyncio.CancelledError` are deliberately not
caught anywhere here: those must still propagate out of the process for the
same shutdown reasons `run_job`'s own docstring gives.

Heartbeat (Important 1+2, final review): the lease `claim_next` grants is
extended only by run-scoped HTTP calls (`claimed_run`'s own docstring), and
during a fit those come only from `TrainContext.report` -- which only
chemprop's engine calls at all; ecfp4-xgboost/ecfp4-randomforest never do.
A healthy fit longer than `lease_seconds` would otherwise have its claim
expire mid-job, get requeued, and restart from zero on another runner, up to
`runner_max_attempts` times, for a job that was never broken. `poll_once`
now runs a background heartbeat for the duration of the job -- a plain
`GET /runs/{id}` on a timer derived from `lease_seconds`, independent of
whatever the engine itself does -- cancelled in `finally` so it cannot
outlive the job.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid

import httpx
import structlog
from pydantic_settings import BaseSettings, SettingsConfigDict

from daikonstudio.infrastructure import jobs
from daikonstudio.infrastructure.engines.registry import default_registry
from daikonstudio.infrastructure.runner.ports import build_http_ctx
from daikonstudio.infrastructure.runner.wire import ClaimResponse
from daikonstudio.logging import configure_logging

_logger = structlog.get_logger(__name__)

# The lease is a soft budget, not a hard deadline -- three heartbeats per
# lease window leaves margin for one missed tick (a slow request, a jittery
# network) without losing the claim, while still renewing well before it
# would otherwise expire.
_HEARTBEATS_PER_LEASE = 3

# The hard kill. The claim's deadline is cooperative (TrainContext.report), and
# the heartbeat above renews the lease whatever the fit is doing, so an engine
# that never returns would hold its run RUNNING and its runner busy forever.
# Past the deadline plus this grace, the agent records the failure and exits the
# process -- the only way to stop a thread -- so the container restarts clean.
_HARD_KILL_GRACE_SECONDS = 300
_exit = os._exit  # indirection so a test can observe the exit instead of dying


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STUDIO_", env_file=".env", extra="ignore")

    url: str  # STUDIO_URL, e.g. https://studio.example.org
    runner_token: str  # STUDIO_RUNNER_TOKEN, the drt_... secret this runner claims with
    poll_seconds: float = 3.0
    # Same two knobs as the API's Settings, same meaning (see daikonstudio.logging).
    log_level: str = "INFO"
    log_format: str = "console"


async def _heartbeat(api: httpx.AsyncClient, run_id: uuid.UUID, interval: float) -> None:
    """Runs until cancelled: sleeps `interval`, then a run-scoped read, forever.
    `claimed_run` (`interface/dependencies/runner_auth.py`) extends the lease
    as a side effect of ANY authenticated call against a run it holds, so a
    plain `GET` here is enough -- no dedicated heartbeat endpoint needed.

    A failed tick is logged and swallowed, not raised: this task's only job
    is to keep the lease alive for `run_job`, which is running concurrently
    in the same process and must not be disturbed by a transient heartbeat
    failure. If every tick fails for a whole `lease_seconds`, the claim
    expires and the run gets requeued the normal way -- exactly the
    pre-heartbeat behaviour, not a new failure mode.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            response = await api.get(f"/api/v1/runner/runs/{run_id}")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _logger.warning("heartbeat failed", run_id=str(run_id), error=str(exc))


async def _abandon_hung_job(ctx: dict[str, object], run_id: uuid.UUID, limit: float) -> None:
    """Fail the run, then exit. Cancelling the awaiting task did not stop the
    fit's thread; nothing can. Writing FAILED first means the user sees why
    instead of a lease-expiry requeue re-running the same hung fit three times."""
    _logger.error(
        "job exceeded its deadline plus the grace period; failing it and exiting",
        run_id=str(run_id),
        limit_seconds=limit,
    )
    try:
        async with asyncio.timeout(10):
            await jobs.fail_run(
                ctx, run_id, f"the runner gave up after {limit:.0f}s: the fit never returned"
            )
    except Exception:
        _logger.exception("could not record the failure before exiting", run_id=str(run_id))
    _exit(3)


async def poll_once(api: httpx.AsyncClient, settings: AgentSettings) -> bool:
    """One claim/execute cycle. `api` is the long-lived client `main()` holds
    open across iterations, already carrying the runner's bearer token.

    Returns True iff a claim came back and its job ran (whether it succeeded
    or failed) -- `main()` uses that to skip the poll-interval sleep while
    work keeps arriving.
    """
    try:
        response = await api.post("/api/v1/runner/claim")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        # HTTPError covers both a non-2xx claim response (HTTPStatusError)
        # and the studio being unreachable at all (ConnectError, ReadTimeout,
        # ...). Either way this is a transient poll failure, not a reason to
        # die: return False so main() falls through to its normal
        # sleep(poll_seconds) and retries next cycle, same as an empty claim.
        status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
        _logger.warning("claim failed", error=str(exc), status_code=status)
        return False
    if response.status_code == 204:
        return False

    claimed = ClaimResponse.model_validate(response.json())
    run_id = claimed.run.id
    ctx = build_http_ctx(
        settings.url, settings.runner_token, run_id, deadline_seconds=claimed.deadline_seconds
    )
    heartbeat = asyncio.create_task(
        _heartbeat(api, run_id, claimed.lease_seconds / _HEARTBEATS_PER_LEASE)
    )
    hard_limit = claimed.deadline_seconds + _HARD_KILL_GRACE_SECONDS
    try:
        await asyncio.wait_for(jobs.run_job(ctx, run_id), timeout=hard_limit)
    except TimeoutError:
        await _abandon_hung_job(ctx, run_id, hard_limit)
    except Exception:
        # run_job already persisted FAILED on the row before re-raising --
        # this is purely so the operator sees it, not a retry path.
        _logger.exception("runner job failed", run_id=str(run_id))
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat
        await ctx["_client"].aclose()
    return True


async def main() -> None:
    # url/runner_token are required but come from the environment at runtime
    # (pydantic-settings), not from this call site -- mypy has no way to know
    # that, the same known gap pydantic-settings' own docs call out.
    settings = AgentSettings()  # type: ignore[call-arg]
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    if not settings.runner_token.strip():
        # An empty token never reaches the API (httpx refuses a bare `Bearer `
        # header), so without this the agent polls and logs "claim failed" every
        # 3 s forever. The compose stack starts with the token unset on purpose
        # (it is minted in the UI after first boot); say so and stop.
        _logger.error("STUDIO_RUNNER_TOKEN is empty; mint one in the Runners page and set it")
        raise SystemExit(2)
    async with httpx.AsyncClient(
        base_url=settings.url,
        headers={"Authorization": f"Bearer {settings.runner_token}"},
        timeout=60.0,
    ) as api:
        # The engine ids are logged because this process does not hot-reload: it holds
        # whatever `_ENGINES` contained when it imported, and adding an engine without
        # restarting it fails the run with a bare `UnknownEngineError(<id>)` after the
        # API -- which does reload -- has already listed and accepted that engine. One
        # line of log turns that into a diagnosis you can read instead of infer.
        _logger.info(
            "runner agent started",
            url=settings.url,
            engines=sorted(m.id for m in default_registry().manifests()),
        )
        while True:
            executed = await poll_once(api, settings)
            if not executed:
                await asyncio.sleep(settings.poll_seconds)
