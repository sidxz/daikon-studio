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
"""

from __future__ import annotations

import asyncio

import httpx
import structlog
from pydantic_settings import BaseSettings, SettingsConfigDict

from daikonstudio.infrastructure import jobs
from daikonstudio.infrastructure.runner.ports import build_http_ctx
from daikonstudio.infrastructure.runner.wire import ClaimResponse

_logger = structlog.get_logger(__name__)


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STUDIO_", env_file=".env", extra="ignore")

    url: str  # STUDIO_URL, e.g. https://studio.example.org
    runner_token: str  # STUDIO_RUNNER_TOKEN, the drt_... secret this runner claims with
    poll_seconds: float = 3.0


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
    try:
        await jobs.run_job(ctx, run_id)
    except Exception:
        # run_job already persisted FAILED on the row before re-raising --
        # this is purely so the operator sees it, not a retry path.
        _logger.exception("runner job failed", run_id=str(run_id))
    finally:
        await ctx["_client"].aclose()
    return True


async def main() -> None:
    # url/runner_token are required but come from the environment at runtime
    # (pydantic-settings), not from this call site -- mypy has no way to know
    # that, the same known gap pydantic-settings' own docs call out.
    settings = AgentSettings()  # type: ignore[call-arg]
    async with httpx.AsyncClient(
        base_url=settings.url,
        headers={"Authorization": f"Bearer {settings.runner_token}"},
        timeout=60.0,
    ) as api:
        _logger.info("runner agent started", url=settings.url)
        while True:
            executed = await poll_once(api, settings)
            if not executed:
                await asyncio.sleep(settings.poll_seconds)
