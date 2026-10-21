"""Anthropic adapter for the `LanguageModel` port.

Follows `infrastructure/chemcellar/client.py`: an empty key means the integration
is off, and the refusal names the environment variable rather than saying
"unauthorized", because nothing the caller did was wrong. Validated at the point
of use, not at boot -- a deployment without a model key still trains models.
"""

from __future__ import annotations

import anthropic
import structlog
from pydantic import BaseModel, ValidationError

from daikonstudio.application.ports.llm import ModelChoice
from daikonstudio.domain.shared.errors import ServiceUnavailableError

_logger = structlog.get_logger(__name__)

# A plan is a few thousand tokens at most. Generous enough that a long one is not
# truncated into invalid JSON, small enough to bound a runaway.
_MAX_TOKENS = 8000

# Planning is one call and the caller is a person waiting on a page. Longer than
# the ChemCellar client's 60 s because a reasoning model on a hard goal genuinely
# takes longer than a database lookup, and a truncated plan is worse than a slow one.
_TIMEOUT_SECONDS = 120.0

# Server-side fallback, on by default.
#
# The classifiers decline some requests, and this one is aimed squarely at the
# categories they watch: a goal like "which changes buy potency without pulling
# hERG in" on an antibacterial series reads a great deal like the thing they
# exist to catch, and it is also exactly the work a medicinal chemist does all
# day. A refusal on a legitimate series is a product defect, not a safety win.
#
# "default" routes by refusal category, so there is no model list here to go
# stale. The fallback is a different model, which is why the plan records what
# actually wrote it rather than what was asked for.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLanguageModel:
    def __init__(self, *, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client: anthropic.AsyncAnthropic | None = None

    @property
    def choice(self) -> ModelChoice:
        return ModelChoice(provider="anthropic", model=self._model)

    def _require(self) -> anthropic.AsyncAnthropic:
        if not self._api_key:
            raise ServiceUnavailableError(
                "No Anthropic model is configured on this server.",
                detail="Set STUDIO_ANTHROPIC_API_KEY to an Anthropic API key.",
            )
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=self._api_key, timeout=_TIMEOUT_SECONDS
            )
        return self._client

    async def parse[T: BaseModel](self, *, system: str, prompt: str, schema: type[T]) -> T:
        client = self._require()
        try:
            reply = await client.beta.messages.parse(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.APIStatusError as exc:
            # Every status failure is this deployment's problem, not the caller's:
            # a 401 means our key is wrong, a 429 that we are over our own limit.
            # Mapping any of them to AuthorizationError would tell a scientist
            # they lack permission for something they are perfectly entitled to.
            _logger.warning("anthropic refused the request", status=exc.status_code)
            raise ServiceUnavailableError(
                "The Anthropic model could not be reached.",
                detail=f"The provider answered {exc.status_code}.",
            ) from exc
        except anthropic.APIError as exc:
            _logger.warning("anthropic call failed", error=str(exc)[:200])
            raise ServiceUnavailableError("The Anthropic model could not be reached.") from exc

        if reply.stop_reason == "refusal":
            # Distinct from a malformed answer, and worth saying so: the remedy
            # is to rephrase or ask the other provider, not to retry. Reaching
            # here means even the fallback declined.
            category = getattr(reply.stop_details, "category", None)
            _logger.warning("anthropic declined the request", category=category)
            raise ServiceUnavailableError(
                "The Anthropic model declined to answer this one.",
                detail=(
                    "Safety classifiers sometimes decline ordinary medicinal chemistry. "
                    "Try rewording the goal, or ask a different provider."
                ),
            )

        parsed = reply.parsed_output
        if parsed is None:
            # The model answered but not in the shape asked for. Reported as the
            # provider failing rather than as a bad request, because the caller
            # supplied a goal and has no way to act on this.
            raise ServiceUnavailableError(
                "The Anthropic model did not answer in the expected form.",
                detail=f"Stopped because: {reply.stop_reason}.",
            )
        try:
            return schema.model_validate(parsed.model_dump())
        except ValidationError as exc:
            raise ServiceUnavailableError(
                "The Anthropic model did not answer in the expected form."
            ) from exc
