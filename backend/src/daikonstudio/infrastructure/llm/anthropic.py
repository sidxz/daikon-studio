"""Anthropic adapter for the `LanguageModel` port.

Follows `infrastructure/chemcellar/client.py`: an empty key means the integration
is off, and the refusal names the environment variable rather than saying
"unauthorized", because nothing the caller did was wrong. Validated at the point
of use, not at boot -- a deployment without a model key still trains models.

A turn with tools is two requests when the model uses them: the first offers the
tools and the answer's schema together, and a model that calls tools stops there;
each call is acknowledged as recorded and the second request, with tool choice off,
gets the answer. Nothing is carried out between the two: the caller checks every
call against the study after the turn, which is why the tools are not strict and the
provider's grammar is spent on the answer alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import anthropic
import structlog
from anthropic import transform_schema
from anthropic.types.beta import BetaMessage, BetaTextBlockParam
from pydantic import BaseModel, ValidationError

from daikonstudio.application.ports.llm import Answer, ModelChoice, Tool
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

# What each tool call is answered with before the model answers. It is not carried out
# yet, and the numbers the model used still mean the plan it was shown.
_RECORDED = (
    "Recorded. It is checked and carried out, or offered to the scientist, after you "
    "answer; step numbers in your answer still mean the plan as you were shown it."
)


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

    async def parse[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T], stable: str = ""
    ) -> T:
        answer = await self.act(
            system=system, prompt=prompt, schema=schema, tools=(), stable=stable
        )
        return answer.parsed

    async def act[T: BaseModel](
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        tools: Sequence[Tool],
        stable: str = "",
    ) -> Answer[T]:
        # Cache breakpoints: the system prompt, the end of the stable prefix, and, when a
        # second request may follow, the prompt itself, so that request reads it back.
        # The prefix is cached together with everything before it, so a later call that
        # changes only `prompt` reads both from the cache.
        blocks: list[BetaTextBlockParam] = [{"type": "text", "text": prompt}]
        if tools:
            blocks[0]["cache_control"] = {"type": "ephemeral"}
        if stable:
            blocks.insert(
                0, {"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}}
            )
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": _MAX_TOKENS,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "output_config": {
                "format": {"type": "json_schema", "schema": transform_schema(schema)}
            },
            "betas": [_FALLBACK_BETA],
            "fallbacks": "default",
        }
        if tools:
            request["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.schema.model_json_schema(),
                }
                for tool in tools
            ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": blocks}]
        reply = await self._send(messages=messages, **request)

        calls: list[BaseModel] = []
        malformed = 0
        used = [block for block in reply.content if block.type == "tool_use"]
        if used:
            by_name = {tool.name: tool.schema for tool in tools}
            for block in used:
                try:
                    calls.append(by_name[block.name].model_validate(block.input))
                except (KeyError, ValidationError):
                    malformed += 1
            messages += [
                {"role": "assistant", "content": reply.content},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": block.id, "content": _RECORDED}
                        for block in used
                    ],
                },
            ]
            reply = await self._send(messages=messages, tool_choice={"type": "none"}, **request)

        text = "".join(block.text for block in reply.content if block.type == "text")
        if not text:
            # The model answered but not in the shape asked for. Reported as the
            # provider failing rather than as a bad request, because the caller
            # supplied a goal and has no way to act on this.
            raise ServiceUnavailableError(
                "The Anthropic model did not answer in the expected form.",
                detail=f"Stopped because: {reply.stop_reason}.",
            )
        try:
            parsed = schema.model_validate_json(text)
        except ValidationError as exc:
            raise ServiceUnavailableError(
                "The Anthropic model did not answer in the expected form."
            ) from exc
        return Answer(parsed, tuple(calls), malformed)

    async def _send(self, **request: Any) -> BetaMessage:
        """One request, with the provider's failures mapped to this deployment's."""
        client = self._require()
        try:
            reply: BetaMessage = await client.beta.messages.create(**request)
        except anthropic.APIStatusError as exc:
            # Every status failure is this deployment's problem, not the caller's:
            # a 401 means our key is wrong, a 429 that we are over our own limit.
            # Mapping any of them to AuthorizationError would tell a scientist
            # they lack permission for something they are perfectly entitled to.
            # The provider's own words go with it: a status alone names no remedy.
            error = exc.body.get("error") if isinstance(exc.body, dict) else None
            said = error.get("message") if isinstance(error, dict) else None
            said = said or exc.message
            _logger.warning("anthropic refused the request", status=exc.status_code, said=said)
            raise ServiceUnavailableError(
                "The Anthropic model could not be reached.",
                detail=f"The provider answered {exc.status_code}: {said}",
            ) from exc
        except anthropic.APIError as exc:
            _logger.warning("anthropic call failed", error=str(exc)[:200])
            raise ServiceUnavailableError("The Anthropic model could not be reached.") from exc

        _logger.info(
            "anthropic answered",
            input=reply.usage.input_tokens,
            cache_read=reply.usage.cache_read_input_tokens,
            cache_write=reply.usage.cache_creation_input_tokens,
            output=reply.usage.output_tokens,
            calls=sum(block.type == "tool_use" for block in reply.content),
        )

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
        return reply
