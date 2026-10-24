"""OpenAI adapter for the `LanguageModel` port.

Same contract and the same failure manners as the Anthropic adapter; only the
call shape differs, which is the whole reason for two thin adapters rather than
one compatibility shim.
"""

from __future__ import annotations

import openai
import structlog
from pydantic import BaseModel, ValidationError

from daikonstudio.application.ports.llm import ModelChoice
from daikonstudio.domain.shared.errors import ServiceUnavailableError

_logger = structlog.get_logger(__name__)

_MAX_OUTPUT_TOKENS = 8000
_TIMEOUT_SECONDS = 120.0


class OpenAILanguageModel:
    def __init__(self, *, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client: openai.AsyncOpenAI | None = None

    @property
    def choice(self) -> ModelChoice:
        return ModelChoice(provider="openai", model=self._model)

    def _require(self) -> openai.AsyncOpenAI:
        if not self._api_key:
            raise ServiceUnavailableError(
                "No OpenAI model is configured on this server.",
                detail="Set STUDIO_OPENAI_API_KEY to an OpenAI API key.",
            )
        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=self._api_key, timeout=_TIMEOUT_SECONDS)
        return self._client

    async def parse[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T], stable: str = ""
    ) -> T:
        client = self._require()
        text = f"{stable}\n\n{prompt}" if stable else prompt
        try:
            reply = await client.responses.parse(
                model=self._model,
                instructions=system,
                input=text,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
                text_format=schema,
            )
        except openai.APIStatusError as exc:
            # The SDK hands over the body's `error` object, which carries the provider's words.
            said = exc.body.get("message") if isinstance(exc.body, dict) else None
            said = said or exc.message
            _logger.warning("openai refused the request", status=exc.status_code, said=said)
            raise ServiceUnavailableError(
                "The OpenAI model could not be reached.",
                detail=f"The provider answered {exc.status_code}: {said}",
            ) from exc
        except openai.OpenAIError as exc:
            _logger.warning("openai call failed", error=str(exc)[:200])
            raise ServiceUnavailableError("The OpenAI model could not be reached.") from exc

        parsed = reply.output_parsed
        if parsed is None:
            raise ServiceUnavailableError(
                "The OpenAI model did not answer in the expected form.",
                detail=f"Stopped because: {reply.status}.",
            )
        try:
            return schema.model_validate(parsed.model_dump())
        except ValidationError as exc:
            raise ServiceUnavailableError(
                "The OpenAI model did not answer in the expected form."
            ) from exc
