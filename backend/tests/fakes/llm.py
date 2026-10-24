"""A language model that answers from a script.

Hand-written rather than mocked, like every other fake here: a test says what the
model returns and then asserts on what the planner did with it, which is the only
part worth testing. The real adapters are thin enough that their error mapping is
better covered against `respx` than against a mock.
"""

from __future__ import annotations

from pydantic import BaseModel

from daikonstudio.application.ports.llm import LanguageModel, ModelChoice
from daikonstudio.domain.shared.errors import ServiceUnavailableError


class ScriptedLanguageModel:
    """Returns its replies in order, then the last one for every call after, recording
    the prompt and the schema of each call."""

    def __init__(self, reply: BaseModel, *more: BaseModel, provider: str = "anthropic") -> None:
        self._replies = (reply, *more)
        self._provider = provider
        self.calls: list[tuple[str, str]] = []
        self.stables: list[str] = []
        self.schemas: list[type[BaseModel]] = []

    @property
    def choice(self) -> ModelChoice:
        return ModelChoice(provider=self._provider, model="scripted")

    async def parse[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T], stable: str = ""
    ) -> T:
        reply = self._replies[min(len(self.calls), len(self._replies) - 1)]
        self.calls.append((system, prompt))
        self.stables.append(stable)
        self.schemas.append(schema)
        return schema.model_validate(reply.model_dump())


class ScriptedLanguageModels:
    def __init__(self, model: LanguageModel | None, *, default: str = "anthropic") -> None:
        self._model = model
        self._default = default

    def available(self) -> tuple[ModelChoice, ...]:
        return (self._model.choice,) if self._model is not None else ()

    def get(self, provider: str | None) -> LanguageModel:
        if self._model is None:
            raise ServiceUnavailableError(
                "No anthropic model is configured on this server.",
                detail="Set STUDIO_ANTHROPIC_API_KEY to an API key.",
            )
        return self._model
