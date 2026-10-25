"""A language model that answers from a script.

Hand-written rather than mocked, like every other fake here: a test says what the
model returns and then asserts on what the planner did with it, which is the only
part worth testing. The real adapters are thin enough that their error mapping is
better covered against `respx` than against a mock.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from daikonstudio.application.ports.llm import Answer, LanguageModel, ModelChoice, Tool
from daikonstudio.domain.shared.errors import ServiceUnavailableError


class ScriptedLanguageModel:
    """Returns its replies in order, then the last one for every call after, recording
    the prompt, the schema and the tools of each call.

    A reply is the answer alone, or an `Answer` carrying the tool calls made before it.
    """

    def __init__(
        self,
        reply: BaseModel | Answer[BaseModel],
        *more: BaseModel | Answer[BaseModel],
        provider: str = "anthropic",
    ) -> None:
        self._replies = (reply, *more)
        self._provider = provider
        self.calls: list[tuple[str, str]] = []
        self.stables: list[str] = []
        self.schemas: list[type[BaseModel]] = []
        self.tools: list[tuple[str, ...]] = []

    @property
    def choice(self) -> ModelChoice:
        return ModelChoice(provider=self._provider, model="scripted")

    async def parse[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T], stable: str = ""
    ) -> T:
        return (
            await self.act(system=system, prompt=prompt, schema=schema, tools=(), stable=stable)
        ).parsed

    async def act[T: BaseModel](
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        tools: Sequence[Tool],
        stable: str = "",
    ) -> Answer[T]:
        reply = self._replies[min(len(self.calls), len(self._replies) - 1)]
        self.calls.append((system, prompt))
        self.stables.append(stable)
        self.schemas.append(schema)
        self.tools.append(tuple(tool.name for tool in tools))
        parsed, calls = (reply.parsed, reply.calls) if isinstance(reply, Answer) else (reply, ())
        offered = {tool.schema for tool in tools}
        # The real provider cannot return a call to a tool it was not given; a script that
        # does is a test's mistake, not a turn's.
        assert all(type(call) in offered for call in calls), (
            "a call to a tool this turn was not offered"
        )
        return Answer(schema.model_validate(parsed.model_dump()), tuple(calls))


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
