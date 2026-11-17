"""Which providers this deployment can reach, and which one answers by default.

A provider with no key is absent rather than broken: `available()` leaves it out,
so a client can offer the choice honestly instead of showing an option that fails
on use. Nothing here raises at construction -- the container is built at import
time and must touch no network and no configuration it can reject.
"""

from __future__ import annotations

from daikonstudio.application.ports.llm import LanguageModel, ModelChoice
from daikonstudio.domain.shared.errors import ServiceUnavailableError

_ENV_VAR = {"anthropic": "STUDIO_ANTHROPIC_API_KEY", "openai": "STUDIO_OPENAI_API_KEY"}


class ConfiguredLanguageModels:
    def __init__(
        self, *, models: dict[str, LanguageModel], configured: frozenset[str], default: str
    ) -> None:
        self._models = models
        self._configured = configured
        self._default = default

    def available(self) -> tuple[ModelChoice, ...]:
        return tuple(
            self._models[name].choice for name in sorted(self._configured) if name in self._models
        )

    def get(self, provider: str | None) -> LanguageModel:
        name = provider or self._default
        if name not in self._models:
            known = ", ".join(sorted(self._models))
            raise ServiceUnavailableError(
                f"'{name}' is not a model provider this server knows about.",
                detail=f"Known providers: {known}.",
            )
        if name not in self._configured:
            raise ServiceUnavailableError(
                f"No {name} model is configured on this server.",
                detail=f"Set {_ENV_VAR.get(name, 'the provider API key')} to an API key.",
            )
        return self._models[name]
