"""The language-model port.

Narrow on purpose: one method, which takes a system prompt, a user message and a
schema, and returns an instance of that schema. Everything the planner needs is
expressible that way, and nothing else leaks -- no provider types, no message
lists, no streaming.

**Why a schema rather than text.** The planner's output is a plan, and a plan that
arrives as prose has to be parsed by something that can be wrong in a second way.
Both providers validate against the schema at their end, so a malformed plan fails
at the boundary rather than halfway through rendering. It also means the port can
never be used to ask for free text by accident, which is the shape of request that
would let a model's prose reach a study unchecked.

**Why two implementations rather than one OpenAI-compatible client pointed at
both.** A shim gives up each vendor's own parse helper and diverges quietly on
anything past plain chat. Two thin adapters cost less than one leaky abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel


@dataclass(frozen=True, kw_only=True)
class ModelChoice:
    """Which model answered, recorded on whatever it produced.

    Not decoration. A plan is a judgement, and six months later the only way to
    reread it fairly is to know what wrote it -- the same reason an engine's
    version is pinned on a run.
    """

    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


class LanguageModel(Protocol):
    @property
    def choice(self) -> ModelChoice: ...

    async def parse[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T], stable: str = ""
    ) -> T:
        """Ask once and return the answer as `schema`.

        `stable` is text that is identical from one call to the next (a catalogue
        index, say) and goes ahead of `prompt` so a provider that caches prefixes
        can reuse it. Empty means there is none.

        Raises `ServiceUnavailableError` for anything the provider does wrong or
        cannot do -- including an unconfigured key, which is a deployment problem
        and must say which environment variable is missing rather than failing as
        an authorization error against the caller, who did nothing wrong.
        """
        ...


class LanguageModels(Protocol):
    """The providers this deployment can reach.

    A deployment may have neither configured, which is not an error until someone
    asks for a plan: the rest of Studio works without a model, and refusing to
    boot over an optional integration is how a missing key takes down training.
    """

    def available(self) -> tuple[ModelChoice, ...]: ...

    def get(self, provider: str | None) -> LanguageModel:
        """The named provider, or the configured default when `provider` is None.

        Raises `ServiceUnavailableError` naming the environment variable to set
        when the request names a provider this deployment has no key for, or when
        nothing at all is configured.
        """
        ...
