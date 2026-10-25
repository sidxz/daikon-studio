"""The language-model port.

Narrow on purpose: two methods. `parse` takes a system prompt, a user message and a
schema, and returns an instance of that schema. `act` takes the same and a few tools,
and returns the answer with the calls the model made to them. Nothing else leaks --
no provider types, no message lists, no streaming.

**Why tools for what a turn does.** A provider compiles the answer's schema into a
grammar, and the grammar has a ceiling: a schema that also carried every act a turn
might make, as fields the model fills, outgrew it. A tool is the shape providers give
to an act, small, named and offered only to the turns allowed to make it, and a tool
that is not offered cannot be called, which is the same guarantee a missing field gave.

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

from collections.abc import Sequence
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


@dataclass(frozen=True)
class Tool:
    """Something a turn may do, offered to the model by name; `schema` is its parameters.

    Not strict: the host checks every call against the study before anything changes, so
    the provider's grammar is spent on the answer and not on this.
    """

    name: str
    description: str
    schema: type[BaseModel]


@dataclass(frozen=True)
class Answer[T: BaseModel]:
    """What a turn produced: the answer in its schema, and the calls it made first.

    Each call is an instance of its tool's `schema`; `malformed` counts the ones whose
    arguments did not fit it, which the caller counts as dropped. The calls are recorded
    before the model answers, so the answer can refer to what it did.
    """

    parsed: T
    calls: tuple[BaseModel, ...] = ()
    malformed: int = 0


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

    async def act[T: BaseModel](
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        tools: Sequence[Tool],
        stable: str = "",
    ) -> Answer[T]:
        """Ask once, letting the model call `tools` before it answers in `schema`.

        With no tools this is `parse`. With them, a model that calls some is told each
        call is recorded and asked for its answer; the calls are not carried out here,
        and the answer and the calls come back together. Raises as `parse` does.
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
