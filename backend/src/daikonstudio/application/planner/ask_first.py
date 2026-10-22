"""What the planner needs to know before a plan would mean anything.

The first half of planning, split out from the second deliberately. Asked to do
both at once, a model asks "is IC50 the readout that matters?" and, in the same
answer, writes a plan that uses IC50 throughout. The guess is load-bearing before
the scientist has read the question, and the plan reads as settled when it rests
on an assumption nobody confirmed.

Asking first costs one extra call and makes the plan an answer to something.
"""

from __future__ import annotations

from dataclasses import dataclass

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.planner.catalogue import (
    CATALOGUE_HEADING,
    READING_THE_DATA,
    always_runs,
    brought,
    catalogue,
    profile_text,
    starting_note,
)
from daikonstudio.application.planner.plan import Interrogation
from daikonstudio.application.ports.llm import LanguageModels, ModelChoice
from daikonstudio.application.ports.table_profiler import TableProfile
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.shared.errors import DomainError, ValidationError

# At most this many before planning. A form of fifteen is not answered, it is
# skipped, and skipping silently is the failure this whole step exists to avoid.
_MAX_QUESTIONS = 6

# Across the whole analysis, counting every round. The ceiling exists because a
# model that may always ask one more will: each answer gives it enough context to
# see another choice, and the scientist is three rounds deep in an interview
# instead of reading a plan. At zero it has to proceed on stated assumptions,
# which it can always do -- a question it cannot default is rare, and everything
# else was never a question.
QUESTION_BUDGET = 10

_SYSTEM_TEMPLATE = """\
You are about to plan an analysis for a scientist in Daikon Studio. Before you do, \
work out what you need to know. You are not planning yet, and you must not propose \
steps.

Rules, in order of importance.

1. Ask only what you cannot answer yourself. Before asking anything, decide what \
you would do if nobody replied. If you can defend that choice, put it in `default`: \
the question then reaches the scientist as an assumption they can override, the \
plan is written on it unless they do, and it costs them nothing. Leave `default` \
null only where the plan genuinely cannot be written either way -- which column \
holds the result that matters, what an ambiguous code means -- because those are \
the questions this round exists for. At most %(limit)d, and fewer is better. The \
whole analysis may ask %(budget)d, counting every round, so each one spent here \
is one the plan cannot raise later. Asking none is a fine outcome and means you \
could read the data well enough to proceed.

2. %(reading)s

3. Offer concrete options whenever the plausible answers enumerate: which of \
three columns, whether a value is a bound or was never tested, whether repeated \
rows are re-tests or duplicate registrations. Leave options empty when the answer \
is genuinely open, such as the name of a target. Never offer "not sure" yourself; \
the scientist is given that on every question anyway.

4. Say in one line what changes depending on the answer. That line is how the \
scientist decides which questions are worth their time.

5. %(always)s

Write for a working scientist. Be specific and brief.
"""


@dataclass(frozen=True, kw_only=True)
class AskFirstCommand:
    goal: str
    provider: str | None = None
    profile: TableProfile | None = None


@dataclass(frozen=True, kw_only=True)
class Questions:
    goal: str
    asked: Interrogation
    written_by: ModelChoice


class AskFirst:
    def __init__(self, models: LanguageModels, tools: ToolRegistry) -> None:
        self._models = models
        self._tools = tools

    async def __call__(
        self, command: AskFirstCommand, auth: AuthContext | None = None
    ) -> Result[Questions, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None

        goal = command.goal.strip()
        if not goal:
            return Failure(ValidationError("Say what you want to find out."))

        model = self._models.get(command.provider)
        parts = [f"{CATALOGUE_HEADING}\n{catalogue(self._tools.manifests())}"]
        if command.profile is not None:
            parts.append(profile_text(command.profile))
        parts.append(starting_note(brought(command.profile)))
        parts.append(f"The scientist's goal, in their words:\n{goal}")

        manifests = self._tools.manifests()
        system = _SYSTEM_TEMPLATE % {
            "limit": _MAX_QUESTIONS,
            "budget": QUESTION_BUDGET,
            "reading": READING_THE_DATA,
            "always": always_runs(manifests),
        }
        asked = await model.parse(system=system, prompt="\n\n".join(parts), schema=Interrogation)
        # Trimmed here rather than trusted to the prompt: a form nobody finishes
        # is worse than a shorter one, and the model has already ranked them.
        asked.questions = asked.questions[:_MAX_QUESTIONS]

        return Success(Questions(goal=goal, asked=asked, written_by=model.choice))
