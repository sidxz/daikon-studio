"""How well the model names the right tools from a long, real-world index.

The registry is Daikon's own tools plus the 177 vendored Scientific Agent Skills; the goals
are the vendored examples' objectives, their skill lists the gold. Each goal gets one asking
turn (the study agent reads the index as a cached prefix and names in `wanted` what it wants
to read in full), and that turn's `wanted` is scored against the gold. Costs real calls:

    uv run python -m tests.planner_eval.wanted_recall --provider anthropic --out <dir>
    uv run python -m tests.planner_eval.wanted_recall --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from pydantic import BaseModel

from daikonstudio.application.planner.catalogue import index
from daikonstudio.application.ports.llm import LanguageModel, ModelChoice
from daikonstudio.application.studies.agent import StudyAgent
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.shared.errors import ServiceUnavailableError
from daikonstudio.domain.studies.study import Study
from daikonstudio.infrastructure.llm.anthropic import AnthropicLanguageModel
from daikonstudio.infrastructure.llm.openai import OpenAILanguageModel
from daikonstudio.infrastructure.tools.catalog import default_registry
from daikonstudio.logging import configure_logging
from daikonstudio.settings import Settings
from tests.fakes.blob_store import InMemoryBlobStore
from tests.fakes.registry import Example, load_examples, skills_registry
from tests.fakes.studies import InMemoryStudies

_ROOT = Path(__file__).resolve().parents[3]  # the repository (or worktree) root
_OUT = "../internal-docs/superpowers/reviews/2026-10-24-tool-selection-eval"


class _AskingTurnOnly:
    """Lets the agent's asking call through and stops it there.

    `ask()` plans at once when it has no questions, and that second call is neither what is
    measured nor worth its cost; the stored asking turn is already complete by then.
    """

    def __init__(self, model: LanguageModel) -> None:
        self._model = model
        self._calls = 0

    @property
    def choice(self) -> ModelChoice:
        return self._model.choice

    async def parse[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T], stable: str = ""
    ) -> T:
        self._calls += 1
        if self._calls > 1:
            raise ServiceUnavailableError("Measurement stops after the asking turn.")
        return await self._model.parse(system=system, prompt=prompt, schema=schema, stable=stable)


class _Models:
    def __init__(self, model: LanguageModel) -> None:
        self._model = model

    def get(self, provider: str | None) -> LanguageModel:
        return self._model


class _Advance:
    async def __call__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("an asking turn starts no run")


@dataclass(frozen=True)
class Row:
    title: str
    recall: float | None  # None: the asking call failed
    extra: tuple[str, ...] = ()
    missed: tuple[str, ...] = ()
    target_kinds: tuple[str, ...] = ()
    model: str = ""


def _registry() -> ToolRegistry:
    return ToolRegistry((*default_registry().manifests(), *skills_registry().manifests()))


def _usable(examples: list[Example]) -> tuple[list[Example], list[tuple[str, str]]]:
    """Drop the goals that name a skill: a disclaimer naming one would inflate recall."""
    ids = [m.id for m in skills_registry().manifests()]
    kept: list[Example] = []
    dropped: list[tuple[str, str]] = []
    for e in examples:
        named = next((i for i in ids if i.lower() in e.objective.lower()), None)
        if named:
            dropped.append((e.title, named))
        else:
            kept.append(e)
    return kept, dropped


def _model(provider: str) -> LanguageModel | None:
    # The key lives in `.env`: the backend's own if there is one, else the repository root's.
    settings = Settings(_env_file=(_ROOT / ".env", _ROOT / "backend" / ".env"))  # type: ignore[call-arg]
    if provider == "openai":
        key = settings.openai_api_key
        return OpenAILanguageModel(api_key=key, model=settings.openai_model) if key else None
    key = settings.anthropic_api_key
    return AnthropicLanguageModel(api_key=key, model=settings.anthropic_model) if key else None


async def _ask(example: Example, registry: ToolRegistry, model: LanguageModel) -> Row:
    studies = InMemoryStudies()
    study = Study(
        workspace_id=uuid.uuid4(),
        created_by=uuid.uuid4(),
        goal=example.objective,
        restated_goal="",
        checks=(),
        stop_conditions=(),
        steps=[],
    )
    studies.rows[study.id] = study
    agent = StudyAgent(
        studies,
        registry,
        _Models(_AskingTurnOnly(model)),  # type: ignore[arg-type]
        InMemoryBlobStore(),  # type: ignore[arg-type]
        _Advance(),  # type: ignore[arg-type]
    )
    # The stop after the asking turn, or a real failure: the thread tells which.
    with contextlib.suppress(ServiceUnavailableError):
        await agent.ask(study, None)
    if not study.thread.turns:
        return Row(example.title, None)
    turn = study.thread.turns[0]
    wanted, gold = set(turn.wanted), set(example.skills)
    return Row(
        example.title,
        len(wanted & gold) / len(gold) if gold else 1.0,
        tuple(sorted(wanted - gold)),
        tuple(sorted(gold - wanted)),
        turn.target_kinds,
        turn.written_by,
    )


def _report(
    rows: list[Row], dropped: list[tuple[str, str]], provider: str, size: int, tokens: int
) -> str:
    scored = [r for r in rows if r.recall is not None]
    models = sorted({r.model for r in scored})
    lines = [
        "# Real-world `wanted` recall",
        "",
        f"- Provider: {provider}; model that answered: {', '.join(models) or 'none'}",
        f"- Registry: {size} tools; index about {tokens:,} tokens (characters over four)",
        f"- Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Examples: {len(rows)} asked, {len(dropped)} excluded, "
        f"{len(rows) - len(scored)} failed",
    ]
    if scored:
        full = sum(r.recall == 1.0 for r in scored)
        lines += [
            f"- Mean recall: {mean(r.recall or 0.0 for r in scored):.3f}",
            f"- At full recall: {full}/{len(scored)} ({full / len(scored):.0%})",
            f"- Mean extra tools named: {mean(len(r.extra) for r in scored):.2f}",
        ]
    lines += [
        "",
        "| Example | Recall | Extra | Target kinds | Missed | Named instead |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.recall is None:
            lines.append(f"| {r.title} | failed | | | | |")
        else:
            lines.append(
                f"| {r.title} | {r.recall:.2f} | {len(r.extra)} | {', '.join(r.target_kinds)} "
                f"| {', '.join(r.missed)} | {', '.join(r.extra)} |"
            )
    lines += ["", "## Excluded (the objective names a skill)", ""]
    lines += [f"- {title} (names `{skill}`)" for title, skill in dropped] or ["- none"]
    return "\n".join(lines) + "\n"


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="anthropic", choices=("anthropic", "openai"))
    parser.add_argument("--out", default=_OUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    registry = _registry()
    tokens = len(index(registry.manifests())) // 4
    examples, dropped = _usable(load_examples())
    examples = examples[: args.limit]
    print(f"registry: {len(registry.manifests())} tools; index about {tokens:,} tokens")
    print(f"examples: {len(examples)} after excluding {len(dropped)}: {[d[0] for d in dropped]}")
    if args.dry_run:
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = _model(args.provider)
    if model is None:
        (out / "wanted-recall-not-run.md").write_text(
            f"# Not run\n\nNo {args.provider} API key is configured "
            f"(STUDIO_{args.provider.upper()}_API_KEY in `.env`), so no call was made.\n"
        )
        print("no key; wrote wanted-recall-not-run.md")
        return 1

    configure_logging()  # the "answered" lines show cache_read per call
    rows = [await _ask(e, registry, model) for e in examples]
    if all(r.recall is None for r in rows):
        (out / "wanted-recall-not-run.md").write_text(
            "# Not run\n\nEvery asking call failed; see the run's log for the provider's answer.\n"
        )
        print("every call failed; wrote wanted-recall-not-run.md")
        return 1
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"wanted-recall-{stamp}.md"
    path.write_text(_report(rows, dropped, args.provider, len(registry.manifests()), tokens))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
