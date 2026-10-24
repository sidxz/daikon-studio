"""One-off extractor for the vendored Scientific Agent Skills fixture.

Reads files only, never runs anything from the clone, and prints the fixture JSON:

    uv run python -I tests/fakes/data/extract_scientific_agent_skills.py <clone> \
        > tests/fakes/data/scientific_agent_skills.json

Source: https://github.com/K-Dense-AI/scientific-agent-skills (MIT, K-Dense Inc.).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

KEY = re.compile(r"^[A-Za-z][\w-]*:")


def frontmatter(text: str) -> dict[str, str]:
    lines = text.split("\n")
    assert lines[0] == "---", "no frontmatter"
    end = lines.index("---", 1)
    fields: dict[str, list[str]] = {}
    key = ""
    for line in lines[1:end]:
        if KEY.match(line):
            key, _, rest = line.partition(":")
            rest = rest.strip()
            fields[key] = [] if re.fullmatch(r"[>|][+-]?", rest) else [rest]  # block scalar marker
        elif key and line.strip():
            fields[key].append(line.strip())
    return {k: scalar(" ".join(v)) for k, v in fields.items()}


def scalar(value: str) -> str:
    """The text of a YAML flow scalar: quotes removed, escapes decoded."""
    if len(value) > 1 and value[0] == value[-1] == '"':
        return json.loads(value)
    if len(value) > 1 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return value


def skills(root: Path) -> list[dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    for path in sorted((root / "skills").glob("*/SKILL.md")):
        meta = frontmatter(path.read_text(encoding="utf-8"))
        found[path.parent.name] = {
            "id": path.parent.name,
            "name": meta["name"],
            "description": meta["description"],
        }
    link = re.compile(r"^- \*\*\[[^\]]+\]\(\.\./skills/([^/]+)/SKILL\.md\)\*\*")
    category = ""
    for line in (root / "docs" / "skills.md").read_text(encoding="utf-8").split("\n"):
        if line.startswith("### "):
            category = line[4:].strip()
        elif line.startswith("## "):
            category = line[3:].strip()
        elif m := link.match(line):
            found[m[1]]["category"] = category
    assert all("category" in s for s in found.values()), "a skill sits under no heading"
    return list(found.values())


def worked_examples(root: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    text = (root / "docs" / "examples.md").read_text(encoding="utf-8")
    for block in re.split(r"^(?=### Example )", text, flags=re.M):
        head = re.match(r"### Example \w+: (.+)", block)
        if not head:
            continue
        objective = re.search(r"^\*\*Objective\*\*: (.+(?:\n(?!\n).+)*)", block, re.M)
        used = re.search(r"^\*\*Skills Used\*\*:\n\n?((?:- .*\n)+)", block, re.M)
        assert objective and used, head[1]
        ids = re.findall(r"^- `([^`]+)`", used[1], re.M)
        out.append(
            {"title": head[1].strip(), "objective": " ".join(objective[1].split()), "skills": ids}
        )
    return out


def quick_examples(root: Path) -> list[dict[str, object]]:
    readme = (root / "README.md").read_text(encoding="utf-8")
    section = readme.split("## 💡 Quick Examples")[1].split("## 🔬 Use Cases")[0]
    out: list[dict[str, object]] = []
    for block in re.split(r"^(?=### )", section, flags=re.M)[1:]:
        title = re.match(r"### \S+ (.+)", block)
        goal = re.search(r"^\*\*Goal\*\*: (.+)$", block, re.M)
        used = re.search(r"^\*\*Skills Used\*\*: (.+)$", block, re.M)
        assert title and goal and used, block[:40]
        ids = [s.strip() for s in used[1].split(",")]
        out.append({"title": title[1].strip(), "objective": goal[1].strip(), "skills": ids})
    return out


def main() -> None:
    root = Path(sys.argv[1])
    catalogue = skills(root)
    examples = worked_examples(root) + quick_examples(root)
    ids = {s["id"] for s in catalogue}
    assert len(catalogue) == 177, len(catalogue)
    for e in examples:
        missing = set(e["skills"]) - ids  # type: ignore[arg-type]
        assert not missing, (e["title"], missing)
    assert len(examples) >= 50, len(examples)
    fixture = {
        "source": {
            "repository": "https://github.com/K-Dense-AI/scientific-agent-skills",
            "license": "MIT",
            "copyright": "K-Dense Inc.",
            "version": "2.72.0",
            "commit": "593377b",
            "extracted": "2026-10-24",
        },
        "skills": catalogue,
        "examples": examples,
    }
    print(json.dumps(fixture, indent=1, ensure_ascii=False))


main()
