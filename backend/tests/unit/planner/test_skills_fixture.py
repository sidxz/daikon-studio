"""The vendored Scientific Agent Skills load as a registry, and its goals name real skills."""

from __future__ import annotations

from tests.fakes.registry import load_examples, skills_registry


def test_the_real_world_fixture_loads_as_a_registry() -> None:
    registry = skills_registry()
    assert len(registry.manifests()) == 177
    assert len({m.id for m in registry.manifests()}) == 177
    assert all(m.description for m in registry.manifests())
    assert len(skills_registry(replicate=3).manifests()) == 531


def test_every_example_names_skills_the_fixture_has() -> None:
    ids = {m.id for m in skills_registry().manifests()}
    examples = load_examples()
    assert len(examples) >= 50
    assert all(set(e.skills) <= ids and e.objective for e in examples)
