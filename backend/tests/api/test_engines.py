"""Tests for the engines endpoint."""

from __future__ import annotations


async def test_engines_are_listed_with_their_conditions(client):
    """Verify engines are listed with all condition metadata needed for form rendering."""
    response = await client.get("/api/v1/engines")
    assert response.status_code == 200, response.text
    engines = response.json()
    ids = {e["id"] for e in engines}
    # Every registered engine is listed in every deployment, GPU-lane ones included:
    # the manifest is plain data, so the picker stays identical everywhere and a
    # deployment that cannot run one says so through a PENDING Run, not a missing entry.
    assert ids == {
        "ecfp4-xgboost",
        "ecfp4-randomforest",
        "descriptors-xgboost",
        "chemprop-dmpnn",
    }
    xgb = next(e for e in engines if e["id"] == "ecfp4-xgboost")
    keys = {c["key"] for c in xgb["conditions"]}
    assert keys == {"n_estimators", "max_depth", "learning_rate"}
    depth = next(c for c in xgb["conditions"] if c["key"] == "max_depth")
    assert depth["type"] == "integer" and depth["default"] == 6


async def test_exactly_one_engine_is_marked_as_the_baseline(client):
    """Verify exactly one engine is flagged as baseline."""
    response = await client.get("/api/v1/engines")
    assert response.status_code == 200, response.text
    engines = response.json()
    assert sum(1 for e in engines if e["is_baseline"]) == 1
    baseline = next(e for e in engines if e["is_baseline"])
    assert baseline["id"] == "ecfp4-randomforest"


async def test_conditions_carry_all_form_metadata(client):
    """Verify conditions include every field a form needs to render."""
    response = await client.get("/api/v1/engines")
    assert response.status_code == 200, response.text
    engines = response.json()
    rf = next(e for e in engines if e["id"] == "ecfp4-randomforest")
    n_est = next(c for c in rf["conditions"] if c["key"] == "n_estimators")
    # Every condition carries these fields so the form can render without hardcoding
    assert "key" in n_est
    assert "label" in n_est
    assert "type" in n_est
    assert "required" in n_est
    assert "default" in n_est
    assert "minimum" in n_est
    assert "maximum" in n_est
    assert "options" in n_est
    assert "help" in n_est
    # Verify user-facing copy, not developer notes
    assert "Number of trees" in n_est["label"]


async def test_engines_endpoint_requires_authentication(anonymous_client):
    """Verify unauthenticated requests are rejected."""
    response = await anonymous_client.get("/api/v1/engines")
    assert response.status_code == 401, response.text
