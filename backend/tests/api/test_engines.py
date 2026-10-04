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
        "ecfp4-lightgbm",
        "descriptors-xgboost",
        "tanimoto-gp",
        "chemprop-dmpnn",
        "molformer-xl",
    }
    xgb = next(e for e in engines if e["id"] == "ecfp4-xgboost")
    keys = {c["key"] for c in xgb["conditions"]}
    assert keys == {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "positive_weighting",
        "rdkit_descriptors",
    }
    depth = next(c for c in xgb["conditions"] if c["key"] == "max_depth")
    assert depth["type"] == "integer" and depth["default"] == 6
    weighting = next(c for c in xgb["conditions"] if c["key"] == "positive_weighting")
    assert weighting["tasks"] == ["binary_classification"] and weighting["default"] == "none"
    assert weighting["options"] == ["none", "balanced", "sqrt_balanced"]
    assert weighting["option_labels"] == ["None", "Balanced", "Square-root balanced"]
    descriptors = next(c for c in xgb["conditions"] if c["key"] == "rdkit_descriptors")
    assert descriptors["tasks"] == [] and descriptors["default"] is False
    chemprop = next(e for e in engines if e["id"] == "chemprop-dmpnn")
    chemprop_weighting = next(
        c for c in chemprop["conditions"] if c["key"] == "positive_weighting"
    )
    assert chemprop_weighting["tasks"] == ["binary_classification"]
    assert "rdkit_descriptors" in {c["key"] for c in chemprop["conditions"]}
    molformer = next(e for e in engines if e["id"] == "molformer-xl")
    molformer_weighting = next(
        c for c in molformer["conditions"] if c["key"] == "positive_weighting"
    )
    assert molformer_weighting["tasks"] == ["binary_classification"]


async def test_exactly_one_engine_is_marked_as_the_baseline(client):
    """Verify exactly one engine is flagged as baseline."""
    response = await client.get("/api/v1/engines")
    assert response.status_code == 200, response.text
    engines = response.json()
    assert sum(1 for e in engines if e["is_baseline"]) == 1
    baseline = next(e for e in engines if e["is_baseline"])
    assert baseline["id"] == "ecfp4-randomforest"


async def test_every_manifest_says_whether_it_learns_targets_jointly(client):
    response = await client.get("/api/v1/engines")
    assert response.status_code == 200, response.text
    joint = {e["id"] for e in response.json() if e["supports_multitask"]}
    assert joint == {"chemprop-dmpnn", "molformer-xl"}


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
    # The tasks the setting applies to; empty means all of them.
    assert n_est["tasks"] == []
    # Empty means the raw option values are shown.
    assert n_est["option_labels"] == []
    # Verify user-facing copy, not developer notes
    assert "Number of trees" in n_est["label"]


async def test_engines_endpoint_requires_authentication(anonymous_client):
    """Verify unauthenticated requests are rejected."""
    response = await anonymous_client.get("/api/v1/engines")
    assert response.status_code == 401, response.text


async def test_every_engine_names_the_lane_that_serves_it(client):
    """A pending run can only say "waiting for a gpu runner" if the catalogue
    says which engines need one."""
    engines = (await client.get("/api/v1/engines")).json()
    lanes = {engine["id"]: engine["lane"] for engine in engines}
    assert lanes["chemprop-dmpnn"] == "gpu"
    assert lanes["molformer-xl"] == "gpu"
    assert lanes["ecfp4-randomforest"] == "default"
