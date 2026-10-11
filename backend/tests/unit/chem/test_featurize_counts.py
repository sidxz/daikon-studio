"""Count-based Morgan fingerprints, beside the binary ones rather than replacing them.

Why both exist: on the same leakage-aware peptide benchmark, counts measure +4.03 AUPRC
over binary bits for ECFP, +7.00 for topological torsion and +9.23 for RDKit
fingerprints -- a substitution occurring three times is not the same molecule as one
where it occurs once, and a binary bit cannot say so.

Why `ecfp4` is untouched: `tanimoto_gp`'s kernel needs binary vectors to stay positive
semi-definite. Swapping counts in underneath it would keep producing numbers from a
kernel that is no longer a valid covariance, which is the quiet kind of wrong. MinMax is
the PSD generalisation to counts, and until someone implements and argues for it, the GP
keeps the binary input it was built on.
"""

from __future__ import annotations

import numpy as np

from daikonstudio.infrastructure.chem.featurize import ecfp4, ecfp4_counts


def test_counts_record_how_often_a_substructure_occurs() -> None:
    """The whole point. Hexane repeats one carbon environment six times; a binary bit
    for it says "present" either way, a count says six."""
    binary = ecfp4(["CCCCCC"])
    counts = ecfp4_counts(["CCCCCC"])

    assert binary.max() == 1
    assert counts.max() > 1


def test_counts_are_nonzero_exactly_where_bits_are_set() -> None:
    """Same generator, same radius, same width -- so the support must agree. A mismatch
    would mean the two featurizers disagree about the molecule, not about multiplicity."""
    smiles = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CCN(CC)CC"]

    assert np.array_equal(ecfp4(smiles) > 0, ecfp4_counts(smiles) > 0)


def test_invalid_smiles_is_an_all_zero_row_not_an_exception() -> None:
    """Matching `ecfp4`'s existing contract: one unparseable structure in an upload must
    not fail the whole fit."""
    result = ecfp4_counts(["not-a-molecule", "CCO"])

    assert result.shape == (2, 2048)
    assert result[0].sum() == 0
    assert result[1].sum() > 0


def test_counts_do_not_saturate_at_the_binary_ceiling() -> None:
    """A uint8 row would wrap at 255. Long polymers are exactly the case counts were
    added for, so the dtype has to carry them."""
    result = ecfp4_counts(["C" * 400])

    assert result.max() > 255, result.max()


# --- the user path -----------------------------------------------------------------


def test_the_toggle_picks_the_count_featurizer() -> None:
    """Two independent toggles, four keys. Asserting the whole matrix because the
    combined key is built by string concatenation, where a typo would silently fall back
    to a KeyError at fit time rather than at import."""
    from daikonstudio.infrastructure.engines._scoring import tree_featurizer

    assert tree_featurizer({})[0] == "ecfp4"
    assert tree_featurizer({"count_fingerprints": True})[0] == "ecfp4_counts"
    assert tree_featurizer({"rdkit_descriptors": True})[0] == "ecfp4+rdkit_descriptors"
    assert (
        tree_featurizer({"count_fingerprints": True, "rdkit_descriptors": True})[0]
        == "ecfp4_counts+rdkit_descriptors"
    )


def test_every_tree_engine_offers_the_toggle() -> None:
    """It is a property of the representation, not of one learner, so all three carry
    it -- and a user comparing random forest against XGBoost must be able to hold the
    featurization fixed."""
    from daikonstudio.infrastructure.engines.ecfp4_lightgbm import Ecfp4LightGBM
    from daikonstudio.infrastructure.engines.ecfp4_randomforest import Ecfp4RandomForest
    from daikonstudio.infrastructure.engines.ecfp4_xgboost import Ecfp4XGBoost

    for engine in (Ecfp4RandomForest(), Ecfp4XGBoost(), Ecfp4LightGBM()):
        keys = {spec.key for spec in engine.manifest().conditions}
        assert "count_fingerprints" in keys, engine.manifest().id


def test_the_default_key_is_unchanged_so_old_artifacts_still_read() -> None:
    """`bundle_features` writes nothing for "ecfp4", which is what lets every artifact
    fitted before these toggles existed predict unchanged. Renaming the default would
    break them silently -- `_load_bundle` reads a missing key as ECFP4."""
    from daikonstudio.infrastructure.engines._scoring import bundle_features

    assert bundle_features("ecfp4") == {}
