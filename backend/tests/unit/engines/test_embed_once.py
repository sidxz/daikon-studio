"""Embed the frame once, slice per partition.

A fit's partitions are disjoint, so embedding them separately wastes nothing. What wastes
a forward pass over the whole dataset is that a protocol *run* fits the same rows several
times -- the optimism gap re-partitions them for its comparison leg, and every split draw
adds another leg -- and those legs differ only in the `split` column. So the frame's
sequence column is the right cache key, and these tests pin both halves of that: the
features must not change, and the second leg must not embed anything.

The counting tests matter more than the equality ones. An "optimisation" that silently
still embeds per partition would pass every equality check in this file.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from daikonstudio.infrastructure.engines._scoring import esm2_featurizer
from daikonstudio.infrastructure.protein import embed as embed_module
from daikonstudio.infrastructure.protein.embed import DIM, embedding_map, site_map

_WEIGHTS_DIR = Path(
    os.environ.get("STUDIO_PRETRAINED_WEIGHTS_DIR", "~/.cache/daikon-studio/weights")
).expanduser()

needs_weights = pytest.mark.skipif(
    not (_WEIGHTS_DIR / "hub" / "models--facebook--esm2_t33_650M_UR50D").exists(),
    reason="ESM-2 650M weights not cached; CI does not download 2.6 GB",
)

_PARENT = "MAKVQLSRLSGEQLLDEL"
_FRAME = [
    _PARENT,
    "MAKVQLSRLSGEQLLDEV",
    "MAKVQLSRLSGEQLLDEI",
    "MAKVQLSRLAGEQLLDEL",
    "MAKVQLSRLTGEQLLDEL",
    "MAKVQLSRLCGEQLLDEL",
]
# The two legs a grouped-split run actually fits: same rows, different partitioning.
_LEG_ONE = (_FRAME[:4], _FRAME[4:])
_LEG_TWO = (_FRAME[2:], _FRAME[:2])


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Each test starts cold, or a previous test's whole-frame pass makes the counting
    tests trivially pass."""
    embedding_map.cache_clear()
    site_map.cache_clear()
    embed_module._embed.cache_clear()
    embed_module._site_features.cache_clear()


# --- what the featurizer returns -----------------------------------------------------


@needs_weights
def test_slicing_the_frame_gives_what_embedding_the_partition_gave() -> None:
    """`allclose` and not `array_equal`: the sliced row is embedded beside the whole
    frame while the direct one is embedded beside its partition, and a batch pads to its
    longest member. That tolerance is the documented one, not a new concession."""
    from daikonstudio.infrastructure.protein.embed import esm2_650m

    _key, served, _parent = esm2_featurizer({}, _FRAME)
    train, _test = _LEG_ONE

    assert np.allclose(served(train), esm2_650m(train), atol=1e-4)
    assert served(train).shape == (len(train), DIM)


@needs_weights
def test_the_site_block_slices_identically_too() -> None:
    from daikonstudio.infrastructure.protein.embed import esm2_with_sites, site_dim

    _key, served, parent = esm2_featurizer({"site_features": True}, _FRAME)
    train, _test = _LEG_ONE

    assert parent == _PARENT
    assert served(train).shape == (len(train), site_dim())
    assert np.allclose(served(train), esm2_with_sites(train, parent), atol=1e-4)


@needs_weights
def test_the_likelihood_column_survives_the_slicing() -> None:
    """The LLR is appended after the slice, so this is what breaks if the `+llr` suffix
    is mishandled and the column is silently dropped -- leaving a model fitted on 1,280
    features where its artifact says 1,281."""
    _key, served, _parent = esm2_featurizer({"likelihood_feature": True}, _FRAME)
    train, _test = _LEG_ONE

    assert served(train).shape == (len(train), DIM + 1)


@needs_weights
def test_lowercase_rows_are_normalized_on_the_sliced_path_too() -> None:
    """The bug this file caught while being written: `esm2_with_sites` normalizes before
    scoring positions, and a map built from raw rows would compare a lowercase sequence
    against a normalized parent and read it as mutated at every position."""
    frame = [*_FRAME, _PARENT.lower()]

    _key, served, parent = esm2_featurizer({"site_features": True}, frame)
    block = served([_PARENT.lower()])

    assert parent == _PARENT
    # Normalized, this row *is* the parent: no mutated sites, so the count column is 0.
    assert block[0, -1] == 0.0


# --- that it really is one pass ------------------------------------------------------


@needs_weights
def test_a_second_leg_embeds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The point of the change. Two legs over the same rows in different partitions must
    cost one forward pass over the frame, not two."""
    passes: list[int] = []
    original = embed_module._embed

    def counting(sequences: tuple[str, ...], model: str = "esm2-650m") -> np.ndarray:
        passes.append(len(sequences))
        return original(sequences, model)

    monkeypatch.setattr(embed_module, "_embed", counting)
    embedding_map.cache_clear()

    _key, served, _parent = esm2_featurizer({}, _FRAME)
    for partition in (*_LEG_ONE, *_LEG_TWO):
        served(partition)

    assert passes == [len(_FRAME)], passes


@needs_weights
def test_duplicate_rows_are_embedded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single-parent series repeats the wild type, and the map is keyed by sequence."""
    passes: list[int] = []
    original = embed_module._embed

    def counting(sequences: tuple[str, ...], model: str = "esm2-650m") -> np.ndarray:
        passes.append(len(sequences))
        return original(sequences, model)

    monkeypatch.setattr(embed_module, "_embed", counting)
    embedding_map.cache_clear()

    frame = [*_FRAME, *_FRAME]
    _key, served, _parent = esm2_featurizer({}, frame)
    served(frame)

    assert passes == [len(_FRAME)], passes


# --- the edges -----------------------------------------------------------------------


def test_an_empty_frame_falls_back_rather_than_priming_nothing() -> None:
    """Deliberately not `needs_weights`: an empty partition must featurize without torch
    installed at all, which is why `_embed` answers the empty case above its import. CI
    runs this on a venv with neither torch nor the checkpoint."""
    _key, served, parent = esm2_featurizer({}, [])

    assert parent is None
    assert served([]).shape == (0, DIM)


@needs_weights
def test_a_row_outside_the_frame_is_still_embedded() -> None:
    """Keeps the featurizer total. It cannot happen on the fit path -- the partitions come
    from this very frame -- so this is about not being a trap for a future caller."""
    _key, served, _parent = esm2_featurizer({}, _FRAME)

    stranger = served(["MAKVQLSRLSGEQLLDEW"])

    assert stranger.shape == (1, DIM)
    assert np.isfinite(stranger).all()


@needs_weights
def test_the_map_values_are_views_not_copies() -> None:
    """Memory, asserted rather than assumed: the dict holds rows of `_embed`'s cached
    read-only array, so priming a 10k-row frame does not cost a second 51 MB."""
    found = embedding_map(tuple(_FRAME))

    assert len(found) == len(_FRAME)
    for vector in found.values():
        assert vector.base is not None, "a copy, not a view into the cached array"
        assert not vector.flags.writeable
