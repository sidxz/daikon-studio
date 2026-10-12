"""MolEncoder embeddings: a small ModernBERT over SMILES, read frozen.

`fabikru/MolEncoder` is a 15M-parameter encoder pretrained with masked language
modelling on about a million ChEMBL molecules, published with the paper "MolEncoder:
towards optimal masked language modeling for molecules" (Krüger et al., *Digital
Discovery*, 2025, doi 10.1039/D5DD00369E). Its own finding is the interesting one for
this roster: the authors report that raising the masking ratio to about 30% matters more
than model or dataset size, and that this 15M model matches larger SMILES transformers
pretrained on far more data. So it is here as a *cheap* transformer representation --
60 MB of weights, `model.safetensors` in full -- not as a stronger one. **No accuracy
comparison against our fingerprint or descriptor engines has been measured yet.**

Pooling is the CLS token, `last_hidden_state[:, 0, :]`, because that is what the
authors' own code does: their `examples/cls_embedding.ipynb` extracts exactly that
vector, and the checkpoint's `config.json` sets `classifier_pooling: "cls"`, so a
fine-tune of theirs reads the same position. Their notebook also carries a caveat worth
repeating where a scientist can see it, and the engine's help text does: CLS embeddings
are most meaningful *after* fine-tuning on the task, and nothing fine-tunes here. Mean
pooling over the non-special tokens is the obvious alternative and is not offered,
because the authors' convention is the one their published numbers come from; add the
alternative only if a measurement asks for it.

The tokenizer is character-level over the SMILES vocabulary (82 tokens, no multi-
character atom symbols -- `Cl` is `C` then `l`), and the checkpoint's
`max_position_embeddings` is 502. A longer structure is **refused, not truncated**, the
same rule `molformer_xl._collate` applies and for the same reason: truncation drops the
tail of a molecule without saying so.

Every torch and transformers import lives inside a function, so `_scoring.py` can
register this featurizer at module scope while the API tier installs without the `gpu`
extra.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np

from daikonstudio.domain.shared.errors import ValidationError

logger = logging.getLogger(__name__)

#: The published checkpoint and the full 40-character commit behind the short
#: `6e443fbd` the spec names. Pinned for the same reason every other checkpoint here is:
#: a published Protocol's predictions must not change because the Hub repository did.
#: No `trust_remote_code` -- `modernbert` is a first-class `transformers` architecture,
#: so this loads through plain `AutoModel` on transformers >= 5.19.
CHECKPOINT = "fabikru/MolEncoder"
REVISION = "6e443fbdd46af01728b6f3ddab7f8ed60963869d"

#: `hidden_size` from the checkpoint's own config -- 384, not the 768 a ModernBERT-base
#: would have. Asserted after the forward pass rather than trusted.
DIM = 384

#: `max_position_embeddings` from the checkpoint's own config, counting the two special
#: tokens. See the module docstring: over-length structures are refused.
MAX_TOKENS = 502

#: Molecules per forward pass. A fixed count rather than `protein.embed`'s token budget:
#: 502 tokens is the hard ceiling here, so the worst batch is 64 x 502 and needs no
#: arithmetic to bound.
_BATCH = 64


def embed(smiles: list[str]) -> np.ndarray:
    """CLS-token embeddings, one row per input and aligned with it: `(n, 384)` float32.

    Reproducible to float32 tolerance rather than bitwise, for exactly the reason
    `protein.embed.esm2_650m` documents at length: a batch pads to its longest member,
    so the same molecule embedded beside different neighbours goes through differently
    shaped matmuls. Compare these with `allclose`, never `array_equal`.
    """
    return _embed(tuple(smiles))


@lru_cache(maxsize=6)
def _embed(smiles: tuple[str, ...]) -> np.ndarray:
    """Cached on the exact input tuple, because the callers repeat themselves: one fit
    featurizes train, validation and test separately, and `FanOut` repeats all three per
    target.

    ponytail: three forward passes per fit, one per partition, because each partition is
    a different tuple. The protein side needed the upgrade first and has it
    (`_scoring._sliced_from_one_pass`, which embeds the whole frame once and slices); it
    is ESM-2-specific inside. Generalize it to any featurizer if this engine's embedding
    time ever shows up next to the tree fit -- at 15M parameters it currently does not.
    """
    # Answered before the torch import, so an empty partition does not need the `gpu`
    # extra: the width is a property of the checkpoint and needs nothing loaded. The
    # alternative passes in CI and fails on a runner, or the reverse.
    if not smiles:
        return np.zeros((0, DIM), dtype=np.float32)

    from daikonstudio.infrastructure.protein.embed import torch_and_device

    # Shared with the protein engines, so one weights directory, one fp32-precision
    # guard and one device order serve both modalities.
    torch, device = torch_and_device()
    tokenizer, model = _loaded()

    # Longest first, so a batch's padding is set by molecules of similar length -- and,
    # here, so the over-length refusal below fires in the first batch rather than
    # partway through the embedding pass.
    order = sorted(range(len(smiles)), key=lambda i: len(smiles[i]), reverse=True)
    out = np.zeros((len(smiles), DIM), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(order), _BATCH):
            batch = order[start : start + _BATCH]
            # No `truncation=True`: see the module docstring. The mask counts each row's
            # real tokens, so the error can name the structure that is long.
            encoded = tokenizer([smiles[i] for i in batch], padding=True, return_tensors="pt")
            _refuse_over_length(encoded["attention_mask"], [smiles[i] for i in batch])
            encoded = {key: value.to(device) for key, value in encoded.items()}
            # The CLS token, at position 0 for every row: the tokenizer prepends it and
            # pads on the right, so padding never moves it.
            vectors = model(**encoded).last_hidden_state[:, 0, :].to(torch.float32)
            block = vectors.cpu().numpy()
            if block.shape[1] != DIM:
                raise AssertionError(
                    f"MolEncoder returned {block.shape[1]}-dimensional embeddings, not "
                    f"the expected {DIM}."
                )
            for row, index in enumerate(batch):
                out[index] = block[row]

    # Handed out of a cache, so make it read-only rather than trust every caller.
    out.flags.writeable = False
    return out


def _refuse_over_length(attention_mask: Any, smiles: list[str]) -> None:
    """Refuse a batch holding a structure longer than the model's context.

    Counted and named, not truncated: a tokenized SMILES that runs past 502 tokens is a
    peptide or another large molecule, which this engine is the wrong tool for, and
    silently keeping its first 502 tokens would produce a Scorecard describing a
    fragment of each structure.
    """
    lengths = attention_mask.sum(dim=1)
    longest = int(lengths.max())
    if longest <= MAX_TOKENS:
        return
    over = int((lengths > MAX_TOKENS).sum())
    raise ValidationError(
        f"MolEncoder reads at most {MAX_TOKENS} tokens of a SMILES string, and "
        f"{over:,} of these structures need more (up to {longest:,}). Anything longer "
        "is outside what the model has seen, so this engine is a poor fit for peptides "
        "and other large molecules. Train a fingerprint or descriptor engine instead. "
        f"The longest structure begins: {smiles[int(lengths.argmax())][:60]}"
    )


@lru_cache(maxsize=1)
def _loaded() -> tuple[Any, Any]:
    """The tokenizer and the model, in eval mode on the chosen device, loaded once.

    One entry: this engine offers no choice of checkpoint, so a second entry could only
    ever hold a duplicate of the first.
    """
    from transformers import AutoModel, AutoTokenizer

    from daikonstudio.infrastructure.protein.embed import torch_and_device

    _torch, device = torch_and_device()
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, revision=REVISION)
    model = AutoModel.from_pretrained(CHECKPOINT, revision=REVISION).to(device)
    model.eval()
    logger.info("Loaded %s on %s for embedding.", CHECKPOINT, device)
    return tokenizer, model
