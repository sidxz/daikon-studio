"""ESM-2 650M mean-pooled embeddings -- the protein side of `chem/featurize.py`.

Frozen embeddings into an existing head, rather than a fine-tuned protein model, and
the evidence for that choice is unusually one-sided. On FLIP2's 16 engineering-realistic
splits a naively fine-tuned protein language model was the best model on only 2; a plain
zero-shot likelihood won 6 and ridge on one-hot plus likelihoods won 4. Kermut tops
supervised ProteinGym as a Gaussian process over frozen ESM-2 650M embeddings, in about
ten minutes where ProteinNPT needs upwards of four hundred hours. So the transferable
information sits in the embedding, and the head should stay cheap and tuneable. See
`docs/plans/2026-10-06-protein-peptide-roster.md`.

650M and not 3B or 15B: ESM-2's scaling is flat up to 15B on downstream supervised
tasks, 15B visibly overfits (0.955 train against 0.660 test on Meltome), and every
saved artifact here is already a 150-750 MB upload.

**Every torch and transformers import lives inside a function, never at module scope.**
`_scoring.py` registers this featurizer at import time and the API tier installs without
the `gpu` extra -- a module-level `import torch` here would break `daikonstudio` on the
API image. Same rule as `chemprop_dmpnn.py` and `molformer_xl.py`.

Deliberately not here yet:

* **The log-likelihood ratio feature.** Kermut uses ESM-2's LLR as the GP's prior mean,
  and it is the single highest-value addition to this module -- but an LLR is a variant
  scored against a reference sequence, so it only means anything for a dataset of
  variants of one parent protein. That is the TB resistance-MIC case study's shape, not
  the general one, so it lands with that case study.
* **fp16.** ESM-2 was pretrained in fp16 and fp16 tracks fp32 far better than bf16 here,
  so it is the right speed knob later. fp32 first because it behaves identically on
  CUDA, MPS and CPU, and this engine should work on all three before it is fast on one.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: `facebook/esm2_t33_650M_UR50D`, MIT licence, 652M parameters, 1280-dimensional
#: embeddings, and the most used protein language model by a wide margin (~53.7M
#: downloads a month) -- which matters here mainly because it is a first-class
#: `transformers` architecture, so this module needs no new dependency and no
#: `trust_remote_code`.
CHECKPOINT = "facebook/esm2_t33_650M_UR50D"

#: Embedding width, asserted after the first forward pass rather than trusted.
DIM = 1280

#: ESM-2 has 1026 position embeddings, two of which are the start and end tokens, so
#: 1022 residues is the real ceiling. Sequences longer than this are truncated and
#: **counted** -- see `_clean`. Silence here is a data-integrity bug: `esm-extract`
#: truncates without a word and the Hugging Face tokenizer does not truncate at all by
#: default, and one published dataset had 9.3% of its sequences over the limit. It bites
#: immediately on real targets: M. tuberculosis RpoB is ~1,178 residues and EmbB ~1,098.
MAX_RESIDUES = 1022

#: Tokens per batch rather than sequences per batch, so one long sequence in a set of
#: short ones cannot blow up memory. Paired with the length sort in `_embed`, which is
#: not cosmetic: unsorted, *raising* the batch size made 650M embedding monotonically
#: slower in published measurement, 118s to 258s.
_TOKEN_BUDGET = 16384


def esm2_650m(sequences: list[str]) -> np.ndarray:
    """Mean-pooled last-hidden-state embeddings, one row per input, aligned with it.

    Shape `(len(sequences), DIM)`, float32. Deterministic: the model runs in eval mode
    under `no_grad`, so the same sequence scores the same on every call and a reloaded
    artifact reproduces the numbers its own Scorecard reported.
    """
    return _embed(tuple(sequences))


@lru_cache(maxsize=6)
def _embed(sequences: tuple[str, ...]) -> np.ndarray:
    """Cached on the exact input tuple, because the callers repeat themselves.

    One fit featurizes the train, validation and test partitions separately, and
    `FanOut` repeats all three per target -- so a four-target dataset would otherwise
    run twelve forward passes over the same sequences. Six entries at 10k sequences is
    about 300 MB of float32.

    ponytail: an in-process LRU, sized for one fit. A dataset-scoped embedding cache in
    the blob store is the real answer if embeddings ever need to survive a retrain, and
    is the point of frozen features in the first place -- but it needs a cache key and
    an eviction story, and this proves the pipeline first.
    """
    import torch

    if not sequences:
        return np.zeros((0, DIM), dtype=np.float32)

    cleaned, truncated = _clean(sequences)
    if truncated:
        logger.warning(
            "%d of %d sequences were longer than %d residues and were truncated for "
            "ESM-2; predictions for those rows describe only the first %d residues.",
            truncated,
            len(sequences),
            MAX_RESIDUES,
            MAX_RESIDUES,
        )

    tokenizer, model, device = _loaded()

    # Longest first, so a batch's padding is set by sequences of similar length and the
    # first batch is the one most likely to run out of memory -- better to fail in the
    # first second than forty minutes in.
    order = sorted(range(len(cleaned)), key=lambda i: len(cleaned[i]), reverse=True)
    out = np.zeros((len(cleaned), DIM), dtype=np.float32)

    with torch.no_grad():
        for batch in _batches(cleaned, order):
            encoded = tokenizer(
                [cleaned[i] for i in batch],
                return_tensors="pt",
                padding=True,
                return_special_tokens_mask=True,
            )
            special = encoded.pop("special_tokens_mask")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hidden = model(**encoded).last_hidden_state

            # Pool over residues only: the attention mask keeps padding out, and
            # dropping the special tokens keeps ESM-2's start and end tokens out. Both
            # matter -- a short sequence is otherwise two tokens of noise in a mean over
            # a handful of residues.
            mask = encoded["attention_mask"] * (1 - special.to(device))
            mask = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

            vectors = pooled.to(torch.float32).cpu().numpy()
            if vectors.shape[1] != DIM:
                raise AssertionError(
                    f"{CHECKPOINT} returned {vectors.shape[1]}-dimensional embeddings, "
                    f"not the expected {DIM}."
                )
            for row, index in enumerate(batch):
                out[index] = vectors[row]

    # Handed out of a cache, so make it read-only rather than trust every caller.
    out.flags.writeable = False
    return out


def _clean(sequences: tuple[str, ...]) -> tuple[list[str], int]:
    """Upper-case, drop whitespace and alignment gaps, truncate, and count truncations.

    Residue-alphabet validation is deliberately not done here: that belongs to the
    dataset layer alongside SMILES canonicalization, and ESM-2's tokenizer maps an
    unknown character to `<unk>` rather than failing, so a stray character degrades one
    row instead of killing a fit.
    """
    cleaned: list[str] = []
    truncated = 0
    for sequence in sequences:
        residues = "".join(str(sequence).split()).replace("-", "").replace(".", "").upper()
        if len(residues) > MAX_RESIDUES:
            truncated += 1
            residues = residues[:MAX_RESIDUES]
        cleaned.append(residues)
    return cleaned, truncated


def _batches(cleaned: list[str], order: list[int]) -> list[list[int]]:
    """Indices grouped so that `batch_size * longest_in_batch` stays under the budget."""
    batches: list[list[int]] = []
    current: list[int] = []
    for index in order:
        longest = len(cleaned[current[0]]) if current else len(cleaned[index])
        # +2 for the start and end tokens every sequence carries.
        if current and (len(current) + 1) * (longest + 2) > _TOKEN_BUDGET:
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


@lru_cache(maxsize=1)
def _loaded() -> tuple[Any, Any, Any]:
    """Tokenizer, model and device, built once per process.

    `transformers` caches the weights on disk itself, so there is nothing for
    `_pretrained.py` to do here -- that module exists for checksummed URL downloads, and
    ESM-2 comes off the Hub as a first-class architecture.

    No revision pin, unlike `molformer_xl.py`. That pin exists because MoLFormer needs
    `trust_remote_code=True` and therefore executes Python fetched from the Hub, which
    must be the code that was reviewed. ESM-2 executes nothing: the architecture ships
    inside `transformers`, so only the weights come over the wire.
    """
    # `HF_HOME` is read by transformers at import time, so it is set before the import.
    # Reusing `STUDIO_PRETRAINED_WEIGHTS_DIR` is what keeps one weights directory per
    # deployment -- the same one `Dockerfile.gpu` bakes CheMeleon into -- instead of a
    # second multi-gigabyte cache appearing under the worker's home. `setdefault`, so a
    # deployment already managing `HF_HOME` keeps its own answer. Identical to
    # `molformer_xl._require_transformers`; see the rationale there.
    import os

    from daikonstudio.settings import Settings

    os.environ.setdefault("HF_HOME", str(os.path.expanduser(Settings().pretrained_weights_dir)))

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        from daikonstudio.domain.shared.errors import ValidationError

        raise ValidationError(
            "This runner does not have the GPU dependencies that the ESM-2 engine "
            "requires. An administrator can register a runner for the 'gpu' lane on "
            "the Runners page."
        ) from exc

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModel.from_pretrained(CHECKPOINT).to(device)
    model.eval()
    logger.info("Loaded %s on %s for embedding.", CHECKPOINT, device)
    return tokenizer, model, device
