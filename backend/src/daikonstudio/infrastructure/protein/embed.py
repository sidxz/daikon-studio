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

    Shape `(len(sequences), DIM)`, float32.

    **Reproducible to float32 tolerance, not bitwise, and the distinction is measured
    rather than assumed.** The model is deterministic -- eval mode, `no_grad`, no
    sampling -- but a batch pads every sequence to its longest member, so the same
    sequence embedded beside different neighbours goes through differently shaped
    matmuls and the reduction order changes. Measured here: the same 36-residue sequence
    embedded alone and then beside six 341-residue ones differs by up to **1.4e-06** per
    dimension. PyTorch states the guarantee it does not make: `(A@B)[0]` is not promised
    to be bitwise identical to `A[0]@B[0]`.

    What that does and does not threaten. Within one fit it is harmless: each partition
    is featurized in one call, so a partition's composition is fixed. What it does touch
    is any comparison *across* calls -- `predict` featurizes the frame it is handed, not
    the training partitions, and the optimism-gap comparison re-partitions the same rows
    -- where the same sequence can carry features differing in the sixth decimal. A tree
    threshold sitting inside that interval could land either way.

    The fix is not a tolerance but a single call: embed the whole frame once and slice
    per partition, which also halves the work every protein run currently does twice.
    Until then, treat equality here as `allclose`, never `array_equal`, and see
    `tests/unit/engines/test_esm2_xgboost.py` for the test that pins it.
    """
    return _embed(tuple(sequences))


@lru_cache(maxsize=6)
def _embed(sequences: tuple[str, ...]) -> np.ndarray:
    """Cached on the exact input tuple, because the callers repeat themselves.

    One fit featurizes the train, validation and test partitions separately, and
    `FanOut` repeats all three per target -- so a four-target dataset would otherwise
    run twelve forward passes over the same sequences. Six entries at 10k sequences is
    about 300 MB of float32.

    Keyed on the exact input tuple, which is why a *differently partitioned* view of the
    same dataset misses entirely. That now has a price tag: a grouped split also trains
    the random-split comparison behind the Scorecard's optimism gap, whose partitions
    hold the same sequences in different groupings, so every protein run embeds the
    dataset twice. Measured on 3,996 TEM-1 variants of 286 residues, embedding was 323 s
    of a 465 s fit -- so the second fit is most of a second embedding pass.

    ponytail: an in-process LRU, sized for one fit. A dataset-scoped embedding cache in
    the blob store is the real answer if embeddings ever need to survive a retrain, and
    is the point of frozen features in the first place -- but it needs a cache key and
    an eviction story, and this proves the pipeline first. The cheaper intermediate step,
    if the double fit hurts first: embed the whole frame once and slice per partition,
    which turns the comparison fit into just the tree refit.
    """
    import torch

    if not sequences:
        return np.zeros((0, DIM), dtype=np.float32)

    cleaned, truncated, unrepresentable = _clean(sequences)
    if truncated:
        logger.warning(
            "%d of %d sequences were longer than %d residues and were truncated for "
            "ESM-2; predictions for those rows describe only the first %d residues.",
            truncated,
            len(sequences),
            MAX_RESIDUES,
            MAX_RESIDUES,
        )
    if unrepresentable:
        logger.warning(
            "%d of %d sequences contain a residue ESM-2 has no token for (a stop codon, "
            "most often) and were embedded with an unknown-residue token in its place; "
            "those rows are described less precisely than the rest.",
            unrepresentable,
            len(sequences),
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


#: The residues ESM-2 has real tokens for: the twenty standard ones plus the ambiguity
#: and rare codes it was pretrained with (B Asx, Z Glx, X unknown, U selenocysteine,
#: O pyrrolysine). Anything else tokenizes to `<unk>`.
#:
#: Hardcoded so `_clean` can run without loading 2.6 GB of weights, and kept honest by
#: `test_the_embeddable_alphabet_matches_the_real_tokenizer`, which checks every member
#: against the real tokenizer and checks that `*` really is `<unk>`.
#:
#: `*` is the one character that matters in practice: it is the only member of
#: `domain.data.structure_kind._RESIDUES` -- so a string containing it is classified as a
#: sequence -- that ESM-2 cannot represent.
_EMBEDDABLE = frozenset("ACDEFGHIKLMNPQRSTVWYBXZUO")


def _clean(sequences: tuple[str, ...]) -> tuple[list[str], int, int]:
    """Upper-case, drop whitespace and gaps, truncate, and count both kinds of loss.

    Returns `(cleaned, truncated, unrepresentable)`. Residue-alphabet *validation* still
    belongs to the dataset layer alongside SMILES canonicalization, and the policy here
    is unchanged: ESM-2's tokenizer maps an unknown character to `<unk>` rather than
    failing, so a stray character degrades one row instead of killing a fit.

    What changed is that it is no longer silent. Truncation was already counted and
    warned about; an `<unk>` substitution was not, so a nonsense variant embedded exactly
    like an ordinary protein and nothing said so. Counting it is the same discipline
    applied to the same class of loss.
    """
    cleaned: list[str] = []
    truncated = 0
    unrepresentable = 0
    for sequence in sequences:
        residues = "".join(str(sequence).split()).replace("-", "").replace(".", "").upper()
        if len(residues) > MAX_RESIDUES:
            truncated += 1
            residues = residues[:MAX_RESIDUES]
        if not _EMBEDDABLE.issuperset(residues):
            unrepresentable += 1
        cleaned.append(residues)
    return cleaned, truncated, unrepresentable


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
    _torch, device = torch_and_device()

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModel.from_pretrained(CHECKPOINT).to(device)
    model.eval()
    logger.info("Loaded %s on %s for embedding.", CHECKPOINT, device)
    return tokenizer, model, device


#: oneDNN's fp32 math mode. `BF16` or `TF32` here makes every "fp32" matmul silently
#: lower precision -- it appears in AWS's own recommended Graviton PyTorch recipe, so a
#: deployment can acquire it without anyone choosing it for us. We refuse instead of
#: coping: a precision downgrade nobody declared would move every embedding by far more
#: than the float32 noise documented on `esm2_650m`, and it would move them silently.
#: oneDNN also has no strict-reproducibility mode (unlike oneMKL), so pinning
#: `OMP_NUM_THREADS` -- which both images do -- is necessary and not sufficient.
_FPMATH_MODE = "DNNL_DEFAULT_FPMATH_MODE"


def refuse_lowered_precision() -> None:
    """Raise if the environment has silently downgraded fp32 matmuls.

    Separate from `torch_and_device`, and called by it before the torch import, for two
    reasons. A deployment that has this set is misconfigured whether or not torch is
    installed, so the clearer error is the one about precision. And it makes the guard
    testable without the `gpu` extra -- the API tier and CI both install without it, and
    a test that reached for torch to check an environment variable failed there while
    passing locally.
    """
    import os

    from daikonstudio.domain.shared.errors import ValidationError

    declared = os.environ.get(_FPMATH_MODE, "").strip().upper()
    if declared not in ("", "STRICT"):
        raise ValidationError(
            f"This runner sets {_FPMATH_MODE}={declared}, which makes every fp32 "
            f"matrix multiplication run at lower precision. Scores computed under it "
            f"would not match the ones this model reports. Unset it, or set it to "
            f"STRICT, and retrain."
        )


def torch_and_device() -> tuple[Any, Any]:
    """`torch` plus the device to run on, with the environment prepared first.

    Shared by this module's embedding loader and `likelihood.py`'s masked-LM loader, so
    that one weights directory, one precision guard and one device order serve both.
    """
    # `HF_HOME` is read by transformers at import time, so it is set before the import.
    # Reusing `STUDIO_PRETRAINED_WEIGHTS_DIR` is what keeps one weights directory per
    # deployment -- the same one `Dockerfile.gpu` bakes CheMeleon into -- instead of a
    # second multi-gigabyte cache appearing under the worker's home. `setdefault`, so a
    # deployment already managing `HF_HOME` keeps its own answer. Identical to
    # `molformer_xl._require_transformers`; see the rationale there.
    import os

    from daikonstudio.domain.shared.errors import ValidationError
    from daikonstudio.settings import Settings

    os.environ.setdefault("HF_HOME", str(os.path.expanduser(Settings().pretrained_weights_dir)))

    refuse_lowered_precision()

    try:
        import torch
    except ImportError as exc:
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
    return torch, device
