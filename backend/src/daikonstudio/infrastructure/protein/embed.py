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

from daikonstudio.application.data.assign_split import consensus, mutated_positions
from daikonstudio.domain.shared.errors import ValidationError

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
    -- where the same sequence can carry features differing in the sixth decimal.

    **A tree threshold sitting inside that interval lands either way, and that is
    measured, not hypothetical.** On the eight-sequence fixture in
    `tests/unit/engines/test_esm2_xgboost.py`, predicting one training row alone rather
    than beside the other seven moved its feature vector by 1.9e-06 and its prediction
    from 0.86 to 3.00. A four-row fit is the worst case -- `hist` bins put split
    boundaries on observed values, and with four rows every boundary is on one -- so this
    is not the magnitude a real fit should expect. It is still the same mechanism, and it
    is the clearest argument for the fix below.

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

            mask = encoded["attention_mask"] * (1 - special.to(device))
            vectors = _pool(hidden, mask).cpu().numpy()
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


#: The site block's width: pooled embedding, mutated-site embedding, WT-relative site
#: delta, then the two site scalars (mean relative position, number of mutations).
SITE_DIM = 3 * DIM + 2


def esm2_with_sites(sequences: list[str], parent: str | None = None) -> np.ndarray:
    """Embeddings that say *where* the change is, not only that there was one.

    Shape `(len(sequences), SITE_DIM)`, float32: the mean-pooled embedding this module
    already produced, then the hidden state at the mutated site, then that site's state
    minus the parent's at the same site, then the mean relative position and the number
    of mutations.

    The problem it fixes. A row here is "one sequence -> one number", so a per-sequence
    mean over 286 residues moves by about 1/286 for a point mutation: the model is handed
    the average of a whole protein and cannot express which residue changed. The
    per-residue state at the mutated site can, and it is the published difference between
    0.744 and 0.782 Spearman on ProteinGym's random split, structure-free.

    **The WT-relative delta here is per-*site*, which is the only version that carries
    information to a tree.** A pooled `variant - parent` delta subtracts the same constant
    from every row, and a tree splits one feature at a time, so a threshold on
    `feature_i - c_i` is the same tree as a threshold on `feature_i` -- a pooled delta is
    a no-op, not a feature. (It is also a no-op for an isotropic RBF kernel, which sees
    only `||a - b||`.) The site delta is not, because each row subtracts the parent's
    state *at its own position*, and different rows mutate different positions. It is also
    the half that survives a POSITION holdout, where the site's identity is by
    construction useless.

    A row that mutates nothing -- the wild type itself -- gets zeros for both site blocks
    and `0` for both scalars. Zeros rather than a fabricated vector, with the mutation
    count beside them so the tree can learn that the site block means nothing on that row.
    A multiple mutant averages over its sites, matching how `likelihood.py` sums over
    them: one row, one feature vector.

    `parent` is the reference the fit recorded; see `aligned_parent` for why inferring it
    per call is wrong in every path that matters.
    """
    if not sequences:
        return np.zeros((0, SITE_DIM), dtype=np.float32)
    normalized, reference = aligned_parent(sequences, parent)
    return _site_features(tuple(normalized), reference)


@lru_cache(maxsize=4)
def _site_features(sequences: tuple[str, ...], parent: str) -> np.ndarray:
    """Cached on the exact input, like `_embed`, and for the same repeated callers.

    Pooling happens here rather than through `esm2_650m` so that the whole feature block
    costs **one** forward pass over the variants. Calling both would embed every sequence
    twice, and embedding is already 323 s of a measured 465 s fit.
    """
    import torch

    tokenizer, model, device = _loaded()
    parent_ids, offset = residue_token_offset(tokenizer, parent)
    sites = [sorted(mutated_positions(sequence, parent)) for sequence in sequences]
    out = np.zeros((len(sequences), SITE_DIM), dtype=np.float32)

    # All one length, so the budget divides evenly and no length sort is needed -- the
    # same simplification `likelihood._masked_marginals` gets from an aligned series.
    per_batch = max(1, _TOKEN_BUDGET // (len(parent) + 2))
    with torch.no_grad():
        parent_input = parent_ids.unsqueeze(0).to(device)
        parent_hidden = model(
            input_ids=parent_input, attention_mask=torch.ones_like(parent_input)
        ).last_hidden_state[0]
        if parent_hidden.shape[-1] != DIM:
            raise AssertionError(
                f"{CHECKPOINT} returned {parent_hidden.shape[-1]}-dimensional "
                f"embeddings, not the expected {DIM}."
            )

        for start in range(0, len(sequences), per_batch):
            chunk = sequences[start : start + per_batch]
            encoded = tokenizer(
                list(chunk),
                return_tensors="pt",
                padding=True,
                return_special_tokens_mask=True,
            )
            special = encoded.pop("special_tokens_mask")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            pooled = _pool(hidden, encoded["attention_mask"] * (1 - special.to(device)))

            for index in range(len(chunk)):
                row = start + index
                out[row, :DIM] = pooled[index].cpu().numpy()
                positions = sites[row]
                if not positions:
                    continue
                tokens = [offset + position for position in positions]
                at_site = hidden[index, tokens].to(torch.float32)
                reference = parent_hidden[tokens].to(torch.float32)
                out[row, DIM : 2 * DIM] = at_site.mean(dim=0).cpu().numpy()
                out[row, 2 * DIM : 3 * DIM] = (at_site - reference).mean(dim=0).cpu().numpy()
                out[row, 3 * DIM] = sum(positions) / len(positions) / len(parent)
                out[row, 3 * DIM + 1] = len(positions)

    logger.info(
        "Embedded %d sequences with site features against a %d-residue parent.",
        len(sequences),
        len(parent),
    )
    # Handed out of a cache, so make it read-only rather than trust every caller.
    out.flags.writeable = False
    return out


def aligned_parent(sequences: list[str], parent: str | None = None) -> tuple[list[str], str]:
    """`(normalized sequences, their parent)` for a feature that scores a *position*.

    One definition of "the series' parent", shared by the masked-marginal LLR and the
    site featurizer, for the reason `assign_split.mutated_positions` is shared: two
    notions of what the wild type is would make the two features disagree about which
    residue a variant replaced, and both would still produce plausible numbers.

    **Pass `parent` whenever one is known, and it is known in every path that matters.**
    Inferring it from the list in hand is only right when that list is the whole series,
    and the two paths that featurize are not: a fit featurizes train, validation and test
    separately, and `predict` featurizes whatever frame it is handed. A small test
    partition can elect a different modal residue from the train partition, and a predict
    frame holding *one* variant elects that variant as its own parent -- so it reads as
    zero mutations, its LLR comes out 0 and its site block comes out zeros, against a
    model fitted on real values. Nothing about that is visible in the output. So the
    parent is computed once from the whole frame, recorded in the artifact, and passed
    back in here.

    Normalization comes before the length check and before the consensus, which is not
    cosmetic: an upload with lowercase residues would otherwise have `a` and `A` counted
    as different residues, so every such row would read as a mutation of a parent that
    exists in no well. Gaps are **not** stripped, unlike `_clean` -- removing a `-`
    shifts every position after it, and these features are indexed by position.
    """
    normalized = ["".join(str(sequence).split()).upper() for sequence in sequences]
    lengths = {len(sequence) for sequence in normalized}
    if parent is not None and lengths - {len(parent)}:
        raise ValidationError(
            f"This model scores each variant against the {len(parent)}-residue parent "
            f"sequence recorded when it was trained, but these sequences are "
            f"{', '.join(str(length) for length in sorted(lengths))} residues. Predict "
            f"with variants of the same protein, or train a model on this one."
        )
    if len(lengths) > 1:
        raise ValidationError(
            f"A position-scored feature compares each variant with the series' parent, "
            f"so it needs an aligned single-parent series: every sequence must have the "
            f"same length, but these range from {min(lengths)} to {max(lengths)} "
            f"residues."
        )
    if max(lengths) > MAX_RESIDUES:
        raise ValidationError(
            f"These sequences are {max(lengths)} residues, beyond ESM-2's "
            f"{MAX_RESIDUES}-residue context. Truncating them would move every position "
            f"past the cut, so a position-scored feature cannot be computed for them."
        )
    return normalized, consensus(normalized) if parent is None else parent


def residue_token_offset(tokenizer: Any, sequence: str) -> tuple[Any, int]:
    """`(token ids, the token index holding residue 0)` for one sequence.

    Derived from the special-token mask rather than assumed to be 1: ESM-2 prefixes one
    `<cls>`, but trusting that silently offsets every position by one if a checkpoint
    ever differs, and an off-by-one reads the wrong site while still returning a
    plausible number. Raises unless the residues occupy one contiguous token each, which
    is the condition that makes `offset + position` mean anything at all.
    """
    encoded = tokenizer(sequence, return_tensors="pt", return_special_tokens_mask=True)
    special = encoded.pop("special_tokens_mask")[0]
    residues = [int(index) for index in (special == 0).nonzero().flatten()]
    offset = residues[0] if residues else 0
    if residues != list(range(offset, offset + len(sequence))):
        raise ValidationError(
            "This ESM-2 tokenizer does not map residues to tokens one-for-one, so a "
            "position-scored feature cannot be aligned to a residue position."
        )
    ids: Any = encoded["input_ids"][0]
    return ids, offset


def _pool(hidden: Any, mask: Any) -> Any:
    """Mean over residues only, as float32.

    The attention mask keeps padding out and dropping the special tokens keeps ESM-2's
    start and end tokens out. Both matter -- a short sequence is otherwise two tokens of
    noise in a mean over a handful of residues.

    One function rather than the same four lines in two forward passes: `_site_features`
    pools the same hidden states it gathers per-residue vectors from, and a pooled column
    that disagreed with this engine's other pooled column by two special tokens would be
    a difference no test looks for.
    """
    import torch

    weights = mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
    return pooled.to(torch.float32)


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
