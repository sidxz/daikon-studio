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
from dataclasses import dataclass
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

#: Embedding width of `CHECKPOINT`, asserted after the forward pass rather than trusted.
#: Per-variant widths live on `Variant.dim`; this is the default's, kept as a module
#: constant because `SITE_DIM` and a good deal of test arithmetic are written against it.
DIM = 1280


#: ESM-2 has 1026 position embeddings, two of which are the start and end tokens, so
#: 1022 residues is the real ceiling. Sequences longer than this are truncated and
#: **counted** -- see `_clean`. Silence here is a data-integrity bug: `esm-extract`
#: truncates without a word and the Hugging Face tokenizer does not truncate at all by
#: default, and one published dataset had 9.3% of its sequences over the limit. It bites
#: immediately on real targets: M. tuberculosis RpoB is ~1,178 residues and EmbB ~1,098.
#:
#: ESM-2's number, and the default of `Variant.max_residues`, which is what every
#: truncation and refusal actually reads: ESM C is rotary and has no such ceiling, so its
#: cap is a choice of ours rather than the architecture's.
MAX_RESIDUES = 1022


@dataclass(frozen=True)
class Variant:
    """One set of weights this module can embed and score with.

    Three fields rather than a bare checkpoint string, because the two additions to the
    roster differ along two independent axes. A smaller ESM-2 is a *different checkpoint
    with a different width*; VESM2 is the *same checkpoint with weights laid over it*.
    Collapsing either into the other loses the distinction that matters at load time.
    """

    #: The `transformers` architecture and the base weights, from the Hub.
    checkpoint: str
    #: Embedding width. Asserted against the real forward pass, never trusted.
    dim: int
    #: What the engine picker calls it.
    label: str
    #: A `_pretrained.WEIGHT_SETS` name whose partial state dict is laid over the base
    #: weights, or `None` to use them as published.
    overlay: str | None = None
    #: How a log-likelihood ratio is read off this model: `"masked"` masks each position
    #: of the parent in turn and reads the distribution there, `"wt"` reads them all off
    #: one forward pass of the unmasked parent. **A property of the weights, not a user
    #: preference** -- it is the protocol the publisher trained or distilled under, and
    #: scoring a model the other way is what made VESM2's own LLR look worse than stock
    #: ESM-2's. So it lives here and no engine offers it as a setting.
    #:
    #: `None` means *nobody has read the publisher's scoring code yet*, and every
    #: likelihood path then refuses rather than picking one. See `likelihood_protocol`.
    likelihood_scoring: str | None = "masked"
    #: The `torch.nn.Linear` names a LoRA fine-tune attaches to, for `_lora.attach`.
    #: Per-variant because the names are the architecture's: ESM-2's attention
    #: projections are `query/key/value/dense`, ESM C's are `q_proj ... down_proj`, and
    #: peft has no built-in map for either. A wrong name trains nothing but the head.
    target_modules: tuple[str, ...] = ("query", "key", "value", "dense")
    #: The longest sequence this checkpoint is read over. Truncation and the counting
    #: that goes with it read this, not the module constant.
    max_residues: int = MAX_RESIDUES
    #: A Hub commit to pin `from_pretrained` to, or `None` for whatever the repo's
    #: default branch holds. Pinned where the weights are new enough that the repository
    #: is still moving; ESM-2's snapshots have been stable for years.
    revision: str | None = None


#: The default, and the only one the engines offered before the others existed.
DEFAULT_VARIANT = "esm2-650m"

VARIANTS: dict[str, Variant] = {
    DEFAULT_VARIANT: Variant(checkpoint=CHECKPOINT, dim=DIM, label="ESM-2 650M"),
    # VESM2: ESM-2 650M with a co-distilled partial state dict laid over it. Same
    # architecture, same width, same 1022-residue ceiling, so it costs no new code path
    # -- and MIT, per the model card's body.
    #
    # **Measured on the TEM-1 pilot under its position holdout, and it is the best thing
    # on the roster**: 0.7366 Spearman / 0.5395 r2 with the likelihood feature on, where
    # stock ESM-2's best tree arm is 0.7197 / 0.5084. As a plain embedding it is also
    # ahead, 0.6928 / 0.4725 against 0.6575 / 0.4195.
    #
    # It reached that only after the scoring protocol below was fixed. Scored by ESM-2's
    # masked-marginal protocol its LLR *hurt* -- 0.6860 / 0.4483, worse than VESM2 with
    # no LLR at all -- which read as "VESM2's embeddings are better but its head is
    # worse" and was entirely our bug.
    #
    # **The cause was a scoring-protocol mismatch, since confirmed and fixed.** VESM's
    # released code reads ratios off the logits of the *unmasked* sequence -- wt-marginals
    # -- and `likelihood.py` was masking each position in turn, which is ESM-2's protocol
    # and not theirs. Zero-shot LLR against fitness on the same 512 test rows:
    #
    #     esm2-650m   masked  +0.5545     esm2-650m   wt  +0.5434
    #     vesm2-650m  masked  +0.5092     vesm2-650m  wt  +0.6558
    #
    # So masked is correctly ESM-2's default, and reading VESM2 the wrong way was costing
    # **0.147 Spearman**. Scored its own way its zero-shot LLR beats ESM-2's best by
    # 0.101, which does reproduce the paper's claim. Hence `likelihood_scoring` on this
    # dataclass rather than a user setting. One split draw, no paired test.
    #
    # VESM3 is ESM3-derived and non-commercial -- excluded.
    "vesm2-650m": Variant(
        checkpoint=CHECKPOINT,
        dim=DIM,
        label="VESM2 650M",
        overlay="VESM2-650M",
        # wt-marginals, because that is what VESM's released code does: it reads ratios
        # off the logits of one unmasked forward pass. Scoring it with ESM-2's
        # masked-marginal protocol is the measured reason its own LLR underperformed.
        likelihood_scoring="wt",
    ),
    # 150M rather than 35M or 8M as the small option. The price of shrinking is not
    # uniform: under a leakage-aware split, 650M -> 8M costs about 0.02 Spearman on
    # thermostability but up to 0.16 on hard fitness extrapolation -- nearly free when
    # the model interpolates and expensive exactly when the user extrapolates. 150M
    # keeps most of the speed and much less of that risk.
    "esm2-150m": Variant(
        checkpoint="facebook/esm2_t30_150M_UR50D", dim=640, label="ESM-2 150M (faster)"
    ),
    # ESM C 600M, EvolutionaryScale's 2024 encoder as mirrored by Arc Institute's Biohub
    # under MIT for the weights *and* the code. It is on the roster for three measured
    # properties, none of which is accuracy:
    #
    # * **Context.** Rotary positions and no learned position table, so it reads a
    #   2,500-residue protein without raising -- where ESM-2 stops at 1,022 and every
    #   residue past that is silently dropped by anyone who forgets to count. The cap
    #   below is therefore ours, not the architecture's.
    # * **Speed and width.** 1,152 dimensions against ESM-2 650M's 1,280, and its
    #   published throughput is the reason it exists.
    # * **Licence.** MIT, like ESM-2, and unlike ESM3 and everything derived from it --
    #   our runner images are public, which is distribution.
    #
    # **It is not the more accurate model.** On ProteinGym zero-shot, read fairly, ESM C
    # 600M is *behind* ESM-2 650M (0.405 against 0.414). Nothing here has measured the
    # two against each other on our own data yet, and the help text says so.
    #
    # Loaded through native `transformers` as `EsmcModel` -- not the repository's
    # `modeling_esmc_remote.py`, which needs `trust_remote_code=True` and pulls in the
    # `esm` package and its CUDA dependencies. Pinned to a commit anyway: a mirror
    # republished a week later is a different set of weights under the same name, and
    # this one is what was measured.
    #
    # `likelihood_scoring=None` is the deliberate gap. ESM C's own scoring code has not
    # been read, and the one thing this project knows for certain about likelihood
    # protocols is that guessing one costs 0.147 Spearman and reads as a model
    # regression (see `vesm2-650m` above). So the feature refuses instead.
    "esmc-600m": Variant(
        checkpoint="biohub/ESMC-600M",
        dim=1152,
        label="ESM C 600M (faster)",
        likelihood_scoring=None,
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        max_residues=2048,
        revision="fcc9d36f5fff97d9bb69a36e3252de61cc78c968",
    ),
}

#: The variant names, in the order an engine offers them.
PROTEIN_MODELS: tuple[str, ...] = tuple(VARIANTS)
PROTEIN_MODEL_LABELS: tuple[str, ...] = tuple(one.label for one in VARIANTS.values())


def variant(name: str) -> Variant:
    """The named weight set, or a refusal naming the ones that exist."""
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValidationError(
            f"Unknown protein model '{name}'. Available: {', '.join(sorted(VARIANTS))}."
        ) from None


def likelihood_protocol(name: str) -> str:
    """How a log-likelihood ratio is read off this variant, or a refusal.

    One function so that the refusal cannot be forgotten at one of the several places a
    likelihood is reached from -- the LLR feature column, the engine condition that turns
    it on, and the ranking objective that trains against it.

    **A missing protocol is refused, never guessed**, and the cost of the alternative is
    measured: reading VESM2 by ESM-2's masked-marginal protocol made its own LLR look
    0.147 Spearman worse than stock ESM-2's, which read for weeks as a weakness of the
    weights rather than as our bug. Somebody has to read the publisher's scoring code
    first; until then this is a capability we do not have, which is a better thing to
    report than a number computed the wrong way.
    """
    scoring = variant(name).likelihood_scoring
    if scoring is None:
        raise ValidationError(
            f"The likelihood feature is not yet available for {variant(name).label}. How a "
            "log-likelihood ratio should be read off this model depends on how it was "
            "trained, and reading it the wrong way produces plausible numbers that are "
            "quietly wrong, so it is refused rather than guessed. Turn the setting off, "
            "or choose ESM-2 650M, VESM2 650M or ESM-2 150M."
        )
    return scoring


#: Tokens per batch rather than sequences per batch, so one long sequence in a set of
#: short ones cannot blow up memory. Paired with the length sort in `_embed`, which is
#: not cosmetic: unsorted, *raising* the batch size made 650M embedding monotonically
#: slower in published measurement, 118s to 258s.
_TOKEN_BUDGET = 16384


def esm2_650m(sequences: list[str], model: str = DEFAULT_VARIANT) -> np.ndarray:
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
    return _embed(tuple(sequences), model)


@lru_cache(maxsize=6)
def _embed(sequences: tuple[str, ...], model_name: str = DEFAULT_VARIANT) -> np.ndarray:
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
    # The empty case is answered before the import, not after: no sequences means no
    # forward pass, so it must not require the `gpu` extra to be installed. The width is
    # a property of the variant and needs nothing loaded. A test for an empty partition
    # otherwise passes on this machine and fails in CI, which is the same shape as the
    # failure that turned `main` red in 506e864.
    width = variant(model_name).dim
    if not sequences:
        return np.zeros((0, width), dtype=np.float32)

    import torch

    cap = variant(model_name).max_residues
    cleaned, truncated, unrepresentable = _clean(sequences, cap)
    if truncated:
        logger.warning(
            "%d of %d sequences were longer than %d residues and were truncated for "
            "%s; predictions for those rows describe only the first %d residues.",
            truncated,
            len(sequences),
            cap,
            model_name,
            cap,
        )
    if unrepresentable:
        logger.warning(
            "%d of %d sequences contain a residue ESM-2 has no token for (a stop codon, "
            "most often) and were embedded with an unknown-residue token in its place; "
            "those rows are described less precisely than the rest.",
            unrepresentable,
            len(sequences),
        )

    tokenizer, model, device = _loaded(model_name)

    # Longest first, so a batch's padding is set by sequences of similar length and the
    # first batch is the one most likely to run out of memory -- better to fail in the
    # first second than forty minutes in.
    order = sorted(range(len(cleaned)), key=lambda i: len(cleaned[i]), reverse=True)
    out = np.zeros((len(cleaned), width), dtype=np.float32)

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
            if vectors.shape[1] != width:
                raise AssertionError(
                    f"{model_name} returned {vectors.shape[1]}-dimensional embeddings, "
                    f"not the expected {width}."
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


def as_embeddable(residues: str) -> tuple[str, bool]:
    """`(residues with every unembeddable character replaced by X, whether any was)`.

    `X` is ESM-2's own unknown-residue token, which is what an unrepresentable residue
    *is*; `<unk>` is the tokenizer's "I have never seen this string". The difference
    matters for more than tidiness, because `EsmTokenizer._tokenize` is `text.split()`
    and the trie only splits on vocabulary tokens: **a run of two or more unembeddable
    characters collapses into a single `<unk>` token.** Measured on the real tokenizer,
    `MA**V` is five residues and four tokens, and so is `MAJJKV` at six -- it is not
    specific to stop codons. Every residue after such a run then sits one token earlier
    than its position says, so anything that indexes a token by residue position reads a
    neighbour and returns a finite, plausible, wrong number.

    Substituting rather than refusing, because an internal stop is a legitimate DMS
    control that `_clean` deliberately keeps -- see `050fc8d`. One token per residue is
    the invariant; which token it is was already a degraded answer, counted and warned
    about.
    """
    if _EMBEDDABLE.issuperset(residues):
        return residues, False
    return "".join(residue if residue in _EMBEDDABLE else "X" for residue in residues), True


def _clean(
    sequences: tuple[str, ...], max_residues: int = MAX_RESIDUES
) -> tuple[list[str], int, int]:
    """Upper-case, drop whitespace and gaps, truncate, and count both kinds of loss.

    `max_residues` is the chosen variant's, not the module constant's: ESM C reads four
    times as far as ESM-2, and truncating it at ESM-2's ceiling would throw away two
    thirds of a long protein for no reason. The default keeps ESM-2's number for the
    callers that have no variant in hand.

    Returns `(cleaned, truncated, unrepresentable)`. Residue-alphabet *validation* still
    belongs to the dataset layer alongside SMILES canonicalization, and the policy here
    is unchanged: ESM-2's tokenizer maps an unknown character to `<unk>` rather than
    failing, so a stray character degrades one row instead of killing a fit.

    What changed is that it is no longer silent. Truncation was already counted and
    warned about; an `<unk>` substitution was not, so a nonsense variant embedded exactly
    like an ordinary protein and nothing said so. Counting it is the same discipline
    applied to the same class of loss.

    The unembeddable residues are now also replaced by `X` rather than left for the
    tokenizer to turn into `<unk>`, which changes the embedding of exactly the rows this
    function already counts and warns about. It is a fix, not a tidy-up: a run of two or
    more such characters collapses into a *single* `<unk>`, so this function's own
    promise that the mean covers the sequence's residues was false for them -- a 286-mer
    with a `**` run was pooled over 285 tokens. `as_embeddable` has the measurement.
    """
    cleaned: list[str] = []
    truncated = 0
    unrepresentable = 0
    for sequence in sequences:
        residues = "".join(str(sequence).split()).replace("-", "").replace(".", "").upper()
        if len(residues) > max_residues:
            truncated += 1
            residues = residues[:max_residues]
        residues, substituted = as_embeddable(residues)
        unrepresentable += substituted
        cleaned.append(residues)
    return cleaned, truncated, unrepresentable


#: The site block's width: pooled embedding, mutated-site embedding, WT-relative site
#: delta, then the two site scalars (mean relative position, number of mutations).
SITE_DIM = 3 * DIM + 2


@lru_cache(maxsize=2)
def embedding_map(sequences: tuple[str, ...], model: str = DEFAULT_VARIANT) -> dict[str, Any]:
    """`{sequence: its pooled embedding}` for a whole frame, from **one** forward pass.

    This is the "embed once and slice" that `_embed`'s docstring asks for, and the reason
    it belongs here rather than in a caller is what the key has to be. A fit embeds its
    train, validation and test partitions separately -- which costs nothing extra, since
    they are disjoint -- but a protocol run fits the same rows *several times*: the
    optimism gap re-partitions them randomly for a comparison leg, and every split draw
    adds another leg. Those legs differ only in the `split` column, so the frame's
    sequence column is identical across all of them, and keying on it makes the second
    and later legs free. **Measured on the 4,996-row TEM-1 pilot: the first leg costs
    471.9 s and the second 0.0 s**, where before the change each leg paid the full pass.

    Keyed by sequence rather than by position, so duplicate rows are embedded once. The
    values are views into `_embed`'s cached read-only array, not copies, so this costs
    dictionary overhead and not a second 51 MB.
    """
    unique = tuple(dict.fromkeys(sequences))
    vectors = _embed(unique, model)
    return dict(zip(unique, vectors, strict=True))


@lru_cache(maxsize=2)
def site_map(
    sequences: tuple[str, ...], parent: str, model: str = DEFAULT_VARIANT
) -> dict[str, Any]:
    """`{sequence: its site-feature block}` for a whole frame, from one forward pass.

    The same trick as `embedding_map`, for the representation that costs three embeddings
    a row. `parent` is part of the key because the block is scored against it -- a
    different reference is a different feature, not a cache hit.

    The keys are the sequences **as given** and the blocks are computed from their
    normalized forms, which is not a detail: `esm2_with_sites` normalizes before
    `_site_features`, and skipping that here would compare a raw row against a normalized
    parent and read a lowercase sequence as mutated at every position. The two paths have
    to agree, since a fit can reach either.
    """
    unique = tuple(dict.fromkeys(sequences))
    normalized, reference = aligned_parent(list(unique), parent, model)
    blocks = _site_features(tuple(normalized), reference, model)
    return dict(zip(unique, blocks, strict=True))


def site_dim(model: str = DEFAULT_VARIANT) -> int:
    """`SITE_DIM` for a variant: three embeddings wide plus the two site scalars."""
    return 3 * variant(model).dim + 2


def esm2_with_sites(
    sequences: list[str], parent: str | None = None, model: str = DEFAULT_VARIANT
) -> np.ndarray:
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
        return np.zeros((0, site_dim(model)), dtype=np.float32)
    normalized, reference = aligned_parent(sequences, parent, model)
    return _site_features(tuple(normalized), reference, model)


@lru_cache(maxsize=4)
def _site_features(
    sequences: tuple[str, ...], parent: str, model_name: str = DEFAULT_VARIANT
) -> np.ndarray:
    """Cached on the exact input, like `_embed`, and for the same repeated callers.

    Pooling happens here rather than through `esm2_650m` so that the whole feature block
    costs **one** forward pass over the variants. Calling both would embed every sequence
    twice, and embedding is already 323 s of a measured 465 s fit.

    Four entries and not `_embed`'s six, because a row here is three embeddings wide:
    about 154 MB per entry at 10k sequences against that module's 51 MB, so this cache is
    the larger of the two at 615 MB full. Four is enough for the re-use that exists -- one
    fit featurizes train, then validation and test inside `_scored` -- and the comparison
    fit behind the optimism gap re-partitions the same rows, so its tuples miss anyway.
    That miss is item 16's to fix by embedding the frame once and slicing, not this
    cache's to absorb by growing.
    """
    import torch

    width = variant(model_name).dim
    tokenizer, model, device = _loaded(model_name)

    # Mutated positions come from the residues as given, so a stop codon still reads as
    # the mutation it is; the forward pass gets the X-substituted copy, so that every
    # residue is exactly one token and `offset + position` cannot drift. See
    # `as_embeddable`. This is also the counting `_clean` does for the pooled path, which
    # this path does not go through -- without it a nonsense variant would be embedded at
    # an unknown-residue token and nothing would say so.
    sites = [sorted(mutated_positions(sequence, parent)) for sequence in sequences]
    embeddable = [as_embeddable(sequence) for sequence in sequences]
    unrepresentable = sum(substituted for _residues, substituted in embeddable)
    if unrepresentable:
        logger.warning(
            "%d of %d sequences contain a residue ESM-2 has no token for (a stop codon, "
            "most often) and were embedded with an unknown-residue token in its place; "
            "those rows are described less precisely than the rest.",
            unrepresentable,
            len(sequences),
        )

    parent_ids, offset = residue_token_offset(tokenizer, as_embeddable(parent)[0])
    out = np.zeros((len(sequences), 3 * width + 2), dtype=np.float32)

    # All one length, so the budget divides evenly and no length sort is needed -- the
    # same simplification `likelihood._masked_marginals` gets from an aligned series.
    per_batch = max(1, _TOKEN_BUDGET // (len(parent) + 2))
    with torch.no_grad():
        parent_input = parent_ids.unsqueeze(0).to(device)
        parent_hidden = model(
            input_ids=parent_input, attention_mask=torch.ones_like(parent_input)
        ).last_hidden_state[0]
        if parent_hidden.shape[-1] != width:
            raise AssertionError(
                f"{model_name} returned {parent_hidden.shape[-1]}-dimensional "
                f"embeddings, not the expected {width}."
            )
        parent_states = parent_hidden.to(torch.float32).cpu().numpy()

        for start in range(0, len(sequences), per_batch):
            chunk = [residues for residues, _substituted in embeddable[start : start + per_batch]]
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                return_special_tokens_mask=True,
            )
            special = encoded.pop("special_tokens_mask")
            # The indexing invariant, asserted rather than trusted: one token per residue
            # plus two special tokens, on every row. `as_embeddable` guarantees it, so
            # this can only fire if the tokenizer's contract changes under us -- which is
            # worth a crash, because the alternative is a silently shifted read.
            expected = len(chunk) * (len(parent) + 2)
            if int(encoded["attention_mask"].sum().item()) != expected:
                raise AssertionError(
                    f"{variant(model_name).checkpoint} tokenized a batch of "
                    f"{len(chunk)} sequences of "
                    f"{len(parent)} residues into something other than {expected} "
                    f"tokens, so a residue position cannot be indexed by token."
                )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            pooled = _pool(hidden, encoded["attention_mask"] * (1 - special.to(device)))

            # Both tensors off the device once per batch, not once per row. A batch is
            # about 80 MB of float32 and a row's gather is a handful of 1280-wide slices,
            # so the per-row version paid two device syncs each -- ten thousand of them
            # on the TEM-1 pilot -- for arithmetic numpy does on the host.
            pooled_rows = pooled.cpu().numpy()
            states = hidden.to(torch.float32).cpu().numpy()

            for index in range(len(chunk)):
                row = start + index
                out[row, :width] = pooled_rows[index]
                positions = sites[row]
                if not positions:
                    continue
                tokens = [offset + position for position in positions]
                at_site = states[index, tokens]
                out[row, width : 2 * width] = at_site.mean(axis=0)
                out[row, 2 * width : 3 * width] = (at_site - parent_states[tokens]).mean(axis=0)
                out[row, 3 * width] = sum(positions) / len(positions) / len(parent)
                out[row, 3 * width + 1] = len(positions)

    logger.info(
        "Embedded %d sequences with site features against a %d-residue parent.",
        len(sequences),
        len(parent),
    )
    # Handed out of a cache, so make it read-only rather than trust every caller.
    out.flags.writeable = False
    return out


def aligned_parent(
    sequences: list[str], parent: str | None = None, model: str = DEFAULT_VARIANT
) -> tuple[list[str], str]:
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
    # The chosen variant's ceiling, not ESM-2's: a 1,500-residue protein is inside ESM C's
    # context and outside ESM-2's, and refusing it under the wrong number would read as a
    # limit of the data rather than of the model that was picked.
    cap = variant(model).max_residues
    if max(lengths) > cap:
        raise ValidationError(
            f"These sequences are {max(lengths)} residues, beyond {variant(model).label}'s "
            f"{cap}-residue context. Truncating them would move every position past the "
            f"cut, so a position-scored feature cannot be computed for them."
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
            "This protein model's tokenizer does not map residues to tokens "
            "one-for-one, so a "
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


def overlay_state_dict(name: str) -> dict[str, Any]:
    """The partial state dict for an overlay weight set, keyed as its publisher keys it.

    `weights_only=True`, and not as a formality: these are `.pth` pickles from the Hub,
    so loading one without it executes whatever the file says to execute. The checksum in
    `_pretrained.WEIGHT_SETS` is the other half of that trust boundary.
    """
    import torch

    from daikonstudio.infrastructure.engines._pretrained import weights_path
    from daikonstudio.settings import Settings

    path = weights_path(name, Settings().pretrained_weights_dir)
    loaded: dict[str, Any] = torch.load(path, weights_only=True, map_location="cpu")
    return loaded


#: The prefix an `EsmForMaskedLM` state dict puts on the keys of the `EsmModel` inside it.
_MLM_PREFIX = "esm."


def apply_overlay(model: Any, state: dict[str, Any], *, for_masked_lm: bool) -> int:
    """Lay a published partial state dict over `model`; return how many tensors landed.

    **Returning the count is the point, and the caller must check it.** VESM2 publishes an
    `EsmForMaskedLM` state dict -- every key prefixed `esm.` or `lm_head.` -- while the
    embedding path loads an `EsmModel`, whose keys carry neither. `load_state_dict(...,
    strict=False)` is therefore not a tolerant load here, it is a silent no-op: measured
    on `VESM_650M.pth`, 87 of 87 keys match `EsmForMaskedLM`, **0 of 87** match `EsmModel`
    as published, and 81 of 81 match it once the prefix is stripped. A no-op returns stock
    ESM-2 embeddings from an engine the user selected a different model on, with nothing
    anywhere saying so.

    Partial is expected rather than tolerated: `VESM_650M.pth` is 100M parameters against
    the base model's 652M, because it overrides the embeddings, the top encoder layers and
    the head and leaves the rest of the tower alone.
    """
    if not for_masked_lm:
        state = {
            name[len(_MLM_PREFIX) :]: tensor
            for name, tensor in state.items()
            if name.startswith(_MLM_PREFIX)
        }
    known = set(model.state_dict())
    landed = {name: tensor for name, tensor in state.items() if name in known}
    model.load_state_dict(landed, strict=False)
    return len(landed)


def _overlaid(model: Any, chosen: Variant, name: str, *, for_masked_lm: bool) -> Any:
    """`model` with `chosen`'s overlay applied, or a refusal if none of it applied."""
    if chosen.overlay is None:
        return model
    landed = apply_overlay(model, overlay_state_dict(chosen.overlay), for_masked_lm=for_masked_lm)
    if landed == 0:
        raise ValidationError(
            f"None of the weights published for '{name}' matched the model they are "
            f"meant to override, so the fit would silently have used stock "
            f"{chosen.checkpoint}. This is a packaging mismatch rather than anything a "
            f"scientist can fix by changing settings."
        )
    logger.info("Laid %d tensors from %s over %s.", landed, chosen.overlay, chosen.checkpoint)
    return model


@lru_cache(maxsize=1)
def _loaded(name: str = DEFAULT_VARIANT) -> tuple[Any, Any, Any]:
    """Tokenizer, model and device for one variant, built once per process.

    `transformers` caches the base weights on disk itself, so there is nothing for
    `_pretrained.py` to do for those -- that module exists for checksummed URL downloads,
    and ESM-2 comes off the Hub as a first-class architecture. An *overlay* does come
    through `_pretrained`, because it is a `.pth` fetched by URL and a checksum is the
    only thing standing between it and `torch.load`.

    `maxsize=1`, so selecting a different variant evicts the last. Deliberate: each entry
    is 0.6-2.6 GB of weights, and one fit uses one variant.

    No revision pin, unlike `molformer_xl.py`. That pin exists because MoLFormer needs
    `trust_remote_code=True` and therefore executes Python fetched from the Hub, which
    must be the code that was reviewed. ESM-2 executes nothing: the architecture ships
    inside `transformers`, so only the weights come over the wire.
    """
    chosen = variant(name)
    _torch, device = torch_and_device()

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(chosen.checkpoint, revision=chosen.revision)
    model = AutoModel.from_pretrained(chosen.checkpoint, revision=chosen.revision)
    model = _overlaid(model, chosen, name, for_masked_lm=False).to(device)
    model.eval()
    logger.info("Loaded %s (%s) on %s for embedding.", name, chosen.checkpoint, device)
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
