"""ESM-2's log-likelihood ratio -- what the model thinks of a variant before it has
seen a single label.

`embed.py` named this the single highest-value addition to the protein side and deferred
it for one reason: an LLR is a variant scored against a reference sequence, and an upload
carries no reference. `assign_split.consensus` removes that blocker -- the parent of a
single-parent series is its modal residue at each position, which is also exactly the
reference the POSITION holdout already uses, so the feature and the split agree on what
the wild type is.

Worth the forward pass on published measurement: ESM-2 zero-shot reaches ~0.414 Spearman
on ProteinGym with no labels at all, and Kermut's ablations attribute roughly half of its
advantage over a plain embedding head to using this as a Gaussian-process prior mean
rather than to the kernel (-0.036 contiguous and -0.032 modulo when the zero-shot mean is
removed, against -0.008 random). It lands on the hard splits, which is where a
variant-effect model is actually judged.

**Cost is per distinct mutated position, not per variant.** Masked-marginal scoring masks
one site of the *parent* and reads the distribution there, so a 4,000-variant series over
200 mutated positions needs 200 forward passes, not 4,000. That is the whole reason this
is affordable as a feature. The wt-marginal protocol needs exactly one.

**Two protocols, chosen by the weights rather than by the user.** Masked marginals are
ESM-2's own and score better for it; wt-marginals read every position off a single
unmasked pass, which is what VESM's released code does. `Variant.likelihood_scoring`
carries the choice, because it is a property of how a checkpoint was trained -- scoring
VESM2 by masking is the measured reason its own LLR came out behind stock ESM-2's.

Sign convention, stated once because inverting it would flip every downstream correlation
while still producing plausible numbers: **LLR = log p(mutant) - log p(wild type)**, so a
substitution the model rates worse than the residue it replaced is negative.

Same module rule as `embed.py`: **every torch and transformers import lives inside a
function.** `_scoring.py` registers featurizers at import time and the API tier installs
without the `gpu` extra.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np

from daikonstudio.application.data.assign_split import mutated_positions
from daikonstudio.infrastructure.protein.embed import (
    DEFAULT_VARIANT,
    _overlaid,
    aligned_parent,
    likelihood_protocol,
    residue_token_offset,
    torch_and_device,
    variant,
)

logger = logging.getLogger(__name__)

#: The residues a masked-marginal distribution is read for. ESM-2's vocabulary also holds
#: `X`, `B`, `U`, `Z`, `O` and the gap token, but a log-probability for "unknown residue"
#: is not a statement about any particular substitution, so they are left out and the
#: positions that need them are skipped and counted instead -- the same discipline
#: `embed.py` applies to truncation.
_RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


def esm2_llr(
    sequences: list[str], parent: str | None = None, model: str = DEFAULT_VARIANT
) -> np.ndarray:
    """One column: each variant's LLR against the series' parent.

    Which protocol reads it is `model`'s to say, not the caller's -- see the module
    docstring and `Variant.likelihood_scoring`.

    Shape `(len(sequences), 1)`, float32, so it concatenates onto any feature matrix.
    Deterministic: the model runs in eval mode under `no_grad`, and the parent is the
    modal residue per position with a tie broken by residue letter, so re-sorting the
    upload cannot move a score.

    `parent` is the reference the fit recorded, and passing it is what makes this column
    a property of the row rather than of the list the row arrived in -- see
    `embed.aligned_parent`.
    """
    if not sequences:
        return np.zeros((0, 1), dtype=np.float32)

    # Before the alignment check and long before any weights load: a model whose scoring
    # protocol nobody has read yet cannot produce this column at all, and that is a
    # clearer thing to say than a refusal about sequence lengths.
    scoring = likelihood_protocol(model)
    sequences, parent = aligned_parent(sequences, parent, model)
    positions = sorted(
        {position for sequence in sequences for position in mutated_positions(sequence, parent)}
    )
    read = _wt_marginals if scoring == "wt" else _masked_marginals
    return _llr_from_logprobs(sequences, parent, read(parent, positions, model))


def _llr_from_logprobs(
    sequences: list[str], parent: str, logprobs: dict[int, dict[str, float]]
) -> np.ndarray:
    """Sum of `log p(mutant) - log p(wild type)` over the sites a variant mutates.

    A position with no scored distribution, or a residue outside `_RESIDUES`, contributes
    zero -- a ratio of one, i.e. "no evidence". Not "equally good": the alternative is
    inventing a number for a column a Scorecard reports as measured.
    """
    scores = np.zeros((len(sequences), 1), dtype=np.float32)
    unscored = 0
    for row, sequence in enumerate(sequences):
        total = 0.0
        for position in mutated_positions(sequence, parent):
            at_site = logprobs.get(position)
            mutant, wild_type = sequence[position], parent[position]
            if at_site is None or mutant not in at_site or wild_type not in at_site:
                unscored += 1
                continue
            total += at_site[mutant] - at_site[wild_type]
        scores[row, 0] = total
    if unscored:
        logger.warning(
            "%d mutated site(s) had no scorable residue pair and contributed nothing to "
            "their variant's log-likelihood ratio.",
            unscored,
        )
    return scores


def _wt_marginals(
    parent: str, positions: list[int], model_name: str = DEFAULT_VARIANT
) -> dict[int, dict[str, float]]:
    """`{position: {residue: log p(residue | the unmasked parent)}}`, in **one** pass.

    The other protocol in the variant-effect literature, and the one VESM's released code
    uses: run the parent as-is and read the distribution at each position without hiding
    the residue that is already there. Masked marginals score better for ESM-2 itself and
    remain its default -- but a model distilled under this protocol has to be read under
    it, which is the measured reason VESM2's own LLR looked worse than stock ESM-2's when
    both were scored by masking.

    It is also *cheaper*, by a lot: one forward pass for the whole parent rather than one
    per distinct mutated position. The catch that makes it a different quantity rather
    than a free lunch is that the model can see the wild-type residue it is being asked
    about, so `log p(wild type)` is inflated by its own presence.
    """
    if not positions:
        return {}

    import torch

    tokenizer, model, device = _loaded_mlm(model_name)
    residue_ids = {residue: tokenizer.convert_tokens_to_ids(residue) for residue in _RESIDUES}
    ids, offset = residue_token_offset(tokenizer, parent)

    with torch.no_grad():
        batch = ids.unsqueeze(0).to(device)
        logits = model(input_ids=batch, attention_mask=torch.ones_like(batch)).logits
        log_probabilities = torch.log_softmax(logits.float(), dim=-1)[0]

    marginals = {
        position: {
            residue: float(log_probabilities[offset + position, token_id])
            for residue, token_id in residue_ids.items()
            if token_id is not None
        }
        for position in positions
    }
    logger.info(
        "Read %d positions of a %d-residue parent from one unmasked pass.",
        len(positions),
        len(parent),
    )
    return marginals


def _masked_marginals(
    parent: str, positions: list[int], model_name: str = DEFAULT_VARIANT
) -> dict[int, dict[str, float]]:
    """`{position: {residue: log p(residue | parent with that position masked)}}`.

    One forward pass per position, batched. The masked copies are all the same length as
    the parent, so the token budget divides evenly and no length sort is needed -- unlike
    `embed.py`, which sorts because real uploads are ragged.
    """
    if not positions:
        return {}

    import torch

    tokenizer, model, device = _loaded_mlm(model_name)
    mask_id = tokenizer.mask_token_id
    residue_ids = {residue: tokenizer.convert_tokens_to_ids(residue) for residue in _RESIDUES}

    ids, offset = residue_token_offset(tokenizer, parent)

    rows = [ids.clone() for _ in positions]
    for row, position in zip(rows, positions, strict=True):
        row[offset + position] = mask_id

    marginals: dict[int, dict[str, float]] = {}
    # +2 for the special tokens, matching `embed._batches`'s accounting.
    per_batch = max(1, _TOKEN_BUDGET // (len(parent) + 2))
    with torch.no_grad():
        for start in range(0, len(positions), per_batch):
            chunk = positions[start : start + per_batch]
            batch = torch.stack(rows[start : start + per_batch]).to(device)
            logits = model(input_ids=batch, attention_mask=torch.ones_like(batch)).logits
            log_probabilities = torch.log_softmax(logits.float(), dim=-1)
            for index, position in enumerate(chunk):
                at_site = log_probabilities[index, offset + position]
                marginals[position] = {
                    residue: float(at_site[token_id])
                    for residue, token_id in residue_ids.items()
                    if token_id is not None
                }
    logger.info("Scored %d masked positions of a %d-residue parent.", len(positions), len(parent))
    return marginals


#: Mirrors `embed._TOKEN_BUDGET`; imported by value rather than name so that module's
#: private batching constant stays private.
_TOKEN_BUDGET = 16384


@lru_cache(maxsize=1)
def _loaded_mlm(name: str = DEFAULT_VARIANT) -> tuple[Any, Any, Any]:
    """Tokenizer, masked-LM model and device, built once per process.

    `AutoModelForMaskedLM`, not `AutoModel`: the plain class drops `lm_head`, which is
    the only part of the checkpoint that can produce a per-residue distribution. Loading
    it under `AutoModel` yields a model that runs, returns hidden states, and has nothing
    to read a log-probability from.

    The weights are the same 2.6 GB snapshot `embed._loaded` uses, so this is a second
    head on a cached download rather than a second download -- as long as `HF_HOME`
    points where that module points it, which is why the same `setdefault` runs here.
    """
    chosen = variant(name)
    _torch, device = torch_and_device()

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(chosen.checkpoint, revision=chosen.revision)
    model = AutoModelForMaskedLM.from_pretrained(chosen.checkpoint, revision=chosen.revision)
    # `for_masked_lm=True`: a published overlay is keyed for this class, so here the
    # keys land as-is. The embedding path has to strip a prefix; see `apply_overlay`.
    model = _overlaid(model, chosen, name, for_masked_lm=True).to(device)
    model.eval()
    logger.info(
        "Loaded %s (%s) on %s for masked-marginal scoring.", name, chosen.checkpoint, device
    )
    return tokenizer, model, device
