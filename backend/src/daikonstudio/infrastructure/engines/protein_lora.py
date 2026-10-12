"""A LoRA fine-tune of any protein model on the roster, under either of two objectives.

Every other protein engine here reads the pretrained weights **frozen** and puts a cheap
head on top, and `esm2_xgboost.py` records why: on FLIP2's sixteen
engineering-realistic splits a naively fine-tuned protein language model was the best
model on two of them, and the one 2026 sweep that runs frozen, LoRA and full fine-tuning
through the same pipeline has frozen embeddings winning or tying in four of its five
head-to-heads. **So this engine is a capability, not an upgrade**, and the frozen engines
stay the cheaper first thing to try.

What it does add, and both halves are measured by somebody:

* **A small saved model.** LoRA trains rank-8 adapter matrices inside each attention
  projection and leaves the published weights alone, so the artifact is a few megabytes
  where a full fine-tune of ESM-2 650M would be 2.6 GB. That is LoRA's real benefit at
  this scale -- *not* memory, which it barely saves, and not speed. See `_lora.py`.
* **A ranking objective.** The regression objective is the ordinary thing: pool the
  hidden states, put a linear head on them, minimize squared error. The ranking objective
  trains the model's *own likelihood* to order variants correctly, which its authors
  report as worth far more at small training-set sizes than anything LoRA bought in the
  rest of this literature, and as degrading least under a held-out-position split -- this
  platform's hard split. **Nothing on either side of that is measured here yet**, so no
  help text in this file carries a number.

**The ranking objective's provenance, and what in it is ours.** The objective is
Hawkins-Hooker et al., *Likelihood-based fine-tuning of protein language models for
few-shot fitness prediction and design* (bioRxiv 2024.05.28.596156).

**They released no code.** No code-availability statement in any version of the preprint,
nothing in the first author's GitHub, nothing in their lab's. (A third-party
reimplementation exists, `JudeWells/pref_opt_profam`; its author is not an author of the
paper, so it is not evidence about what they did.) So each piece below says where it comes
from, and the pieces that are ours say so:

* **The loss is theirs and is published verbatim**, as the PyTorch listing in Appendix
  A.6: binary cross-entropy with logits on the raw score difference, over all B² ordered
  pairs of a batch, the diagonal zeroed, halved to undo the double counting, and averaged
  over B². `_ranking_loss` reproduces that listing, including its treatment of ties --
  which is a behaviour the paper's Equation 2 does not describe and its code does, and
  which matters on assay data with repeated values. See that function.
* **The score is theirs, from Equation 5 and Appendix A.1**: the sum of log-likelihood
  ratios between the mutant and wild-type residues over the mutated positions, read from
  **one unmasked forward pass of the wild type** -- "for ESM models, we computed the loss
  by scoring all sequences using the wt-marginal strategy". The same function scores
  training and evaluation; their masked and masked-modulo variants are ablations.
* **Ours, because the formula does not state it:** how a token index is derived from a
  residue position (`embed.residue_token_offset`, rather than assuming one special token),
  and what happens to a substitution involving a residue outside the standard twenty
  (skipped and counted -- see `_mutation_rows`).
* **Ours, and not in the paper at all:** the score-to-label line (`_affine`). The
  likelihood ratio *is* their prediction, and a fitted regression head is their losing
  baseline; this platform reports RMSE and r2 in a target's own unit, which a
  log-likelihood is not. It cannot move a rank correlation.
* **Ours, with no published result in either direction:** they fine-tune the whole model
  and this trains LoRA adapters. Also, under ESM-2's module names one adapter lands on the
  masked-LM head's own projection, which their full fine-tune trains anyway.

**The protocol the ranking objective reads a likelihood under is wt-marginal for every
variant**, which is the paper's own choice for ESM models and is separately VESM2's own
protocol. It is *not* ESM-2's preferred masked-marginal protocol: that is a different
quantity and costs one forward pass per mutated position per step rather than one, and it
is not what the published result was produced with. A variant whose scoring protocol
nobody has read at all (`Variant.likelihood_scoring is None`, which is ESM C today) is
refused rather than scored under a guess -- see `_refuse_unrankable`.

The ranking score is cheap in a way worth knowing: it depends on the weights only through
the *parent's* forward pass, so a training step costs one forward of one sequence however
many variants the batch holds. A larger batch is nearly free and gives quadratically more
pairs.

**Every torch, transformers and peft import lives inside a function, never at module
scope.** `registry.py` builds this class at import time and the API tier installs without
the `gpu` extra -- the same rule as `molformer_xl.py` and `protein/embed.py`.
"""

from __future__ import annotations

import io
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from daikonstudio.application.data.assign_split import mutated_positions
from daikonstudio.application.engines.context import PredictContext, TrainContext, TrainResult
from daikonstudio.application.engines.manifest import (
    ConditionSpec,
    ConditionType,
    EngineManifest,
    TaskType,
    validate_conditions,
)
from daikonstudio.domain.data.structure_kind import StructureKind
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.engines import _lora
from daikonstudio.infrastructure.engines._lightning import (
    keep_best_epoch,
    record_epochs,
    save_training_state,
    saved_training_state,
    training_state_scope,
)
from daikonstudio.infrastructure.engines._options import (
    EPOCH_SELECTION,
    POSITIVE_WEIGHTING,
    PROTEIN_MODEL,
    positive_weight,
)
from daikonstudio.infrastructure.engines._scoring import (
    classification_by_column,
    regression_metrics,
    tuned_cutoffs,
)
from daikonstudio.infrastructure.protein.embed import (
    _clean,
    _overlaid,
    _pool,
    aligned_parent,
    as_embeddable,
    residue_token_offset,
    torch_and_device,
    variant,
)
from daikonstudio.infrastructure.protein.likelihood import _RESIDUES

logger = logging.getLogger(__name__)

#: The two objectives. `REGRESSION` is the default and the only one that accepts
#: active/inactive targets, several targets, or sequences that are not variants of one
#: parent.
OBJECTIVE_REGRESSION = "regression"
OBJECTIVE_RANKING = "ranking"

#: Sequences per forward pass when predicting. Small, and deliberately not the training
#: batch size: a sequence here can be two thousand residues where a SMILES is two hundred
#: tokens, and nothing is being accumulated for a backward pass.
_PREDICT_BATCH_SIZE = 8

_MANIFEST = EngineManifest(
    id="protein-lora",
    version="1.0.0",
    name="Protein model, LoRA fine-tuned",
    description=(
        "Fine-tunes a pretrained protein model on your own measurements, training small "
        "adapter matrices inside each attention layer rather than the whole model, so "
        "the saved model is a few megabytes instead of a few gigabytes. Reads the "
        "structure column as an amino-acid sequence. It either predicts the measured "
        "value through an output layer, or trains the model's own likelihood to rank the "
        "variants of one parent protein. Published comparisons put fine-tuning level "
        "with or behind a frozen protein model under a tree or a Gaussian process on "
        "most tasks, so the frozen engines remain the cheaper first choice."
    ),
    tasks=(TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION),
    conditions=(
        PROTEIN_MODEL,
        ConditionSpec(
            key="objective",
            label="What the fit optimizes",
            type=ConditionType.ENUM,
            default=OBJECTIVE_REGRESSION,
            options=(OBJECTIVE_REGRESSION, OBJECTIVE_RANKING),
            option_labels=(
                "Predict the measured value",
                "Rank variants against their parent",
            ),
            help=(
                "Predicting the measured value trains an output layer on top of the "
                "pooled representation, and suits any sequence dataset. Ranking variants "
                "instead trains the protein model's own likelihood to put the variants "
                "of one parent protein in the right order, so it needs substitution "
                "variants of a single sequence, all the same length, and one continuous "
                "measurement; it learns from the order of your measurements rather than "
                "their values. The published work it comes from reports its advantage "
                "over an output layer growing as the number of measurements falls, and "
                "holding up best when the test variants mutate positions the training "
                "variants did not. That work fine-tunes the whole model where this "
                "engine trains adapters, and we have not measured either objective on "
                "any dataset yet, so treat the choice as open."
            ),
        ),
        ConditionSpec(
            key="epochs",
            label="Training epochs",
            type=ConditionType.INTEGER,
            default=10,
            minimum=1,
            maximum=100,
            help="Number of passes over the training sequences. A pretrained protein "
            "model adapts quickly; longer training risks degrading the pretrained "
            "representation.",
        ),
        ConditionSpec(
            key="batch_size",
            label="Batch size",
            type=ConditionType.INTEGER,
            default=16,
            minimum=2,
            maximum=128,
            help="Number of sequences per gradient update. Reduce it if training runs "
            "out of GPU memory. The ranking objective compares every pair within a "
            "batch, so it has nothing to learn from a batch of one and improves with "
            "larger batches at almost no extra cost.",
        ),
        ConditionSpec(
            key="learning_rate",
            label="Learning rate",
            type=ConditionType.NUMBER,
            default=1e-4,
            minimum=1e-6,
            maximum=1e-2,
            help="Step size of each weight update. Adapters tolerate a larger value "
            "than a full fine-tune does, because the published weights themselves are "
            "not moving; about 10⁻⁴ to 10⁻³ is the usual range. A published sweep "
            "found that this setting matters more to a LoRA fit than the choice of LoRA "
            "itself, so it is worth trying more than one value.",
        ),
        POSITIVE_WEIGHTING,
        EPOCH_SELECTION,
    ),
    lane="gpu",
    is_baseline=False,
    supports_multitask=True,
    structure_kinds=(StructureKind.SEQUENCE,),
)


def _load_backbone(model_name: str, *, ranking: bool) -> tuple[Any, Any]:
    """`(tokenizer, the LoRA-wrapped model)` for one variant, left on the CPU.

    Left on the CPU deliberately: Lightning's Trainer moves the module it is given, and a
    model moved here and again there is two copies of 2.6 GB in flight.

    `AutoModelForMaskedLM` for the ranking objective and `AutoModel` for the regression
    one, because they need different halves of the checkpoint: a log-likelihood needs the
    `lm_head` that `AutoModel` silently drops, and a pooled representation does not.

    Gradient checkpointing is on, with `use_reentrant=False`, and that flag is
    load-bearing rather than tidy. Reentrant checkpointing decides whether a block's
    output needs a backward pass by looking at its *inputs*, and here the inputs are token
    ids coming out of frozen embeddings -- so every adapter inside the block would receive
    no gradient at all and the fit would run to completion having learned nothing. The
    non-reentrant implementation tracks the parameters too.

    `_lora.attach` refuses when none of the variant's `target_modules` matched, which is
    the failure worth guarding: peft would otherwise return the base model under a new
    name with only the head trainable.
    """
    chosen = variant(model_name)
    # Not for the device, which Lightning decides: for the `HF_HOME` this deployment
    # keeps its weights in, the lowered-precision refusal, and the one error message that
    # explains a runner without the gpu extra.
    torch_and_device()

    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(chosen.checkpoint, revision=chosen.revision)
    loader = AutoModelForMaskedLM if ranking else AutoModel
    model = loader.from_pretrained(chosen.checkpoint, revision=chosen.revision)
    model = _overlaid(model, chosen, model_name, for_masked_lm=ranking)
    # ESM-2's `AutoModel` adds a pooler that is **not in the checkpoint** -- it is
    # randomly initialised -- and this engine never reads it, pooling the hidden states
    # with `embed._pool` instead. Dropped before the adapters go on, because otherwise
    # peft attaches to `pooler.dense`, those adapters can never receive a gradient, and
    # they are still written into the artifact. `EsmModel.forward` supports a `None`
    # pooler explicitly; the masked-LM class already builds without one, and ESM C has
    # none at all.
    if getattr(model, "pooler", None) is not None:
        model.pooler = None
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    attached = _lora.attach(model, chosen.target_modules)
    logger.info(
        "Attached LoRA to %s (%s) for the %s objective.",
        model_name,
        chosen.checkpoint,
        OBJECTIVE_RANKING if ranking else OBJECTIVE_REGRESSION,
    )
    return tokenizer, attached


# --- the regression objective --------------------------------------------------------


def _collate(tokenizer: Any) -> Any:
    """Tokenize per batch, padding to the longest sequence in the batch rather than in
    the dataset -- the same saving `molformer_xl._collate` takes, and a larger one here
    because a protein is an order of magnitude longer than a SMILES.

    Returns `(input_ids, attention_mask, residue_mask, targets)`. The third is the second
    with the start and end tokens removed, which is what `_pool` means over: a short
    sequence is otherwise two tokens of punctuation in a mean over a handful of residues.

    The sequences arrive already cleaned and truncated (see `_cleaned`), so nothing here
    can silently shorten one.
    """
    import torch

    def collate(batch: Sequence[tuple[str, Sequence[float]]]) -> tuple[Any, Any, Any, Any]:
        targets = torch.tensor([list(item[1]) for item in batch], dtype=torch.float32)
        encoded = tokenizer(
            [item[0] for item in batch],
            return_tensors="pt",
            padding=True,
            return_special_tokens_mask=True,
        )
        special = encoded.pop("special_tokens_mask")
        attention = encoded["attention_mask"]
        return encoded["input_ids"], attention, attention * (1 - special), targets

    return collate


def _build_regression_module(
    *,
    backbone: Any,
    dim: int,
    n_targets: int,
    learning_rate: float,
    is_classification: bool,
    pos_weight: list[float] | None = None,
) -> Any:
    """The LightningModule for the regression objective, modelled on
    `molformer_xl._build_module`. Defined inside a function so lightning imports lazily.

    The head is a plain `torch.nn.Linear` outside the peft wrapper. It has to be trained
    in full -- it is randomly initialised, so a low-rank correction to it would be a
    low-rank correction to noise -- and keeping it outside means the artifact can hold it
    as an ordinary state dict rather than as a peft `modules_to_save` entry.
    """
    import torch
    from lightning import pytorch as lightning
    from torchmetrics.classification import (
        BinaryAUROC,
        BinaryAveragePrecision,
        BinaryMatthewsCorrCoef,
    )
    from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

    # Logged under the keys `_lightning.SCORE_METRICS` reads, so the run page's live
    # charts work for this engine exactly as they do for chemprop and MoLFormer.
    score_keys = (
        {"roc": "val/roc", "prc": "val/prc", "mcc": "val/binary-mcc"}
        if is_classification
        else {"rmse": "val/rmse", "mae": "val/mae", "r2": "val/r2"}
    )

    class _ProteinLoraModule(lightning.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.head = torch.nn.Linear(dim, n_targets)
            # Any: mypy reads a ModuleDict's values as Tensor | Module, not as metrics.
            self._scores: Any = torch.nn.ModuleDict(
                {
                    "roc": BinaryAUROC(),
                    "prc": BinaryAveragePrecision(),
                    "mcc": BinaryMatthewsCorrCoef(),
                }
                if is_classification
                else {
                    "rmse": MeanSquaredError(squared=False),
                    "mae": MeanAbsoluteError(),
                    "r2": R2Score(),
                }
            )
            self._loss = (
                torch.nn.BCEWithLogitsLoss(
                    pos_weight=None if pos_weight is None else torch.tensor(pos_weight)
                )
                if is_classification
                else torch.nn.MSELoss()
            )

        def forward(self, input_ids: Any, attention_mask: Any, residue_mask: Any) -> Any:
            hidden = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            pooled = _pool(hidden, residue_mask)
            if pooled.shape[-1] != dim:
                raise AssertionError(
                    f"This protein model returned {pooled.shape[-1]}-dimensional "
                    f"embeddings, not the expected {dim}."
                )
            return self.head(pooled.to(self.head.weight.dtype))

        def training_step(self, batch: Any, _index: int) -> Any:
            *inputs, targets = batch
            loss = self._loss(self(*inputs), targets)
            self.log("train_loss", loss, batch_size=len(targets), on_epoch=True)
            return loss

        def validation_step(self, batch: Any, _index: int) -> Any:
            *inputs, targets = batch
            logits = self(*inputs)
            loss = self._loss(logits, targets)
            self.log("val_loss", loss, batch_size=len(targets))
            predicted = torch.sigmoid(logits) if is_classification else logits
            truth = targets.int() if is_classification else targets
            for metric in self._scores.values():
                metric.update(predicted.flatten(), truth.flatten())
            return loss

        def on_validation_epoch_end(self) -> None:
            for name, metric in self._scores.items():
                self.log(score_keys[name], metric.compute())
                metric.reset()

        def predict_step(self, batch: Any, _index: int) -> Any:
            *inputs, _targets = batch
            return self(*inputs)

        def configure_optimizers(self) -> Any:
            # Only the adapters and the head: handing AdamW the frozen tower would
            # allocate optimizer state for parameters that never receive a gradient.
            return torch.optim.AdamW(
                [p for p in self.parameters() if p.requires_grad], lr=learning_rate
            )

    return _ProteinLoraModule()


def _validation_scores(module: Any, batch: Any) -> tuple[Any, Any]:
    """A validation batch's logits and labels, each (rows, targets): what
    `keep_best_epoch` scores a classification epoch by. Logits rank as probabilities do."""
    *inputs, targets = batch
    return module(*inputs), targets


# --- the ranking objective -----------------------------------------------------------


def _mutation_rows(
    sequences: list[str], parent: str, offset: int, residue_ids: dict[str, int]
) -> tuple[list[tuple[tuple[int, int, int], ...]], int]:
    """Each sequence as `((token index, mutant token, wild-type token), ...)`.

    The whole input a ranking score needs, computed once from strings rather than per
    step: the score reads the parent's log-probabilities at these token indices, so a
    variant never goes through the model at all.

    A site whose mutant or wild-type residue is outside the twenty standard ones -- a stop
    codon, most often -- is skipped and counted, exactly as `likelihood._llr_from_logprobs`
    skips it. Contributing nothing is a ratio of one, i.e. "no evidence"; the alternative
    is inventing a log-probability for "unknown residue" and calling it a measurement.
    """
    rows: list[tuple[tuple[int, int, int], ...]] = []
    skipped = 0
    for sequence in sequences:
        row: list[tuple[int, int, int]] = []
        for position in sorted(mutated_positions(sequence, parent)):
            mutant, wild_type = sequence[position], parent[position]
            if mutant not in residue_ids or wild_type not in residue_ids:
                skipped += 1
                continue
            row.append((offset + position, residue_ids[mutant], residue_ids[wild_type]))
        rows.append(tuple(row))
    return rows, skipped


def _ranking_collate(batch: Sequence[tuple[Sequence[tuple[int, int, int]], float]]) -> Any:
    """`(token index, mutant, wild type, mask, y)`, each padded to the batch's widest row.

    Padded rather than ragged because the score is one gather over the parent's
    log-probability matrix; a zero mask row contributes zero to the sum, and the wild type
    itself -- zero mutations -- correctly scores zero.
    """
    import torch

    width = max(1, max(len(mutations) for mutations, _y in batch))
    index = torch.zeros((len(batch), width), dtype=torch.long)
    mutant = torch.zeros_like(index)
    wild_type = torch.zeros_like(index)
    mask = torch.zeros((len(batch), width), dtype=torch.float32)
    for row, (mutations, _y) in enumerate(batch):
        for column, (token, new, old) in enumerate(mutations):
            index[row, column] = token
            mutant[row, column] = new
            wild_type[row, column] = old
            mask[row, column] = 1.0
    return (
        index,
        mutant,
        wild_type,
        mask,
        torch.tensor([y for _m, y in batch], dtype=torch.float32),
    )


def _ranking_loss(scores: Any, y: Any) -> Any:
    """Bradley-Terry over every pair in the batch, as the paper's Appendix A.6 listing.

    `P(a ranks above b) = sigmoid(s_a - s_b)`, so this is binary cross-entropy with
    logits on the raw score difference against the indicator `y_a > y_b`, over all B²
    ordered pairs, with the diagonal zeroed, halved, and averaged over B².

    Three details are the listing's rather than the obvious thing, and each changes a
    number or a behaviour:

    * **The halving.** Every unordered pair appears twice -- once as target 1 and once,
      transposed, as target 0 -- and the two terms are numerically identical. The factor
      corrects that double count.
    * **The normalizer is B², diagonal included**, not the number of orderable pairs. So
      a batch in which few pairs are comparable produces a proportionally smaller
      gradient. It is a constant factor that could be absorbed into the learning rate,
      and it is not ours to absorb: it is what the published runs were trained under.
    * **Ties are not ignored, they are pushed together.** For `y_a == y_b` the indicator
      is false in *both* directions, so the pair contributes
      `softplus(d) + softplus(-d)`, which is minimized at `d = 0`. The paper's Equation 2
      describes only the `y_a > y_b` sum, and its code does this; on assay data with
      repeated values -- which is most dose-response data, and all of it once it is
      rounded -- the difference is real, so the listing wins. Dropping the ties instead
      is a one-word change (`above` from a `>` on both halves) and would be a deviation
      from the only released arithmetic there is.

    Zero for a batch of one row: the single entry is the diagonal, which is zeroed.
    """
    import torch

    difference = scores.unsqueeze(1) - scores.unsqueeze(0)
    above = (y.unsqueeze(1) > y.unsqueeze(0)).to(difference.dtype)
    pairwise = torch.nn.functional.binary_cross_entropy_with_logits(
        difference, above, reduction="none"
    )
    off_diagonal = 1.0 - torch.eye(len(y), dtype=pairwise.dtype, device=pairwise.device)
    return 0.5 * (pairwise * off_diagonal).mean()


def _build_ranking_module(*, backbone: Any, parent_ids: Any, learning_rate: float) -> Any:
    """The LightningModule for the ranking objective.

    One forward pass of the parent per step, whatever the batch size, because the
    wt-marginal score is a function of the parent's logits alone. The parent's ids are a
    buffer rather than a plain tensor so that Lightning moves them to the device with the
    rest of the module, and so that a saved training state restores them.
    """
    import torch
    from lightning import pytorch as lightning

    class _RankingModule(lightning.LightningModule):
        # Annotated because `register_buffer` types its attribute `Tensor | Module`, and
        # every use of it here is a tensor.
        parent_ids: Any

        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.register_buffer("parent_ids", parent_ids.unsqueeze(0))

        def forward(self, index: Any, mutant: Any, wild_type: Any, mask: Any) -> Any:
            logits = self.backbone(
                input_ids=self.parent_ids,
                attention_mask=torch.ones_like(self.parent_ids),
            ).logits
            # float() before the softmax: a half-precision log-softmax over a 33-token
            # vocabulary loses the differences this score is made of.
            log_probabilities = torch.log_softmax(logits[0].float(), dim=-1)
            ratios = log_probabilities[index, mutant] - log_probabilities[index, wild_type]
            return (ratios * mask).sum(dim=1)

        def training_step(self, batch: Any, _index: int) -> Any:
            *inputs, y = batch
            loss = _ranking_loss(self(*inputs), y)
            self.log("train_loss", loss, batch_size=len(y), on_epoch=True)
            return loss

        def validation_step(self, batch: Any, _index: int) -> Any:
            *inputs, y = batch
            loss = _ranking_loss(self(*inputs), y)
            self.log("val_loss", loss, batch_size=len(y))
            return loss

        def predict_step(self, batch: Any, _index: int) -> Any:
            *inputs, _y = batch
            # (rows, 1), so the scoring path downstream is shaped like the other
            # objective's single-target output.
            return self(*inputs).unsqueeze(1)

        def configure_optimizers(self) -> Any:
            return torch.optim.AdamW(
                [p for p in self.parameters() if p.requires_grad], lr=learning_rate
            )

    return _RankingModule()


def _ranking_parent(sequences: list[str], model_name: str) -> tuple[list[str], str]:
    """`(normalized sequences, their parent)`, or a refusal naming what is wrong.

    The parent is the consensus of the whole frame -- `assign_split.consensus`, through
    `embed.aligned_parent`, so that this engine, the POSITION holdout, the LLR feature
    column and the site featurizer all agree on what the wild type is. Four definitions of
    "the parent" would each produce plausible numbers and disagree about which residue a
    variant replaced.

    The length check comes first and says what *this objective* needs, because
    `aligned_parent`'s own refusal is written for a feature column and a scientist
    reaching it here would not know which setting to change. Both happen before any
    weights load, so a dataset of the wrong shape is refused in a second rather than
    after a 2.6 GB download.
    """
    normalized = ["".join(str(sequence).split()).upper() for sequence in sequences]
    lengths = {len(sequence) for sequence in normalized}
    if len(lengths) > 1:
        raise ValidationError(
            f"Ranking needs substitution variants of one parent sequence, so every "
            f"sequence must be the same length -- these range from {min(lengths)} to "
            f"{max(lengths)} residues. An insertion or a deletion changes the length, and "
            "a variant cannot be scored against a parent it is not aligned to. Choose "
            "'Predict the measured value' instead, which places no such requirement on "
            "the sequences."
        )
    # This also refuses a series longer than the chosen model reads, where truncating
    # would move every position past the cut.
    return aligned_parent(normalized, None, model_name)


def _affine(scores: Any, y: Any) -> list[float]:
    """`[a, b]` of the least-squares line `y ~= a*s + b` over the training rows.

    **Ours, and nowhere in the paper.** For its authors the likelihood ratio *is* the
    prediction and a fitted regression head is the baseline they beat; they report rank
    correlation, which a monotone map cannot move. This platform reports RMSE and r2 in
    the target's own unit on every Scorecard, and a log-likelihood ratio is not that unit
    -- without this line a perfectly ranked test partition would show an RMSE of about ten
    log units and an r2 far below zero, which reads as a broken model.

    Fitted on the training rows only: it is a parameter of the model, and fitting it on
    anything the model is scored against would be fitting on the test set.

    A rank-deficient design -- every score identical -- gives lstsq's minimum-norm answer,
    a zero slope and the mean as the intercept. That is the right degenerate behaviour for
    a model that distinguishes nothing, and needs no guard of its own; it is the same
    answer `esm2_gp` relies on for its prior mean.
    """
    import numpy as np

    design = np.column_stack([np.asarray(scores, dtype=float), np.ones(len(scores))])
    slope, intercept = np.linalg.lstsq(design, np.asarray(y, dtype=float), rcond=None)[0]
    return [float(slope), float(intercept)]


# --- shared plumbing -----------------------------------------------------------------


def _loader(
    examples: Sequence[Any],
    *,
    collate: Any,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
) -> Any:
    """A DataLoader over already-prepared examples. A plain list is a map-style dataset,
    so there is no Dataset subclass here to re-implement `__len__`."""
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(
        examples,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        generator=None if seed is None else torch.Generator().manual_seed(seed),
    )


def _outputs(trainer: Any, module: Any, loader: Any) -> Any:
    """The module's raw outputs over `loader`, shaped (rows, n_tasks).

    `trainer.predict` rather than calling the module directly, so that eval mode and
    `no_grad` are applied the one way, and so scoring during training and `predict` can
    never drift into running the model differently."""
    import torch

    batches: Any = trainer.predict(module, loader)
    return torch.cat(batches).float().cpu().numpy()


def _to_values(
    outputs: Any,
    *,
    is_classification: bool,
    target_mean: Any,
    target_std: Any,
    affine: list[float] | None,
) -> Any:
    """Raw model outputs to the numbers a scientist sees, for both objectives.

    One function, called by the training-time scoring and by `predict`, so a Scorecard
    cannot describe a different transform from the one production applies."""
    import numpy as np

    if affine is not None:
        slope, intercept = affine
        return np.asarray(outputs) * slope + intercept
    if is_classification:
        return 1.0 / (1.0 + np.exp(-np.asarray(outputs)))
    return np.asarray(outputs) * np.asarray(target_std) + np.asarray(target_mean)


def _standardize(targets: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    """Per-target mean and standard deviation of the training values, as
    `molformer_xl._standardize` does it and for the same reason: squared error against raw
    assay values scales the gradients by whatever unit the column happens to use.

    A zero deviation (every training value identical) would divide by zero; 1.0 leaves the
    values alone, which is the honest answer for a constant column.
    """
    import numpy as np

    array = np.asarray(targets, dtype=float)
    deviation = array.std(axis=0)
    return array.mean(axis=0).tolist(), np.where(deviation > 0.0, deviation, 1.0).tolist()


def _cleaned(sequences: list[str], max_residues: int) -> list[str]:
    """Upper-cased, de-gapped, truncated at the variant's own ceiling, and **counted**.

    Once per fit rather than once per batch, so the warning is issued once -- and so the
    truncation that a Scorecard's numbers were computed under is recorded in the log
    beside the fit rather than scattered through its epochs.
    """
    prepared, truncated, unrepresentable = _clean(tuple(sequences), max_residues)
    if truncated:
        logger.warning(
            "%d of %d sequences were longer than the %d residues this protein model "
            "reads and were truncated; predictions for those rows describe only the "
            "first %d residues.",
            truncated,
            len(sequences),
            max_residues,
            max_residues,
        )
    if unrepresentable:
        logger.warning(
            "%d of %d sequences contain a residue this protein model has no token for (a "
            "stop codon, most often) and were read with an unknown-residue token in its "
            "place; those rows are described less precisely than the rest.",
            unrepresentable,
            len(sequences),
        )
    return prepared


def _refuse_unmeasured(ctx: TrainContext) -> None:
    """Refuse a dataset with blank targets, before a multi-gigabyte checkpoint loads.

    This engine declares `supports_multitask`, so `FanOut` does not split its context and
    does not drop unmeasured rows, and neither loss below masks anything -- a null would
    be trained on as a number, with no exception and no visible symptom. The same refusal
    `molformer_xl` makes, for the same reason.
    """
    for column in ctx.targets:
        missing = int(ctx.frame[column].null_count())
        if missing:
            raise ValidationError(
                f"{_MANIFEST.name} does not support datasets with unmeasured targets, "
                f"and '{column}' is blank for {missing:,} of {ctx.frame.height:,} rows. "
                "Use a dataset where every row is measured, or train one target at a "
                "time."
            )


def _refuse_unrankable(ctx: TrainContext, *, model_name: str, is_classification: bool) -> None:
    """What the ranking objective cannot do: two shapes of dataset, and one weight set.

    Every refusal here is about the *objective* rather than the data or the model -- the
    same dataset and the same weights train perfectly well under the other objective, and
    each message says so.

    The weight-set refusal reads the same field `embed.likelihood_protocol` gates the LLR
    feature column on, for the same reason and with its own wording: a scoring protocol
    nobody has read is refused rather than guessed, because reading VESM2 the wrong way
    cost 0.147 Spearman and read for weeks as a weakness of the weights. The two call
    sites need different messages -- one is a checkbox, this is a training objective --
    but the fact they test lives on the dataclass.
    """
    chosen = variant(model_name)
    if chosen.likelihood_scoring is None:
        raise ValidationError(
            f"Ranking variants trains {chosen.label}'s own likelihood, and how a "
            "likelihood should be read off this particular model has not been "
            "established yet. Reading it the wrong way produces plausible numbers that "
            "are quietly wrong, so it is refused rather than guessed. Choose 'Predict "
            "the measured value', which uses this model's representation rather than "
            "its likelihood, or rank with ESM-2 650M, VESM2 650M or ESM-2 150M."
        )
    if is_classification:
        raise ValidationError(
            "Ranking variants against their parent needs a continuous measurement: it "
            "learns to put the variants in order, and an active/inactive label gives it "
            "only two groups to order. Choose 'Predict the measured value' for this "
            "dataset."
        )
    if len(ctx.targets) > 1:
        raise ValidationError(
            f"Ranking variants against their parent produces one score per sequence, so "
            f"it fits one target at a time, and this dataset has "
            f"{len(ctx.targets)} ({', '.join(ctx.targets)}). Choose 'Predict the "
            "measured value', which learns every target in one model."
        )


class ProteinLora:
    @staticmethod
    def manifest() -> EngineManifest:
        return _MANIFEST

    def train(self, ctx: TrainContext) -> TrainResult:
        # Before anything else, including loading 2.6 GB of weights: these are decisions
        # about the shape of the data, not failed fits.
        _refuse_unmeasured(ctx)

        conditions: dict[str, Any] = validate_conditions(_MANIFEST, ctx.conditions)
        model_name = str(conditions["protein_model"])
        objective = str(conditions["objective"])
        epochs = int(conditions["epochs"])
        batch_size = int(conditions["batch_size"])
        learning_rate = float(conditions["learning_rate"])
        weighting = str(conditions["positive_weighting"])
        selection = str(conditions["epoch_selection"])
        columns = ctx.target_columns
        is_classification = ctx.task is TaskType.BINARY_CLASSIFICATION
        ranking = objective == OBJECTIVE_RANKING
        structures = ctx.frame[ctx.structure_column].to_list()
        # The whole frame's column, not the training partition's, for the reason
        # `_scoring.esm2_featurizer` settles the parent there: a small test partition can
        # elect a different modal residue, which is a train/test mismatch no metric would
        # attribute to its cause.
        # Empty unless the ranking objective is chosen, which is the only one that has
        # a reference sequence at all.
        parent = ""
        normalized: list[str] = []
        if ranking:
            _refuse_unrankable(ctx, model_name=model_name, is_classification=is_classification)
            normalized, parent = _ranking_parent(structures, model_name)

        # Before `import torch`: this is the one that refuses, with a cause a human can
        # act on, on a runner that has no gpu extra -- and it is what points the Hub
        # cache at this deployment's weights directory.
        torch, _device = torch_and_device()

        from lightning import pytorch as lightning
        from lightning.pytorch.callbacks import LambdaCallback

        chosen = variant(model_name)
        lightning.seed_everything(ctx.seed, workers=True)

        ctx.report(0.0, f"Loading {chosen.label}")
        tokenizer, backbone = _load_backbone(model_name, ranking=ranking)

        train_rows = ctx.frame.filter(pl.col("split") == "train")
        validation_rows = ctx.frame.filter(pl.col("split") == "validation")
        test_rows = ctx.frame.filter(pl.col("split") == "test")
        target_mean, target_std = [0.0] * len(columns), [1.0] * len(columns)
        module: Any
        collate: Any
        examples: Any

        if ranking:
            # The X-substituted parent goes through the model, so that every residue is
            # exactly one token and `offset + position` cannot drift -- see
            # `embed.as_embeddable`, where a run of two unembeddable characters collapsing
            # into a single token is measured.
            parent_ids, offset = residue_token_offset(tokenizer, as_embeddable(parent)[0])
            residue_ids = {
                residue: token
                for residue in _RESIDUES
                if (token := tokenizer.convert_tokens_to_ids(residue)) is not None
            }
            mutations, skipped = _mutation_rows(normalized, parent, offset, residue_ids)
            if skipped:
                logger.warning(
                    "%d mutated site(s) had no scorable residue pair and contribute "
                    "nothing to their variant's score.",
                    skipped,
                )
            by_sequence = dict(zip(normalized, mutations, strict=True))
            module = _build_ranking_module(
                backbone=backbone, parent_ids=parent_ids, learning_rate=learning_rate
            )
            collate = _ranking_collate

            def ranking_examples(rows_frame: pl.DataFrame) -> list[Any]:
                return [
                    (by_sequence["".join(str(sequence).split()).upper()], float(value))
                    for sequence, value in zip(
                        rows_frame[ctx.structure_column].to_list(),
                        rows_frame[ctx.target_column].to_list(),
                        strict=True,
                    )
                ]

            examples = ranking_examples
        else:
            prepared = dict(
                zip(structures, _cleaned(structures, chosen.max_residues), strict=True)
            )
            if not is_classification:
                target_mean, target_std = _standardize(train_rows.select(columns).rows())
            pos_weight = (
                [
                    positive_weight(train_rows[column].to_numpy(), weighting) or 1.0
                    for column in columns
                ]
                if is_classification and weighting != "none"
                else None
            )
            module = _build_regression_module(
                backbone=backbone,
                dim=chosen.dim,
                n_targets=len(columns),
                learning_rate=learning_rate,
                is_classification=is_classification,
                pos_weight=pos_weight,
            )
            collate = _collate(tokenizer)

            def pooled_examples(rows_frame: pl.DataFrame) -> list[Any]:
                return [
                    (
                        prepared[sequence],
                        tuple(
                            (float(value) - mean) / std
                            for value, mean, std in zip(row, target_mean, target_std, strict=True)
                        ),
                    )
                    for sequence, row in zip(
                        rows_frame[ctx.structure_column].to_list(),
                        rows_frame.select(columns).rows(),
                        strict=True,
                    )
                ]

            examples = pooled_examples

        train_loader = _loader(
            examples(train_rows),
            collate=collate,
            batch_size=batch_size,
            shuffle=True,
            seed=ctx.seed,
        )
        # Not shuffled: it would not corrupt the loss, but it would make the number
        # depend on the seed for no reason.
        validation_loader = (
            _loader(examples(validation_rows), collate=collate, batch_size=batch_size)
            if validation_rows.height > 0
            else None
        )

        def _report_epoch(trainer: Any, _module: Any) -> None:
            # The interruption point: `report` may raise RunInterrupted, which propagates
            # out of `fit` and is the only way to stop a fit already running on the
            # worker thread. Naming the device is not decoration -- see chemprop_dmpnn.
            ctx.report(
                (trainer.current_epoch + 1) / epochs,
                f"Training {chosen.label} adapters on {trainer.strategy.root_device}",
            )

        keep_best = (
            keep_best_epoch(by=selection, predict=_validation_scores)
            if is_classification
            else keep_best_epoch()
        )
        callbacks: list[Any] = [
            # Before the reporter, which may raise to stop the fit.
            record_epochs(
                ctx.record_epoch,
                epochs=epochs,
                # The ranking objective logs no error metrics -- they would be in
                # log-likelihood units, which are not the target's.
                unit_scale=float(target_std[0])
                if not ranking and not is_classification and len(columns) == 1
                else None,
                kept=keep_best if validation_loader is not None else None,
            ),
            LambdaCallback(on_train_epoch_end=_report_epoch),
        ]
        if validation_loader is not None:
            callbacks.append(keep_best)

        # The scratch directory holds the training state a save writes and a resume
        # reads; it only has to outlive `fit`.
        with tempfile.TemporaryDirectory() as state_dir:
            state = training_state_scope(ctx.checkpoints, "transformers", "peft")
            resume_from = saved_training_state(state, Path(state_dir), module)
            # After the reporter: it raises in `on_train_epoch_end` when the time limit
            # stops the fit, and the save then happens in `on_exception`.
            if state is not None:
                callbacks.append(save_training_state(state, Path(state_dir)))
            if resume_from is not None:

                def _report_resume(trainer: Any, _module: Any) -> None:
                    ctx.report(
                        trainer.current_epoch / epochs,
                        f"Resuming {_MANIFEST.name} from epoch "
                        f"{trainer.current_epoch + 1} of {epochs}",
                    )

                callbacks.append(LambdaCallback(on_train_start=_report_resume))

            trainer = lightning.Trainer(
                accelerator="auto",
                devices=1,
                max_epochs=epochs,
                enable_checkpointing=False,
                logger=False,
                enable_progress_bar=False,
                callbacks=callbacks,
            )
            trainer.fit(
                module, train_loader, validation_loader, ckpt_path=resume_from, weights_only=False
            )

        # Restore the selected epoch before anything is scored or saved, so the numbers on
        # the Scorecard and the weights in the artifact are one model.
        if keep_best.best_state is not None:
            module.load_state_dict(keep_best.best_state)

        def infer(rows: pl.DataFrame) -> Any:
            loader = _loader(examples(rows), collate=collate, batch_size=_PREDICT_BATCH_SIZE)
            return _outputs(trainer, module, loader)

        # The score-to-label line is fitted on the training rows only -- it is a parameter
        # of the model, and fitting it on anything the model is scored against would be
        # fitting on the test set.
        affine = (
            _affine(infer(train_rows)[:, 0], train_rows[ctx.target_column].to_numpy())
            if ranking
            else None
        )

        def values(rows: pl.DataFrame) -> Any:
            return _to_values(
                infer(rows),
                is_classification=is_classification,
                target_mean=target_mean,
                target_std=target_std,
                affine=affine,
            )

        # The validation partition is inferred once: it picks the cutoffs, then is scored
        # at them. Test is scored at those cutoffs and never sees them chosen.
        validation_values = values(validation_rows) if validation_rows.height > 0 else None
        cutoffs: dict[str, float] = {}
        if ctx.tune_cutoffs and is_classification and validation_values is not None:
            cutoffs = tuned_cutoffs(columns, validation_rows, validation_values)

        def score(rows: pl.DataFrame, predicted: Any) -> dict[str, dict[str, float]]:
            if is_classification:
                return classification_by_column(columns, rows, predicted, cutoffs, train_rows)
            return {
                column: regression_metrics(rows[column].to_numpy(), predicted[:, index])
                for index, column in enumerate(columns)
            }

        metrics = score(test_rows, values(test_rows))
        validation_metrics = (
            None if validation_values is None else score(validation_rows, validation_values)
        )

        # Only what trained: the adapters, and the output layer for the regression
        # objective. The published weights are rebuilt from the variant's pinned revision
        # at predict time, which is the Hub dependency the frozen protein engines already
        # have -- and the reason this artifact is megabytes rather than gigabytes.
        buffer = io.BytesIO()
        torch.save(
            {
                "lora_state": _lora.trainable_state(module.backbone),
                "head_state": None if ranking else module.head.state_dict(),
                "protein_model": model_name,
                "objective": objective,
                "parent": parent or None,
                "affine": affine,
                "target_mean": target_mean,
                "target_std": target_std,
                "n_tasks": len(columns),
                "is_classification": is_classification,
            },
            buffer,
        )

        return TrainResult(
            artifact=buffer.getvalue(),
            metrics=metrics,
            validation_metrics=validation_metrics,
            cutoffs=cutoffs or None,
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        torch, _device = torch_and_device()

        from lightning import pytorch as lightning

        # weights_only=True: the artifact holds tensors, strings, plain scalars and lists
        # of them, so there is no reason to let the pickle in it execute anything.
        bundle = torch.load(io.BytesIO(ctx.artifact), weights_only=True)
        objective = str(bundle["objective"])
        ranking = objective == OBJECTIVE_RANKING
        model_name = str(bundle["protein_model"])
        is_classification = bool(bundle["is_classification"])
        n_tasks = int(bundle["n_tasks"])
        chosen = variant(model_name)

        tokenizer, backbone = _load_backbone(model_name, ranking=ranking)
        structures = ctx.frame[ctx.structure_column].to_list()

        if ranking:
            parent = str(bundle["parent"])
            # Raises unless these rows are variants of the parent the fit recorded, with
            # a message naming both lengths -- a frame of a different protein would
            # otherwise be scored at positions that mean nothing in it.
            normalized, _parent = aligned_parent(structures, parent, model_name)
            parent_ids, offset = residue_token_offset(tokenizer, as_embeddable(parent)[0])
            residue_ids = {
                residue: tokenizer.convert_tokens_to_ids(residue) for residue in _RESIDUES
            }
            rows, _skipped = _mutation_rows(
                normalized, parent, offset, {r: i for r, i in residue_ids.items() if i is not None}
            )
            module = _build_ranking_module(
                backbone=backbone, parent_ids=parent_ids, learning_rate=1e-4
            )
            examples: list[Any] = [(row, 0.0) for row in rows]
            collate: Any = _ranking_collate
        else:
            module = _build_regression_module(
                backbone=backbone,
                dim=chosen.dim,
                n_targets=n_tasks,
                learning_rate=1e-4,
                is_classification=is_classification,
            )
            # Placeholder targets: `_collate` builds a target tensor unconditionally and
            # nothing computes a loss on a predict pass.
            examples = [
                (sequence, (0.0,) * n_tasks)
                for sequence in _cleaned(structures, chosen.max_residues)
            ]
            collate = _collate(tokenizer)

        _lora.load_trainable(module.backbone, bundle["lora_state"])
        if not ranking:
            module.head.load_state_dict(bundle["head_state"])

        trainer = lightning.Trainer(
            accelerator="auto",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )
        values = _to_values(
            _outputs(
                trainer,
                module,
                _loader(examples, collate=collate, batch_size=_PREDICT_BATCH_SIZE),
            ),
            is_classification=is_classification,
            target_mean=bundle["target_mean"],
            target_std=bundle["target_std"],
            affine=bundle["affine"],
        )
        # As for chemprop and MoLFormer: a bundle with fewer tasks than requested would
        # raise an IndexError below, and one with more would drop a task silently.
        if values.shape[1] != len(ctx.target_columns):
            raise ValueError(
                f"The model predicts {values.shape[1]} targets, but "
                f"{len(ctx.target_columns)} were requested."
            )

        # Explicit dtypes, matching every other engine: an all-None uncertainty list would
        # infer as polars' Null dtype and make this engine's output schema-incompatible
        # with theirs.
        #
        # ponytail: uncertainty is always null, as for chemprop and MoLFormer. A LoRA
        # fine-tune can produce one through MC-dropout or an ensemble of adapters -- the
        # second is cheap here, since an adapter is megabytes -- but a fabricated number
        # would be plotted by the triage grid as "the model is confident here", which is
        # worse than an admitted absent one.
        #
        # Long format, one block per target: a joint engine is not wrapped in `FanOut`,
        # so it tags its own rows.
        row_ids = pl.Series(range(values.shape[0]), dtype=pl.Int64)
        return pl.concat(
            [
                pl.DataFrame(
                    {
                        "row_id": row_ids,
                        "value": pl.Series(
                            [float(value) for value in values[:, index]], dtype=pl.Float64
                        ),
                        "uncertainty": pl.Series([None] * values.shape[0], dtype=pl.Float64),
                        "target": pl.Series([column] * values.shape[0], dtype=pl.String),
                    }
                )
                for index, column in enumerate(ctx.target_columns)
            ]
        )
