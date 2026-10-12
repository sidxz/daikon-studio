"""MoLFormer-XL: a transformer over the SMILES string itself.

The roster's missing architectural family. Everything else here reads a molecule as a
graph (chemprop), a fingerprint (the ECFP4 engines and the GP) or a descriptor vector;
this reads it as a sequence of tokens, which is a different inductive bias rather than
a better one. Expect no average accuracy gain -- the capability is the family.

IBM's checkpoint is Apache-2.0, 46.8M parameters, pretrained on ~1.1B molecules from
ZINC and PubChem (`arxiv:2106.09553`), and ships its own
`MolformerForSequenceClassification` head with the skip connections the paper
describes. That head is used as-is rather than reimplemented on top of the bare
encoder: it is the configuration the published numbers come from.

**Every torch and transformers import lives inside a function, never at module scope**,
for the same reason as `chemprop_dmpnn.py` -- `default_registry()` builds this class at
import time and the API tier installs without the gpu extra.

Two things about this checkpoint are load-bearing and easy to get wrong; both are
guarded below and neither fails loudly:

1. `trust_remote_code=True` is required -- the `molformer` architecture is not in
   transformers, it is Python fetched from the Hub and executed here. `_REVISION` pins
   that code to one immutable commit, so what runs on a worker is what was reviewed,
   not whatever the repo holds today.
2. `deterministic_eval=True` is required for reproducible predictions. MoLFormer uses
   linear attention with random Fourier features, and its `MolformerFeatureMap.forward`
   resamples those features on *every* forward pass unless this is set -- so the
   default configuration scores the same molecule differently on each call. With it
   set, resampling happens only in training mode and the final draw is carried in the
   `weight` buffer, which `state_dict()` saves -- so a reloaded artifact reproduces the
   numbers its own Scorecard reported.

Peptides are out of scope here, and now loudly so. A peptide SMILES runs past the
202-token context the model was pretrained on -- a dipeptide alone is 23 tokens, about
eleven per residue, so roughly eighteen residues exhaust it and anything longer is out
of distribution. `_collate` refuses an over-long structure instead of truncating it;
those datasets belong on a fingerprint or descriptor engine.

What this tokenizer does **not** do, against a claim repeated in our own research notes
and checked here on the pinned snapshot: it does not collapse stereochemistry. `[C@H]`
and `[C@@H]` are separate tokens (ids 15 and 16) and the pre-tokenizer keeps bracket
atoms whole, so a D- and an L-residue tokenize differently. Length is the real
limitation; do not justify this guard with stereochemistry.

Two settings choose *what* trains and *which* published weights it starts from, and
neither is an accuracy claim. `training` offers the whole model, LoRA adapters, or the
output layer alone; LoRA's measured value at this size is the artifact -- a few
megabytes instead of about 174 -- and not speed or memory, which it saves neither of
(`_lora.py`). `checkpoint` offers IBM's 10%-sample weights or DeepChem's c3 release of
the same architecture pretrained on all 1.1 billion molecules. Nothing here has been
compared on a dataset yet, so neither setting carries an accuracy claim in either
direction.

The LoRA artifact holds the adapters, the output layer **and** the linear-attention
random-feature buffers, which is not a detail: see `_random_features`.

**Measured 2026-10-12, and the result is that none of these choices matters here.**
Polaris SARS-CoV-2 Mpro potency, the published partition (758 train / 84 validation /
263 test), every arm fit through `train` and scored through `predict` on identical rows,
each compared with `ecfp4-randomforest` by a paired bootstrap of the Spearman difference
over 10,000 resamples:

    molformer-xl  full   0.8538   +0.0233 [-0.0039, +0.0494]
    molformer-xl  lora   0.8501   +0.0197 [-0.0083, +0.0473]
    molformer-xl  head   0.8482   +0.0178 [-0.0111, +0.0453]
    molformer-c3  full   0.8465   +0.0160 [-0.0091, +0.0411]
    molformer-c3  lora   0.8355   +0.0051 [-0.0240, +0.0334]
    molformer-c3  head   0.8334   +0.0029 [-0.0277, +0.0326]
    ecfp4-randomforest   0.8304   (reference, one second to fit)

Every interval contains zero. So on this dataset a fine-tuned 46.8M-parameter transformer
is **not** distinguishable from a fingerprint forest that fits in a second, more
pretraining data does not help, and LoRA matches a full fine-tune while saving 165 MB per
Protocol. Read the ordering as noise, not as a ranking: `full` leads `head` by 0.0056,
which is a fifth of the width of its own interval. One dataset, one split, regression
only -- the honest claim is "no measured difference", and the reason to pick a mode is
cost and artifact size.
"""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from daikonstudio.application.engines.context import PredictContext, TrainContext, TrainResult
from daikonstudio.application.engines.manifest import (
    ConditionSpec,
    ConditionType,
    EngineManifest,
    TaskType,
    validate_conditions,
)
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
    positive_weight,
)
from daikonstudio.infrastructure.engines._scoring import (
    classification_by_column,
    regression_metrics,
    tuned_cutoffs,
)

#: The published checkpoint, pinned to an immutable commit. The `ibm/` namespace
#: redirects here; naming the canonical one keeps the pin meaningful.
_MODEL_ID = "ibm-research/MoLFormer-XL-both-10pct"

#: Pinned because `trust_remote_code=True` executes this repo's Python on the worker.
#: An unpinned `main` means the code that runs can change under a published Protocol
#: without anything in this repository changing. Bump it deliberately, having read the
#: diff -- that is the whole point of it being here.
_REVISION = "361063d0ad524ef77cf39b08469f6be770dc550f"

#: `max_position_embeddings` from the checkpoint's own config. A longer SMILES is
#: refused, not truncated -- see `_collate`. Most drug-like SMILES tokenize well under
#: half of this; what reaches the limit is a peptide or another large molecule, which
#: this engine is the wrong tool for in the first place.
_MAX_TOKENS = 202

_PREDICT_BATCH_SIZE = 64

#: The two `checkpoint` options. `CHECKPOINT_XL` is what every artifact written before
#: this setting existed was trained on, so it is both the default and what a bundle with
#: no `checkpoint` key means.
CHECKPOINT_XL = "molformer-xl"
CHECKPOINT_C3 = "molformer-c3"

#: DeepChem's release of the same 44M-parameter architecture, pretrained on *all* of
#: ZINC20 and PubChem rather than the 10% sample IBM published weights for.
#:
#: Only its tensors are used. The repository's own `config.json` and
#: `tokenizer_config.json` point `auto_map` at IBM's repo **@main** with no revision,
#: and IBM deleted the tokenizer code there on 2026-07-23 -- so loading it the way it
#: declares raises today, and on any day it did not it would execute unpinned Python.
#: The tokenizer and the model class therefore come from `_MODEL_ID` at `_REVISION`
#: exactly as for the 10% weights, and only the state dict is laid over the result.
#: Verified on this snapshot: the vocabularies are identical token for token, and the
#: load reports 0 missing and 0 unexpected keys.
_C3_MODEL_ID = "DeepChem/MoLFormer-c3-1.1B"
_C3_REVISION = "9f1b9ea3590833bd0ea1a70e789c5d3da11ba7ed"
_C3_WEIGHTS = "model.safetensors"

#: What the c3 file prefixes its encoder keys with. `MolformerForSequenceClassification`
#: holds the same encoder as `.molformer`, whose own keys carry no prefix, and the
#: `lm_head.*` the file also contains has no home in a classification model.
_C3_PREFIX = "molformer."

#: The Linear layers LoRA adapts, by their leaf name. Read off `named_modules()` of the
#: real checkpoint, not from peft's built-in map, which has no entry for the `molformer`
#: architecture at all -- see `_lora.py` for why a wrong name is the dangerous case.
_LORA_TARGETS = ("query", "key", "value", "dense")

#: What `training` replaced. A stored condition dict can still carry it -- an in-flight
#: run being resumed reads back the conditions it was enqueued with -- and
#: `validate_conditions` rejects any key the manifest does not declare, so without the
#: map below a resumed fit would fail instead of continuing.
_LEGACY_FREEZE = "freeze_encoder"

_MANIFEST = EngineManifest(
    id="molformer-xl",
    version="1.1.0",
    name="MoLFormer-XL",
    description=(
        "A transformer pretrained on ~1.1 billion molecules that reads the SMILES "
        "string directly instead of the molecular graph. Learns every target of a "
        "dataset in one model. Fine-tuning takes minutes to hours on a GPU; training "
        "only the output layer takes a fraction of the time."
    ),
    tasks=(TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION),
    conditions=(
        ConditionSpec(
            key="epochs",
            label="Training epochs",
            type=ConditionType.INTEGER,
            default=10,
            minimum=1,
            maximum=200,
            help="Number of passes over the training set. A pretrained transformer "
            "adapts quickly; longer training risks degrading the pretrained "
            "representation.",
        ),
        ConditionSpec(
            key="batch_size",
            label="Batch size",
            type=ConditionType.INTEGER,
            default=32,
            minimum=8,
            maximum=256,
            help="Number of molecules per gradient update. Reduce it if training runs "
            "out of GPU memory.",
        ),
        ConditionSpec(
            key="learning_rate",
            label="Learning rate",
            type=ConditionType.NUMBER,
            default=3e-5,
            minimum=1e-6,
            maximum=1e-2,
            help="Step size of each weight update. Fine-tuning a pretrained transformer "
            "needs a small value (about 3 × 10⁻⁵); increase it toward 10⁻³ when only the "  # noqa: RUF001
            "output layer is trained, and to about 10⁻⁴–10⁻³ for LoRA adapters, which "  # noqa: RUF001
            "start at zero and have far fewer weights to move.",
        ),
        ConditionSpec(
            key="training",
            label="What to train",
            type=ConditionType.ENUM,
            default="full",
            options=("full", "lora", "head"),
            option_labels=(
                "Fine-tune the whole model",
                "LoRA adapters",
                "Output layer only",
            ),
            help="Fine-tuning the whole model updates all 46 million weights. LoRA "
            "trains small adapter matrices inside each attention layer and leaves the "
            "pretrained weights untouched, so the saved model is about 8 megabytes "
            "instead of 174. Output layer only trains on top of the frozen pretrained "
            "representation, which is the fastest and the hardest to overfit. On the one "
            "dataset we have measured — 758 training molecules of SARS-CoV-2 Mpro "
            "potency — the three choices scored 0.85, 0.85 and 0.85 in rank "
            "correlation, differences too small to separate from measurement noise, and "
            "none of them was distinguishable from the fingerprint baseline. Choose on "
            "cost and model size rather than on expected accuracy.",
        ),
        ConditionSpec(
            key="checkpoint",
            label="Pretrained weights",
            type=ConditionType.ENUM,
            default=CHECKPOINT_XL,
            options=(CHECKPOINT_XL, CHECKPOINT_C3),
            option_labels=(
                "MoLFormer-XL (10% of pretraining data)",
                "MoLFormer-c3 (full 1.1B-molecule pretraining)",
            ),
            help="Which published weights the fit starts from. Both are the same "
            "44-million-parameter architecture with the same tokenizer. MoLFormer-XL is "
            "IBM's release, pretrained on a 10% sample of 1.1 billion molecules, and is "
            "the default. MoLFormer-c3 is DeepChem's release of the same model "
            "pretrained on all of them. More pretraining data did not help on the one "
            "dataset we have measured: on SARS-CoV-2 Mpro potency, MoLFormer-c3 scored "
            "at or slightly below MoLFormer-XL in every training mode, by margins "
            "inside measurement noise. Neither is established as the better starting "
            "point.",
        ),
        POSITIVE_WEIGHTING,
        EPOCH_SELECTION,
    ),
    lane="gpu",
    supports_multitask=True,
)


def _require_transformers(weights_dir: str) -> None:
    """Fail with a cause a human can act on, and point the Hub cache at our own dir.

    `HF_HOME` is read by transformers at import time, so it is set first. Reusing
    `STUDIO_PRETRAINED_WEIGHTS_DIR` means one weights directory per deployment --
    the same one `Dockerfile.gpu` bakes CheMeleon into -- rather than a second cache
    appearing under the worker's home directory. `setdefault`, so a deployment that
    already manages `HF_HOME` keeps its own answer.
    """
    os.environ.setdefault("HF_HOME", str(os.path.expanduser(weights_dir)))
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ValidationError(
            "This runner does not have the GPU dependencies that MoLFormer-XL requires. "
            "An administrator can register a runner for the 'gpu' lane on the Runners "
            "page."
        ) from exc


def _load_c3_weights(model: Any) -> None:
    """Lay the full-data c3 tensors over `model.molformer`, refusing a partial fit.

    `strict=False` is unavoidable -- the file also carries a masked-LM head this model
    does not have -- and on its own it is exactly the silent no-op that
    `protein/embed.apply_overlay` documents: a prefix that stopped matching would leave
    every weight at the 10% checkpoint's value and report a Scorecard for weights the
    scientist did not choose. So both key counts are checked, not just one.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download(_C3_MODEL_ID, _C3_WEIGHTS, revision=_C3_REVISION)
    published = load_file(path)
    encoder = {
        name[len(_C3_PREFIX) :]: tensor
        for name, tensor in published.items()
        if name.startswith(_C3_PREFIX)
    }
    result = model.molformer.load_state_dict(encoder, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValidationError(
            f"The MoLFormer-c3 weights do not fit this model: "
            f"{len(result.missing_keys)} of its tensors are absent from the published "
            f"file and {len(result.unexpected_keys)} of the file's have no home in it. "
            "The two checkpoints have diverged, which needs a code change rather than a "
            "different setting."
        )


def _lora_targets(model: Any) -> list[str]:
    """`_LORA_TARGETS` inside the encoder, by full module name.

    Full names rather than the leaf names peft matches by suffix, because the
    classification head's first layer is **also** called `dense`: a bare `"dense"` would
    put an adapter inside the head at the same time as `modules_to_save` asks for the
    head to train in full, which is two owners for one layer. Naming the encoder's own
    is unambiguous, and an architecture that stopped having them would reach `attach`'s
    zero-match refusal rather than quietly training only the head.
    """
    return [
        f"molformer.{name}"
        for name, _child in model.molformer.named_modules()
        if name.rsplit(".", 1)[-1] in _LORA_TARGETS
    ]


def _load_backbone(*, training: str, checkpoint: str, num_labels: int) -> tuple[Any, Any]:
    """The tokenizer and the classification model, both pinned to `_REVISION`.

    One logit per target (`num_labels`), read as a value for regression and through a
    sigmoid for classification. The loss is computed by the Lightning module
    rather than by passing `labels=` into the model, because the model's own
    `problem_type` inference would pick mean-squared error for a single label -- which
    is wrong for a probability. Logits and an explicit loss keep both tasks on one
    code path with nothing inferred.

    `training` decides what carries a gradient, and `predict` calls this too -- with
    `"lora"` whenever the artifact holds adapters, because the adapters have to exist
    before their saved weights can be loaded into them.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        _MODEL_ID, revision=_REVISION, trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        _MODEL_ID,
        revision=_REVISION,
        trust_remote_code=True,
        num_labels=num_labels,
        # See this module's docstring: without it, every forward pass redraws the
        # linear-attention random features and the same molecule scores differently
        # on each call.
        deterministic_eval=True,
    )
    if checkpoint == CHECKPOINT_C3:
        _load_c3_weights(model)
    if training == "head":
        # The `classifier` head stays trainable; only the pretrained trunk is pinned.
        model.molformer.requires_grad_(False)
    elif training == "lora":
        # The head trains in full beside the adapters: it is randomly initialized here,
        # so a rank-8 correction to it would be a correction to noise.
        model = _lora.attach(model, _lora_targets(model), modules_to_save=("classifier",))
    return tokenizer, model


def _random_features(model: Any) -> dict[str, Any]:
    """The linear-attention random features, which are buffers rather than parameters.

    `_lora.trainable_state` saves what trained, and these did not -- but a *training*
    forward pass redraws them (see this module's docstring), so the draw behind the
    numbers on the Scorecard is the one the fit ended on, not the one in the published
    checkpoint. Left out of a LoRA artifact, `predict` would rebuild the base from the
    Hub, get the published draw, and quietly return different numbers from the ones the
    model reported about itself. Twelve tensors of 64 x 32, about 100 kB.
    """
    return {
        name: buffer.detach().cpu()
        for name, buffer in model.named_buffers()
        if name.endswith("feature_map.weight")
    }


def _without_legacy_freezing(conditions: dict[str, Any]) -> dict[str, Any]:
    """`conditions` with a stored `freeze_encoder` read as the `training` mode it meant.

    `true` trained the head alone and `false` trained everything, which are two of the
    three modes that replaced it. An explicit `training` wins, so a caller sending both
    is not overridden by the older key.
    """
    if _LEGACY_FREEZE not in conditions:
        return conditions
    mapped = dict(conditions)
    frozen = bool(mapped.pop(_LEGACY_FREEZE))
    mapped.setdefault("training", "head" if frozen else "full")
    return mapped


def _validation_scores(module: Any, batch: Any) -> tuple[Any, Any]:
    """A validation batch's logits and labels, each (rows, targets): what
    `keep_best_epoch` scores an epoch by. Logits rank exactly as probabilities do."""
    input_ids, attention_mask, targets = batch
    return module(input_ids, attention_mask), targets


def _collate(tokenizer: Any) -> Any:
    """Tokenize per batch rather than up front.

    Tokenizing a whole split eagerly would hold `n x 202` token ids in memory for the
    entire fit -- hundreds of megabytes on a large assay, most of it padding. Batching
    it also pads to the longest SMILES *in the batch* instead of the longest in the
    dataset, which is the bulk of the saving.

    Also the one place a structure's length is checked, for both training and predicting,
    so neither can accept a molecule the other would refuse.
    """
    import torch

    def collate(batch: Sequence[tuple[str, Sequence[float]]]) -> tuple[Any, Any, Any]:
        smiles = [item[0] for item in batch]
        # (batch, n_tasks): one column per target.
        targets = torch.tensor([list(item[1]) for item in batch], dtype=torch.float32)
        # No `truncation=True`: truncating cuts the tail off the structure without
        # saying so, and a peptide is almost all tail. (It would not have fired anyway --
        # this tokenizer declares no `model_max_length`, so `truncation=True` alone was
        # always a no-op and the 202-token ceiling was never actually enforced.) The mask
        # counts each row's real tokens, so the error can name the structure that is long.
        #
        # ponytail: checked per batch, so a single over-long row in a large assay fails
        # the fit partway through the first epoch rather than before it starts. A pass
        # over the whole structure column up front would buy a few minutes, at the cost
        # of tokenizing every molecule twice.
        encoded = tokenizer(smiles, padding=True, return_tensors="pt")
        lengths = encoded["attention_mask"].sum(dim=1)
        if int(lengths.max()) > _MAX_TOKENS:
            raise ValidationError(
                f"MoLFormer-XL was pretrained on at most {_MAX_TOKENS} tokens of a SMILES "
                f"string, and one structure here needs {int(lengths.max())}. Anything "
                "longer is outside what the model has seen, so this engine is a poor fit "
                "for peptides and other large molecules. Train a fingerprint or descriptor "
                f"engine instead. The structure begins: {smiles[int(lengths.argmax())][:60]}"
            )
        return encoded["input_ids"], encoded["attention_mask"], targets

    return collate


def _loader(
    examples: Sequence[tuple[str, Sequence[float]]],
    *,
    collate: Any,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
) -> Any:
    """A DataLoader over `(smiles, targets)` pairs, one target per task.

    A plain list is already a map-style dataset, so there is no Dataset subclass here
    to hold two parallel lists and re-implement `__len__`.
    """
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(
        examples,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        generator=None if seed is None else torch.Generator().manual_seed(seed),
    )


def _logits(trainer: Any, module: Any, loader: Any) -> Any:
    """The model's raw outputs over `loader`, shaped (molecules, n_tasks).

    `trainer.predict` rather than calling the module directly: it is what puts the
    module in eval mode, which is what freezes MoLFormer's random features (see this
    module's docstring). Shared by scoring and by `predict` so those two can never
    drift into running the model differently.
    """
    import torch

    batches: Any = trainer.predict(module, loader)
    return torch.cat(batches).cpu().numpy()


def _unlabelled(structures: list[str], n_tasks: int) -> list[tuple[str, tuple[float, ...]]]:
    """Structures paired with placeholder targets, for inference.

    `_collate` builds a target tensor unconditionally; zero is never read, because
    nothing computes a loss on a predict pass.
    """
    return [(smiles, (0.0,) * n_tasks) for smiles in structures]


def _to_values(logits: Any, *, is_classification: bool, target_mean: Any, target_std: Any) -> Any:
    """Raw logits to the numbers a scientist sees.

    One function, called by both the training-time scoring and `predict`, so the
    Scorecard can never describe a different transform than the one production uses --
    a divergence here would be invisible in both places.

    `target_mean` and `target_std` broadcast against the (molecules, n_tasks) logits:
    a per-column list from a current bundle, or the single scalar an older one holds.
    """
    import numpy as np

    if is_classification:
        return 1.0 / (1.0 + np.exp(-logits))
    return np.asarray(logits) * np.asarray(target_std) + np.asarray(target_mean)


def _build_module(
    *,
    model: Any,
    learning_rate: float,
    is_classification: bool,
    pos_weight: list[float] | None = None,
) -> Any:
    """The LightningModule. Defined inside a function so `lightning` imports lazily.

    `pos_weight` is one positive-class weight per target for the classification loss,
    which broadcasts it over the (batch, n_tasks) logits; None leaves the loss as it was.
    The loss only runs in training and validation, so `predict` never passes one.
    """
    import torch
    from lightning import pytorch as lightning
    from torchmetrics.classification import (
        BinaryAUROC,
        BinaryAveragePrecision,
        BinaryMatthewsCorrCoef,
    )
    from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

    # Validation scores per epoch for the run page's live charts, logged under the keys
    # chemprop uses so one recorder reads both engines (`_lightning.SCORE_METRICS`).
    # Pooled over every target the fit predicts, as chemprop's are; `keep_best_epoch`
    # logs AUROC and PR AUC averaged over targets, which replace these on the run page.
    score_keys = (
        {"roc": "val/roc", "prc": "val/prc", "mcc": "val/binary-mcc"}
        if is_classification
        else {"rmse": "val/rmse", "mae": "val/mae", "r2": "val/r2"}
    )

    class _MolformerModule(lightning.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = model
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

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            return output.logits

        def training_step(self, batch: Any, _index: int) -> Any:
            input_ids, attention_mask, targets = batch
            loss = self._loss(self(input_ids, attention_mask), targets)
            # on_epoch: the run page charts the epoch's mean, not its last batch.
            self.log("train_loss", loss, batch_size=len(targets), on_epoch=True)
            return loss

        def validation_step(self, batch: Any, _index: int) -> Any:
            input_ids, attention_mask, targets = batch
            logits = self(input_ids, attention_mask)
            loss = self._loss(logits, targets)
            # The key `keep_best_epoch` reads for a measured target. `batch_size` is
            # explicit because Lightning cannot infer it from a tuple batch and warns per step.
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
            input_ids, attention_mask, _targets = batch
            return self(input_ids, attention_mask)

        def configure_optimizers(self) -> Any:
            # Only the parameters that are actually learning: under `head` or `lora`,
            # handing AdamW the frozen trunk would allocate optimizer state for 46M
            # parameters that never receive a gradient.
            trainable = [p for p in self.parameters() if p.requires_grad]
            return torch.optim.AdamW(trainable, lr=learning_rate)

    return _MolformerModule()


def _standardize(targets: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    """Per-target mean and standard deviation of the training values, for regression.

    A transformer fine-tuned with mean-squared error against raw assay values --
    percent inhibition, IC50 in micromolar, log units -- has its gradients scaled by
    whatever unit the column happens to use. Standardizing on the *training* split
    only, and inverting at predict time, is the same thing chemprop's
    `UnscaleTransform` does, so predictions come back in the target's own unit and the
    Scorecard compares them against `actual` without rescaling anything itself.

    A zero standard deviation (every training value identical) would divide by zero;
    1.0 leaves the values as they are, which is the honest answer for a constant column.
    Each column gets its own, so a target on a small scale is not dragged toward one on
    a large scale.
    """
    import numpy as np

    array = np.asarray(targets, dtype=float)
    deviation = array.std(axis=0)
    return array.mean(axis=0).tolist(), np.where(deviation > 0.0, deviation, 1.0).tolist()


class MolformerXL:
    @staticmethod
    def manifest() -> EngineManifest:
        return _MANIFEST

    def train(self, ctx: TrainContext) -> TrainResult:
        # Before anything else, including loading a 1.1-billion-parameter checkpoint:
        # this is a decision about the shape of the data, not a failed fit.
        #
        # MoLFormer declares `supports_multitask`, so `FanOut` does not split this
        # context and does not drop unmeasured rows. The loss below masks nothing, so a
        # null target would be trained on as a number -- no exception, no visible
        # symptom, meaningless weights. Refused rather than masked: neither of the
        # sparse benchmarks this platform targets needs MoLFormer, both reproduce a
        # D-MPNN, and this message is what tells the next person the capability is
        # missing rather than broken.
        for column in ctx.targets:
            missing = int(ctx.frame[column].null_count())
            if missing:
                raise ValidationError(
                    f"MoLFormer-XL does not support datasets with unmeasured targets, "
                    f"and '{column}' is blank for {missing:,} of {ctx.frame.height:,} "
                    "compounds. Use Chemprop D-MPNN, which learns every target jointly "
                    "and handles unmeasured ones, or a dataset where every compound is "
                    "measured."
                )

        from daikonstudio.settings import Settings

        _require_transformers(Settings().pretrained_weights_dir)

        import torch
        from lightning import pytorch as lightning
        from lightning.pytorch.callbacks import LambdaCallback

        conditions: dict[str, Any] = validate_conditions(
            _MANIFEST, _without_legacy_freezing(ctx.conditions)
        )
        epochs = int(conditions["epochs"])
        batch_size = int(conditions["batch_size"])
        learning_rate = float(conditions["learning_rate"])
        training = str(conditions["training"])
        checkpoint = str(conditions["checkpoint"])
        weighting = str(conditions["positive_weighting"])
        selection = str(conditions["epoch_selection"])
        columns = ctx.target_columns
        # One task for the whole fit: a joint engine is only ever handed a dataset
        # whose targets share a kind (`joint_kind_error`, at enqueue).
        is_classification = ctx.task is TaskType.BINARY_CLASSIFICATION

        lightning.seed_everything(ctx.seed, workers=True)

        train_rows = ctx.frame.filter(pl.col("split") == "train")
        validation_rows = ctx.frame.filter(pl.col("split") == "validation")
        test_rows = ctx.frame.filter(pl.col("split") == "test")

        # Regression targets are standardized; classification targets are already
        # 0/1 and BCEWithLogitsLoss expects them that way.
        target_mean, target_std = (
            ([0.0] * len(columns), [1.0] * len(columns))
            if is_classification
            else _standardize(train_rows.select(columns).rows())
        )

        def examples(rows: pl.DataFrame) -> list[tuple[str, tuple[float, ...]]]:
            structures = rows[ctx.structure_column].to_list()
            return [
                (
                    smiles,
                    tuple(
                        (float(value) - mean) / std
                        for value, mean, std in zip(row, target_mean, target_std, strict=True)
                    ),
                )
                for smiles, row in zip(structures, rows.select(columns).rows(), strict=True)
            ]

        tokenizer, model = _load_backbone(
            training=training, checkpoint=checkpoint, num_labels=len(columns)
        )
        pos_weight = (
            [
                positive_weight(train_rows[column].to_numpy(), weighting) or 1.0
                for column in columns
            ]
            if is_classification and weighting != "none"
            else None
        )
        module = _build_module(
            model=model,
            learning_rate=learning_rate,
            is_classification=is_classification,
            pos_weight=pos_weight,
        )
        collate = _collate(tokenizer)

        train_loader = _loader(
            examples(train_rows),
            collate=collate,
            batch_size=batch_size,
            shuffle=True,
            seed=ctx.seed,
        )
        # Shuffling a validation loader would not corrupt the loss, but it makes the
        # number depend on the seed for no reason.
        validation_loader = (
            _loader(examples(validation_rows), collate=collate, batch_size=batch_size)
            if validation_rows.height > 0
            else None
        )

        def _report_epoch(trainer: Any, _module: Any) -> None:
            # Naming the device is not decoration -- see chemprop_dmpnn.py. This is
            # also the interruption point: `report` may raise RunInterrupted, which
            # propagates out of `fit` and is the only way to stop a fit already
            # running on the worker thread.
            ctx.report(
                (trainer.current_epoch + 1) / epochs,
                f"Training {_MANIFEST.name} on {trainer.strategy.root_device}",
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
                unit_scale=float(target_std[0])
                if not is_classification and len(columns) == 1
                else None,
                kept=keep_best if validation_loader is not None else None,
            ),
            LambdaCallback(on_train_epoch_end=_report_epoch),
        ]
        if validation_loader is not None:
            callbacks.append(keep_best)

        # The scratch directory holds the training state a save writes and a resume
        # reads; it only has to outlive `fit`. The trainer does not need it afterwards.
        with tempfile.TemporaryDirectory() as state_dir:
            state = training_state_scope(ctx.checkpoints, "transformers")
            resume_from = saved_training_state(state, Path(state_dir), module)
            # After the reporter: it raises in `on_train_epoch_end` when the time limit
            # stops the fit, and the save then happens in `on_exception`.
            if state is not None:
                callbacks.append(save_training_state(state, Path(state_dir)))
            if resume_from is not None:

                def _report_resume(trainer: Any, _module: Any) -> None:
                    ctx.report(
                        trainer.current_epoch / epochs,
                        f"Resuming {_MANIFEST.name} from epoch {trainer.current_epoch + 1} "
                        f"of {epochs}",
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

        # Restore the selected epoch before anything is scored or saved, so the
        # numbers on the Scorecard and the weights in the artifact are one model.
        if keep_best.best_state is not None:
            # strict=False: the snapshot deliberately omits frozen parameters, which
            # cannot have changed between epochs (see `keep_best_epoch`).
            module.load_state_dict(keep_best.best_state, strict=False)

        def infer(rows: pl.DataFrame) -> Any:
            loader = _loader(
                _unlabelled(rows[ctx.structure_column].to_list(), len(columns)),
                collate=collate,
                batch_size=_PREDICT_BATCH_SIZE,
            )
            return _to_values(
                _logits(trainer, module, loader),
                is_classification=is_classification,
                target_mean=target_mean,
                target_std=target_std,
            )

        # The validation partition is inferred once: it picks the cutoffs, then is scored
        # at them. Test is scored at those cutoffs and never sees them chosen.
        validation_values = infer(validation_rows) if validation_rows.height > 0 else None
        cutoffs: dict[str, float] = {}
        if ctx.tune_cutoffs and is_classification and validation_values is not None:
            cutoffs = tuned_cutoffs(columns, validation_rows, validation_values)

        def score(rows: pl.DataFrame, values: Any) -> dict[str, dict[str, float]]:
            if is_classification:
                return classification_by_column(columns, rows, values, cutoffs, train_rows)
            return {
                column: regression_metrics(rows[column].to_numpy(), values[:, index])
                for index, column in enumerate(columns)
            }

        metrics = score(test_rows, infer(test_rows))
        validation_metrics = (
            None if validation_values is None else score(validation_rows, validation_values)
        )

        # `full` and `head` save the whole fine-tuned model, not just the head. Under
        # `head` most of those ~174 MB duplicate the public checkpoint, which is
        # wasteful -- and deliberate: a published Protocol then keeps predicting even if
        # the Hub is unreachable or the repository is withdrawn.
        #
        # `lora` is the upgrade path the ponytail comment here used to name, now taken:
        # it saves only the parameters whose `requires_grad` was true and rebuilds the
        # rest from `_REVISION` at predict time, which is a few megabytes per Protocol
        # instead of 174 -- in exchange for needing the checkpoint to be reachable (or
        # already in the weights directory) to predict at all.
        bundle: dict[str, Any] = {
            "is_classification": is_classification,
            "target_mean": target_mean,
            "target_std": target_std,
            "n_tasks": len(columns),
            "revision": _REVISION,
            "training": training,
            "checkpoint": checkpoint,
        }
        if training == "lora":
            bundle["lora_state"] = _lora.trainable_state(module.backbone) | _random_features(
                module.backbone
            )
            # So a future edit to _lora's constants cannot reinterpret this artifact.
            bundle["lora_settings"] = _lora.settings(_lora_targets(module.backbone))
        else:
            # Without the loss: a weighted fit's `pos_weight` is a buffer of it, and
            # `predict` builds its module with no loss weights to load it into.
            bundle["state_dict"] = {
                key: value
                for key, value in module.state_dict().items()
                if not key.startswith("_loss.")
            }
        buffer = io.BytesIO()
        torch.save(bundle, buffer)

        return TrainResult(
            artifact=buffer.getvalue(),
            metrics=metrics,
            validation_metrics=validation_metrics,
            cutoffs=cutoffs or None,
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        from daikonstudio.settings import Settings

        _require_transformers(Settings().pretrained_weights_dir)

        import torch
        from lightning import pytorch as lightning

        # weights_only=True: the artifact holds tensors, plain scalars and lists of them,
        # so there is no reason to allow the pickle in it to execute anything.
        bundle = torch.load(io.BytesIO(ctx.artifact), weights_only=True)
        is_classification = bool(bundle["is_classification"])
        # An artifact from before targets could be several has no `n_tasks` and a scalar
        # mean and deviation; one task, and `_to_values` broadcasts the scalars.
        n_tasks = int(bundle.get("n_tasks", 1))
        # An artifact written before these settings existed holds a full state dict
        # trained on the 10% checkpoint, which is what both defaults say.
        training = str(bundle.get("training", "full"))
        checkpoint = str(bundle.get("checkpoint", CHECKPOINT_XL))

        # `"full"` for a `head` artifact regardless of how it was trained: that setting
        # only controls which parameters receive gradients, and nothing is training now.
        # `lora` is the one case the distinction survives into prediction -- the adapter
        # layers have to be in the model before their saved weights can land in them.
        tokenizer, model = _load_backbone(
            training="lora" if training == "lora" else "full",
            checkpoint=checkpoint,
            num_labels=n_tasks,
        )
        module = _build_module(
            model=model, learning_rate=1e-4, is_classification=is_classification
        )
        if training == "lora":
            # The adapters, the head, and the linear-attention random features the fit
            # ended on -- see `_random_features` for why those travel with them.
            _lora.load_trainable(
                module.backbone, bundle["lora_state"], bundle.get("lora_settings")
            )
        else:
            # Restores the fine-tuned weights *and* the linear-attention `weight`
            # buffer, which is what makes these predictions identical to the ones the
            # Scorecard was built from.
            module.load_state_dict(bundle["state_dict"])

        loader = _loader(
            _unlabelled(ctx.frame[ctx.structure_column].to_list(), n_tasks),
            collate=_collate(tokenizer),
            batch_size=_PREDICT_BATCH_SIZE,
        )
        trainer = lightning.Trainer(
            accelerator="auto",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )
        values = _to_values(
            _logits(trainer, module, loader),
            is_classification=is_classification,
            target_mean=bundle["target_mean"],
            target_std=bundle["target_std"],
        )
        # As for chemprop: a bundle with fewer tasks than requested would raise an
        # IndexError below, and one with more would drop a task silently.
        if values.shape[1] != len(ctx.target_columns):
            raise ValueError(
                f"The model predicts {values.shape[1]} targets, but {len(ctx.target_columns)} "
                "were requested."
            )

        # Explicit dtypes, matching every other engine: an all-None uncertainty list
        # would infer as polars' Null dtype and make this engine's output
        # schema-incompatible with theirs.
        #
        # ponytail: uncertainty is always null, as for chemprop. A transformer can
        # produce one through MC-dropout or a deep ensemble; a fabricated number would
        # be plotted by the triage grid as "the model is confident here", which is
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
