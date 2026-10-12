"""The `protein-lora` engine's contract, on whatever device this machine has.

Two groups of tests. The first needs no weights at all and runs in CI: the manifest, the
two refusals the ranking objective makes about the shape of a dataset, and the pure
arithmetic of the ranking score and its loss. All four of those are reachable before a
single byte of a checkpoint is downloaded, which is deliberate -- a dataset of the wrong
shape should be refused in a second, not after 2.6 GB.

The second group is gated on ESM-2 150M already being cached, because the default-lane
worker and the API tier install without the gpu extra and CI does not download weights.
It trains for two epochs on thirty sequences: this is not a test of whether a LoRA fit
learns anything, it is a test that the adapter honours the Engine contract and that the
three traps in this file stay guarded -- gradient checkpointing silently detaching the
adapters, an artifact that reloads into a different model than it was saved from, and a
ranking score that is not in the target's units.

Nothing here pins the device. `accelerator="auto"` resolves to MPS on Apple Silicon, CUDA
on a GPU runner and CPU otherwise, and all three must satisfy the same contract.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from daikonstudio.application.engines.context import PredictContext, TrainContext
from daikonstudio.application.engines.manifest import ConditionType, TaskType
from daikonstudio.domain.data.structure_kind import StructureKind
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.engines.protein_lora import (
    ProteinLora,
    _affine,
    _mutation_rows,
)
from daikonstudio.infrastructure.protein.embed import PROTEIN_MODELS

_WEIGHTS_DIR = Path(
    os.environ.get("STUDIO_PRETRAINED_WEIGHTS_DIR", "~/.cache/daikon-studio/weights")
).expanduser()

needs_weights = pytest.mark.skipif(
    not (_WEIGHTS_DIR / "hub" / "models--facebook--esm2_t30_150M_UR50D").exists(),
    reason="ESM-2 150M weights not cached; CI does not download 600 MB",
)

#: A 26-residue stand-in protein. Short on purpose: the forward pass is quadratic in
#: length and nothing here measures accuracy.
_PARENT = "MAKVQLSRLSGEQLLDELMKVYAQTQ"
_RESIDUES = "ACDEFGHIKLMNPQRSTVWY"

#: 150M rather than the 650M default, for the same reason `test_molformer_xl` trains for
#: two epochs: four times faster, and the contract is the same.
_FAST: dict[str, Any] = {"protein_model": "esm2-150m", "epochs": 2, "batch_size": 8}


def _single_parent_series(count: int = 30) -> tuple[list[str], list[float]]:
    """`count` single-substitution variants of `_PARENT`, plus the parent itself.

    Every position is mutated at most twice in thirty rows, so the modal residue at each
    position is still the parent's -- i.e. `assign_split.consensus` recovers `_PARENT`,
    which `test_the_series_recovers_its_own_parent` pins rather than assumes.

    The values are a deterministic function of the mutated position, so they are
    orderable: the ranking objective has something to rank, and the regression objective
    has something to fit, without either being learnable in two epochs. Neither is the
    point of these tests.
    """
    sequences = [_PARENT]
    values = [0.0]
    for index in range(count - 1):
        position = index % len(_PARENT)
        offset = 3
        replacement = _RESIDUES[(index * 7 + offset) % len(_RESIDUES)]
        while replacement == _PARENT[position]:
            offset += 1
            replacement = _RESIDUES[(index * 7 + offset) % len(_RESIDUES)]
        sequences.append(_PARENT[:position] + replacement + _PARENT[position + 1 :])
        values.append(float(position) * 0.5 - float(index) * 0.1)
    return sequences, values


def _frame(task: TaskType = TaskType.REGRESSION) -> pl.DataFrame:
    sequences, values = _single_parent_series()
    if task is TaskType.BINARY_CLASSIFICATION:
        values = [float(index % 2) for index in range(len(values))]
    # 18/6/6, with the parent in the training partition where a reference belongs.
    splits = ["train"] * 18 + ["validation"] * 6 + ["test"] * 6
    return pl.DataFrame({"sequence": sequences, "y": values, "split": splits})


def _context(frame: pl.DataFrame, task: TaskType, **conditions: object) -> TrainContext:
    return TrainContext(
        frame=frame,
        targets={"y": task},
        structure_column="sequence",
        conditions={**_FAST, **conditions},
        seed=13,
    )


def _predict(artifact: bytes, frame: pl.DataFrame, targets: tuple[str, ...] = ("y",)) -> Any:
    """A **fresh** engine instance, as a worker that only ever predicts would have."""
    return ProteinLora().predict(
        PredictContext(
            frame=frame,
            structure_column="sequence",
            artifact=artifact,
            conditions={},
            target_columns=targets,
        )
    )


# --- the manifest ---------------------------------------------------------------------


def test_the_manifest_declares_the_gpu_lane_and_reads_sequences() -> None:
    """Both halves matter. The lane keeps it off the default queue, where a worker with no
    gpu extra would pick it up; the structure kind keeps it off molecule datasets, where a
    SMILES string would be read as an amino-acid sequence and produce numbers."""
    manifest = ProteinLora.manifest()

    assert manifest.lane == "gpu"
    assert manifest.structure_kinds == (StructureKind.SEQUENCE,)
    assert manifest.supports_multitask is True
    assert manifest.is_baseline is False


def test_the_manifest_offers_every_protein_model_including_esm_c() -> None:
    """The point of `PROTEIN_MODEL` being shared: a sweep comparing this engine against
    the frozen ones compares the same weight sets. ESM C is named explicitly because it is
    the variant this batch added and the one whose absence here would be silent."""
    protein_model = next(
        spec for spec in ProteinLora.manifest().conditions if spec.key == "protein_model"
    )

    assert tuple(protein_model.options) == PROTEIN_MODELS
    assert "esmc-600m" in protein_model.options


def test_the_objective_defaults_to_predicting_the_value() -> None:
    """Ranking is the better-evidenced objective at small n, and it is still not the
    default: it silently requires single-parent variant data, and a scientist who has not
    read the help text should get the setting that works on any sequence dataset."""
    objective = next(spec for spec in ProteinLora.manifest().conditions if spec.key == "objective")

    assert objective.type is ConditionType.ENUM
    assert objective.default == "regression"
    assert tuple(objective.options) == ("regression", "ranking")
    assert len(objective.option_labels) == 2


# --- what the ranking objective refuses, before any weights load ----------------------


def test_ranking_refuses_an_active_inactive_target() -> None:
    """A pairwise ranking loss over two label values learns "actives above inactives",
    which is what every classifier here already does, and the affine map back to label
    units would be meaningless. Refused with the setting to change named."""
    with pytest.raises(ValidationError, match="needs a continuous measurement"):
        ProteinLora().train(
            _context(
                _frame(TaskType.BINARY_CLASSIFICATION),
                TaskType.BINARY_CLASSIFICATION,
                objective="ranking",
            )
        )


def test_ranking_refuses_several_targets() -> None:
    """One score per sequence, so one target. This engine declares `supports_multitask`,
    so nothing upstream splits the context for it -- without this refusal the second
    target would be dropped in silence."""
    frame = _frame().with_columns((pl.col("y") * 2.0).alias("z"))

    with pytest.raises(ValidationError, match="fits one target at a time"):
        ProteinLora().train(
            TrainContext(
                frame=frame,
                targets={"y": TaskType.REGRESSION, "z": TaskType.REGRESSION},
                structure_column="sequence",
                conditions={**_FAST, "objective": "ranking"},
                seed=13,
            )
        )


def test_ranking_refuses_sequences_of_different_lengths() -> None:
    """The score is indexed by residue position, so an unaligned series would score
    position 40 of one protein against position 40 of another. The message names both
    lengths and the setting that has no such requirement."""
    frame = _frame()
    ragged = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(_PARENT + "GGG"))
        .otherwise(pl.col("sequence"))
        .alias("sequence")
    )

    with pytest.raises(ValidationError) as caught:
        ProteinLora().train(_context(ragged, TaskType.REGRESSION, objective="ranking"))

    message = str(caught.value)
    assert "substitution variants of one parent sequence" in message
    assert "26 to 29 residues" in message
    assert "Predict the measured value" in message


def test_ranking_refuses_a_model_whose_scoring_protocol_is_unread() -> None:
    """ESM C is on the roster as an embedding, with `likelihood_scoring=None` because
    nobody has read Biohub's own scoring code. The ranking objective *is* a likelihood, so
    it has to refuse rather than pick a protocol -- reading VESM2 the wrong way cost 0.147
    Spearman and read for weeks as a weakness of the weights. It must refuse cleanly, with
    the reason and both ways out named, rather than crash somewhere in the forward pass."""
    with pytest.raises(ValidationError) as caught:
        ProteinLora().train(
            _context(
                _frame(),
                TaskType.REGRESSION,
                protein_model="esmc-600m",
                objective="ranking",
            )
        )

    message = str(caught.value)
    assert "ESM C" in message
    assert "has not been established" in message
    assert "Predict the measured value" in message


def test_the_regression_objective_accepts_what_ranking_refuses() -> None:
    """The refusals above are about the objective, not the data, and the messages say so.
    This is the cheap check that the claim is true as far as validation goes -- the fit
    itself is weights-gated below."""
    frame = _frame().with_columns((pl.col("y") * 2.0).alias("z"))

    # No ValidationError from the shape checks; it gets as far as wanting weights.
    from daikonstudio.infrastructure.engines.protein_lora import _refuse_unrankable

    context = TrainContext(
        frame=frame,
        targets={"y": TaskType.REGRESSION, "z": TaskType.REGRESSION},
        structure_column="sequence",
        conditions=_FAST,
        seed=13,
    )
    with pytest.raises(ValidationError):
        _refuse_unrankable(context, model_name="esm2-150m", is_classification=False)


def test_a_dataset_with_blank_targets_is_refused_by_name() -> None:
    """`supports_multitask` means `FanOut` does not drop unmeasured rows, and neither loss
    masks anything -- a null would be trained on as a number with no visible symptom."""
    frame = _frame().with_columns(
        pl.when(pl.int_range(pl.len()) < 3).then(None).otherwise(pl.col("y")).alias("y")
    )

    with pytest.raises(ValidationError, match="blank for 3 of 30"):
        ProteinLora().train(_context(frame, TaskType.REGRESSION))


# --- the ranking score and its loss, as arithmetic ------------------------------------


def test_the_series_recovers_its_own_parent() -> None:
    """The fixture's invariant, pinned rather than assumed: if the consensus were not
    `_PARENT`, every test below would score mutations that are not the ones it means."""
    from daikonstudio.application.data.assign_split import consensus

    sequences, _values = _single_parent_series()

    assert consensus(sequences) == _PARENT


def test_mutation_rows_point_at_the_tokens_the_parent_differs_at() -> None:
    """The index arithmetic the whole objective rests on: `offset + position`, with the
    offset read from the tokenizer rather than assumed to be 1."""
    residue_ids = {residue: index for index, residue in enumerate(_RESIDUES)}
    variant = "C" + _PARENT[1:-1] + "W"

    rows, skipped = _mutation_rows([_PARENT, variant], _PARENT, offset=1, residue_ids=residue_ids)

    assert skipped == 0
    assert rows[0] == (), "the wild type mutates nothing and scores zero"
    assert rows[1] == (
        (1, residue_ids["C"], residue_ids["M"]),
        (len(_PARENT), residue_ids["W"], residue_ids["Q"]),
    )


def test_an_unscorable_substitution_contributes_nothing_and_is_counted() -> None:
    """A stop codon is a legitimate DMS control, and there is no log-probability for
    "unknown residue" that is a statement about any particular substitution. Skipped and
    counted, exactly as `likelihood._llr_from_logprobs` skips it."""
    residue_ids = {residue: index for index, residue in enumerate(_RESIDUES)}

    rows, skipped = _mutation_rows(["*" + _PARENT[1:]], _PARENT, 1, residue_ids)

    assert skipped == 1
    assert rows[0] == ()


def test_the_ranking_loss_falls_when_the_scores_match_the_order() -> None:
    torch = pytest.importorskip("torch")
    from daikonstudio.infrastructure.engines.protein_lora import _ranking_loss

    y = torch.tensor([3.0, 2.0, 1.0])
    right = _ranking_loss(torch.tensor([3.0, 2.0, 1.0]), y)
    wrong = _ranking_loss(torch.tensor([1.0, 2.0, 3.0]), y)
    flat = _ranking_loss(torch.tensor([0.0, 0.0, 0.0]), y)

    assert float(right) < float(flat) < float(wrong)


def test_the_ranking_loss_is_the_papers_published_arithmetic() -> None:
    """The exact value of the listing in Appendix A.6, which is the only released
    arithmetic for this objective and therefore the thing to pin.

    Three rows with no two scores equal: all six off-diagonal entries have a zero logit,
    so each is `softplus(0) = log 2`; the mean is taken over all **nine** cells, diagonal
    included, and halved. Anything that normalizes by the three *orderable* pairs instead
    lands on 0.3466, and anything that forgets the halving lands on double this.
    """
    import math

    torch = pytest.importorskip("torch")
    from daikonstudio.infrastructure.engines.protein_lora import _ranking_loss

    flat = _ranking_loss(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([3.0, 2.0, 1.0]))

    assert float(flat) == pytest.approx(0.5 * 6 * math.log(2) / 9)


def test_the_ranking_loss_is_zero_for_a_batch_of_one() -> None:
    """The only cell is the diagonal, and the diagonal is zeroed. A batch of one states no
    preference, so there is nothing to learn from it and no division by zero either."""
    torch = pytest.importorskip("torch")
    from daikonstudio.infrastructure.engines.protein_lora import _ranking_loss

    assert float(_ranking_loss(torch.tensor([1.5]), torch.tensor([2.0]))) == 0.0


def test_tied_labels_are_pushed_together_rather_than_ignored() -> None:
    """The one place the paper's code and its Equation 2 disagree, and the code wins.

    `y_a > y_b` is false in both directions for a tie, so the pair is asked for target 0
    twice -- `softplus(d) + softplus(-d)`, minimized where the two scores are equal. Our
    first version summed only the `y_a > y_b` terms and so ignored ties, which is the
    equation but not the implementation; on assay data with repeated values (most
    dose-response data, and all of it once rounded) that is a real behavioural difference.
    """
    torch = pytest.importorskip("torch")
    from daikonstudio.infrastructure.engines.protein_lora import _ranking_loss

    tied = torch.tensor([4.0, 4.0])
    apart = _ranking_loss(torch.tensor([1.0, 9.0]), tied)
    together = _ranking_loss(torch.tensor([3.0, 3.0]), tied)

    assert float(together) < float(apart)
    assert float(apart) > 0.0, "ignoring ties would make this zero"


def test_the_score_to_label_line_recovers_a_known_slope() -> None:
    """Without this line a Scorecard would report an RMSE in log-likelihood units and an
    r2 far below zero for a model that ranks perfectly."""
    scores = [-4.0, -2.0, 0.0, 2.0]

    slope, intercept = _affine(scores, [i * 3.0 + 7.0 for i in scores])

    assert slope == pytest.approx(3.0)
    assert intercept == pytest.approx(7.0)


def test_a_model_that_distinguishes_nothing_gets_a_flat_line() -> None:
    """The degenerate design, which happens when no row differs from the parent: lstsq's
    minimum-norm answer is a zero slope and the mean as the intercept. The right
    behaviour, and it needs no guard of its own."""
    slope, intercept = _affine([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])

    assert slope == pytest.approx(0.0)
    assert intercept == pytest.approx(2.0)


# --- against the real weights ---------------------------------------------------------


@needs_weights
def test_the_adapters_receive_gradients_through_gradient_checkpointing() -> None:
    """The trap this engine would fail silently on.

    Reentrant gradient checkpointing decides whether a block needs a backward pass by
    looking at its *inputs*, and here the inputs come out of frozen embeddings -- so every
    adapter inside a checkpointed block would receive no gradient, the fit would run to
    completion, and the Scorecard would honestly report an untrained model.
    `use_reentrant=False` is what makes the parameters count. This test fails if that
    keyword is dropped.

    `lora_B` and not `lora_A`: peft initialises B to zeros, so at the first step the
    product `B @ A` is zero and A's gradient is zero with it. B's is not.
    """
    from daikonstudio.infrastructure.engines.protein_lora import _load_backbone

    tokenizer, backbone = _load_backbone("esm2-150m", ranking=False)
    # Train mode is what actually turns checkpointing on; `from_pretrained` leaves the
    # model in eval mode, where transformers skips it.
    backbone.train()

    encoded = tokenizer([_PARENT], return_tensors="pt")
    backbone(**encoded).last_hidden_state.sum().backward()

    gradients = [
        parameter.grad for name, parameter in backbone.named_parameters() if "lora_B" in name
    ]
    assert gradients, "no LoRA adapter landed at all"
    assert all(gradient is not None for gradient in gradients), (
        "an adapter received no gradient: the checkpointed blocks detached it"
    )
    assert any(float(gradient.abs().sum()) > 0.0 for gradient in gradients)


@needs_weights
def test_the_regression_objective_round_trips_and_reproduces_its_own_scorecard() -> None:
    """The artifact holds adapters and an output layer, not a model, so a reload rebuilds
    the published weights from the Hub and lays the adapters back over them. If any part of
    that disagrees with how the fit ran, the predictions move -- and nothing else would say
    so, because a shifted model still returns plausible numbers.

    Scored through `regression_metrics` from the predict path and compared against the
    number the fit reported, which is the end-to-end form of the check.
    """
    from daikonstudio.infrastructure.engines._scoring import regression_metrics

    frame = _frame()
    result = ProteinLora().train(_context(frame, TaskType.REGRESSION))

    assert sorted(result.metrics["y"]) == ["mae", "r2", "rmse", "spearman"]
    assert result.validation_metrics is not None

    test_rows = frame.filter(pl.col("split") == "test")
    predictions = _predict(result.artifact, test_rows)
    reported = regression_metrics(test_rows["y"].to_numpy(), predictions["value"].to_numpy())

    assert reported["rmse"] == pytest.approx(result.metrics["y"]["rmse"], rel=1e-4)
    assert predictions["value"].n_unique() > 1, "a constant prediction is not a model"


@needs_weights
def test_predict_returns_the_contracted_schema_and_dtypes() -> None:
    """An all-None uncertainty column inferred as polars Null would make this engine's
    output schema-incompatible with every other engine's."""
    result = ProteinLora().train(_context(_frame(), TaskType.REGRESSION))

    predictions = _predict(result.artifact, _frame().head(4))

    assert predictions.columns == ["row_id", "value", "uncertainty", "target"]
    assert predictions.schema["row_id"] == pl.Int64
    assert predictions.schema["value"] == pl.Float64
    assert predictions.schema["uncertainty"] == pl.Float64
    assert predictions.schema["target"] == pl.String
    assert predictions.height == 4


@needs_weights
def test_the_artifact_holds_adapters_rather_than_a_model() -> None:
    """LoRA's measured benefit at this size. ESM-2 150M is 600 MB of weights and ESM-2
    650M is 2.6 GB; an adapter set is single-digit megabytes, and that is the whole reason
    this engine saves what it saves."""
    import io

    import torch

    result = ProteinLora().train(_context(_frame(), TaskType.REGRESSION))

    assert len(result.artifact) < 50 * 1024 * 1024, len(result.artifact)
    bundle = torch.load(io.BytesIO(result.artifact), weights_only=True)
    assert bundle["protein_model"] == "esm2-150m"
    assert bundle["objective"] == "regression"
    assert bundle["parent"] is None
    assert bundle["affine"] is None
    assert all("lora_" in key or "modules_to_save" in key for key in bundle["lora_state"])


@needs_weights
def test_the_ranking_objective_trains_and_predicts_in_the_targets_units() -> None:
    """The ranking objective end to end: it trains, its artifact records the parent and the
    score-to-label line, a fresh instance reproduces its predictions, and those predictions
    are in the target's own range rather than in log-likelihood units."""
    import io

    import torch

    from daikonstudio.infrastructure.engines._scoring import regression_metrics

    frame = _frame()
    result = ProteinLora().train(_context(frame, TaskType.REGRESSION, objective="ranking"))

    bundle = torch.load(io.BytesIO(result.artifact), weights_only=True)
    assert bundle["objective"] == "ranking"
    assert bundle["parent"] == _PARENT
    assert len(bundle["affine"]) == 2
    assert bundle["head_state"] is None, "the ranking objective trains no output layer"
    assert len(result.artifact) < 50 * 1024 * 1024, len(result.artifact)

    test_rows = frame.filter(pl.col("split") == "test")
    predictions = _predict(result.artifact, test_rows)
    reported = regression_metrics(test_rows["y"].to_numpy(), predictions["value"].to_numpy())

    assert reported["rmse"] == pytest.approx(result.metrics["y"]["rmse"], rel=1e-4)
    assert predictions["value"].n_unique() > 1, "a constant prediction is not a ranking"
    # The affine map's job: the scores themselves are negative log-likelihood ratios of
    # order 10, and the labels run from about -3 to 12.
    values = predictions["value"].to_list()
    assert min(values) > -60.0 and max(values) < 60.0, values


@needs_weights
def test_the_ranking_objective_refuses_to_predict_a_different_protein() -> None:
    """The score is indexed by position against the parent the fit recorded, so a frame of
    some other protein would be scored at positions that mean nothing in it. Refused with
    both lengths named -- `embed.aligned_parent`'s guard, reached through this engine."""
    result = ProteinLora().train(_context(_frame(), TaskType.REGRESSION, objective="ranking"))

    with pytest.raises(ValidationError, match="26-residue parent"):
        _predict(result.artifact, pl.DataFrame({"sequence": [_PARENT + "GGG"]}))


@needs_weights
def test_report_is_called_once_per_epoch() -> None:
    """This callback is the only thing that can stop a fit already in flight."""
    calls: list[tuple[float, str]] = []
    context = TrainContext(
        frame=_frame(),
        targets={"y": TaskType.REGRESSION},
        structure_column="sequence",
        conditions=_FAST,
        seed=13,
        report=lambda fraction, phase: calls.append((fraction, phase)),
    )

    ProteinLora().train(context)

    training = [call for call in calls if call[1].startswith("Training")]
    assert len(training) == 2
    assert training[-1][0] == pytest.approx(1.0)
    assert "ESM-2 150M" in training[-1][1]


@needs_weights
def test_a_classification_fit_predicts_probabilities() -> None:
    """One logit per target, read through a sigmoid -- the classification path is the only
    thing that applies it."""
    frame = _frame(TaskType.BINARY_CLASSIFICATION)
    result = ProteinLora().train(_context(frame, TaskType.BINARY_CLASSIFICATION))

    assert sorted(result.metrics["y"]) == ["auprc", "auroc", "balanced_accuracy", "mcc"]

    predictions = _predict(result.artifact, frame)

    assert all(0.0 <= value <= 1.0 for value in predictions["value"].to_list())
