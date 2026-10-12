"""LoRA adapters for the transformer engines, and the artifact that holds only them.

What LoRA buys here is **size, not memory**. At the parameter counts on this roster
(15M-650M) a LoRA fit still backpropagates activations through every layer, so it saves
little GPU memory -- gradient checkpointing is the lever for that. What it does change is
the saved model: a few megabytes of adapter weights instead of the whole checkpoint
(174 MB for MoLFormer, 2.6 GB for ESM-2 650M), with the base rebuilt from its pinned
revision at predict time.

`target_modules` always comes from the caller, never from peft's built-in map: the names
differ between architectures (ESM-2 `query/key/value/dense`, ESM C `q_proj ... down_proj`),
and peft has no entry at all for some of them. A wrong name is the dangerous case --
peft would then train only the head and return a model that looks fine -- so `attach`
refuses when no adapter landed, the same rule `protein/embed.apply_overlay` applies to an
overlay that matches zero keys.

Rank, alpha and dropout are constants, not settings. Add a setting only when a
measurement shows one matters.

Same module rule as every other engine file: torch and peft are imported inside the
functions, because the API tier installs without the `gpu` extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from daikonstudio.domain.shared.errors import ValidationError

RANK = 8
ALPHA = 16
DROPOUT = 0.05


def attach(
    model: Any, target_modules: Sequence[str], *, modules_to_save: Sequence[str] = ()
) -> Any:
    """`model` wrapped with LoRA adapters on `target_modules`; everything else frozen
    except `modules_to_save`, which train in full and are saved with the adapters."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=DROPOUT,
        target_modules=list(target_modules),
        modules_to_save=list(modules_to_save) or None,
    )
    try:
        wrapped = get_peft_model(model, config)
    except ValueError as exc:
        # peft raises when none of the names exist in the model.
        raise ValidationError(
            f"No layer named {', '.join(target_modules)} exists in this model, so LoRA "
            "would train nothing but the output layer."
        ) from exc
    landed = sum(1 for name, _ in wrapped.named_modules() if name.endswith("lora_A"))
    if landed == 0:
        raise ValidationError(
            f"LoRA attached to no layer named {', '.join(target_modules)}; refusing to "
            "train only the output layer under the LoRA name."
        )
    return wrapped


def trainable_state(module: Any) -> dict[str, Any]:
    """Only the parameters that trained -- the adapters and any `modules_to_save`."""
    return {
        name: parameter.detach().cpu()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def settings(target_modules: Sequence[str]) -> dict[str, Any]:
    """What `attach` was configured with, to be stored beside the weights.

    Without this an artifact is interpreted by whatever the *current* constants say, and
    two edits to this file would change old Protocols' predictions in silence rather than
    failing: widening a target list leaves the new adapters at their zero init (a no-op
    the key check below now catches), and changing `ALPHA` alone rescales every adapter's
    contribution while the state dict still loads cleanly. Same reasoning as persisting
    `EngineManifest.version` -- a stored model has to be read the way it was written.
    """
    return {"r": RANK, "alpha": ALPHA, "dropout": DROPOUT, "targets": list(target_modules)}


def load_trainable(
    module: Any, state: dict[str, Any], saved: dict[str, Any] | None = None
) -> None:
    """Lay `trainable_state` output back over a freshly attached model.

    Refuses three ways a reload can disagree with the fit, each of which would otherwise
    predict plausible numbers from a model that is not the one the Scorecard scored:
    a saved tensor with no home, a trainable parameter the artifact does not cover, and
    `saved` settings that do not match what this code would attach today.
    """
    if saved is not None:
        now = settings(saved.get("targets", ()))
        differing = [key for key in ("r", "alpha", "dropout") if saved.get(key) != now[key]]
        if differing:
            raise ValidationError(
                "This saved model was fitted with different LoRA settings ("
                + ", ".join(f"{key} {saved.get(key)} not {now[key]}" for key in differing)
                + "), so reloading it here would not reproduce its own results. Train a "
                "new Protocol on this version."
            )
    result = module.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValidationError(
            f"{len(result.unexpected_keys)} of {len(state)} saved adapter weights do not "
            "match this model; the saved model and its base checkpoint disagree."
        )
    uncovered = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and name not in state
    ]
    if uncovered:
        raise ValidationError(
            f"{len(uncovered)} trainable weights are missing from this saved model, so "
            f"they would predict untrained (first: {uncovered[0]}). The saved model and "
            "this version of the engine disagree about what LoRA trains."
        )
