"""LoRA adapters for the transformer engines, and the artifact that holds only them.

What LoRA buys here is **size, not memory**. At the parameter counts on this roster
(15M-650M) a LoRA fit still backpropagates activations through every layer, so it saves
little GPU memory -- gradient checkpointing is the lever for that. What it does change is
the saved model: a few megabytes of adapter weights instead of the whole checkpoint
(190 MB for MoLFormer, 2.6 GB for ESM-2 650M), with the base rebuilt from its pinned
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


def load_trainable(module: Any, state: dict[str, Any]) -> None:
    """Lay `trainable_state` output back over a freshly attached model, refusing if any
    saved tensor found no home -- a silent partial load predicts with untrained adapters."""
    result = module.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValidationError(
            f"{len(result.unexpected_keys)} of {len(state)} saved adapter weights do not "
            "match this model; the saved model and its base checkpoint disagree."
        )
