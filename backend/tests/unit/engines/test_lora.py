"""The shared LoRA helper: it must refuse a zero-match, and its artifact must round-trip."""

from __future__ import annotations

import pytest

pytest.importorskip("peft")

import torch

from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.engines import _lora


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = torch.nn.Linear(6, 6)
        self.value = torch.nn.Linear(6, 6)
        self.head = torch.nn.Linear(6, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.query(x)) + self.value(x))


def _attached() -> torch.nn.Module:
    torch.manual_seed(0)
    return _lora.attach(_Tiny(), ("query", "value"), modules_to_save=("head",))


def test_a_wrong_layer_name_is_refused_not_silently_head_only() -> None:
    with pytest.raises(ValidationError):
        _lora.attach(_Tiny(), ("q_proj",))


def test_only_adapters_and_saved_modules_train() -> None:
    model = _attached()
    state = _lora.trainable_state(model)
    assert any("lora_A" in name for name in state)
    assert any("head" in name for name in state)
    assert not any(name.endswith("query.base_layer.weight") for name in state)


def test_trainable_state_round_trips_into_a_fresh_model() -> None:
    model = _attached()
    with torch.no_grad():
        # Adapters start at zero; move them so the comparison means something.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.1)
    model.eval()  # LoRA dropout is live in training mode
    x = torch.randn(4, 6)
    expected = model(x)

    fresh = _attached()
    _lora.load_trainable(fresh, _lora.trainable_state(model))
    fresh.eval()
    assert torch.allclose(fresh(x), expected)


def test_a_state_from_another_model_is_refused() -> None:
    with pytest.raises(ValidationError):
        _lora.load_trainable(_attached(), {"nowhere.lora_A.weight": torch.zeros(1)})


def test_a_trainable_weight_missing_from_the_artifact_is_refused() -> None:
    """Widening the target list after a fit leaves new adapters at their zero init --
    a no-op that would predict confidently. The reload has to notice."""
    model = _attached()
    state = _lora.trainable_state(model)
    dropped = next(name for name in state if "lora_A" in name)
    del state[dropped]
    with pytest.raises(ValidationError, match="missing from this saved model"):
        _lora.load_trainable(_attached(), state)


def test_settings_that_disagree_with_this_version_are_refused() -> None:
    """`alpha` rescales every adapter's contribution while the state dict still loads
    cleanly, so matching key names are not enough to prove the artifact is readable."""
    model = _attached()
    stale = _lora.settings(("query", "value")) | {"alpha": _lora.ALPHA + 1}
    with pytest.raises(ValidationError, match="different LoRA settings"):
        _lora.load_trainable(_attached(), _lora.trainable_state(model), stale)


def test_matching_settings_load() -> None:
    model = _attached()
    _lora.load_trainable(
        _attached(), _lora.trainable_state(model), _lora.settings(("query", "value"))
    )
