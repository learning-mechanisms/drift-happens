"""Freeze-strategy tests for the ConvNeXt transfer-learning backbone."""

from __future__ import annotations

import torchvision.models as tvmodels

from drift_happens.model.dataset.image.transfer_learning import convnext as convnext_mod
from drift_happens.model.dataset.image.transfer_learning.convnext import (
    ConvNeXtConfig,
    ConvNeXtTransferLearning,
)


def _build_offline(monkeypatch, *, fine_tune: bool) -> ConvNeXtTransferLearning:
    real = tvmodels.convnext_tiny
    monkeypatch.setattr(
        convnext_mod.models,
        "convnext_tiny",
        lambda weights=None: real(weights=None),
    )
    return ConvNeXtTransferLearning(
        num_classes=2,
        config=ConvNeXtConfig(model_size="tiny", fine_tune=fine_tune),
    )


def test_frozen_convnext_freezes_pre_head_layernorm(monkeypatch) -> None:
    model = _build_offline(monkeypatch, fine_tune=False)
    flags = {name: p.requires_grad for name, p in model.named_parameters()}

    # classifier[0] is the LayerNorm forward_only_backend runs as the backbone.
    assert flags["model.classifier.0.weight"] is False
    assert flags["model.classifier.0.bias"] is False
    assert all(
        rg is False for name, rg in flags.items() if name.startswith("model.features.")
    )
    # The replaced head stays trainable.
    assert flags["model.classifier.2.weight"] is True
    assert flags["model.classifier.2.bias"] is True


def test_finetuned_convnext_keeps_everything_trainable(monkeypatch) -> None:
    model = _build_offline(monkeypatch, fine_tune=True)

    assert all(p.requires_grad for p in model.parameters())
