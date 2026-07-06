"""Tests that SigLIP loads only the vision tower, not the full dual-encoder."""

import torch
import torch.nn as nn

import drift_happens.model.dataset.image.transfer_learning.siglip as siglip_mod
from drift_happens.model.dataset.image.transfer_learning.siglip import (
    SigLIPConfig,
    SigLIPTransferLearning,
)


class _FakeVisionTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 768)
        self.calls = 0

    def forward(self, pixel_values: torch.Tensor):
        self.calls += 1
        pooled = pixel_values.flatten(1)[:, :768]
        return type("Out", (), {"pooler_output": pooled})()


class _FakeSiglipVisionModel:
    @staticmethod
    def from_pretrained(model_name: str) -> _FakeVisionTower:
        return _FakeVisionTower()


def test_siglip_loads_vision_tower_only(monkeypatch) -> None:
    monkeypatch.setattr(siglip_mod, "SiglipVisionModel", _FakeSiglipVisionModel)

    model = SigLIPTransferLearning(
        num_classes=2, config=SigLIPConfig(model_size="base")
    )

    # The full SiglipModel would carry a text tower; the vision-only model does not.
    assert not hasattr(model.siglip, "text_model")

    # forward must call self.siglip directly, not self.siglip.vision_model.
    out = model.forward_only_backend(torch.randn(2, 3, 224, 224))
    assert model.siglip.calls == 1
    assert out.shape == (2, 768)
