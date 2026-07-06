"""Tests for the DINOv3 backbone pooling contract."""

from types import SimpleNamespace

import pytest
import torch

from drift_happens.model.dataset.image.transfer_learning.dino_v3 import (
    DINOv3TransferLearning,
)


def _model_with_outputs(outputs: object) -> DINOv3TransferLearning:
    model = DINOv3TransferLearning.__new__(DINOv3TransferLearning)
    torch.nn.Module.__init__(model)
    model.dinov3 = lambda pixel_values: outputs
    return model


def test_forward_only_backend_returns_pooler_output() -> None:
    pooled = torch.arange(6.0).reshape(2, 3)
    model = _model_with_outputs(
        SimpleNamespace(pooler_output=pooled, last_hidden_state=torch.zeros(2, 5, 3))
    )

    out = model.forward_only_backend(torch.zeros(2, 3, 224, 224))

    assert torch.equal(out, pooled)


def test_forward_only_backend_raises_without_pooler_output() -> None:
    model = _model_with_outputs(
        SimpleNamespace(pooler_output=None, last_hidden_state=torch.ones(2, 5, 3))
    )

    with pytest.raises(RuntimeError, match="pooler_output"):
        model.forward_only_backend(torch.zeros(2, 3, 224, 224))
