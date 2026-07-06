from __future__ import annotations

import torch

from drift_happens.model.text.weighted_mse_loss import WeightedMSELoss


def test_weighted_mse_loss_applies_class_weights() -> None:
    loss = WeightedMSELoss(weights=torch.tensor([2.0, 4.0]))

    out = loss(torch.tensor([2.0, 1.0]), torch.tensor([1, 2]))

    assert out == torch.tensor(3.0)


def test_weighted_mse_loss_squeezes_column_outputs() -> None:
    loss = WeightedMSELoss(weights=torch.ones(2))

    out = loss(torch.tensor([[1.0], [2.0]]), torch.tensor([1, 2]))

    assert out == torch.tensor(0.0)
