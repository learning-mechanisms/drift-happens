"""Smoke test for the gradient-saliency attribution used by the saliency figure."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from drift_happens.evaluation.interpretability.saliency import compute_saliency_map


class _TinyNet(nn.Module):
    """Minimal conv classifier so saliency backprop has something to flow through."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.relu(self.conv(x)).mean(dim=(2, 3)))


class _ViewNet(nn.Module):
    """Minimal classifier with a view-based flattening path."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.shape[0], -1))


def test_compute_saliency_map_smoke() -> None:
    torch.manual_seed(0)
    saliency = compute_saliency_map(_TinyNet(), torch.randn(3, 32, 32))

    assert saliency.shape == (32, 32)
    assert np.all(np.isfinite(saliency))
    assert saliency.min() >= 0.0
    assert saliency.max() <= 1.0


def test_compute_saliency_map_handles_noncontiguous_images() -> None:
    torch.manual_seed(0)
    image = torch.randn(32, 32, 3).permute(2, 0, 1)
    saliency = compute_saliency_map(_ViewNet(), image)

    assert saliency.shape == (32, 32)
