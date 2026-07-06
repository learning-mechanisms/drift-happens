"""Gradient-based saliency maps for image classifiers."""

import numpy as np
import torch
import torch.nn as nn


def compute_saliency_map(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Compute saliency map via backprop."""
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    # Add batch dim if needed
    if image.ndim == 3:
        image = image.unsqueeze(0)

    image = image.to(device).contiguous()
    image.requires_grad = True

    output = model(image)

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    score = output[0, target_class]
    (gradients,) = torch.autograd.grad(score, image)
    gradients = gradients.squeeze(0)  # (C,H,W)

    # Max absolute gradient across channels
    saliency = gradients.abs().max(dim=0)[0]

    saliency_np = saliency.cpu().numpy()
    saliency_np = normalize_map(saliency_np)

    return saliency_np


def normalize_map(map_array: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1]."""
    map_min = map_array.min()
    map_max = map_array.max()

    if map_max - map_min < 1e-8:
        return np.zeros_like(map_array)

    return (map_array - map_min) / (map_max - map_min)
