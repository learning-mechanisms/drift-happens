import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    # Targets are 1-indexed integer class labels (e.g. ratings 1..len(weights)),
    # used both as regression targets and as weight indices.
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.register_buffer("weights", weights, persistent=False)

    def forward(self, inputs, targets):
        if inputs.dim() > 1:
            if inputs.shape[-1] != 1:
                raise ValueError(
                    f"expected inputs of shape (N,) or (N, 1), got {tuple(inputs.shape)}"
                )
            inputs = inputs.squeeze(-1)

        fixed_targets = targets - 1

        assert isinstance(self.weights, torch.Tensor)
        assert fixed_targets.min() >= 0
        assert fixed_targets.max() < self.weights.shape[0]

        return (((inputs - targets) ** 2) * self.weights[fixed_targets]).mean()
