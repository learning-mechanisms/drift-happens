from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel

from drift_happens.model.blocks.activation import ActivationFunc, build_activation


class MLPConfig(BaseModel):
    """Configuration for the fully-connected image MLP classifier."""

    hidden_layers: Sequence[int] = (256, 128, 64)
    """
    Sizes of the hidden layers, e.g. (256, 128, 64).

    [] means 'no hidden layers', i.e. direct linear classifier.
    """

    activation: ActivationFunc = "relu"
    """Activation function to use in hidden layers."""

    normalization: Literal["none", "batchnorm", "layernorm"] = "none"
    """Optional normalization layer after each Linear:
    - 'none': no normalization
    - 'batchnorm': nn.BatchNorm1d
    - 'layernorm': nn.LayerNorm
    """

    dropout: float = 0.0
    """Dropout probability applied after activation in each hidden layer."""

    weight_decay: float = 0.0
    """L2 weight decay to use in the optimizer (not applied inside the module)."""


class ImageMLP(nn.Module):
    """
    A fully-connected MLP for image classification.

    Args:
        in_channels: Number of input channels (e.g. 3 for RGB, 13 for multispectral).
        height: Image height in pixels.
        width: Image width in pixels.
        num_classes: Number of output classes.
        config: MLPConfig with architectural hyperparameters (hidden layers,
            activation, normalization, dropout, weight decay).
    """

    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        num_classes: int,
        config: MLPConfig,
    ):
        super().__init__()
        self.config = config

        input_dim = in_channels * height * width
        layers: list[nn.Module] = []
        prev_dim = input_dim

        # hidden layers
        for hidden_dim in config.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # normalization (if any)
            if config.normalization == "batchnorm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif config.normalization == "layernorm":
                layers.append(nn.LayerNorm(hidden_dim))

            # activation
            layers.append(build_activation(config.activation))

            # dropout
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))

            prev_dim = hidden_dim

        # final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))

        # wrap as Sequential
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b = x.shape[0]
        x = x.view(b, -1)  # flatten
        logits = self.net(x)  # (B, num_classes)
        return logits
