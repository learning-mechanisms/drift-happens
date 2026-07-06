from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from drift_happens.model.blocks.activation import ActivationFunc, build_activation


class CNNConfig(BaseModel):
    """Configuration for the generic image CNN backbone."""

    # Convolutional layers
    channels: Sequence[int] = (32, 32, 32, 32)
    kernel_size: int = 3

    # Normalization and activation
    use_batchnorm: bool = True
    activation: ActivationFunc = "relu"

    # Pooling
    pool_type: Literal["max", "avg", "none"] = "max"
    pool_every: int = Field(default=1, ge=1)  # pool after every N conv blocks
    global_pool: Literal["avg", "max"] = "avg"

    # MLP head
    mlp_head_dims: Sequence[int] | None = None  # e.g. (128,) or (256,128)

    # Regularization / experimentation dimensions
    conv_dropout: float = 0.0
    """Dropout probability applied as 2D dropout after conv+activation blocks (0.0 =
    off)."""

    head_dropout: float = 0.0
    """Dropout probability applied in the MLP head after activations (0.0 = off)."""

    weight_decay: float = 0.0
    """L2 weight decay to be used in the optimizer (not applied inside the module)."""


class ImageCNN(nn.Module):
    """
    Flexible CNN backbone for image classification.

    Examples:
    - VGG-ish: channels=[64,64,128,128,256,256], pool_every=2
    - AlexNet-ish: channels=[64,192,384,256,256], larger kernels, etc.
    """

    def __init__(
        self,
        num_input_channels: int,
        num_classes: int,
        config: CNNConfig | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = CNNConfig()

        self.config = config
        in_ch = num_input_channels
        layers: list[nn.Module] = []

        # ---------------------------------- ENCODER --------------------------------- #
        for i, out_ch in enumerate(config.channels):
            # conv
            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=config.kernel_size,
                    padding=config.kernel_size // 2,
                )
            )
            if config.use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(build_activation(config.activation))

            # optional conv dropout (spatial)
            if config.conv_dropout > 0.0:
                layers.append(nn.Dropout2d(config.conv_dropout))

            # optional pooling
            if config.pool_type != "none" and (i + 1) % config.pool_every == 0:
                if config.pool_type == "max":
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                elif config.pool_type == "avg":
                    layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

            in_ch = out_ch

        self.encoder = nn.Sequential(*layers)
        self.feature_channels = in_ch  # channels after encoder

        # pooling head
        self.global_pool = config.global_pool

        # --------------------------------- HEAD MLP --------------------------------- #
        mlp_dims = list(config.mlp_head_dims) if config.mlp_head_dims else []
        head_layers: list[nn.Module] = []

        prev_dim = self.feature_channels

        for hidden_dim in mlp_dims:
            head_layers.append(nn.Linear(prev_dim, hidden_dim))
            head_layers.append(build_activation(config.activation))
            if config.head_dropout > 0.0:
                head_layers.append(nn.Dropout(config.head_dropout))
            prev_dim = hidden_dim

        # final classifier
        head_layers.append(nn.Linear(prev_dim, num_classes))
        self.head = nn.Sequential(*head_layers)

    def _apply_global_pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        if self.global_pool == "avg":
            return torch.mean(x, dim=(2, 3))  # [B, C]
        if self.global_pool == "max":
            return torch.amax(x, dim=(2, 3))  # [B, C]
        raise ValueError(f"Unsupported global_pool: {self.global_pool}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self._apply_global_pool(x)
        x = self.head(x)
        return x
