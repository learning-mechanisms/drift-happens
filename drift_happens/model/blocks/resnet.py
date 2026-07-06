from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel

from drift_happens.model.blocks.activation import ActivationFunc, build_activation


class ResNetConfig(BaseModel):
    block_channels: Sequence[int] = (64, 128, 256)
    blocks_per_stage: Sequence[int] = (2, 2, 2)
    use_batchnorm: bool = True
    activation: ActivationFunc = "relu"
    initial_conv_channels: int = 64
    initial_kernel_size: int = 3
    initial_stride: int = 1
    initial_pool: bool = False  # e.g. for ImageNet-style ResNets
    global_pool: Literal["avg", "max"] = "avg"


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_batchnorm: bool = True,
        activation: ActivationFunc = "relu",
    ) -> None:
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.act = build_activation(activation)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not use_batchnorm,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not use_batchnorm,
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        self.downsample: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=not use_batchnorm,
                ),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(identity)
        out += identity
        out = self.act(out)

        return out


class ImageResNet(nn.Module):
    """
    Configurable ResNet-like architecture for small images.

    Example (ResNet-18-ish): block_channels=[64, 128, 256], blocks_per_stage=[2, 2, 2].
    """

    def __init__(
        self,
        num_input_channels: int,
        num_classes: int,
        config: ResNetConfig | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = ResNetConfig()
        self.config = config

        self.act = build_activation(config.activation)

        # Initial stem
        self.conv1 = nn.Conv2d(
            num_input_channels,
            config.initial_conv_channels,
            kernel_size=config.initial_kernel_size,
            stride=config.initial_stride,
            padding=config.initial_kernel_size // 2,
            bias=not config.use_batchnorm,
        )
        self.bn1 = (
            nn.BatchNorm2d(config.initial_conv_channels)
            if config.use_batchnorm
            else nn.Identity()
        )
        self.pool1 = (
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if config.initial_pool
            else nn.Identity()
        )

        # Residual stages
        if len(config.block_channels) != len(config.blocks_per_stage):
            raise ValueError(
                "block_channels and blocks_per_stage must have equal length"
            )
        in_channels = config.initial_conv_channels
        stages: list[nn.Module] = []
        for out_channels, num_blocks in zip(
            config.block_channels, config.blocks_per_stage, strict=True
        ):
            stages.append(self._make_stage(in_channels, out_channels, num_blocks))
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)

        # Global pooling & classifier
        if config.global_pool == "avg":
            self.global_pool: nn.Module = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.fc = nn.Linear(in_channels, num_classes)

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
    ) -> nn.Sequential:
        if num_blocks < 1:
            raise ValueError("each stage needs at least one block")
        blocks: list[nn.Module] = []
        # First block of the stage may downsample (stride=2) if in!=out
        stride = 1 if in_channels == out_channels else 2
        blocks.append(
            BasicBlock(
                in_channels,
                out_channels,
                stride=stride,
                use_batchnorm=self.config.use_batchnorm,
                activation=self.config.activation,
            )
        )
        for _ in range(1, num_blocks):
            blocks.append(
                BasicBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    use_batchnorm=self.config.use_batchnorm,
                    activation=self.config.activation,
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool1(x)

        x = self.stages(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
