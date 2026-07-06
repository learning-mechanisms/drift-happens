from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from drift_happens.model.blocks.activation import build_activation
from drift_happens.model.blocks.cnn import CNNConfig
from drift_happens.model.blocks.mlp import ImageMLP, MLPConfig
from drift_happens.model.blocks.resnet import ImageResNet, ResNetConfig
from drift_happens.model.blocks.vit import ImageViT, ViTConfig


def test_image_mlp_direct_linear_forward_shape() -> None:
    model = ImageMLP(1, 4, 4, 3, MLPConfig(hidden_layers=()))

    out = model(torch.randn(2, 1, 4, 4))

    assert out.shape == (2, 3)


def test_image_resnet_max_pool_forward_shape() -> None:
    model = ImageResNet(
        1,
        2,
        ResNetConfig(
            block_channels=(4,),
            blocks_per_stage=(1,),
            initial_conv_channels=4,
            initial_pool=True,
            global_pool="max",
        ),
    )

    out = model(torch.randn(2, 1, 8, 8))

    assert out.shape == (2, 2)


def test_image_vit_rejects_non_divisible_patch_size() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ImageViT(
            in_channels=1, image_size=10, num_classes=2, config=ViTConfig(patch_size=4)
        )


def test_cnn_config_rejects_non_positive_pool_every() -> None:
    # pool_every=0 would divide by zero when deciding which blocks pool.
    with pytest.raises(ValidationError, match="pool_every"):
        CNNConfig(pool_every=0)


def test_activation_factory_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="Unknown activation"):
        build_activation("unknown")  # type: ignore[arg-type]
