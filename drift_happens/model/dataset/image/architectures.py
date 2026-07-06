"""Aggregate different neural architectures for image classification."""

from typing import Literal

import torch.nn as nn

from drift_happens.model.blocks.cnn import CNNConfig, ImageCNN
from drift_happens.model.blocks.mlp import ImageMLP, MLPConfig
from drift_happens.model.blocks.resnet import ImageResNet, ResNetConfig
from drift_happens.model.blocks.vit import ImageViT, ViTConfig
from drift_happens.model.dataset.image.transfer_learning.clip import (
    ClipConfig,
    ClipModelSize,
    CLIPTransferLearning,
)
from drift_happens.model.dataset.image.transfer_learning.convnext import (
    ConvNeXtConfig,
    ConvNeXtModelSize,
    ConvNeXtTransferLearning,
)
from drift_happens.model.dataset.image.transfer_learning.dino import DinoModelSize
from drift_happens.model.dataset.image.transfer_learning.dino_v2 import (
    DinoV2Config,
    DINOv2TransferLearning,
)
from drift_happens.model.dataset.image.transfer_learning.dino_v3 import (
    DinoV3Config,
    DINOv3TransferLearning,
)
from drift_happens.model.dataset.image.transfer_learning.eva02 import (
    EVA02Config,
    EVA02ModelSize,
    EVA02TransferLearning,
)
from drift_happens.model.dataset.image.transfer_learning.siglip import (
    SigLIPConfig,
    SigLIPModelSize,
    SigLIPTransferLearning,
)
from drift_happens.model.dataset.image.transfer_learning.timm_backbone import (
    TimmBackboneConfig,
    TimmBackbonePreset,
    TimmBackboneTransferLearning,
)

CNNPreset = Literal["cnn_s", "cnn_m", "cnn_l"]
ResNetPreset = Literal["resnet_s", "resnet_m", "resnet_l"]
ViTPreset = Literal["vit_s", "vit_m", "vit_l"]
MLPPreset = Literal["mlp_s", "mlp_m", "mlp_l"]

DinoV2Preset = DinoModelSize
DinoV3Preset = DinoModelSize
ClipPreset = ClipModelSize
ConvNeXtPreset = ConvNeXtModelSize
SigLIPPreset = SigLIPModelSize
EVA02Preset = EVA02ModelSize
TimmPreset = TimmBackbonePreset


class ImageModelFactory:
    """
    Factory for all image model families: CNN, ResNet, MLP, ViT, DINOv2, DINOv3, CLIP,
    ConvNeXt, SigLIP, EVA-02, and timm backbones.

    You can either:
        - pass explicit configs
        - or use named presets like "cnn_s", "resnet_m", "mlp_l".
    """

    # -------------------------------------------------------------------------------- #
    #                                  FACTORY METHODS                                 #
    # -------------------------------------------------------------------------------- #

    @staticmethod
    def create_mlp(
        num_input_channels: int,
        height: int,
        width: int,
        num_classes: int,
        config: MLPConfig | None = None,
        preset: MLPPreset | None = None,
    ) -> ImageMLP:
        """Create an ImageMLP from an explicit config or a named preset."""
        if preset is not None:
            config = ImageModelFactory._mlp_preset(preset)
        if config is None:
            config = MLPConfig()

        return ImageMLP(
            in_channels=num_input_channels,
            height=height,
            width=width,
            num_classes=num_classes,
            config=config,
        )

    @staticmethod
    def create_cnn(
        num_input_channels: int,
        num_classes: int,
        config: CNNConfig | None = None,
        preset: CNNPreset | None = None,
    ) -> ImageCNN:
        if preset is not None:
            config = ImageModelFactory._cnn_preset(preset)
        if config is None:
            config = CNNConfig()
        return ImageCNN(num_input_channels, num_classes, config=config)

    @staticmethod
    def create_resnet(
        num_input_channels: int,
        num_classes: int,
        config: ResNetConfig | None = None,
        preset: ResNetPreset | None = None,
    ) -> ImageResNet:
        if preset is not None:
            config = ImageModelFactory._resnet_preset(preset)
        if config is None:
            config = ResNetConfig()
        return ImageResNet(num_input_channels, num_classes, config=config)

    @staticmethod
    def create_vit(
        num_input_channels: int,
        image_size: int,
        num_classes: int,
        config: ViTConfig | None = None,
        preset: ViTPreset | None = None,
    ) -> ImageViT:
        if preset is not None:
            config = ImageModelFactory._vit_preset(preset)
        if config is None:
            config = ViTConfig()
        return ImageViT(
            in_channels=num_input_channels,
            image_size=image_size,
            num_classes=num_classes,
            config=config,
        )

    @staticmethod
    def create_dinov2(
        num_classes: int = 2,
        config: DinoV2Config | None = None,
        preset: DinoV2Preset | None = None,
    ) -> nn.Module:
        """Create a DINO-based transfer learning model."""
        if preset is not None:
            config = ImageModelFactory._dinov2_preset(preset)
        if config is None:
            config = DinoV2Config()

        return DINOv2TransferLearning(num_classes=num_classes, config=config)

    @staticmethod
    def create_dinov3(
        num_classes: int = 2,
        config: DinoV3Config | None = None,
        preset: DinoV3Preset | None = None,
    ) -> nn.Module:
        """Create a DINOv3-based transfer learning model."""
        if preset is not None:
            config = ImageModelFactory._dinov3_preset(preset)
        if config is None:
            config = DinoV3Config()

        return DINOv3TransferLearning(num_classes=num_classes, config=config)

    @staticmethod
    def create_clip(
        num_classes: int = 2,
        config: ClipConfig | None = None,
        preset: ClipPreset | None = None,
    ) -> nn.Module:
        """Create a CLIP-based transfer learning model."""
        if preset is not None:
            config = ImageModelFactory._clip_preset(preset)
        if config is None:
            config = ClipConfig()

        return CLIPTransferLearning(num_classes=num_classes, config=config)

    @staticmethod
    def create_convnext(
        num_classes: int = 2,
        config: ConvNeXtConfig | None = None,
        preset: ConvNeXtPreset | None = None,
    ) -> nn.Module:
        """Create a ConvNeXt-based transfer learning model."""
        if preset is not None:
            config = ImageModelFactory._convnext_preset(preset)
        if config is None:
            config = ConvNeXtConfig()

        return ConvNeXtTransferLearning(num_classes=num_classes, config=config)

    @staticmethod
    def create_siglip(
        num_classes: int = 2,
        config: SigLIPConfig | None = None,
        preset: SigLIPPreset | None = None,
    ) -> nn.Module:
        """Create a SigLIP-based transfer learning model."""
        if preset is not None:
            config = ImageModelFactory._siglip_preset(preset)
        if config is None:
            config = SigLIPConfig()

        return SigLIPTransferLearning(num_classes=num_classes, config=config)

    @staticmethod
    def create_eva02(
        num_classes: int = 2,
        config: EVA02Config | None = None,
        preset: EVA02Preset | None = None,
    ) -> nn.Module:
        """Create an EVA-02-based transfer learning model."""
        if preset is not None:
            config = ImageModelFactory._eva02_preset(preset)
        if config is None:
            config = EVA02Config()

        return EVA02TransferLearning(num_classes=num_classes, config=config)

    @staticmethod
    def create_timm_backbone(
        num_classes: int = 2,
        config: TimmBackboneConfig | None = None,
        preset: TimmPreset | None = None,
    ) -> nn.Module:
        if preset is not None:
            config = ImageModelFactory._timm_preset(preset)
        if config is None:
            config = TimmBackboneConfig()
        return TimmBackboneTransferLearning(num_classes=num_classes, config=config)

    # -------------------------------------------------------------------------------- #
    #                                      PRESETS                                     #
    # -------------------------------------------------------------------------------- #

    @staticmethod
    def _mlp_preset(name: MLPPreset) -> MLPConfig:
        if name == "mlp_s":
            return MLPConfig(
                hidden_layers=(32,),
                activation="relu",
                normalization="none",
                dropout=0.0,
                weight_decay=0.0,
            )

        if name == "mlp_m":
            return MLPConfig(
                hidden_layers=(128, 128),
                activation="relu",
                normalization="none",
                dropout=0.0,
                weight_decay=0.0,
            )

        if name == "mlp_l":
            return MLPConfig(
                hidden_layers=(512, 512, 512),
                activation="relu",
                normalization="none",
                dropout=0.0,
                weight_decay=0.0,
            )

        raise ValueError(f"Unknown MLP preset: {name}")

    @staticmethod
    def _cnn_preset(name: CNNPreset) -> CNNConfig:
        if name == "cnn_s":
            return CNNConfig(
                channels=(32, 64, 128),
                kernel_size=3,
                pool_every=1,
                pool_type="max",
                global_pool="avg",
                mlp_head_dims=None,
                conv_dropout=0.0,
                head_dropout=0.0,
                weight_decay=0.0,
            )

        if name == "cnn_m":
            return CNNConfig(
                channels=(48, 96, 192, 192),
                kernel_size=3,
                pool_every=1,
                pool_type="max",
                global_pool="avg",
                mlp_head_dims=None,
                conv_dropout=0.0,
                head_dropout=0.0,
                weight_decay=0.0,
            )

        if name == "cnn_l":
            return CNNConfig(
                channels=(64, 128, 256, 256, 512),
                kernel_size=3,
                pool_every=1,
                pool_type="max",
                global_pool="avg",
                mlp_head_dims=None,
                conv_dropout=0.0,
                head_dropout=0.0,
                weight_decay=0.0,
            )

        raise ValueError(f"Unknown CNN preset: {name}")

    @staticmethod
    def _resnet_preset(name: ResNetPreset) -> ResNetConfig:
        if name == "resnet_s":
            return ResNetConfig(
                block_channels=(18, 36, 72),
                blocks_per_stage=(1, 1, 1),
                initial_conv_channels=18,
                initial_kernel_size=3,
                initial_stride=1,
                initial_pool=False,
                global_pool="avg",
            )
        if name == "resnet_m":
            return ResNetConfig(
                block_channels=(32, 64, 128),
                blocks_per_stage=(2, 2, 1),
                initial_conv_channels=32,
                initial_kernel_size=3,
                initial_stride=1,
                initial_pool=False,
                global_pool="avg",
            )
        if name == "resnet_l":
            return ResNetConfig(
                block_channels=(56, 112, 224),
                blocks_per_stage=(2, 2, 2),
                initial_conv_channels=56,
                initial_kernel_size=3,
                initial_stride=1,
                initial_pool=False,
                global_pool="avg",
            )
        raise ValueError(f"Unknown ResNet preset: {name}")

    @staticmethod
    def _vit_preset(name: ViTPreset) -> ViTConfig:
        if name == "vit_s":
            return ViTConfig(
                patch_size=4,
                embed_dim=64,
                num_layers=2,
                num_heads=4,
                mlp_ratio=2.0,
                dropout=0.1,
            )
        if name == "vit_m":
            return ViTConfig(
                patch_size=4,
                embed_dim=128,
                num_layers=4,
                num_heads=4,
                mlp_ratio=2.0,
                dropout=0.1,
            )
        if name == "vit_l":
            return ViTConfig(
                patch_size=4,
                embed_dim=192,
                num_layers=6,
                num_heads=6,
                mlp_ratio=3.0,
                dropout=0.1,
            )
        raise ValueError(f"Unknown ViT preset: {name}")

    @staticmethod
    def _dinov2_preset(name: DinoV2Preset) -> DinoV2Config:
        return DinoV2Config(model_size=name)

    @staticmethod
    def _dinov3_preset(name: DinoV3Preset) -> DinoV3Config:
        return DinoV3Config(model_size=name)

    @staticmethod
    def _clip_preset(name: ClipPreset) -> ClipConfig:
        return ClipConfig(model_size=name)

    @staticmethod
    def _convnext_preset(name: ConvNeXtPreset) -> ConvNeXtConfig:
        return ConvNeXtConfig(model_size=name)

    @staticmethod
    def _siglip_preset(name: SigLIPPreset) -> SigLIPConfig:
        return SigLIPConfig(model_size=name)

    @staticmethod
    def _eva02_preset(name: EVA02Preset) -> EVA02Config:
        return EVA02Config(model_size=name)

    @staticmethod
    def _timm_preset(name: TimmPreset) -> TimmBackboneConfig:
        return TimmBackboneConfig(preset=name)
