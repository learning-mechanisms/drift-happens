"""The conference image-model lineup shared by the image datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ImageModelTier = Literal["small", "medium", "large"]
ImageScratchFamily = Literal["mlp", "cnn", "resnet", "vit"]
ImageScratchModelId = Literal[
    "mlp_s",
    "mlp_m",
    "mlp_l",
    "cnn_s",
    "cnn_m",
    "cnn_l",
    "resnet_s",
    "resnet_m",
    "resnet_l",
    "vit_s",
    "vit_m",
    "vit_l",
]
ImageFrozenFamily = Literal[
    "resnet50",
    "vit",
    "dinov2",
    "dinov3",
    "convnext",
    "mae",
    "eva02",
    "clip",
    "siglip",
]
ImageFrozenModelId = Literal[
    "resnet50_in_frozen",
    "vit_s16_in21k_frozen",
    "dinov2_s_frozen",
    "dinov3_s_frozen",
    "convnext_s_frozen",
    "mae_b_frozen",
    "eva02_b_frozen",
    "clip_b32_frozen",
    "siglip_b_frozen",
]


@dataclass(frozen=True, slots=True)
class ImageScratchModel:
    """One scratch-trained model in the conference image lineup."""

    model_id: ImageScratchModelId
    family: ImageScratchFamily
    tier: ImageModelTier


@dataclass(frozen=True, slots=True)
class ImageFrozenModel:
    """One frozen-backbone model in the conference image lineup."""

    model_id: ImageFrozenModelId
    family: ImageFrozenFamily
    producer: str


IMAGE_SCRATCH_MODELS: tuple[ImageScratchModel, ...] = (
    ImageScratchModel(model_id="mlp_s", family="mlp", tier="small"),
    ImageScratchModel(model_id="mlp_m", family="mlp", tier="medium"),
    ImageScratchModel(model_id="mlp_l", family="mlp", tier="large"),
    ImageScratchModel(model_id="cnn_s", family="cnn", tier="small"),
    ImageScratchModel(model_id="cnn_m", family="cnn", tier="medium"),
    ImageScratchModel(model_id="cnn_l", family="cnn", tier="large"),
    ImageScratchModel(model_id="resnet_s", family="resnet", tier="small"),
    ImageScratchModel(model_id="resnet_m", family="resnet", tier="medium"),
    ImageScratchModel(model_id="resnet_l", family="resnet", tier="large"),
    ImageScratchModel(model_id="vit_s", family="vit", tier="small"),
    ImageScratchModel(model_id="vit_m", family="vit", tier="medium"),
    ImageScratchModel(model_id="vit_l", family="vit", tier="large"),
)

IMAGE_FROZEN_MODELS: tuple[ImageFrozenModel, ...] = (
    ImageFrozenModel(
        model_id="resnet50_in_frozen",
        family="resnet50",
        producer="resnet50.a1_in1k",
    ),
    ImageFrozenModel(
        model_id="vit_s16_in21k_frozen",
        family="vit",
        producer="vit_small_patch16_224.augreg_in21k",
    ),
    ImageFrozenModel(
        model_id="dinov2_s_frozen",
        family="dinov2",
        producer="dinov2-small",
    ),
    ImageFrozenModel(
        model_id="dinov3_s_frozen",
        family="dinov3",
        producer="dinov3-small",
    ),
    ImageFrozenModel(
        model_id="convnext_s_frozen",
        family="convnext",
        producer="torchvision.convnext_small.imagenet1k_v1",
    ),
    ImageFrozenModel(
        model_id="mae_b_frozen",
        family="mae",
        producer="vit_base_patch16_224.mae",
    ),
    ImageFrozenModel(
        model_id="eva02_b_frozen",
        family="eva02",
        producer="eva02-base",
    ),
    ImageFrozenModel(
        model_id="clip_b32_frozen",
        family="clip",
        producer="openai/clip-vit-base-patch32",
    ),
    ImageFrozenModel(
        model_id="siglip_b_frozen",
        family="siglip",
        producer="google/siglip-base-patch16-224",
    ),
)
