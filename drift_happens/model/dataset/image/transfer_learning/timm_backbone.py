"""Generic timm-backed frozen image feature adapters."""

from __future__ import annotations

from typing import Literal, NamedTuple

import timm
import torch
import torch.nn as nn

from drift_happens.model.dataset.image.transfer_learning.base import (
    TransferLearningConfig,
    TransferLearningModel,
)

TimmBackbonePreset = Literal[
    "resnet50_in",
    "vit_s16_in21k",
    "mae_b",
]


class TimmBackboneConfig(TransferLearningConfig):
    preset: TimmBackbonePreset = "resnet50_in"


class TimmBackboneSpec(NamedTuple):
    model_name: str
    feature_dim: int
    image_size: int


_TIMM_BACKBONES: dict[TimmBackbonePreset, TimmBackboneSpec] = {
    "resnet50_in": TimmBackboneSpec("resnet50.a1_in1k", 2048, 224),
    "vit_s16_in21k": TimmBackboneSpec("vit_small_patch16_224.augreg_in21k", 384, 224),
    "mae_b": TimmBackboneSpec("vit_base_patch16_224.mae", 768, 224),
}


class TimmBackboneTransferLearning(TransferLearningModel):
    """Frozen timm feature extractor plus a linear task head."""

    def __init__(
        self,
        num_classes: int = 2,
        *,
        config: TimmBackboneConfig | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = TimmBackboneConfig()
        self.config = config
        try:
            model_name, feature_dim, image_size = _TIMM_BACKBONES[config.preset]
        except KeyError as exc:
            raise ValueError(f"Unknown timm backbone preset: {config.preset}") from exc

        self.image_size = image_size
        self.backbone = timm.create_model(
            model_name,
            pretrained=config.pretrained,
            num_classes=0,
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        self._apply_freeze_strategy(config)

    def _apply_freeze_strategy(self, config: TimmBackboneConfig) -> None:
        if not config.fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.maybe_forward_backend(x)
        return self.classifier(features)

    def forward_only_backend(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != self.image_size or x.shape[3] != self.image_size:
            x = torch.nn.functional.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.backbone(x)
