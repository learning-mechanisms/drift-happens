"""DINOv2 transfer-learning model definitions."""

from typing import cast

import torch
import torch.nn as nn

from drift_happens.model.dataset.image.transfer_learning.base import (
    TransferLearningModel,
)
from drift_happens.model.dataset.image.transfer_learning.dino import DinoConfig
from drift_happens.utils.log import get_logger

logger = get_logger()

_DINO_INPUT_SIZE = 224  # spatial resolution expected by all DINOv2 variants


class DinoV2Config(DinoConfig):
    pass


class DINOv2TransferLearning(TransferLearningModel):
    def __init__(self, num_classes: int = 2, *, config: DinoV2Config | None = None):
        """
        DINOv2 backbone (torch hub) with a linear classification head.

        ``config`` picks the variant and whether the backbone is fine-tuned or frozen.
        """
        super().__init__()

        if config is None:
            config = DinoV2Config()

        self.config = config

        # Map model size to torch.hub model name and feature dimensions
        model_map = {
            "small": ("dinov2_vits14", 384),  # 21M params
            "base": ("dinov2_vitb14", 768),  # 86M params
            "large": ("dinov2_vitl14", 1024),  # 300M params
            "giant": ("dinov2_vitg14", 1536),  # 1.1B params
        }

        if config.model_size not in model_map:
            raise ValueError(f"Unknown model_size: {config.model_size}")

        model_name, feature_dim = model_map[config.model_size]

        self.backbone = _load_dinov2_backbone(model_name, pretrained=config.pretrained)

        # Classification head
        self.head = nn.Linear(feature_dim, num_classes, bias=True)

        # Apply freezing strategy based on config
        self._apply_freeze_strategy(config)

    def _apply_freeze_strategy(self, config: DinoV2Config) -> None:
        """
        Apply the appropriate freezing strategy to the backbone.

        Supports two modes:
        - Full freeze (fine_tune=False): freeze entire backbone
        - No freeze (fine_tune=True): all parameters trainable
        """
        if not config.fine_tune:
            # Standard transfer learning: freeze entire backbone
            for param in self.backbone.parameters():
                param.requires_grad = False

    @property
    def classifier(self) -> nn.Linear:
        """Expose the head under the name the cached-embedding handoff expects."""
        return self.head

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
               Resized to 224x224 if not already that resolution.

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features = self.maybe_forward_backend(x)
        return self.head(features)

    def forward_only_backend(self, x):
        """Only run the forward pass of the fixed pretrained model."""
        # Resize to the expected input resolution if not already correct.
        if x.shape[2] != _DINO_INPUT_SIZE or x.shape[3] != _DINO_INPUT_SIZE:
            x = torch.nn.functional.interpolate(
                x,
                size=(_DINO_INPUT_SIZE, _DINO_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        return self.backbone(x)


def _load_dinov2_backbone(model_name: str, *, pretrained: bool = True) -> nn.Module:
    """Load a DINOv2 backbone, refreshing a corrupt torch hub cache once."""
    kwargs = {} if pretrained else {"pretrained": False}
    try:
        return cast(
            nn.Module,
            torch.hub.load("facebookresearch/dinov2", model_name, **kwargs),
        )
    except ModuleNotFoundError as error:
        # A partially corrupt cache fails on a dinov2 submodule, whose name is the
        # dotted path; refresh for those too, not just the top-level package.
        if error.name is None or not (
            error.name == "dinov2" or error.name.startswith("dinov2.")
        ):
            raise
        logger.warning(
            "DINOv2 torch hub cache is missing package modules; forcing reload."
        )
        return cast(
            nn.Module,
            torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                force_reload=True,
                **kwargs,
            ),
        )
