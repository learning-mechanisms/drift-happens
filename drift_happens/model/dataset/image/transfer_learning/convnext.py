from typing import Literal, cast

import torch
import torch.nn as nn
import torchvision.models as models

from drift_happens.model.dataset.image.transfer_learning.base import (
    TransferLearningConfig,
    TransferLearningModel,
)

ConvNeXtModelSize = Literal["tiny", "small", "base", "large"]

# Variant builder paired with its ImageNet weights.
_CONVNEXT_VARIANTS = {
    "tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "small": (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1),
    "base": (models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1),
    "large": (models.convnext_large, models.ConvNeXt_Large_Weights.IMAGENET1K_V1),
}


class ConvNeXtConfig(TransferLearningConfig):
    """Configuration for ConvNeXt-based models."""

    model_size: ConvNeXtModelSize = "small"
    """ConvNeXt variant."""


class ConvNeXtTransferLearning(TransferLearningModel):
    def __init__(
        self,
        num_classes: int = 2,
        *,
        config: ConvNeXtConfig | None = None,
    ):
        """
        Torchvision ConvNeXt with its final linear layer swapped for ``num_classes``.

        ``config`` picks the variant and whether the features are fine-tuned or frozen.
        """
        super().__init__()
        if config is None:
            config = ConvNeXtConfig()

        self.config = config

        if config.model_size not in _CONVNEXT_VARIANTS:
            raise ValueError(
                f"Unknown model_size: {config.model_size}. "
                f"Choose from: {list(_CONVNEXT_VARIANTS.keys())}"
            )
        builder, weights = _CONVNEXT_VARIANTS[config.model_size]
        self.model = builder(weights=weights if config.pretrained else None)

        # Replace final classification layer
        # ConvNeXt classifier structure: Sequential([LayerNorm, Flatten, Linear])
        # Only replace the Linear layer (index 2) with custom head
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_classes)

        # Apply freezing strategy based on config
        self._apply_freeze_strategy(config)

    def _apply_freeze_strategy(self, config: ConvNeXtConfig) -> None:
        """
        Apply the appropriate freezing strategy to the backbone.

        Supports two modes:
        - Full freeze (fine_tune=False): freeze the features and the pre-head
          LayerNorm; only the replaced head (classifier[2]) stays trainable
        - No freeze (fine_tune=True): all parameters trainable
        """
        if not config.fine_tune:
            # classifier[0] is the pre-head LayerNorm forward_only_backend runs.
            for param in self.model.features.parameters():
                param.requires_grad = False
            for param in self.model.classifier[0].parameters():
                param.requires_grad = False

    @property
    def classifier(self) -> nn.Linear:
        """Expose the head under the name the cached-embedding handoff expects."""
        return cast(nn.Linear, self.model.classifier[2])

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
               Will be automatically upscaled to 224x224 if needed

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features = self.maybe_forward_backend(x)
        return self.classifier(features)

    def forward_only_backend(self, x):
        """Only run the forward pass of the fixed pretrained model."""
        # ConvNeXt expects 224x224 input, upscale if needed
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )

        x = self.model.features(x)
        x = self.model.avgpool(x)

        # Match the existing classifier structure comment:
        # Sequential([LayerNorm, Flatten, Linear])
        x = self.model.classifier[0](x)
        x = self.model.classifier[1](x)

        return x
