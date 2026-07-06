from typing import Literal

import timm
import torch
import torch.nn as nn

from drift_happens.model.dataset.image.transfer_learning.base import (
    TransferLearningConfig,
    TransferLearningModel,
)

EVA02ModelSize = Literal["tiny", "small", "base", "large"]


class EVA02Config(TransferLearningConfig):
    """Configuration for EVA-02-based models."""

    model_size: EVA02ModelSize = "base"
    """EVA-02 variant."""


class EVA02TransferLearning(TransferLearningModel):
    def __init__(self, num_classes: int = 2, *, config: EVA02Config | None = None):
        """
        EVA-02 feature extractor (timm) with a linear classification head.

        ``config`` picks the variant and whether the backbone is fine-tuned or frozen.
        """
        super().__init__()
        if config is None:
            config = EVA02Config()

        self.config = config

        # Map model size to timm model identifier, feature dimensions, and input size
        model_map = {
            "tiny": ("eva02_tiny_patch14_224.mim_in22k", 192, 224),  # 5.5M params
            "small": ("eva02_small_patch14_224.mim_in22k", 384, 224),  # 21.6M params
            "base": ("eva02_base_patch14_224.mim_in22k", 768, 224),  # 85.8M params
            "large": ("eva02_large_patch14_224.mim_in22k", 1024, 224),  # 303.3M params
        }

        if config.model_size not in model_map:
            raise ValueError(
                f"Unknown model_size: {config.model_size}. Choose from: {list(model_map.keys())}"
            )

        model_name, feature_dim, self.image_size = model_map[config.model_size]

        # num_classes=0 creates a feature extractor without classification head.
        self.eva02 = timm.create_model(
            model_name,
            pretrained=config.pretrained,
            num_classes=0,  # Remove classification head, use as feature extractor
        )

        # Create custom classification head
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Apply freezing strategy based on config
        self._apply_freeze_strategy(config)

    def _apply_freeze_strategy(self, config: EVA02Config) -> None:
        """
        Apply the appropriate freezing strategy to the backbone.

        Supports two modes:
        - Full freeze (fine_tune=False): freeze entire backbone
        - No freeze (fine_tune=True): all parameters trainable
        """
        if not config.fine_tune:
            # Standard transfer learning: freeze entire backbone
            for param in self.eva02.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: images (batch_size, 3, H, W) when needs_backend_fw_pass,
               otherwise precomputed embeddings (batch_size, feature_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features = self.maybe_forward_backend(x)

        # Pass through classification head
        logits = self.classifier(features)

        return logits

    def forward_only_backend(self, x):
        """Only run the forward pass of the fixed pretrained model."""
        # EVA-02 expects square inputs, upscale if needed
        if x.shape[2] != self.image_size or x.shape[3] != self.image_size:
            x = torch.nn.functional.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Extract features from EVA-02 backbone
        # With num_classes=0, timm models return pooled features
        features = self.eva02(x)  # (batch_size, feature_dim)

        return features
