"""CLIP vision encoder transfer-learning models."""

from typing import Literal

import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, CLIPVisionModel

from drift_happens.model.dataset.image.transfer_learning.base import (
    TransferLearningConfig,
    TransferLearningModel,
)

ClipModelSize = Literal["base-patch32", "base-patch16", "large-patch14"]


class ClipConfig(TransferLearningConfig):
    """Configuration for Clip-based models."""

    model_size: ClipModelSize = "base-patch32"
    """Clip variant."""


class CLIPTransferLearning(TransferLearningModel):
    def __init__(
        self,
        num_classes: int = 2,
        *,
        config: ClipConfig | None = None,
    ):
        """
        Linear classifier on the pooled features of a CLIP vision encoder.

        ``config`` picks the variant and whether the encoder is fine-tuned or frozen.
        """
        super().__init__()
        if config is None:
            config = ClipConfig()

        self.config = config

        # Map model size to Hugging Face model identifier
        model_map = {
            "base-patch32": "openai/clip-vit-base-patch32",
            "base-patch16": "openai/clip-vit-base-patch16",
            "large-patch14": "openai/clip-vit-large-patch14",
        }
        model_name = model_map[config.model_size]

        if config.pretrained:
            self.vision_encoder = CLIPVisionModel.from_pretrained(model_name)
        else:
            vision_config = CLIPVisionConfig.from_pretrained(model_name)
            self.vision_encoder = CLIPVisionModel(vision_config)
        image_size = self.vision_encoder.config.image_size
        if not isinstance(image_size, int):
            raise TypeError(f"CLIP image_size must be an int, got {image_size!r}")
        self.image_size: int = image_size

        # Classification head on the pooled features
        self.classifier = nn.Linear(self.vision_encoder.config.hidden_size, num_classes)

        # Apply freezing strategy based on config
        self._apply_freeze_strategy(config)

    def _apply_freeze_strategy(self, config: ClipConfig) -> None:
        """
        Apply the appropriate freezing strategy to the backbone.

        Supports two modes:
        - Full freeze (fine_tune=False): freeze entire backbone
        - No freeze (fine_tune=True): all parameters trainable
        """
        if not config.fine_tune:
            # Standard transfer learning: freeze entire backbone
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: raw images (batch_size, 3, H, W) when needs_backend_fw_pass is True,
               otherwise precomputed embeddings (batch_size, hidden_size).

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        image_features = self.maybe_forward_backend(x)

        # Pass through classification head
        logits = self.classifier(image_features)

        return logits

    def forward_only_backend(self, x):
        """Only run the forward pass of the fixed pretrained model."""
        # CLIP expects square inputs of the encoder's image_size, resize if needed
        if x.shape[2] != self.image_size or x.shape[3] != self.image_size:
            x = torch.nn.functional.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Get image features from CLIP vision encoder
        # pooler_output gives us the pooled CLS token representation
        vision_outputs = self.vision_encoder(pixel_values=x)
        image_features = vision_outputs.pooler_output  # (batch_size, hidden_size)

        return image_features
