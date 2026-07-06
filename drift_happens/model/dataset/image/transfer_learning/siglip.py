from typing import Literal

import torch
import torch.nn as nn
from transformers import SiglipVisionConfig, SiglipVisionModel

from drift_happens.model.dataset.image.transfer_learning.base import (
    TransferLearningConfig,
    TransferLearningModel,
)

SigLIPModelSize = Literal["base", "large", "so400m"]


class SigLIPConfig(TransferLearningConfig):
    """Configuration for SigLip-based models."""

    model_size: SigLIPModelSize = "base"
    """SigLIP variant."""


class SigLIPTransferLearning(TransferLearningModel):
    def __init__(self, num_classes: int = 2, *, config: SigLIPConfig | None = None):
        """
        Linear classifier on the pooled features of a SigLIP vision encoder.

        ``config`` picks the variant (which also sets the expected image size) and
        whether the encoder is fine-tuned or frozen.
        """
        super().__init__()
        if config is None:
            config = SigLIPConfig()

        self.config = config

        # Map model size to Hugging Face model identifier, hidden dimensions, and image size
        model_map = {
            "base": (
                "google/siglip-base-patch16-224",
                768,
                224,
            ),  # 86M params, ViT-B, 224x224
            "large": (
                "google/siglip-large-patch16-256",
                1024,
                256,
            ),  # 303M params, ViT-L, 256x256
            "so400m": (
                "google/siglip-so400m-patch14-384",
                1152,
                384,
            ),  # 400M params, ViT-So400m, 384x384
        }

        if config.model_size not in model_map:
            raise ValueError(
                f"Unknown model_size: {config.model_size}. Choose from: {list(model_map.keys())}"
            )

        model_name, hidden_size, self.image_size = model_map[config.model_size]

        # Load the vision tower only; AutoModel would resolve to the full
        # dual-encoder SiglipModel and pull in an unused text tower.
        if config.pretrained:
            self.siglip = SiglipVisionModel.from_pretrained(model_name)
        else:
            siglip_config = SiglipVisionConfig.from_pretrained(model_name)
            self.siglip = SiglipVisionModel(siglip_config)

        # Create custom classification head
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Apply freezing strategy based on config
        self._apply_freeze_strategy(config)

    def _apply_freeze_strategy(self, config: SigLIPConfig) -> None:
        """
        Apply the appropriate freezing strategy to the backbone.

        Supports two modes:
        - Full freeze (fine_tune=False): freeze entire backbone
        - No freeze (fine_tune=True): all parameters trainable
        """
        if not config.fine_tune:
            # Standard transfer learning: freeze entire backbone
            for param in self.siglip.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
               Will be automatically upscaled to required size if needed

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        image_features = self.maybe_forward_backend(x)

        # Pass through classification head
        logits = self.classifier(image_features)

        return logits

    def forward_only_backend(self, x):
        """Only run the forward pass of the fixed pretrained model."""
        # SigLIP expects specific input sizes depending on variant (224, 256, or 384)
        if x.shape[2] != self.image_size or x.shape[3] != self.image_size:
            x = torch.nn.functional.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # SiglipVisionModel.forward delegates to the vision transformer.
        vision_outputs = self.siglip(pixel_values=x)

        # pooler_output is SigLIP's multihead-attention pooling over patch tokens
        image_features = vision_outputs.pooler_output  # (batch_size, hidden_size)

        return image_features
