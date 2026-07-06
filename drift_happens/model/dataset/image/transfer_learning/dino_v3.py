import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from drift_happens.model.dataset.image.transfer_learning.base import (
    TransferLearningModel,
)
from drift_happens.model.dataset.image.transfer_learning.dino import DinoConfig


class DinoV3Config(DinoConfig):
    pass


class DINOv3TransferLearning(TransferLearningModel):
    def __init__(self, num_classes: int = 2, *, config: DinoV3Config | None = None):
        """
        DINOv3 backbone (Hugging Face) with a linear classification head.

        ``config`` picks the variant and whether the backbone is fine-tuned or frozen.
        """
        super().__init__()
        if config is None:
            config = DinoV3Config()

        self.config = config

        # Map model size to Hugging Face model identifier
        # Using DINOv3 ViT models
        model_map = {
            "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "huge": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        }

        if config.model_size not in model_map:
            raise ValueError(
                f"Unknown model_size: {config.model_size}. Choose from: {list(model_map.keys())}"
            )

        model_name = model_map[config.model_size]
        if config.pretrained:
            self.dinov3 = AutoModel.from_pretrained(model_name)
        else:
            self.dinov3 = AutoModel.from_config(AutoConfig.from_pretrained(model_name))

        # Get hidden dimension from model config
        # ViT models typically have a single hidden_size
        hidden_size = self.dinov3.config.hidden_size

        # Create custom classification head
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Apply freezing strategy based on config
        self._apply_freeze_strategy(config)

    def _apply_freeze_strategy(self, config: DinoV3Config) -> None:
        """
        Apply the appropriate freezing strategy to the backbone.

        Supports two modes:
        - Full freeze (fine_tune=False): freeze entire backbone
        - No freeze (fine_tune=True): all parameters trainable
        """
        if not config.fine_tune:
            # Standard transfer learning: freeze entire backbone
            for param in self.dinov3.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Images (batch_size, 3, H, W) when needs_backend_fw_pass is True
               (resized to 224×224 internally), or precomputed backbone embeddings
               (batch_size, hidden_size) when needs_backend_fw_pass is False.

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features = self.maybe_forward_backend(x)

        # Pass features through custom classification head
        logits = self.classifier(features)

        return logits

    def forward_only_backend(self, x):
        """Only run the forward pass of the fixed pretrained model."""
        # DINOv3 expects 224x224 images, upscale if needed
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )

        # Extract features from DINOv3 backbone
        outputs = self.dinov3(pixel_values=x)

        # DINOv3 returns the pooled CLS token; fail loud rather than silently
        # switching to a different (mean-pooled) embedding space.
        features = getattr(outputs, "pooler_output", None)
        if features is None:
            raise RuntimeError("DINOv3 backbone returned no pooler_output")

        return features
