from typing import Literal

from drift_happens.model.dataset.image.transfer_learning.base import (
    TransferLearningConfig,
)

# Covers shared size names across DINOv2 + DINOv3 families.
# DINOv2: small/base/large/giant
# DINOv3: small/base/large/huge
DinoModelSize = Literal["small", "base", "large", "giant", "huge"]


class DinoConfig(TransferLearningConfig):
    """Configuration for DINO-based models."""

    model_size: DinoModelSize = "small"
    """DINO variant."""
