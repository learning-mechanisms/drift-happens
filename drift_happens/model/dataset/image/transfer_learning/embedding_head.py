"""Head-only image classifiers for cached frozen-backbone embeddings."""

import torch
import torch.nn as nn


class CachedEmbeddingHead(nn.Module):
    """Linear classifier over precomputed image backbone embeddings."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
