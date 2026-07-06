"""Linear heads for cached pooled text-backbone embeddings."""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch
import torch.nn as nn

FrozenTextBackboneId = Literal[
    "minilm_l6_frozen",
    "distilbert_base_frozen",
    "bert_base_frozen",
    "roberta_base_frozen",
    "deberta_v3_base_frozen",
    "electra_base_frozen",
    "mpnet_base_frozen",
    "modernbert_base_frozen",
]


class FrozenTextBackbone(NamedTuple):
    backbone_id: FrozenTextBackboneId
    producer: str
    embedding_dim: int


FROZEN_TEXT_BACKBONES: tuple[FrozenTextBackbone, ...] = (
    FrozenTextBackbone(
        "minilm_l6_frozen", "sentence-transformers/all-MiniLM-L6-v2", 384
    ),
    FrozenTextBackbone("distilbert_base_frozen", "distilbert-base-uncased", 768),
    FrozenTextBackbone("bert_base_frozen", "bert-base-uncased", 768),
    FrozenTextBackbone("roberta_base_frozen", "roberta-base", 768),
    FrozenTextBackbone("deberta_v3_base_frozen", "microsoft/deberta-v3-base", 768),
    FrozenTextBackbone("electra_base_frozen", "google/electra-base-discriminator", 768),
    FrozenTextBackbone("mpnet_base_frozen", "microsoft/mpnet-base", 768),
    FrozenTextBackbone("modernbert_base_frozen", "answerdotai/ModernBERT-base", 768),
)

FROZEN_TEXT_BACKBONE_IDS: tuple[FrozenTextBackboneId, ...] = tuple(
    backbone.backbone_id for backbone in FROZEN_TEXT_BACKBONES
)
FROZEN_TEXT_BACKBONE_DIMS: dict[str, int] = {
    backbone.backbone_id: backbone.embedding_dim for backbone in FROZEN_TEXT_BACKBONES
}
FROZEN_TEXT_BACKBONE_PRODUCERS: dict[str, str] = {
    backbone.backbone_id: backbone.producer for backbone in FROZEN_TEXT_BACKBONES
}


class PooledEmbeddingHead(nn.Module):
    """Head-only model for cached pooled text features."""

    def __init__(self, *, input_dim: int, dim_output: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, dim_output)

    def forward(self, pooled_embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled_embeddings.float())


def pooled_embedding_head_for_backbone(
    backbone_id: FrozenTextBackboneId | str, *, dim_output: int
) -> PooledEmbeddingHead:
    try:
        input_dim = FROZEN_TEXT_BACKBONE_DIMS[backbone_id]
    except KeyError as exc:
        raise ValueError(f"Unknown frozen text backbone: {backbone_id}") from exc
    return PooledEmbeddingHead(input_dim=input_dim, dim_output=dim_output)
