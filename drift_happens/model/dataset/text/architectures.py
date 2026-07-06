"""Conference text architectures over cached backbone features."""

from collections.abc import Callable
from functools import partial
from typing import Literal, NamedTuple

from pydantic import BaseModel
from torch import nn

from drift_happens.model.text.feature_models import (
    EmbeddingFeatureCNN,
    EmbeddingFeatureCNNConfig,
    EmbeddingFeatureFFN,
    EmbeddingFeatureFFNConfig,
    EmbeddingFeatureRNN,
    EmbeddingFeatureRNNConfig,
    EmbeddingFeatureTransformer,
    EmbeddingFeatureTransformerConfig,
)
from drift_happens.model.text.frozen_backbone import (
    FROZEN_TEXT_BACKBONE_IDS,
    pooled_embedding_head_for_backbone,
)

TextModelArchitecture = Literal[
    # cached sequence-embedding conference models
    "ffn_s",
    "ffn_m",
    "ffn_l",
    "textcnn_s",
    "textcnn_m",
    "textcnn_l",
    "bigru_s",
    "bilstm_m",
    "bilstm_attn_l",
    "tx_s",
    "tx_m",
    "tx_l",
    # cached pooled frozen-backbone heads
    "minilm_l6_frozen",
    "distilbert_base_frozen",
    "bert_base_frozen",
    "roberta_base_frozen",
    "deberta_v3_base_frozen",
    "electra_base_frozen",
    "mpnet_base_frozen",
    "modernbert_base_frozen",
]
TextScratchFamily = Literal["ffn", "textcnn", "rnn", "transformer"]
TextScratchTier = Literal["small", "medium", "large"]


class TextScratchMetadata(NamedTuple):
    family: TextScratchFamily
    tier: TextScratchTier


# Scratch sequence architectures: model class plus the factory for its config.
# Both are instantiated lazily by text_model_factory, so every call gets a fresh
# config instance.
_SEQUENCE_MODEL_REGISTRY: dict[
    TextModelArchitecture, tuple[Callable[..., nn.Module], Callable[[], BaseModel]]
] = {
    "ffn_s": (
        EmbeddingFeatureFFN,
        partial(EmbeddingFeatureFFNConfig, hidden_dims=(128,)),
    ),
    "ffn_m": (
        EmbeddingFeatureFFN,
        partial(EmbeddingFeatureFFNConfig, hidden_dims=(512,)),
    ),
    "ffn_l": (
        EmbeddingFeatureFFN,
        partial(EmbeddingFeatureFFNConfig, hidden_dims=(2048,)),
    ),
    "textcnn_s": (
        EmbeddingFeatureCNN,
        partial(
            EmbeddingFeatureCNNConfig,
            num_filters=16,
            kernel_sizes=(3, 4),
            projection_dim=64,
        ),
    ),
    "textcnn_m": (
        EmbeddingFeatureCNN,
        partial(
            EmbeddingFeatureCNNConfig,
            num_filters=48,
            kernel_sizes=(3, 4, 5),
            projection_dim=128,
        ),
    ),
    "textcnn_l": (
        EmbeddingFeatureCNN,
        partial(
            EmbeddingFeatureCNNConfig,
            num_filters=192,
            kernel_sizes=(3, 4, 5),
            projection_dim=256,
        ),
    ),
    "bigru_s": (
        EmbeddingFeatureRNN,
        partial(
            EmbeddingFeatureRNNConfig,
            projection_dim=64,
            hidden_dim=64,
            num_layers=1,
        ),
    ),
    "bilstm_m": (
        EmbeddingFeatureRNN,
        partial(
            EmbeddingFeatureRNNConfig,
            projection_dim=160,
            hidden_dim=160,
            num_layers=1,
            rnn_type="lstm",
        ),
    ),
    "bilstm_attn_l": (
        EmbeddingFeatureRNN,
        partial(
            EmbeddingFeatureRNNConfig,
            projection_dim=224,
            hidden_dim=224,
            num_layers=2,
            rnn_type="lstm",
            use_attention=True,
        ),
    ),
    "tx_s": (
        EmbeddingFeatureTransformer,
        partial(
            EmbeddingFeatureTransformerConfig,
            projection_dim=64,
            num_heads=4,
            num_layers=1,
            dim_feedforward=128,
        ),
    ),
    "tx_m": (
        EmbeddingFeatureTransformer,
        partial(
            EmbeddingFeatureTransformerConfig,
            projection_dim=112,
            num_heads=4,
            num_layers=4,
            dim_feedforward=224,
        ),
    ),
    "tx_l": (
        EmbeddingFeatureTransformer,
        partial(
            EmbeddingFeatureTransformerConfig,
            projection_dim=192,
            num_heads=6,
            num_layers=5,
            dim_feedforward=512,
        ),
    ),
}

CONFERENCE_SEQUENCE_TEXT_ARCHITECTURES: tuple[TextModelArchitecture, ...] = tuple(
    _SEQUENCE_MODEL_REGISTRY
)

# (family, tier) for each scratch sequence architecture. Keys mirror
# CONFERENCE_SEQUENCE_TEXT_ARCHITECTURES (enforced by a test); used by the arxiv
# and amazon conference preset builders for descriptive metadata/tags.
TEXT_SCRATCH_FAMILIES: dict[str, TextScratchMetadata] = {
    "ffn_s": TextScratchMetadata("ffn", "small"),
    "ffn_m": TextScratchMetadata("ffn", "medium"),
    "ffn_l": TextScratchMetadata("ffn", "large"),
    "textcnn_s": TextScratchMetadata("textcnn", "small"),
    "textcnn_m": TextScratchMetadata("textcnn", "medium"),
    "textcnn_l": TextScratchMetadata("textcnn", "large"),
    "bigru_s": TextScratchMetadata("rnn", "small"),
    "bilstm_m": TextScratchMetadata("rnn", "medium"),
    "bilstm_attn_l": TextScratchMetadata("rnn", "large"),
    "tx_s": TextScratchMetadata("transformer", "small"),
    "tx_m": TextScratchMetadata("transformer", "medium"),
    "tx_l": TextScratchMetadata("transformer", "large"),
}

CONFERENCE_FROZEN_TEXT_ARCHITECTURES: tuple[TextModelArchitecture, ...] = tuple(
    FROZEN_TEXT_BACKBONE_IDS
)

CONFERENCE_TEXT_MODEL_ARCHITECTURES: tuple[TextModelArchitecture, ...] = (
    *CONFERENCE_SEQUENCE_TEXT_ARCHITECTURES,
    *CONFERENCE_FROZEN_TEXT_ARCHITECTURES,
)

CONFERENCE_ARXIV_MAX_SEQ_LEN = 256
CONFERENCE_AMAZON_REVIEWS_MAX_SEQ_LEN = 128


def text_model_factory(
    architecture_name: TextModelArchitecture,
    dim_output: int,
    feature_input_dim: int = 768,
) -> nn.Module:
    """
    Factory function for text classification/regression models.

    Sequence models consume cached sequence embeddings plus masks; the frozen-backbone
    heads consume cached pooled embeddings.
    """
    if architecture_name in _SEQUENCE_MODEL_REGISTRY:
        model_cls, config_factory = _SEQUENCE_MODEL_REGISTRY[architecture_name]
        return model_cls(
            input_dim=feature_input_dim,
            dim_output=dim_output,
            config=config_factory(),
        )

    if architecture_name in FROZEN_TEXT_BACKBONE_IDS:
        return pooled_embedding_head_for_backbone(
            architecture_name, dim_output=dim_output
        )

    raise ValueError(f"unknown text architecture: {architecture_name}")
