"""Text models that consume cached sequence embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel


class EmbeddingFeatureFFNConfig(BaseModel):
    hidden_dims: Sequence[int]
    dropout: float = 0.1


class EmbeddingFeatureCNNConfig(BaseModel):
    num_filters: int
    kernel_sizes: Sequence[int]
    projection_dim: int | None = None
    dropout: float = 0.1


class EmbeddingFeatureRNNConfig(BaseModel):
    projection_dim: int
    hidden_dim: int
    num_layers: int
    rnn_type: Literal["gru", "lstm"] = "gru"
    bidirectional: bool = True
    use_attention: bool = False
    dropout: float = 0.1


class EmbeddingFeatureTransformerConfig(BaseModel):
    projection_dim: int
    num_heads: int
    num_layers: int
    dim_feedforward: int
    dropout: float = 0.1


class EmbeddingFeatureFFN(nn.Module):
    """Masked mean-pool over sequence embeddings followed by an MLP."""

    def __init__(
        self,
        *,
        input_dim: int,
        dim_output: int,
        config: EmbeddingFeatureFFNConfig,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, dim_output))
        self.net = nn.Sequential(*layers)

    def forward(
        self, sequence_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        pooled = masked_mean_pool(sequence_embeddings.float(), attention_mask)
        return self.net(pooled)


class EmbeddingFeatureCNN(nn.Module):
    """Temporal Conv1d classifier over cached sequence embeddings."""

    def __init__(
        self,
        *,
        input_dim: int,
        dim_output: int,
        config: EmbeddingFeatureCNNConfig,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(input_dim, config.num_filters, kernel_size=kernel_size)
                for kernel_size in config.kernel_sizes
            ]
        )
        conv_dim = config.num_filters * len(config.kernel_sizes)
        self.projection: nn.Module
        if config.projection_dim is None:
            self.projection = nn.Identity()
            head_dim = conv_dim
        else:
            self.projection = nn.Sequential(
                nn.Linear(conv_dim, config.projection_dim),
                nn.ReLU(),
            )
            head_dim = config.projection_dim
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(head_dim, dim_output)

    def forward(
        self, sequence_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        x = _zero_masked_tokens(sequence_embeddings.float(), attention_mask)
        x = x.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))
            pooled.append(torch.amax(h, dim=2))
        features = torch.cat(pooled, dim=1)
        features = self.projection(features)
        return self.head(self.dropout(features))


class EmbeddingFeatureRNN(nn.Module):
    """GRU/LSTM body over cached sequence embeddings."""

    def __init__(
        self,
        *,
        input_dim: int,
        dim_output: int,
        config: EmbeddingFeatureRNNConfig,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, config.projection_dim),
            nn.ReLU(),
        )
        rnn_cls: type[nn.GRU] | type[nn.LSTM] = (
            nn.GRU if config.rnn_type == "gru" else nn.LSTM
        )
        self.rnn = rnn_cls(
            input_size=config.projection_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        direction_factor = 2 if config.bidirectional else 1
        output_dim = config.hidden_dim * direction_factor
        self.attention: nn.Linear | None
        if config.use_attention:
            self.attention = nn.Linear(output_dim, 1)
        else:
            self.attention = None
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(output_dim, dim_output)

    def forward(
        self, sequence_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        projected = self.projection(sequence_embeddings.float())
        # Pack so the recurrence never runs over padding; the backward direction
        # would otherwise consume the whole pad tail before any real token.
        lengths = attention_mask.sum(dim=1).to(torch.int64).cpu().clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(
            projected, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=projected.shape[1]
        )
        if self.attention is not None:
            scores = self.attention(output).squeeze(-1)
            scores = scores.masked_fill(
                ~attention_mask.bool(), torch.finfo(scores.dtype).min
            )
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = torch.sum(output * weights, dim=1)
        else:
            pooled = masked_mean_pool(output, attention_mask)
        return self.head(self.dropout(pooled))


class EmbeddingFeatureTransformer(nn.Module):
    """Transformer encoder over projected cached sequence embeddings."""

    def __init__(
        self,
        *,
        input_dim: int,
        dim_output: int,
        config: EmbeddingFeatureTransformerConfig,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, config.projection_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.projection_dim,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.projection_dim, dim_output)

    def forward(
        self, sequence_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.projection(sequence_embeddings.float())
        padding_mask = ~attention_mask.bool()
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        pooled = masked_mean_pool(x, attention_mask)
        return self.head(self.dropout(pooled))


def masked_mean_pool(
    sequence_embeddings: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    masked = _zero_masked_tokens(sequence_embeddings, attention_mask)
    denom = attention_mask.to(dtype=sequence_embeddings.dtype).sum(dim=1, keepdim=True)
    return masked.sum(dim=1) / denom.clamp_min(1.0)


def _zero_masked_tokens(
    sequence_embeddings: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask.to(dtype=sequence_embeddings.dtype).unsqueeze(-1)
    return sequence_embeddings * mask
