"""Small Vision Transformer blocks for 32x32 image experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import BaseModel


class ViTConfig(BaseModel):
    patch_size: int = 4
    embed_dim: int = 96
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1


class ImageViT(nn.Module):
    """Minimal ViT for fixed-size small images."""

    def __init__(
        self,
        *,
        in_channels: int,
        image_size: int,
        num_classes: int,
        config: ViTConfig,
    ) -> None:
        super().__init__()
        if image_size % config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.config = config
        num_patches = (image_size // config.patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.embed_dim * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(x + self.pos_embed)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)
