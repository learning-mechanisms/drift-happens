from __future__ import annotations

from typing import Literal

import torch.nn as nn

ActivationFunc = Literal["relu", "gelu", "silu", "leaky_relu"]


def build_activation(name: ActivationFunc) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    raise ValueError(f"Unknown activation: {name}")
