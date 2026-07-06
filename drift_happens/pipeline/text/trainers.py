from typing import Literal

import torch.nn as nn
from pydantic import model_validator

from drift_happens.model.dataset.text.architectures import (
    TextModelArchitecture,
    text_model_factory,
)
from drift_happens.model.text.frozen_backbone import FROZEN_TEXT_BACKBONE_DIMS
from drift_happens.model.trainer.pytorch import EpochPrintMode
from drift_happens.pipeline.training_config import TrainingConfig


class TextTrainingConfig(TrainingConfig):
    architecture_name: TextModelArchitecture

    weight_decay: float = 0.0
    optimizer: Literal["adam", "adamw"] = "adam"
    gradient_clip_norm: float | None = None

    feature_input_dim: int = 768
    print_mode: EpochPrintMode = False

    @model_validator(mode="after")
    def _use_frozen_backbone_dim(self) -> "TextTrainingConfig":
        # A frozen-backbone head is sized from its backbone, ignoring the passed
        # feature_input_dim; record the real dim so the config does not misreport.
        backbone_dim = FROZEN_TEXT_BACKBONE_DIMS.get(self.architecture_name)
        if backbone_dim is not None:
            self.feature_input_dim = backbone_dim
        return self


def model_factory(config: TextTrainingConfig, dim_output: int) -> nn.Module:
    return text_model_factory(
        architecture_name=config.architecture_name,
        dim_output=dim_output,
        feature_input_dim=config.feature_input_dim,
    )
