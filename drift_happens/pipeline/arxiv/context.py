from dataclasses import dataclass

import torch

from drift_happens.pipeline.arxiv.trainers import (
    ArxivTrainingConfig,
)
from drift_happens.pipeline.context import PipelineContext


@dataclass(frozen=True)
class ArxivPipelineContext(PipelineContext):
    trainer_configs: dict[str, ArxivTrainingConfig]

    category_to_idx: dict[str, int]
    pos_weight: torch.Tensor
