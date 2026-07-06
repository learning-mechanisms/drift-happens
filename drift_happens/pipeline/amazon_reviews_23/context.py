from dataclasses import dataclass

import torch

from drift_happens.pipeline.amazon_reviews_23.trainers import (
    AmazonReviewsTrainingConfig,
)
from drift_happens.pipeline.context import PipelineContext


@dataclass(frozen=True)
class AmazonReviewsPipelineContext(PipelineContext):
    trainer_configs: dict[str, AmazonReviewsTrainingConfig]

    class_weights: torch.Tensor
