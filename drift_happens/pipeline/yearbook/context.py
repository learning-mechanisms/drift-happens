from dataclasses import dataclass

from drift_happens.pipeline.context import PipelineContext
from drift_happens.pipeline.yearbook.trainers import (
    YearbookTrainingConfig,
)


@dataclass(frozen=True)
class YearbookPipelineContext(PipelineContext):
    trainer_configs: dict[str, YearbookTrainingConfig]
