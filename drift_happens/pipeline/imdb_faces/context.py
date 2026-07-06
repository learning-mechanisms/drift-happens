from dataclasses import dataclass

from drift_happens.pipeline.context import PipelineContext
from drift_happens.pipeline.imdb_faces.trainers import ImdbTrainingConfig


@dataclass(frozen=True)
class ImdbPipelineContext(PipelineContext):
    trainer_configs: dict[str, ImdbTrainingConfig]
