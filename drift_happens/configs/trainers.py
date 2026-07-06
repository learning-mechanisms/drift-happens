"""Typed trainer config validators used at runtime boundaries."""

from __future__ import annotations

from typing import Literal, NamedTuple

from pydantic import Field

from drift_happens.configs.base import BaseConfig
from drift_happens.configs.experiment import TrainerConfig


class SyntheticLinearModelConfig(BaseConfig):
    architecture: str = "linear"


class SyntheticTrainingConfig(BaseConfig):
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    n_features: int = Field(gt=0)
    n_samples: int = Field(gt=0)
    num_epochs: int = Field(gt=0)


class ImageModelConfig(BaseConfig):
    architecture: str
    preset: str | None = None
    input_channels: int | None = None
    image_size: list[int] | None = None
    fine_tune: bool | None = None
    needs_backend_fw_pass: bool | None = None


class ImageTrainingConfig(BaseConfig):
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    num_epochs: int = Field(gt=0)


ModelConfigName = Literal["synthetic-linear", "image"]
TrainingConfigName = Literal["synthetic", "image"]


class TrainerConfigSpec(NamedTuple):
    model: ModelConfigName
    training: TrainingConfigName


_MODEL_CONFIGS: dict[ModelConfigName, type[BaseConfig]] = {
    "synthetic-linear": SyntheticLinearModelConfig,
    "image": ImageModelConfig,
}
_TRAINING_CONFIGS: dict[TrainingConfigName, type[BaseConfig]] = {
    "synthetic": SyntheticTrainingConfig,
    "image": ImageTrainingConfig,
}
_REGISTRY: dict[str, TrainerConfigSpec] = {
    "linear-classifier": TrainerConfigSpec("synthetic-linear", "synthetic"),
    "mlp_s": TrainerConfigSpec("image", "image"),
    "dinov2_s_frozen": TrainerConfigSpec("image", "image"),
}


def validate_registered_trainer_config(trainer: TrainerConfig) -> None:
    """
    Validate typed config for registered trainer keys.

    Unknown trainer keys are intentionally left to external factories.
    """
    spec = _REGISTRY.get(trainer.key)
    if spec is None:
        return
    model_name, training_name = spec
    _MODEL_CONFIGS[model_name].model_validate(trainer.model)
    _TRAINING_CONFIGS[training_name].model_validate(trainer.training)
