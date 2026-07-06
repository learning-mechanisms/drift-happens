"""
Single source of truth for conference trainer hyperparameters.

The experiment preset builders (which materialise the snapshots) and the pipeline
trainer factories (which build the runtime trainers) both read these values, so a
snapshot can never silently disagree with what actually trains.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from pydantic import JsonValue


@dataclass(frozen=True)
class ConferenceTextTraining:
    """Hyperparameters shared by every conference text trainer."""

    batch_size: int = 64
    learning_rate: float = 5e-4
    num_epochs: int = 10
    weight_decay: float = 0.01
    optimizer: Literal["adam", "adamw"] = "adamw"
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class ConferenceImageTraining:
    """
    Hyperparameters shared by every conference image trainer.

    Image trainers split the learning rate between scratch and frozen-backbone models;
    weight decay is derived from the model preset and the optimizer is fixed, so neither
    belongs here.
    """

    batch_size: int = 64
    num_epochs: int = 10
    scratch_learning_rate: float = 5e-4
    frozen_learning_rate: float = 1e-4


CONFERENCE_TEXT_TRAINING = ConferenceTextTraining()
CONFERENCE_IMAGE_TRAINING = ConferenceImageTraining()


def conference_text_training() -> dict[str, JsonValue]:
    """Return the conference text hyperparameters as a fresh snapshot dict."""
    return asdict(CONFERENCE_TEXT_TRAINING)


def conference_image_training(*, frozen: bool) -> dict[str, JsonValue]:
    """Return the conference image hyperparameters as a fresh snapshot dict."""
    return {
        "batch_size": CONFERENCE_IMAGE_TRAINING.batch_size,
        "learning_rate": (
            CONFERENCE_IMAGE_TRAINING.frozen_learning_rate
            if frozen
            else CONFERENCE_IMAGE_TRAINING.scratch_learning_rate
        ),
        "num_epochs": CONFERENCE_IMAGE_TRAINING.num_epochs,
    }
