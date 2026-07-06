"""Construct different Arxiv trainers determining various architectures and training
configurations."""

from functools import partial

import torch
from pydantic import field_serializer

from drift_happens.model.dataset.text.architectures import (
    CONFERENCE_TEXT_MODEL_ARCHITECTURES,
)
from drift_happens.model.trainer.pytorch import (
    EpochPrintMode,
    PytorchTrainer,
    PytorchTrainerConfig,
)
from drift_happens.pipeline._shared.conference_defaults import (
    CONFERENCE_TEXT_TRAINING,
)
from drift_happens.pipeline._shared.optimizers import make_optimizer_factory
from drift_happens.pipeline.text.trainers import TextTrainingConfig, model_factory
from drift_happens.utils.pytorch import device_manual_mps_or_cuda_if_available

# ------------------------------ TRAINER CONFIGURATIONS ------------------------------ #


class ArxivTrainingConfig(TextTrainingConfig):
    category_to_idx: dict[str, int]
    pos_weight: torch.Tensor | None = None

    @field_serializer("pos_weight")
    def serialize_pos_weight(self, value: torch.Tensor | None) -> list[float] | None:
        return None if value is None else value.tolist()

    @property
    def num_classes(self) -> int:
        return len(self.category_to_idx)


def arxiv_conference_trainer_configs(
    category_to_idx: dict[str, int],
    *,
    pos_weight: torch.Tensor | None = None,
    print_mode: EpochPrintMode = False,
) -> dict[str, ArxivTrainingConfig]:
    """Build the cached-feature text lineup for the conference arXiv scope."""
    return {
        architecture_name: ArxivTrainingConfig(
            architecture_name=architecture_name,
            category_to_idx=category_to_idx,
            pos_weight=pos_weight,
            batch_size=CONFERENCE_TEXT_TRAINING.batch_size,
            learning_rate=CONFERENCE_TEXT_TRAINING.learning_rate,
            num_epochs=CONFERENCE_TEXT_TRAINING.num_epochs,
            weight_decay=CONFERENCE_TEXT_TRAINING.weight_decay,
            optimizer=CONFERENCE_TEXT_TRAINING.optimizer,
            gradient_clip_norm=CONFERENCE_TEXT_TRAINING.gradient_clip_norm,
            print_mode=print_mode,
        )
        for architecture_name in CONFERENCE_TEXT_MODEL_ARCHITECTURES
    }


# ------------------------------- INSTANTIATE TRAINERS ------------------------------- #


def build_trainers_from_configs(
    trainer_configs: dict[str, ArxivTrainingConfig],
    *,
    device: str | torch.device | None = None,
) -> dict[str, PytorchTrainer]:
    # None keeps the auto-detected device for direct module-CLI runs; the staged
    # runtime passes the device it resolved and recorded as effective_device.
    resolved_device = (
        str(device) if device is not None else device_manual_mps_or_cuda_if_available()
    )
    trainers: dict[str, PytorchTrainer] = {}

    for key, config in trainer_configs.items():
        if config.pos_weight is None:
            raise ValueError(
                f"trainer '{key}' has no pos_weight; compute it from the label "
                "frequencies before building the arxiv trainers"
            )
        pos_weight = config.pos_weight

        trainers[key] = PytorchTrainer(
            model_factory=partial(
                model_factory, config=config, dim_output=config.num_classes
            ),
            optimizer_factory=make_optimizer_factory(
                config.optimizer,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            ),
            criterion=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight),
            config=PytorchTrainerConfig(
                num_epochs=config.num_epochs,
                batch_size=config.batch_size,
                device=resolved_device,
                gradient_clip_norm=config.gradient_clip_norm,
            ),
            print_mode=config.print_mode,
            multi_label=True,  # arxiv is multi-label
        )

    return trainers
