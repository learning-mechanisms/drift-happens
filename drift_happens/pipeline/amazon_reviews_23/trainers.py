from functools import partial

import torch
from pydantic import field_serializer

from drift_happens.model.dataset.text.architectures import (
    CONFERENCE_TEXT_MODEL_ARCHITECTURES,
)
from drift_happens.model.text.weighted_mse_loss import WeightedMSELoss
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


class AmazonReviewsTrainingConfig(TextTrainingConfig):
    class_weights: torch.Tensor

    @field_serializer("class_weights")
    def serialize_class_weights(self, value: torch.Tensor) -> list[float]:
        return value.tolist()


def amazon_reviews_conference_trainer_configs(
    class_weights: torch.Tensor,
    *,
    print_mode: EpochPrintMode = False,
) -> dict[str, AmazonReviewsTrainingConfig]:
    """Build the cached-feature text lineup for the conference Amazon scope."""
    return {
        architecture_name: AmazonReviewsTrainingConfig(
            architecture_name=architecture_name,
            feature_input_dim=768,
            class_weights=class_weights,
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
    trainer_configs: dict[str, AmazonReviewsTrainingConfig],
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
        trainers[key] = PytorchTrainer(
            model_factory=partial(
                model_factory,
                config=config,
                dim_output=1,  # Regression: 1 output
            ),
            optimizer_factory=make_optimizer_factory(
                config.optimizer,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            ),
            criterion=WeightedMSELoss(weights=config.class_weights),
            config=PytorchTrainerConfig(
                num_epochs=config.num_epochs,
                batch_size=config.batch_size,
                device=resolved_device,
                gradient_clip_norm=config.gradient_clip_norm,
            ),
            print_mode=config.print_mode,
            multi_label=False,
        )

    return trainers
