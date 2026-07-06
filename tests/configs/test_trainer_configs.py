from __future__ import annotations

import pytest
from pydantic import ValidationError

from drift_happens.configs import TrainerConfig
from drift_happens.configs.trainers import validate_registered_trainer_config


def test_registered_trainer_config_rejects_misspelled_fields() -> None:
    # num_epochs satisfies the required field so only the extra-forbidden error fires.
    trainer = TrainerConfig(
        key="linear-classifier",
        model={"architecture": "linear"},
        training={
            "batch_size": 16,
            "learning_rate": 0.1,
            "n_features": 6,
            "n_samples": 64,
            "num_epochs": 2,
            "num_epocs": 2,
        },
    )

    with pytest.raises(ValidationError, match=r"(?s)num_epocs.*extra_forbidden"):
        validate_registered_trainer_config(trainer)


def test_registered_image_trainer_config_validates_schema() -> None:
    # Exercises the image registry branch (mlp_s / dinov2_s_frozen).
    for key in ("mlp_s", "dinov2_s_frozen"):
        trainer = TrainerConfig(
            key=key,
            model={
                "architecture": "mlp",
                "preset": key,
                "image_size": [32, 32],
                "input_channels": 3,
            },
            training={"batch_size": 32, "learning_rate": 1e-3, "num_epochs": 5},
        )
        validate_registered_trainer_config(trainer)


def test_registered_image_trainer_config_rejects_misspelled_fields() -> None:
    # batch_size satisfies the required field so only the extra-forbidden error fires.
    trainer = TrainerConfig(
        key="mlp_s",
        model={"architecture": "mlp"},
        training={
            "batch_size": 32,
            "learning_rate": 1e-3,
            "num_epochs": 5,
            "batch_sz": 32,
        },
    )

    with pytest.raises(ValidationError, match=r"(?s)batch_sz.*extra_forbidden"):
        validate_registered_trainer_config(trainer)


def test_unknown_trainer_keys_remain_unvalidated_for_external_factories() -> None:
    trainer = TrainerConfig(
        key="external-new-key",
        model={"anything": "json"},
        training={"anything": 1},
    )

    validate_registered_trainer_config(trainer)
