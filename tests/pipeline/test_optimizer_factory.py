"""Tests for shared optimizer factory construction."""

import pytest
import torch

from drift_happens.model.blocks.mlp import MLPConfig
from drift_happens.pipeline._shared.optimizers import (
    OptimizerName,
    make_optimizer_factory,
)
from drift_happens.pipeline.image.trainers import (
    ConferenceImageTrainingConfig,
    build_image_trainers_from_configs,
)
from drift_happens.pipeline.imdb_faces import trainers as imdb_trainers
from drift_happens.pipeline.yearbook import trainers as yearbook_trainers


def _build_optimizer(
    name: OptimizerName, *, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    factory = make_optimizer_factory(
        name, learning_rate=learning_rate, weight_decay=weight_decay
    )
    return factory(torch.nn.Linear(1, 1))


@pytest.mark.parametrize(
    ("name", "expected"),
    [("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW)],
)
def test_selects_optimizer_class(
    name: OptimizerName, expected: type[torch.optim.Optimizer]
) -> None:
    optimizer = _build_optimizer(name, learning_rate=1e-3, weight_decay=0.0)
    assert type(optimizer) is expected


def test_binds_hyperparameters() -> None:
    optimizer = _build_optimizer("adam", learning_rate=5e-4, weight_decay=0.01)
    assert optimizer.defaults["lr"] == 5e-4
    assert optimizer.defaults["weight_decay"] == 0.01


def test_each_factory_binds_its_own_values_in_a_loop() -> None:
    # The whole reason the helper exists: built inside a loop, each factory must
    # keep its own hyperparameters. A closure over the loop variable would make
    # every optimizer share the last iteration's values.
    settings: list[tuple[OptimizerName, float, float]] = [
        ("adam", 1e-3, 0.0),
        ("adamw", 5e-4, 0.01),
    ]
    factories = [
        make_optimizer_factory(name, learning_rate=lr, weight_decay=wd)
        for name, lr, wd in settings
    ]
    optimizers = [factory(torch.nn.Linear(1, 1)) for factory in factories]

    assert type(optimizers[0]) is torch.optim.Adam
    assert optimizers[0].defaults["lr"] == 1e-3
    assert optimizers[0].defaults["weight_decay"] == 0.0
    assert type(optimizers[1]) is torch.optim.AdamW
    assert optimizers[1].defaults["lr"] == 5e-4
    assert optimizers[1].defaults["weight_decay"] == 0.01


def test_dataset_trainer_aliases_point_to_shared_builder() -> None:
    # yearbook and imdb_faces re-export the shared image builder; catch future divergence.
    assert (
        yearbook_trainers.build_trainers_from_configs
        is build_image_trainers_from_configs
    )
    assert (
        imdb_trainers.build_trainers_from_configs is build_image_trainers_from_configs
    )
    assert yearbook_trainers.YearbookTrainingConfig is ConferenceImageTrainingConfig
    assert imdb_trainers.ImdbTrainingConfig is ConferenceImageTrainingConfig


def test_builder_binds_each_key_to_its_own_optimizer() -> None:
    # The shared image builder constructs the optimizer factory inside a loop over
    # the configs; each key must keep its own hyperparameters. Invoke the stored
    # factory AFTER the loop has finished: a factory that closed over the loop
    # variable would resolve every key to the last config's values.
    configs = {
        "low": ConferenceImageTrainingConfig(
            architecture_specific_config=MLPConfig(hidden_layers=[8], weight_decay=0.0),
            batch_size=2,
            learning_rate=1e-3,
            num_epochs=1,
        ),
        "high": ConferenceImageTrainingConfig(
            architecture_specific_config=MLPConfig(
                hidden_layers=[8], weight_decay=0.05
            ),
            batch_size=2,
            learning_rate=7e-4,
            num_epochs=1,
        ),
    }
    trainers = build_image_trainers_from_configs(configs)

    for key, learning_rate, weight_decay in [("low", 1e-3, 0.0), ("high", 7e-4, 0.05)]:
        optimizer = trainers[key]._optimizer_factory(torch.nn.Linear(1, 1))
        assert optimizer.defaults["lr"] == learning_rate
        assert optimizer.defaults["weight_decay"] == weight_decay
