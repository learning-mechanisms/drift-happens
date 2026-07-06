from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from drift_happens.model.trainer.pytorch import PytorchTrainer, PytorchTrainerConfig
from drift_happens.utils.pytorch import seed_everything

SEED = 17
INPUT_DIM = 6
NUM_CLASSES = 3
TOTAL_EPOCHS = 6
INTERRUPT_AT = 4


def _devices() -> list[str]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def _model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 16),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(16, NUM_CLASSES),
    )


def _dataset() -> TensorDataset:
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(40, INPUT_DIM, generator=generator)
    labels = torch.randint(0, NUM_CLASSES, (40,), generator=generator)
    return TensorDataset(features, labels)


def _epoch_seeded_sampler_factory(num_rows: int, batch_size: int, seed: int):
    def factory(epoch: int) -> list[list[int]]:
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(num_rows, generator=generator).tolist()
        return [order[i : i + batch_size] for i in range(0, num_rows, batch_size)]

    return factory


def _train(
    num_epochs: int,
    checkpoint_dir: Path | None,
    device: str,
    *,
    batch_sampler_factory=None,
) -> dict[str, torch.Tensor]:
    seed_everything(SEED)
    trainer = PytorchTrainer(
        model_factory=_model,
        optimizer_factory=lambda model: torch.optim.Adam(model.parameters(), lr=1e-2),
        criterion=nn.CrossEntropyLoss(),
        config=PytorchTrainerConfig(num_epochs=num_epochs, batch_size=8, device=device),
    )
    fit_kwargs = {}
    if batch_sampler_factory is not None:
        fit_kwargs["train_batch_sampler_factory"] = batch_sampler_factory
    trainer.fit(train=_dataset(), checkpoint_dir=checkpoint_dir, **fit_kwargs)
    return {
        key: value.cpu().clone() for key, value in trainer._model.state_dict().items()
    }


def _assert_identical(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key])


@pytest.mark.parametrize("device", _devices())
def test_checkpointing_does_not_change_training(tmp_path: Path, device: str) -> None:
    plain = _train(TOTAL_EPOCHS, checkpoint_dir=None, device=device)
    checkpointed = _train(TOTAL_EPOCHS, checkpoint_dir=tmp_path / "ckpt", device=device)
    _assert_identical(plain, checkpointed)


@pytest.mark.parametrize("device", _devices())
def test_resumed_run_matches_uninterrupted_run(tmp_path: Path, device: str) -> None:
    uninterrupted = _train(
        TOTAL_EPOCHS, checkpoint_dir=tmp_path / "reference", device=device
    )

    resume_dir = tmp_path / "resume"
    _train(INTERRUPT_AT, checkpoint_dir=resume_dir, device=device)
    resumed = _train(TOTAL_EPOCHS, checkpoint_dir=resume_dir, device=device)

    _assert_identical(uninterrupted, resumed)


@pytest.mark.parametrize("device", _devices())
def test_resumed_run_matches_with_chunk_blocked_sampler(
    tmp_path: Path, device: str
) -> None:
    factory = _epoch_seeded_sampler_factory(num_rows=40, batch_size=8, seed=SEED)

    uninterrupted = _train(
        TOTAL_EPOCHS,
        checkpoint_dir=tmp_path / "reference",
        device=device,
        batch_sampler_factory=factory,
    )

    resume_dir = tmp_path / "resume"
    _train(
        INTERRUPT_AT,
        checkpoint_dir=resume_dir,
        device=device,
        batch_sampler_factory=factory,
    )
    resumed = _train(
        TOTAL_EPOCHS,
        checkpoint_dir=resume_dir,
        device=device,
        batch_sampler_factory=factory,
    )

    _assert_identical(uninterrupted, resumed)
