from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from drift_happens.evaluation.metrics import (
    ClassificationMetrics,
    MultiLabelClassificationMetrics,
    RegressiveClassificationMetrics,
)
from drift_happens.model.text.weighted_mse_loss import WeightedMSELoss
from drift_happens.model.trainer.base import TrainingHistory
from drift_happens.model.trainer.pytorch import PytorchTrainer, PytorchTrainerConfig


class TwoInputClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.linear(x * mask.float())


class OneOutputRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SimpleClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FixedMultiLabel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.tensor([[0.0, 0.0], [2.0, -2.0]], device=x.device)
        return logits[: x.shape[0]] + self.bias * 0


def _classifier_trainer(**config_updates) -> PytorchTrainer:
    config = PytorchTrainerConfig(num_epochs=1, batch_size=2, seed=0).model_copy(
        update=config_updates
    )
    return PytorchTrainer(
        model_factory=SimpleClassifier,
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.01),
        criterion=nn.CrossEntropyLoss(),
        config=config,
    )


def test_task_type_reflects_configured_strategy() -> None:
    assert _classifier_trainer().task_type == "classification"

    multilabel = PytorchTrainer(
        model_factory=SimpleClassifier,
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.01),
        criterion=nn.BCEWithLogitsLoss(),
        config=PytorchTrainerConfig(num_epochs=1, batch_size=2),
        multi_label=True,
    )
    assert multilabel.task_type == "multilabel"

    regression = PytorchTrainer(
        model_factory=OneOutputRegressor,
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.01),
        criterion=WeightedMSELoss(weights=torch.ones(5)),
        config=PytorchTrainerConfig(num_epochs=1, batch_size=2),
    )
    assert regression.task_type == "regression"


def test_trainer_accepts_multiple_input_tensors() -> None:
    x = torch.randn(12, 3)
    mask = torch.ones(12, 3, dtype=torch.bool)
    y = torch.tensor([0, 1] * 6)
    dataset = TensorDataset(x, mask, y)
    trainer = PytorchTrainer(
        model_factory=TwoInputClassifier,
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.01),
        criterion=nn.CrossEntropyLoss(),
        config=PytorchTrainerConfig(num_epochs=1, batch_size=4, seed=0),
    )

    history = trainer.fit(dataset)
    probs = trainer.predict_proba((x, mask))

    assert len(history.train_epochs) == 1
    assert probs.shape == (12, 2)
    assert probs.device.type == "cpu"


@pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="needs an accelerator to observe the device round-trip",
)
def test_predict_proba_returns_cpu_tensors_from_accelerator() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    trainer = PytorchTrainer(
        model_factory=SimpleClassifier,
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.01),
        criterion=nn.CrossEntropyLoss(),
        config=PytorchTrainerConfig(num_epochs=1, batch_size=4, seed=0, device=device),
    )

    probs = trainer.predict_proba(torch.randn(6, 2))

    assert probs.device.type == "cpu"


def test_regression_predict_proba_returns_raw_scalar_outputs() -> None:
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    trainer = PytorchTrainer(
        model_factory=OneOutputRegressor,
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.01),
        criterion=WeightedMSELoss(weights=torch.ones(5)),
        config=PytorchTrainerConfig(num_epochs=1, batch_size=2, seed=0),
    )

    preds = trainer.predict_proba(x)

    assert preds.shape == (4,)
    assert not torch.allclose(preds, torch.ones_like(preds))


def test_fit_records_validation_history() -> None:
    x = torch.randn(8, 2)
    y = torch.tensor([0, 1] * 4)
    trainer = _classifier_trainer()

    history = trainer.fit(TensorDataset(x, y), val=TensorDataset(x[:4], y[:4]))

    assert len(history.train_epochs) == 1
    assert len(history.val_epochs) == 1
    assert isinstance(history.train_epochs[0], ClassificationMetrics)


def test_fit_records_multilabel_metrics() -> None:
    x = torch.zeros(2, 1)
    y = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    trainer = PytorchTrainer(
        model_factory=FixedMultiLabel,
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.01),
        criterion=nn.BCEWithLogitsLoss(),
        config=PytorchTrainerConfig(num_epochs=1, batch_size=2, seed=0),
        multi_label=True,
    )

    history = trainer.fit(TensorDataset(x, y))

    assert isinstance(history.train_epochs[0], MultiLabelClassificationMetrics)
    assert len(history.train_epochs[0].confusion_matrices) == 2


def test_fit_records_regression_metrics() -> None:
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y = torch.tensor([1, 2, 3, 4])
    trainer = PytorchTrainer(
        model_factory=OneOutputRegressor,
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.01),
        criterion=WeightedMSELoss(weights=torch.ones(5)),
        config=PytorchTrainerConfig(num_epochs=1, batch_size=2, seed=0),
    )

    history = trainer.fit(TensorDataset(x, y))

    assert isinstance(history.train_epochs[0], RegressiveClassificationMetrics)
    assert history.train_epochs[0].loss is not None


def test_save_and_load_model_round_trip(tmp_path) -> None:
    x = torch.randn(3, 2)
    first = _classifier_trainer()
    expected = first.predict_proba(x)
    first.save_model(tmp_path / "trained_model.pt")
    second = _classifier_trainer(seed=99)

    second.load_model(tmp_path / "trained_model.pt")

    torch.testing.assert_close(second.predict_proba(x), expected)


def test_save_model_uses_requested_path_and_loads_legacy_suffix(tmp_path) -> None:
    requested = tmp_path / "trained_model.pt"
    trainer = _classifier_trainer()

    trainer.save_model(requested)
    requested.rename(requested.with_suffix(".model"))

    assert not requested.exists()
    assert requested.with_suffix(".model").exists()
    _classifier_trainer().load_model(requested)


def test_predict_handles_empty_input() -> None:
    preds = _classifier_trainer().predict(torch.empty(0, 2))

    assert preds.shape == (0,)
    assert preds.dtype == torch.long


def test_gradient_clipping_path_runs() -> None:
    x = torch.randn(4, 2)
    y = torch.tensor([0, 1, 0, 1])

    history = _classifier_trainer(gradient_clip_norm=0.1).fit(TensorDataset(x, y))

    assert len(history.train_epochs) == 1


def test_multilabel_thresholds_are_used_for_prediction() -> None:
    trainer = PytorchTrainer(
        model_factory=FixedMultiLabel,
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.01),
        criterion=nn.BCEWithLogitsLoss(),
        config=PytorchTrainerConfig(num_epochs=1, batch_size=2),
        multi_label=True,
    )
    trainer._threshold = np.array([0.6, 0.4])

    preds = trainer.predict(torch.zeros(2, 1))

    torch.testing.assert_close(preds, torch.tensor([[False, True], [True, False]]))


def test_epoch_predictions_use_tuned_thresholds() -> None:
    trainer = PytorchTrainer(
        model_factory=FixedMultiLabel,
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.01),
        criterion=nn.BCEWithLogitsLoss(),
        config=PytorchTrainerConfig(num_epochs=1, batch_size=2),
        multi_label=True,
    )
    probs = np.array([[0.3, 0.2], [0.9, 0.4]])
    y = np.array([[0, 1], [1, 0]])

    thresholds = trainer.find_optimal_threshold(probs, y)

    np.testing.assert_allclose(thresholds, [0.9, 0.2])
    assert trainer._threshold is thresholds
    # Epoch metrics threshold zero logits (probability 0.5) with the tuned
    # per-class thresholds, not a hardcoded 0.5, so they agree with predict().
    preds = trainer._outputs_to_predictions(torch.zeros(2, 2))
    torch.testing.assert_close(preds, torch.tensor([[False, True], [False, True]]))


def test_batch_to_device_requires_labels() -> None:
    with pytest.raises(ValueError, match="at least one input and labels"):
        _classifier_trainer()._batch_to_device((torch.ones(2, 2),), torch.device("cpu"))


def test_same_seed_training_is_reproducible() -> None:
    torch.manual_seed(0)
    x = torch.randn(16, 2)
    y = torch.tensor([0, 1] * 8)
    dataset = TensorDataset(x, y)
    probe = torch.randn(4, 2)

    first = _classifier_trainer(seed=0)
    first.fit(dataset)
    second = _classifier_trainer(seed=0)
    second.fit(dataset)

    torch.testing.assert_close(
        first.predict_proba(probe), second.predict_proba(probe), rtol=0, atol=0
    )


def test_training_history_round_trips_multilabel_val_epochs() -> None:
    # val_epochs must accept the same metric types fit() appends (multilabel here),
    # and survive a model_dump_json round-trip without dropping subclass fields.
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    metrics = MultiLabelClassificationMetrics.from_predictions(3, y_true, y_true)

    history = TrainingHistory(val_epochs=[metrics])
    restored = TrainingHistory.model_validate_json(history.model_dump_json())

    assert isinstance(restored.val_epochs[0], MultiLabelClassificationMetrics)
