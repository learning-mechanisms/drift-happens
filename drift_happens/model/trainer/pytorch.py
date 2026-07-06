import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, override

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader, Dataset

from drift_happens.evaluation.metrics import (
    ClassificationMetrics,
    MultiLabelClassificationMetrics,
    RegressiveClassificationMetrics,
)
from drift_happens.model.text.weighted_mse_loss import WeightedMSELoss
from drift_happens.model.trainer.base import ModelTrainer, TrainingHistory
from drift_happens.utils.log import get_logger
from drift_happens.utils.pytorch import seed_everything
from drift_happens.utils.tensor import ArrayLike, to_tensor

logger = get_logger()


class PytorchTrainerConfig(BaseModel):
    num_epochs: int
    batch_size: int = 32
    device: str | None = None
    seed: int | None = None
    gradient_clip_norm: float | None = None


EpochPrintMode = Literal[False, "short", "long"]
EpochMetrics = (
    ClassificationMetrics
    | MultiLabelClassificationMetrics
    | RegressiveClassificationMetrics
)


TaskType = Literal["classification", "multilabel", "regression"]


class _TaskStrategy(Protocol):
    task_type: TaskType
    prediction_dtype: torch.dtype

    def outputs_to_probabilities(self, outputs: torch.Tensor) -> torch.Tensor: ...

    def outputs_to_predictions(
        self, outputs: torch.Tensor, *, threshold: float | np.ndarray | None = None
    ) -> torch.Tensor: ...

    def metrics_from_epoch(
        self,
        *,
        true_labels: Sequence[Any],
        predicted_probs: Sequence[Any],
        predicted_labels: Sequence[Any],
        loss: float,
    ) -> EpochMetrics: ...


class _ClassificationTask:
    task_type: TaskType = "classification"
    prediction_dtype = torch.long

    def outputs_to_probabilities(self, outputs: torch.Tensor) -> torch.Tensor:
        return nn.functional.softmax(outputs, dim=1)

    def outputs_to_predictions(
        self, outputs: torch.Tensor, *, threshold: float | np.ndarray | None = None
    ) -> torch.Tensor:
        return outputs.argmax(dim=1)

    def metrics_from_epoch(
        self,
        *,
        true_labels: Sequence[Any],
        predicted_probs: Sequence[Any],
        predicted_labels: Sequence[Any],
        loss: float,
    ) -> ClassificationMetrics:
        return ClassificationMetrics.from_predictions(
            y_true=np.asarray(true_labels),
            y_pred=np.asarray(predicted_labels),
            y_prob=np.asarray(predicted_probs),
            loss=loss,
        )


class _MultiLabelTask:
    task_type: TaskType = "multilabel"
    prediction_dtype = torch.bool

    def outputs_to_probabilities(self, outputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(outputs)

    def outputs_to_predictions(
        self, outputs: torch.Tensor, *, threshold: float | np.ndarray | None = None
    ) -> torch.Tensor:
        raw_threshold: float | np.ndarray = 0.5 if threshold is None else threshold
        threshold_tensor = torch.as_tensor(raw_threshold, device=outputs.device)
        return torch.sigmoid(outputs) > threshold_tensor

    def metrics_from_epoch(
        self,
        *,
        true_labels: Sequence[Any],
        predicted_probs: Sequence[Any],
        predicted_labels: Sequence[Any],
        loss: float,
    ) -> MultiLabelClassificationMetrics:
        probs = np.asarray(predicted_probs)
        return MultiLabelClassificationMetrics.from_predictions(
            num_classes=probs.shape[1],
            y_true=np.stack(true_labels),
            y_pred=np.stack(predicted_labels),
        )


class _RegressionTask:
    task_type: TaskType = "regression"
    prediction_dtype = torch.float32

    def outputs_to_probabilities(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs.reshape(outputs.shape[0], -1).squeeze(-1)

    def outputs_to_predictions(
        self, outputs: torch.Tensor, *, threshold: float | np.ndarray | None = None
    ) -> torch.Tensor:
        return self.outputs_to_probabilities(outputs)

    def metrics_from_epoch(
        self,
        *,
        true_labels: Sequence[Any],
        predicted_probs: Sequence[Any],
        predicted_labels: Sequence[Any],
        loss: float,
    ) -> RegressiveClassificationMetrics:
        return RegressiveClassificationMetrics.from_predictions(
            y_true=np.asarray(true_labels),
            predicted=np.asarray(predicted_probs).squeeze(),
            loss=loss,
        )


def _task_strategy(criterion: nn.Module, *, multi_label: bool) -> _TaskStrategy:
    if isinstance(criterion, WeightedMSELoss):
        return _RegressionTask()
    if multi_label:
        return _MultiLabelTask()
    return _ClassificationTask()


# Saved-model extension is ``.pt``; ``.pth``/``.model`` are legacy fallbacks kept so
# run directories produced before the standardization still load.
_LEGACY_MODEL_SUFFIXES = (".pth", ".model")

_EPOCH_CHECKPOINT_FILENAME = "epoch.pt"


def _resolve_model_path(path: Path) -> Path:
    """Resolve a model artifact path, falling back to legacy extensions."""
    if path.exists():
        return path
    for suffix in _LEGACY_MODEL_SUFFIXES:
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return path


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    mps_state = state.get("mps")
    if mps_state is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(mps_state)


class PytorchTrainer(ModelTrainer):
    """Trains, evaluates, and persists PyTorch models."""

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        optimizer_factory: Callable[[nn.Module], optim.Optimizer],
        criterion: nn.Module,
        config: PytorchTrainerConfig,
        multi_label: bool = False,
        print_mode: EpochPrintMode = False,
        threshold: float = 0.5,
    ):
        # Stateful components require factories for re-initialization
        self._model_factory = model_factory
        self._optimizer_factory = optimizer_factory

        # Static components
        self._criterion = criterion
        self._config = config
        self._multi_label = multi_label
        self._print_mode = print_mode
        self._task = _task_strategy(criterion, multi_label=multi_label)

        self._threshold: float | np.ndarray = threshold

        # Initialize stateful components
        self.reset_model()

    # ----------------------------------- TRAINING ----------------------------------- #

    def fit(
        self,
        train: Dataset,
        *,
        val: Dataset | None = None,
        train_batch_sampler_factory: Callable[[int], Any] | None = None,
        checkpoint_dir: Path | None = None,
    ) -> TrainingHistory:
        """
        Train the model.

        ``train_batch_sampler_factory`` maps an epoch index to a batch sampler that
        yields lists of indices. When supplied, the training loader uses it instead of
        global shuffling; this is the chunk-blocked path for out-of-core sequence caches
        (see ``ChunkBlockedBatchSampler``). The pooled/materialized paths leave it
        ``None`` and keep exact global shuffling.

        ``checkpoint_dir`` persists the model, optimizer, history, and RNG state after
        each epoch so a later fit resumes at the next epoch and reproduces an
        uninterrupted run.
        """
        val_loader = self._data_loader(val, shuffle=False) if val is not None else None

        device = self.device
        self.print_epoch(f"Training on device: {device}")
        self._model.to(device)
        self._criterion.to(device)

        history = TrainingHistory()
        start_epoch = 0
        if (
            checkpoint_dir is not None
            and (checkpoint_dir / _EPOCH_CHECKPOINT_FILENAME).exists()
        ):
            start_epoch, history = self._load_epoch_checkpoint(checkpoint_dir)
            self.print_epoch(
                f"Resuming at epoch {start_epoch + 1}/{self._config.num_epochs}"
            )

        for epoch in range(start_epoch, self._config.num_epochs):
            self.print_epoch(f"Starting epoch {epoch + 1}/{self._config.num_epochs}")

            # -------------------------------- TRAIN LOOP -------------------------------- #
            self._model.train()
            train_true_labels: list[int] = []
            train_predicted_probs: list[float] = []
            train_predicted_labels: list[int] | list[np.ndarray] = []
            train_running_loss = 0.0

            if train_batch_sampler_factory is not None:
                train_loader = self._data_loader(
                    train,
                    shuffle=False,
                    batch_sampler=train_batch_sampler_factory(epoch),
                )
            else:
                train_loader = self._data_loader(train, shuffle=True, epoch=epoch)
            for batch in train_loader:
                inputs, labels = self._batch_to_device(batch, device)

                self._optimizer.zero_grad()

                outputs = self._model(*inputs)
                loss = self._criterion(outputs, labels)
                loss.backward()
                if self._config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), self._config.gradient_clip_norm
                    )
                self._optimizer.step()

                train_true_labels.extend(labels.cpu().numpy())
                probs = self._outputs_to_probabilities(outputs).detach().cpu().numpy()
                train_predicted_probs.extend(probs)

                pred = self._outputs_to_predictions(outputs).detach().cpu().numpy()
                train_predicted_labels.extend(pred)
                train_running_loss += loss.item() * len(labels)

            train_epoch_results = self._task.metrics_from_epoch(
                true_labels=train_true_labels,
                predicted_probs=train_predicted_probs,
                predicted_labels=train_predicted_labels,
                loss=train_running_loss / len(train_true_labels),
            )

            history.train_epochs.append(train_epoch_results)  # type: ignore

            if self._print_mode:
                if self._print_mode == "long":
                    self.print_epoch("=" * 40)
                self.print_epoch(
                    "Train Epoch Results: \n"
                    + train_epoch_results.format(long=self._print_mode == "long")
                )

            # ------------------------ VALIDATION LOOP (optional) ------------------------ #
            if val_loader is not None:
                self._model.eval()
                val_true_labels: list[int] = []
                val_predicted_probs: list[float] = []
                val_predicted_labels: list[int] = []
                val_running_loss = 0.0

                with torch.no_grad():
                    for batch in val_loader:
                        inputs, labels = self._batch_to_device(batch, device)
                        outputs = self._model(*inputs)
                        loss = self._criterion(outputs, labels)

                        batch_size = labels.size(0)
                        val_running_loss += float(loss.item()) * batch_size
                        val_true_labels.extend(labels.cpu().numpy())
                        probs = (
                            self._outputs_to_probabilities(outputs)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        pred = (
                            self._outputs_to_predictions(outputs).detach().cpu().numpy()
                        )

                        val_predicted_probs.extend(probs)
                        val_predicted_labels.extend(pred)

                val_epoch_results = self._task.metrics_from_epoch(
                    true_labels=val_true_labels,
                    predicted_probs=val_predicted_probs,
                    predicted_labels=val_predicted_labels,
                    loss=float(val_running_loss) / len(val_true_labels),
                )

                history.val_epochs.append(val_epoch_results)  # type: ignore

                if self._print_mode:
                    if self._print_mode == "long":
                        self.print_epoch("=" * 40)
                    self.print_epoch(
                        "Validation Epoch Results: \n"
                        + val_epoch_results.format(long=self._print_mode == "long")
                    )

            self.print_epoch(f"Epoch {epoch + 1}/{self._config.num_epochs} completed")

            if checkpoint_dir is not None:
                self._save_epoch_checkpoint(
                    checkpoint_dir, next_epoch=epoch + 1, history=history
                )

        return history

    # ------------------------------------ PREDICT ----------------------------------- #

    def predict_proba(self, X: ArrayLike | tuple[ArrayLike, ...]) -> torch.Tensor:
        """
        Predict probabilities, always returned on the CPU.

        Callers hand the result straight to numpy/scikit-learn, which cannot read
        accelerator tensors.
        """
        if isinstance(X, tuple):
            inputs = tuple(to_tensor(item) for item in X)
        else:
            inputs = (to_tensor(X),)
        device = self.device
        self._model.to(device)

        all_probs: list[torch.Tensor] = []

        self._model.eval()
        with torch.no_grad():
            for batch in self._prediction_loader(inputs):
                batch_inputs, _ = self._batch_to_device(batch, device)
                outputs = self._model(*batch_inputs)
                probs = self._outputs_to_probabilities(outputs).detach().cpu()
                all_probs.append(probs)

        if not all_probs:
            return torch.empty(0)
        return torch.cat(all_probs, dim=0)

    def predict(self, X: ArrayLike | tuple[ArrayLike, ...]) -> torch.Tensor:
        probs = self.predict_proba(X)
        if probs.numel() == 0:
            return torch.empty(
                0, dtype=self._task.prediction_dtype, device=probs.device
            )
        if isinstance(self._task, _MultiLabelTask):
            return probs > torch.as_tensor(self._threshold, device=probs.device)
        if isinstance(self._task, _RegressionTask):
            return probs
        return probs.argmax(dim=1)

    def find_optimal_threshold(self, probs: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Optimal threshold for F1."""
        num_classes = probs.shape[1]
        thresholds = np.zeros(num_classes)

        for c in range(num_classes):
            precision, recall, thresh = precision_recall_curve(y[:, c], probs[:, c])
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            thresholds[c] = thresh[np.argmax(f1)] if len(thresh) > 0 else 0.5

        self._threshold = thresholds

        return thresholds

    # ------------------------------- STATE MANAGEMENT ------------------------------- #

    @override
    def reset_model(self) -> None:
        if self._config.seed is not None:
            seed_everything(self._config.seed)
        self._model = self._model_factory()
        self._optimizer = self._optimizer_factory(self._model)

    def replace_model_factory(
        self,
        model_factory: Callable[[], nn.Module],
        *,
        reset: bool = False,
    ) -> None:
        """Replace the model factory used by future resets and model loads."""
        self._model_factory = model_factory
        if reset:
            self.reset_model()

    @override
    def save_model(self, path: Path) -> None:
        """Save the model to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)

    @override
    def load_model(self, path: Path) -> "PytorchTrainer":
        """Load the model state dict from a file."""
        self.reset_model()
        load_path = _resolve_model_path(path)
        self._model.load_state_dict(
            torch.load(load_path, map_location=self.device, weights_only=True)
        )
        return self

    def _save_epoch_checkpoint(
        self, directory: Path, *, next_epoch: int, history: TrainingHistory
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "next_epoch": next_epoch,
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "history": history,
            "rng": _capture_rng_state(),
        }
        target = directory / _EPOCH_CHECKPOINT_FILENAME
        staging = target.with_suffix(".tmp")
        torch.save(payload, staging)
        staging.replace(target)

    def _load_epoch_checkpoint(self, directory: Path) -> tuple[int, TrainingHistory]:
        checkpoint = torch.load(
            directory / _EPOCH_CHECKPOINT_FILENAME,
            map_location="cpu",
            weights_only=False,
        )
        self._model.load_state_dict(checkpoint["model"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        for state in self._optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)
        _restore_rng_state(checkpoint["rng"])
        return checkpoint["next_epoch"], checkpoint["history"]

    # -------------------------------------- IO -------------------------------------- #

    def print_epoch(self, s: str) -> None:
        """Log ``s`` if print_mode is enabled."""
        if self._print_mode:
            logger.info(s)

    # ------------------------------------- UTILS ------------------------------------ #

    def _data_loader(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        epoch: int = 0,
        batch_sampler: Any | None = None,
    ) -> DataLoader:
        if batch_sampler is not None:
            # A batch_sampler already encodes batching and ordering, so the
            # DataLoader must not also set batch_size/shuffle/drop_last.
            return DataLoader(dataset, batch_sampler=batch_sampler)
        generator = None
        if shuffle and self._config.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self._config.seed + epoch)
        return DataLoader(
            dataset,
            batch_size=self._config.batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def _prediction_loader(self, inputs: tuple[torch.Tensor, ...]) -> DataLoader:
        row_count = inputs[0].shape[0]
        dummy = torch.zeros(row_count, dtype=torch.uint8)
        dataset = torch.utils.data.TensorDataset(*inputs, dummy)
        return DataLoader(dataset, batch_size=self._config.batch_size, shuffle=False)

    def _batch_to_device(
        self, batch: tuple[torch.Tensor, ...] | list[torch.Tensor], device: torch.device
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if len(batch) < 2:
            raise ValueError(
                "PyTorch batches must contain at least one input and labels"
            )
        *inputs, labels = batch
        return tuple(item.to(device) for item in inputs), labels.to(device)

    def _outputs_to_probabilities(self, outputs: torch.Tensor) -> torch.Tensor:
        return self._task.outputs_to_probabilities(outputs)

    def _outputs_to_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        # Forward the stored threshold so epoch metrics and predict() agree.
        return self._task.outputs_to_predictions(outputs, threshold=self._threshold)

    @property
    def task_type(self) -> TaskType:
        """The prediction task this trainer was configured for."""
        return self._task.task_type

    @property
    def batch_size(self) -> int:
        """The configured training batch size."""
        return self._config.batch_size

    @property
    def device(self) -> torch.device:
        """Get the device the model is currently on."""
        return (
            torch.device(self._config.device)
            if self._config.device is not None
            else (next(self._model.parameters()).device)
        )
