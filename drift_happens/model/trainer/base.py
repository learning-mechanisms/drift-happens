import abc
from pathlib import Path

from pydantic import BaseModel, Field
from torch.utils.data import Dataset

from drift_happens.evaluation.metrics import (
    ClassificationMetrics,
    MultiLabelClassificationMetrics,
    RegressiveClassificationMetrics,
)
from drift_happens.utils.tensor import ArrayLike


class TrainingHistory(BaseModel):
    train_epochs: (
        list[ClassificationMetrics]
        | list[MultiLabelClassificationMetrics]
        | list[RegressiveClassificationMetrics]
    ) = Field(default_factory=list)
    """Index i corresponds to the training results after epoch i."""

    val_epochs: (
        list[ClassificationMetrics]
        | list[MultiLabelClassificationMetrics]
        | list[RegressiveClassificationMetrics]
    ) = Field(default_factory=list)
    """Index i corresponds to the validation results after epoch i."""


class ModelTrainer(abc.ABC):
    """Abstract trainer class for predictive models."""

    @abc.abstractmethod
    def fit(
        self,
        train: Dataset,
        *,
        val: Dataset | None = None,
    ) -> TrainingHistory:
        """Train the model on the provided data."""

    @abc.abstractmethod
    def predict(self, X: ArrayLike) -> ArrayLike:
        """Make predictions using the trained model."""

    @abc.abstractmethod
    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """Predict class probabilities using the trained model."""

    # ------------------------------- STATE MANAGEMENT ------------------------------- #

    @abc.abstractmethod
    def reset_model(self) -> None:
        """Reset the model to its initial state."""

    @abc.abstractmethod
    def save_model(self, path: Path) -> None:
        """Save the model to a file."""

    @abc.abstractmethod
    def load_model(self, path: Path) -> "ModelTrainer":
        """Load the model from a file."""
