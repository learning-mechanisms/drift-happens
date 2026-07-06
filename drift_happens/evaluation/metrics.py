import math
from typing import Annotated, Any, Literal

import numpy as np
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_serializer,
    field_validator,
)
from sklearn.metrics import roc_auc_score
from tabulate import tabulate

from drift_happens.utils.tensor import ArrayLike

LossFunc = Literal["cross_entropy"]


def _finite_scalars(values: dict[str, Any]) -> dict[str, float]:
    """
    Keep only finite numeric scalars, coerced to ``float``.

    Drops booleans, non-numeric values, and non-finite floats (``nan``/``inf``) so
    callers such as metric sinks never receive values they cannot log.
    """
    out: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float, np.integer, np.floating)):
            continue
        fvalue = float(value)
        if math.isfinite(fvalue):
            out[str(key)] = fvalue
    return out


def format(
    metrics: dict[str, float],
    metrics_per_class: dict[str, dict[int, float]],
    loss: float | None = None,
    long: bool = False,
    confusion_matrix: np.ndarray | None = None,
) -> str:
    """Format metrics as a string."""
    # SHORT FORMAT
    if not long:
        parts = [
            f"ACC={metrics['accuracy']:.4f}",
            f"PREC={metrics['precision_balanced']:.4f}",
            f"REC={metrics['recall_balanced']:.4f}",
            f"F1_macro={metrics['f1_score_macro']:.4f}",
        ]
        if loss is not None:
            parts.append(f"LOSS={loss:.4f}")
        return ", ".join(parts)

    # LONG / TABLE FORMAT
    # Define order + pretty headers
    metric_order = ["accuracy", "precision", "recall", "f1_score"]
    header_map = {
        "accuracy": "ACC",
        "precision": "PREC",
        "recall": "REC",
        "f1_score": "F1",
    }
    headers = [header_map[m] for m in metric_order]

    # Global row (mostly for binary classification); multiclass rows show "—" for F1.
    global_row = []
    for m in metric_order:
        v = metrics.get(m)
        global_row.append(f"{v:.4f}" if v is not None else "—")

    # Macro row from self.metrics. There is no macro-averaged accuracy: balanced
    # accuracy is macro recall, which already shows under the REC column.
    macro_row = ["—"] + [
        f"{metrics[m]:.4f}" if metrics.get(m) is not None else "—"
        for m in [
            "precision_balanced",
            "recall_balanced",
            "f1_score_macro",
        ]
    ]

    # Per-class metrics from self.metrics_per_class
    metrics_pc = metrics_per_class

    if metrics_pc:
        # all per-class metric dicts share the same keys
        first_metric = next(iter(metrics_pc.values()))
        class_ids = sorted(first_metric.keys())
    else:
        class_ids = []

    class_rows: list[list[str]] = []
    for cls in class_ids:
        row: list[str] = []
        for m in metric_order:
            v = metrics_pc[m][cls]
            row.append(f"{v:.4f}" if v is not None else "—")
        class_rows.append(row)

    # Main metrics table
    table = "Scores:\n" + tabulate(
        [global_row] + [macro_row] + class_rows,
        headers=headers,
        tablefmt="github",
        showindex=["", "MACRO AVG"] + [f"class {c}" for c in class_ids],
    )

    # ---------- CONFUSION MATRIX ----------
    cm = confusion_matrix
    if cm is not None:
        cm = np.asarray(cm)
        n_classes = cm.shape[0]

        # If we already have class_ids, prefer them, otherwise use range
        if not class_ids:
            class_ids_cm = list(range(n_classes))
        else:
            class_ids_cm = class_ids

        headers_cm = [f"pred {c}" for c in class_ids_cm]
        index_cm = [f"true {c}" for c in class_ids_cm]

        cm_table = "\nConfusion matrix:\n" + tabulate(
            cm,
            headers=headers_cm,
            showindex=index_cm,
            tablefmt="github",
        )

        table = table + "\n" + cm_table

    return table


class ClassificationMetrics(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    confusion_matrix: np.ndarray
    """Rows: true classes, Columns: predicted classes."""

    loss: float | None = None
    loss_func: str | None = None

    @staticmethod
    def from_predictions(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        y_prob: ArrayLike | None = None,
        *,
        loss: float | None = None,
        loss_func: LossFunc | None = None,
        num_classes: int | None = None,
    ) -> "ClassificationMetrics":
        """Compute classification metrics from true and predicted labels."""
        requested_loss_func = loss_func is not None
        if loss is not None and loss_func is not None:
            raise ValueError("Provide either loss and loss_func, or neither.")
        if loss is None and loss_func is None:
            loss_func = "cross_entropy"
        if requested_loss_func and y_prob is None:
            raise ValueError(
                "loss_func was requested but y_prob is None; "
                "provide y_prob to compute the loss"
            )

        if len(y_true) == 0:
            # Return empty metrics if there are no samples
            num_classes = num_classes or 0
            cm = np.zeros((num_classes, num_classes), dtype=int)
            # No loss is computed for an empty batch, so do not advertise one.
            if loss is None:
                loss_func = None
            return ClassificationMetrics(
                confusion_matrix=cm, loss=loss, loss_func=loss_func
            )

        # convert to numpy if needed
        y_true = np.asarray(
            y_true.cpu() if isinstance(y_true, torch.Tensor) else y_true
        )
        y_pred = np.asarray(
            y_pred.cpu() if isinstance(y_pred, torch.Tensor) else y_pred
        )
        y_prob = (
            np.asarray(y_prob.cpu() if isinstance(y_prob, torch.Tensor) else y_prob)
            if y_prob is not None
            else None
        )

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        labels = np.concatenate([y_true, y_pred]).astype(int)
        if np.any(labels < 0):
            raise ValueError("class labels must be non-negative integers")
        inferred_classes = int(labels.max()) + 1
        if num_classes is None:
            num_classes = inferred_classes
        elif inferred_classes > num_classes:
            raise ValueError("class labels are outside num_classes")
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            np.add.at(cm, (int(t), int(p)), 1)

        if loss is None:
            if loss_func == "cross_entropy" and y_prob is not None:
                if y_prob.ndim != 2 or y_prob.shape[0] != len(y_true):
                    raise ValueError("y_prob must have shape (n_samples, n_classes)")
                if y_prob.shape[1] < num_classes:
                    raise ValueError("y_prob has fewer columns than num_classes")
                # Clip probabilities to avoid log(0)
                eps = 1e-15
                y_prob = np.clip(y_prob, eps, 1 - eps)
                # One-hot encode true labels
                y_true_one_hot = np.zeros_like(y_prob)
                y_true_one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1
                # Compute cross-entropy loss
                loss = -np.mean(np.sum(y_true_one_hot * np.log(y_prob), axis=1))
            else:
                loss_func = None

        return ClassificationMetrics(
            confusion_matrix=cm, loss=loss, loss_func=loss_func
        )

    # ---------------------------------- PROPERTIES ---------------------------------- #

    @property
    def accuracy_per_class(self) -> dict[int, float]:
        total = np.sum(self.confusion_matrix)
        accuracies = {}

        for c in range(
            self.confusion_matrix.shape[0]
        ):  # accuracy = tp+tn / total (per class)
            tp = self.confusion_matrix[c, c]
            fp = np.sum(self.confusion_matrix[:, c]) - tp

            fn = np.sum(self.confusion_matrix[c, :]) - tp
            tn = total - tp - fp - fn
            accuracy_c = (tp + tn) / total if total > 0 else 0.0
            accuracies[c] = accuracy_c
        return accuracies

    @property
    def recall_per_class(self) -> dict[int, float]:
        recalls = {}
        for c in range(self.confusion_matrix.shape[0]):
            tp = self.confusion_matrix[c, c]
            fn = np.sum(self.confusion_matrix[c, :]) - tp
            recall_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls[c] = recall_c
        return recalls

    @property
    def precision_per_class(self) -> dict[int, float]:
        precisions = {}
        for c in range(self.confusion_matrix.shape[0]):
            tp = self.confusion_matrix[c, c]
            fp = np.sum(self.confusion_matrix[:, c]) - tp
            precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precisions[c] = precision_c
        return precisions

    @property
    def f1_score_per_class(self) -> dict[int, float]:
        """F1-score for each class."""
        f1s: dict[int, float] = {}
        for c in range(self.confusion_matrix.shape[0]):
            tp = self.confusion_matrix[c, c]
            fp = np.sum(self.confusion_matrix[:, c]) - tp
            fn = np.sum(self.confusion_matrix[c, :]) - tp
            precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_c = (
                2 * precision_c * recall_c / (precision_c + recall_c)
                if (precision_c + recall_c) > 0
                else 0.0
            )
            f1s[c] = f1_c
        return f1s

    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        return (
            np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)
            if np.sum(self.confusion_matrix) > 0
            else 0.0
        )

    @property
    def recall_balanced(self) -> float:
        """Macro-averaged recall."""
        return float(np.mean(list(self.recall_per_class.values())))

    @property
    def balanced_accuracy(self) -> float:
        """Balanced accuracy, i.e. macro-averaged recall (scikit-learn's definition)."""
        return self.recall_balanced

    @property
    def precision_balanced(self) -> float:
        """Macro-averaged precision."""
        return float(np.mean(list(self.precision_per_class.values())))

    @property
    def f1_score_macro(self) -> float:
        """Macro-averaged F1-score."""
        return float(np.mean(list(self.f1_score_per_class.values())))

    @property
    def metrics_per_class(self) -> dict[str, dict[int, float]]:
        return {
            "accuracy": self.accuracy_per_class,
            "recall": self.recall_per_class,
            "precision": self.precision_per_class,
            "f1_score": self.f1_score_per_class,
        }

    @property
    def metrics(self):
        return {
            # if binary classification, include positive class precision
            **(
                {
                    "precision": self.precision_per_class[1],
                    "recall": self.recall_per_class[1],
                    "f1_score": self.f1_score_per_class[1],
                }
                if self.confusion_matrix.shape[0] == 2
                else {}
            ),
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision_balanced": self.precision_balanced,
            "recall_balanced": self.recall_balanced,
            "f1_score_macro": self.f1_score_macro,
        }

    def scalar_metrics(self) -> dict[str, float]:
        """Flat mapping of finite scalar metrics for logging."""
        values = dict(self.metrics)
        if self.loss is not None:
            values["loss"] = self.loss
        return _finite_scalars(values)

    # -------------------------------------- IO -------------------------------------- #

    def format(self, long: bool = False) -> str:
        """Format metrics as a string."""
        return format(
            metrics=self.metrics,
            metrics_per_class=self.metrics_per_class,
            loss=self.loss,
            long=long,
            confusion_matrix=self.confusion_matrix,
        )

    def __str__(self) -> str:
        # Short one-liner format.
        return self.format(long=False)

    # ------------------------------- (DE)SERIALIZATION ------------------------------ #

    @field_validator("confusion_matrix", mode="before")
    @classmethod
    def to_ndarray(cls, v: Any) -> np.ndarray:
        if isinstance(v, np.ndarray):
            return v
        return np.array(v)

    @field_serializer("confusion_matrix")
    def serialize_array(self, v: np.ndarray):
        # simplest: just store as nested lists
        return v.tolist()


class MultiLabelROCAUCTracker(BaseModel):
    """
    Exact per-class ROC-AUC computed from the raw per-sample scores.

    The on-disk JSON holds only the scalar AUCs (macro and per class); the raw per-row
    ``labels``/``scores`` matrices stay in memory for as long as the tracker lives (so a
    freshly built tracker computes the exact AUC) but are never serialized; at
    production scale they reach tens of gigabytes per run. A tracker loaded from JSON
    therefore carries empty arrays and reports the stored scalar AUCs instead of
    recomputing them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    labels: np.ndarray = Field(default_factory=lambda: np.zeros((0, 0)), exclude=True)
    """In-memory binary ground truth, shape (num_samples, num_classes)."""

    scores: np.ndarray = Field(default_factory=lambda: np.zeros((0, 0)), exclude=True)
    """In-memory predicted scores, shape (num_samples, num_classes)."""

    auc_per_class: list[float] = Field(default_factory=list)
    """Persisted per-class ROC-AUC (``nan`` where undefined)."""

    @staticmethod
    def from_predictions(
        num_classes: int, y_true: ArrayLike, y_pred: ArrayLike, y_prob: ArrayLike
    ) -> "MultiLabelROCAUCTracker":
        """Store the per-sample labels and scores of a (batch_size, num_classes) batch
        and precompute the per-class AUC for serialization."""
        labels = np.asarray(y_true).reshape(-1, num_classes)
        scores = np.asarray(y_prob).reshape(-1, num_classes)
        tracker = MultiLabelROCAUCTracker(labels=labels, scores=scores)
        tracker.auc_per_class = tracker._auc_from_arrays(num_classes)
        return tracker

    def _auc_from_arrays(self, num_classes: int) -> list[float]:
        per_class: list[float] = []
        for c in range(num_classes):
            labels_c = self.labels[:, c]
            if not (np.any(labels_c == 1) and np.any(labels_c == 0)):
                per_class.append(float("nan"))
                continue
            per_class.append(float(roc_auc_score(labels_c, self.scores[:, c])))
        return per_class

    @property
    def auc(self) -> dict:
        """
        Per-class ROC-AUC and their macro average.

        A class without both a positive and a negative sample has no defined AUC and is
        reported as ``nan``; the macro average ignores those classes. Recomputed from
        the raw arrays when present, otherwise read from the persisted
        ``auc_per_class``.
        """
        if self.scores.ndim == 2 and self.scores.shape[1] > 0:
            per_class = self._auc_from_arrays(self.scores.shape[1])
        else:
            per_class = list(self.auc_per_class)

        finite = [value for value in per_class if not math.isnan(value)]
        macro = float(np.mean(finite)) if finite else float("nan")
        return {"per_class": per_class, "macro": macro}

    @property
    def auc_macro(self) -> float:
        """Macro ROC-AUC across all classes."""
        return self.auc["macro"]

    # ---------------------------------- PROPERTIES ---------------------------------- #

    @property
    def num_classes(self) -> int:
        # Prefer the in-memory width; a JSON-loaded tracker reads it from the
        # persisted per-class AUC list.
        if self.scores.ndim == 2 and self.scores.shape[1] > 0:
            return int(self.scores.shape[1])
        return len(self.auc_per_class)

    def scalar_metrics(self) -> dict[str, float]:
        """
        Flat mapping of finite scalar metrics for logging.

        Per-class AUC is ``nan`` for classes without both positive and negative samples;
        those entries are dropped by the finite filter.
        """
        auc = self.auc
        values: dict[str, Any] = {"auc_macro": auc["macro"]}
        for class_idx, class_auc in enumerate(auc["per_class"]):
            values[f"auc_class_{class_idx}"] = class_auc
        return _finite_scalars(values)

    # -------------------------------------- IO -------------------------------------- #

    def format(self, long: bool = False) -> str:
        """Format metrics as a string."""
        return ", ".join(
            f"{name.upper()}={value:.4f}"
            for name, value in self.scalar_metrics().items()
        )

    def __str__(self) -> str:
        return self.format(long=False)

    # ------------------------------- (DE)SERIALIZATION ------------------------------ #

    @field_validator("labels", "scores", mode="before")
    @classmethod
    def to_ndarray(cls, v: Any) -> np.ndarray:
        if isinstance(v, np.ndarray):
            return v
        return np.array(v)

    @field_validator("auc_per_class", mode="before")
    @classmethod
    def coerce_undefined_auc(cls, v: Any) -> Any:
        # Undefined per-class AUCs are ``nan``, which JSON renders as ``null``;
        # map them back to ``nan`` so a serialized tracker round-trips.
        if isinstance(v, list):
            return [float("nan") if item is None else item for item in v]
        return v


class MultiLabelClassificationMetrics(BaseModel):
    """
    Similar to ClassificationMetrics but for multi-label classification.

    Keeps tracks of a confusion matrix per class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    confusion_matrices: dict[int, np.ndarray]
    """
    Confusion matrices per class.

    Each confusion matrix is 2x2:
        [[tn, fp],
         [fn, tp]]
    """

    @staticmethod
    def from_predictions(
        num_classes: int, y_true: ArrayLike, y_pred: ArrayLike
    ) -> "MultiLabelClassificationMetrics":
        """Compute multi-label classification metrics from true and predicted labels."""
        cms: dict[int, np.ndarray]

        if len(y_true) == 0:
            # Return empty metrics if there are no samples
            num_classes = num_classes or 0
            cms = {c: np.zeros((2, 2), dtype=int) for c in range(num_classes)}
            return MultiLabelClassificationMetrics(confusion_matrices=cms)

        # convert to numpy if needed
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        num_classes = y_true.shape[1]
        cms = {c: np.zeros((2, 2), dtype=int) for c in range(num_classes)}

        for i in range(len(y_true)):
            for c in range(num_classes):
                true_label = int(y_true[i, c])
                pred_label = int(y_pred[i, c])
                cms[c][true_label, pred_label] += 1

        return MultiLabelClassificationMetrics(confusion_matrices=cms)

    # ---------------------------------- PROPERTIES ---------------------------------- #

    @property
    def accuracies(self) -> dict[int, float]:
        accuracies = {}
        for c, cm in self.confusion_matrices.items():
            tp = cm[1, 1]
            tn = cm[0, 0]
            fp = cm[0, 1]
            fn = cm[1, 0]
            total = tp + tn + fp + fn
            accuracy_c = (tp + tn) / total if total > 0 else 0.0
            accuracies[c] = accuracy_c
        return accuracies

    @property
    def f1_scores(self) -> dict[int, float]:
        f1_scores = {}
        for c, cm in self.confusion_matrices.items():
            tp = cm[1, 1]
            fp = cm[0, 1]
            fn = cm[1, 0]
            precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_c = (
                2 * precision_c * recall_c / (precision_c + recall_c)
                if (precision_c + recall_c) > 0
                else 0.0
            )
            f1_scores[c] = f1_c
        return f1_scores

    @property
    def f1_scores_macro(self) -> float:
        """Macro-averaged F1-score across all classes."""
        return float(np.mean(list(self.f1_scores.values())))

    @property
    def recalls(self) -> dict[int, float]:
        recalls = {}
        for c, cm in self.confusion_matrices.items():
            tp = cm[1, 1]
            fn = cm[1, 0]
            recall_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls[c] = recall_c
        return recalls

    @property
    def precisions(self) -> dict[int, float]:
        precisions = {}
        for c, cm in self.confusion_matrices.items():
            tp = cm[1, 1]
            fp = cm[0, 1]
            precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precisions[c] = precision_c
        return precisions

    def scalar_metrics(self) -> dict[str, float]:
        """Flat mapping of finite scalar metrics for logging."""
        values: dict[str, Any] = {"f1_score_macro": self.f1_scores_macro}
        for c, value in self.f1_scores.items():
            values[f"f1_score_class_{c}"] = value
        for c, value in self.accuracies.items():
            values[f"accuracy_class_{c}"] = value
        return _finite_scalars(values)

    # -------------------------------------- IO -------------------------------------- #

    def format(self, long: bool = False) -> str:
        """Format metrics as a string."""
        return ", ".join(
            f"{name.upper()}={value:.4f}"
            for name, value in self.scalar_metrics().items()
        )

    def __str__(self) -> str:
        return self.format(long=False)

    # ------------------------------- (DE)SERIALIZATION ------------------------------ #

    @field_validator("confusion_matrices", mode="before")
    @classmethod
    def to_ndarray(cls, v: Any) -> dict[int, np.ndarray]:
        if isinstance(v, dict):
            return {k: np.array(vv) for k, vv in v.items()}
        return v

    @field_serializer("confusion_matrices")
    def serialize_array(self, v: dict[int, np.ndarray]):
        # simplest: just store as nested lists
        return {k: vv.tolist() for k, vv in v.items()}


class RegressiveClassificationMetrics(BaseModel):
    """
    Classification metrics for regression tasks with discrete labels.

    The on-disk JSON holds only the scalar error metrics; the raw per-row
    ``actual``/``predicted`` vectors stay in memory (so a freshly built object computes
    exact metrics) but are never serialized; at production scale they reach tens of
    gigabytes per run. An object loaded from JSON carries empty vectors and reports the
    persisted scalar metrics instead.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    actual: np.ndarray = Field(default_factory=lambda: np.array([]), exclude=True)
    predicted: np.ndarray = Field(default_factory=lambda: np.array([]), exclude=True)
    loss: float | None = None
    scalar_summary: dict[str, float] = Field(default_factory=dict)
    """Persisted scalar metrics, used when the raw vectors are not in memory."""

    @staticmethod
    def from_predictions(
        y_true: ArrayLike,
        predicted: ArrayLike | None = None,
        *,
        loss: float | None = None,
    ) -> "RegressiveClassificationMetrics":
        metrics = RegressiveClassificationMetrics(
            actual=np.asarray(y_true),
            predicted=np.asarray(predicted) if predicted is not None else np.array([]),
            loss=loss,
        )
        metrics.scalar_summary = metrics._scalars_from_arrays()
        return metrics

    def _has_arrays(self) -> bool:
        return self.predicted.size > 0

    # ---------------------------------- PROPERTIES ---------------------------------- #

    @property
    def mae(self) -> float:
        """Mean Absolute Error."""
        if not self._has_arrays():
            return self.scalar_summary.get("mae", float("nan"))
        return np.mean(np.abs(self.actual - self.predicted))

    @property
    def mse(self) -> float:
        """Mean Squared Error."""
        if not self._has_arrays():
            return self.scalar_summary.get("mse", float("nan"))
        return np.mean((self.actual - self.predicted) ** 2)

    @property
    def rmse(self) -> float:
        """Root Mean Squared Error."""
        if not self._has_arrays():
            return self.scalar_summary.get("rmse", float("nan"))
        return np.sqrt(self.mse)

    @property
    def r2_score(self) -> float:
        """R-squared Score."""
        if not self._has_arrays():
            return self.scalar_summary.get("r2_score", float("nan"))
        ss_total = np.sum((self.actual - np.mean(self.actual)) ** 2)
        ss_residual = np.sum((self.actual - self.predicted) ** 2)
        return 1 - (ss_residual / ss_total) if ss_total > 0 else float("nan")

    @property
    def balanced_mse(self) -> float:
        """
        Balanced MSE: the per-class MSE averaged over the classes present.

        Classes (e.g. rating levels) with no samples are omitted from the average rather
        than counted as zero, so the denominator is the number of observed classes.
        """
        if not self._has_arrays():
            return self.scalar_summary.get("balanced_mse", float("nan"))
        unique_classes = np.unique(self.actual)
        mse_per_class = []
        for c in unique_classes:
            mask = self.actual == c
            if np.sum(mask) > 0:
                mse_c = np.mean((self.actual[mask] - self.predicted[mask]) ** 2)
                mse_per_class.append(mse_c)
        return np.mean(mse_per_class) if mse_per_class else float("nan")

    def _scalars_from_arrays(self) -> dict[str, float]:
        """Scalar metrics computed from the raw vectors, for persistence."""
        values: dict[str, float] = {
            "mse": float(self.mse),
            "mae": float(self.mae),
            "rmse": float(self.rmse),
            "r2_score": float(self.r2_score),
            "balanced_mse": float(self.balanced_mse),
        }
        return {key: value for key, value in values.items() if math.isfinite(value)}

    def scalar_metrics(self) -> dict[str, float]:
        """Flat mapping of finite scalar metrics for logging."""
        values: dict[str, Any] = {
            "mse": self.mse,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2_score": self.r2_score,
            "balanced_mse": self.balanced_mse,
        }
        if self.loss is not None:
            values["loss"] = self.loss
        return _finite_scalars(values)

    # -------------------------------------- IO -------------------------------------- #

    def format(self, long: bool = False) -> str:
        """Format metrics as a string."""
        return ", ".join(
            f"{name.upper()}={value:.4f}"
            for name, value in self.scalar_metrics().items()
        )

    def __str__(self) -> str:
        return self.format(long=False)

    # ------------------------------- (DE)SERIALIZATION ------------------------------ #

    @field_validator("actual", "predicted", mode="before")
    @classmethod
    def to_ndarray(cls, v: Any) -> np.ndarray:
        if isinstance(v, np.ndarray):
            return v
        return np.array(v)


def _metric_discriminator(value: Any) -> str:
    """
    Tag a metric payload (dict or model) by its distinguishing fields.

    The multilabel-AUC and regression models serialize only their scalar summaries
    (``auc_per_class`` and ``scalar_summary`` respectively), so field presence
    discriminates them unambiguously; payloads that also carry the raw arrays
    (``labels``/``scores`` or ``actual``/``predicted``) tag the same way.
    """
    if isinstance(value, ClassificationMetrics):
        return "classification"
    if isinstance(value, MultiLabelClassificationMetrics):
        return "multilabel_confusion"
    if isinstance(value, MultiLabelROCAUCTracker):
        return "multilabel_auc"
    if isinstance(value, RegressiveClassificationMetrics):
        return "regression"
    if isinstance(value, dict):
        if "confusion_matrix" in value:
            return "classification"
        if "confusion_matrices" in value:
            return "multilabel_confusion"
        if "auc_per_class" in value or "scores" in value or "labels" in value:
            return "multilabel_auc"
        if "scalar_summary" in value or "actual" in value or "predicted" in value:
            return "regression"
    raise ValueError("payload does not match any known metric model")


ClassificationMetricsUnion = Annotated[
    Annotated[ClassificationMetrics, Tag("classification")]
    | Annotated[MultiLabelClassificationMetrics, Tag("multilabel_confusion")]
    | Annotated[MultiLabelROCAUCTracker, Tag("multilabel_auc")]
    | Annotated[RegressiveClassificationMetrics, Tag("regression")],
    Discriminator(_metric_discriminator),
]
