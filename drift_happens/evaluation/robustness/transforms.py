"""
Utility transforms for mapping raw metrics to [0, 1].

Higher utility always means better performance:
- accuracy, auc_macro: identity (already in [0,1])
- balanced_mse: exp(-x) with λ fixed at 1.0 (inverts loss to utility)
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

_MIN_UTILITY: float = 1e-10


class UtilityTransform(ABC):
    """Base class for metric-to-utility transforms."""

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Transform raw metric to utility in [0, 1]."""

    @abstractmethod
    def inverse(self, u: np.ndarray) -> np.ndarray:
        """Inverse transform: utility back to raw metric."""


class IdentityTransform(UtilityTransform):
    """Identity transform for metrics already in [0, 1] where higher is better."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, 1.0)

    def inverse(self, u: np.ndarray) -> np.ndarray:
        return u


class ExponentialTransform(UtilityTransform):
    """Exponential transform for loss metrics: u = exp(-λx).

    Maps loss ∈ [0, ∞) to utility ∈ (0, 1].
    """

    def __init__(self, lam: float = 1.0):
        if lam <= 0:
            raise ValueError("lambda must be positive")
        self.lam = lam

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-self.lam * np.asarray(x))

    def inverse(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, _MIN_UTILITY, 1.0)
        return -np.log(u) / self.lam


MetricType = Literal["higher_is_better", "lower_is_better"]


def get_transform(metric_type: MetricType, lam: float = 1.0) -> UtilityTransform:
    """Get transform for a given metric type."""
    if metric_type == "higher_is_better":
        return IdentityTransform()
    elif metric_type == "lower_is_better":
        return ExponentialTransform(lam=lam)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")
