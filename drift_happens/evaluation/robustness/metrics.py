"""
Drift robustness metrics for temporal distribution shift evaluation.

This module separates two distinct model properties:
- Strength (S): in-distribution performance (diagonal of the utility matrix)
- Robustness (R): performance retention under temporal drift

Metrics:
    S           Weighted mean of diagonal utilities
    R_abs       1 - mean absolute drop
    R_rel       Mean retention U_ij / U_ii (relative robustness)
    DRS_α       Power-normalized retention family
    LDRS        Geometric mean of retentions (log-space)
    CVaR_τ      Conditional Value at Risk (worst τ% of cases)
    H           Harmonic mean of S and R_rel
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from drift_happens.evaluation.robustness.transforms import (
    IdentityTransform,
    MetricType,
    UtilityTransform,
    get_transform,
)

if TYPE_CHECKING:
    from drift_happens.pipeline.models import TrainerEvaluationResults


MetricName = Literal["accuracy", "auc_macro", "balanced_mse"]


# -----------------------------------------------------------------------------
# Weighting
# -----------------------------------------------------------------------------


@dataclass
class WeightingScheme:
    """
    Weighting scheme for (i, j) pairs in the future region.

    Weight formula: w_ij = π_i * exp(-β * Δt_ij)

    Attributes:
        beta: Temporal decay rate. β=0 means uniform weights.
        row_weights: Per-row weights π_i. None means uniform 1/K.
    """

    beta: float = 0.0
    row_weights: np.ndarray | None = None

    def compute_weights(
        self, time_gaps: np.ndarray, row_indices: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute weights for given time gaps and row indices."""
        w = np.exp(-self.beta * time_gaps)
        if self.row_weights is not None and row_indices is not None:
            w = w * self.row_weights[row_indices]
        return w


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------


class DriftRobustnessResult(BaseModel):
    """Container for drift robustness analysis results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Strength
    strength: float
    """S = Σ π_i U_ii."""

    # Robustness
    robustness_abs: float
    """R_abs = 1 - mean(U_ii - U_ij)"""

    robustness_rel: float
    """R_rel = mean(U_ij / U_ii)"""

    # DRS family
    drs_alpha: float
    """DRS_α = mean(U_ij / U_ii^α)"""

    alpha: float

    # Log-utility
    ldrs: float
    """LDRS = exp(mean(log(U_ij / U_ii)))"""

    # Tail risk
    cvar: float
    """CVaR_τ = E[R | R ≤ q_τ]"""

    cvar_tau: float

    # Combined
    combined_linear: float
    """J_λ = λS + (1-λ)R."""

    lambda_param: float

    combined_harmonic: float
    """H = 2SR / (S + R)"""

    # Diagnostics
    num_future_pairs: int
    utility_matrix: np.ndarray | None = None
    retention_matrix: np.ndarray | None = None

    def __str__(self) -> str:
        return (
            f"S={self.strength:.4f}, "
            f"R_abs={self.robustness_abs:.4f}, "
            f"R_rel={self.robustness_rel:.4f}, "
            f"LDRS={self.ldrs:.4f}, "
            f"CVaR={self.cvar:.4f}, "
            f"H={self.combined_harmonic:.4f}"
        )


# -----------------------------------------------------------------------------
# Core computations
# -----------------------------------------------------------------------------


def compute_utility_matrix(
    M: np.ndarray,
    transform: UtilityTransform | None = None,
    times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert raw performance matrix M to utility matrix U.

    Args:
        M: K×K temporal performance matrix. M[i,j] = perf of model trained
           up to time i, evaluated at time j.
        transform: Utility transform φ. Defaults to identity.
        times: Time values for each slice. Defaults to 0..K-1.

    Returns:
        U: Utility matrix with U_ij = φ(M_ij).
        time_gaps: Matrix of Δ_ij = t_j - t_i.
    """
    M = np.asarray(M, dtype=float)
    K = M.shape[0]

    if M.shape[1] != K:
        raise ValueError(f"Expected square matrix, got {M.shape}")

    if times is None:
        times = np.arange(K, dtype=float)
    else:
        times = np.asarray(times, dtype=float)
        if len(times) != K:
            raise ValueError(f"times has {len(times)} elements, expected {K}")

    if transform is None:
        transform = IdentityTransform()

    U = transform(M)
    time_gaps = times[np.newaxis, :] - times[:, np.newaxis]

    return U, time_gaps


def compute_strength(
    U: np.ndarray,
    row_weights: np.ndarray | None = None,
) -> float:
    """
    Compute strength S = weighted mean of diagonal utilities.

    Excludes the last row since it has no future evaluations.
    """
    diagonal = np.diag(U)
    K = len(diagonal)
    diagonal = diagonal[:-1] if K > 1 else diagonal

    if row_weights is None:
        return float(np.mean(diagonal))

    pi = row_weights[: len(diagonal)]
    pi = pi / np.sum(pi)
    return float(np.sum(pi * diagonal))


def compute_robustness_abs(
    U: np.ndarray,
    time_gaps: np.ndarray,
    weighting: WeightingScheme | None = None,
) -> tuple[float, int]:
    """Compute absolute robustness R_abs = 1 - mean(U_ii - U_ij).

    Returns:
        R_abs score and number of (i,j) pairs used.
    """
    if weighting is None:
        weighting = WeightingScheme()

    K = U.shape[0]
    diagonal = np.diag(U)

    drops_list: list[float] = []
    row_indices_list: list[int] = []
    gaps_list: list[float] = []

    for i in range(K - 1):
        for j in range(i + 1, K):
            if np.isnan(U[i, j]) or np.isnan(diagonal[i]):
                continue
            drops_list.append(diagonal[i] - U[i, j])
            row_indices_list.append(i)
            gaps_list.append(time_gaps[i, j])

    if len(drops_list) == 0:
        return np.nan, 0

    drops = np.array(drops_list)
    row_indices = np.array(row_indices_list)
    gaps = np.array(gaps_list)
    w = weighting.compute_weights(gaps, row_indices)

    mean_drop = np.sum(w * drops) / np.sum(w)
    return float(1.0 - mean_drop), len(drops)


def compute_robustness_rel(
    U: np.ndarray,
    time_gaps: np.ndarray,
    weighting: WeightingScheme | None = None,
    eps: float = 1e-8,
) -> float:
    """Compute relative robustness R_rel = mean(U_ij / U_ii)."""
    if weighting is None:
        weighting = WeightingScheme()

    K = U.shape[0]
    diagonal = np.diag(U)

    retentions_list: list[float] = []
    row_indices_list: list[int] = []
    gaps_list: list[float] = []

    for i in range(K - 1):
        for j in range(i + 1, K):
            if np.isnan(U[i, j]) or np.isnan(diagonal[i]) or diagonal[i] < eps:
                continue
            retentions_list.append(U[i, j] / (diagonal[i] + eps))
            row_indices_list.append(i)
            gaps_list.append(time_gaps[i, j])

    if len(retentions_list) == 0:
        return np.nan

    retentions = np.array(retentions_list)
    row_indices = np.array(row_indices_list)
    gaps = np.array(gaps_list)
    w = weighting.compute_weights(gaps, row_indices)

    return float(np.sum(w * retentions) / np.sum(w))


def compute_drs_alpha(
    U: np.ndarray,
    time_gaps: np.ndarray,
    alpha: float = 1.0,
    weighting: WeightingScheme | None = None,
    eps: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """
    Compute DRS_α = mean(U_ij / U_ii^α).

    α=1 gives R_rel, α=0 gives mean future utility.

    Returns:
        DRS_α score and the full R^(α) matrix.
    """
    if weighting is None:
        weighting = WeightingScheme()

    K = U.shape[0]
    diagonal = np.diag(U)

    R_alpha = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(i + 1, K):
            if np.isnan(U[i, j]):
                continue
            R_alpha[i, j] = U[i, j] / ((diagonal[i] + eps) ** alpha)

    values_list: list[float] = []
    row_indices_list: list[int] = []
    gaps_list: list[float] = []

    for i in range(K - 1):
        for j in range(i + 1, K):
            if not np.isnan(R_alpha[i, j]):
                values_list.append(R_alpha[i, j])
                row_indices_list.append(i)
                gaps_list.append(time_gaps[i, j])

    if len(values_list) == 0:
        return np.nan, R_alpha

    values = np.array(values_list)
    row_indices = np.array(row_indices_list)
    gaps = np.array(gaps_list)
    w = weighting.compute_weights(gaps, row_indices)

    return float(np.sum(w * values) / np.sum(w)), R_alpha


def compute_ldrs(
    U: np.ndarray,
    time_gaps: np.ndarray,
    weighting: WeightingScheme | None = None,
    eps: float = 1e-8,
) -> float:
    """
    Compute LDRS = exp(mean(log(U_ij / U_ii))).

    Geometric mean of retentions. More robust to outliers than arithmetic mean.
    """
    if weighting is None:
        weighting = WeightingScheme()

    K = U.shape[0]
    diagonal = np.diag(U)

    log_retentions_list: list[float] = []
    row_indices_list: list[int] = []
    gaps_list: list[float] = []

    for i in range(K - 1):
        for j in range(i + 1, K):
            if (
                np.isnan(U[i, j])
                or np.isnan(diagonal[i])
                or U[i, j] <= 0
                or diagonal[i] <= 0
            ):
                continue
            log_retentions_list.append(
                np.log(U[i, j] + eps) - np.log(diagonal[i] + eps)
            )
            row_indices_list.append(i)
            gaps_list.append(time_gaps[i, j])

    if len(log_retentions_list) == 0:
        return np.nan

    log_retentions = np.array(log_retentions_list)
    row_indices = np.array(row_indices_list)
    gaps = np.array(gaps_list)
    w = weighting.compute_weights(gaps, row_indices)

    mean_log = np.sum(w * log_retentions) / np.sum(w)
    return float(np.exp(mean_log))


def compute_cvar(R_alpha: np.ndarray, tau: float = 0.1) -> float:
    """
    Compute CVaR_τ = E[R | R ≤ q_τ].

    Mean retention in the worst τ fraction of cases.
    """
    if not 0 < tau < 1:
        raise ValueError(f"tau must be in (0, 1), got {tau}")

    retentions = R_alpha[~np.isnan(R_alpha)]
    if len(retentions) == 0:
        return np.nan

    threshold = np.quantile(retentions, tau)
    tail = retentions[retentions <= threshold]

    return float(np.mean(tail))


def compute_combined_metrics(
    strength: float,
    robustness: float,
    lambda_param: float = 0.5,
) -> tuple[float, float]:
    """
    Compute combined metrics.

    Returns:
        J_λ = λS + (1-λ)R (linear combination)
        H = 2SR / (S+R) (harmonic mean)
    """
    j_lambda = lambda_param * strength + (1 - lambda_param) * robustness

    if strength + robustness > 0:
        harmonic = 2 * strength * robustness / (strength + robustness)
    else:
        harmonic = 0.0

    return float(j_lambda), float(harmonic)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def analyze_drift_robustness(
    M: np.ndarray,
    times: np.ndarray | None = None,
    metric_type: MetricType = "higher_is_better",
    transform: UtilityTransform | None = None,
    weighting: WeightingScheme | None = None,
    alpha: float = 1.0,
    cvar_tau: float = 0.1,
    lambda_param: float = 0.5,
    return_matrices: bool = False,
    eps: float = 1e-8,
) -> DriftRobustnessResult:
    """
    Run full drift robustness analysis.

    Args:
        M: K×K temporal performance matrix.
        times: Time values for each slice.
        metric_type: "higher_is_better" or "lower_is_better" (determines transform).
        transform: Custom utility transform (overrides metric_type).
        weighting: Weighting scheme. Defaults to uniform.
        alpha: Power parameter for DRS_α.
        cvar_tau: τ level for CVaR.
        lambda_param: λ for linear combination.
        return_matrices: Include U and R matrices in result.
        eps: Numerical stability constant.

    Returns:
        DriftRobustnessResult with all computed metrics.
    """
    if transform is None:
        transform = get_transform(metric_type)

    U, time_gaps = compute_utility_matrix(M, transform, times)
    row_weights = weighting.row_weights if weighting else None

    strength = compute_strength(U, row_weights)
    robustness_abs, n_pairs = compute_robustness_abs(U, time_gaps, weighting)
    robustness_rel = compute_robustness_rel(U, time_gaps, weighting, eps)
    drs_alpha, R_alpha = compute_drs_alpha(U, time_gaps, alpha, weighting, eps)
    ldrs = compute_ldrs(U, time_gaps, weighting, eps)
    cvar = compute_cvar(R_alpha, cvar_tau)
    combined_linear, combined_harmonic = compute_combined_metrics(
        strength, robustness_rel, lambda_param
    )

    return DriftRobustnessResult(
        strength=strength,
        robustness_abs=robustness_abs,
        robustness_rel=robustness_rel,
        drs_alpha=drs_alpha,
        alpha=alpha,
        ldrs=ldrs,
        cvar=cvar,
        cvar_tau=cvar_tau,
        combined_linear=combined_linear,
        lambda_param=lambda_param,
        combined_harmonic=combined_harmonic,
        num_future_pairs=n_pairs,
        utility_matrix=U if return_matrices else None,
        retention_matrix=R_alpha if return_matrices else None,
    )


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------


def _extract_metric(metrics_obj, metric_name: str) -> float:
    """Extract a scalar metric from a metrics object."""
    if hasattr(metrics_obj, metric_name):
        return float(getattr(metrics_obj, metric_name))
    raise AttributeError(
        f"{type(metrics_obj).__name__} has no attribute '{metric_name}'"
    )


def build_matrix_from_results(
    results: dict[str | int, "TrainerEvaluationResults"],
    metric: MetricName = "accuracy",
) -> tuple[np.ndarray, list[str | int], list[str]]:
    """
    Build performance matrix from TrainerEvaluationResults.

    Returns:
        M: K×K matrix where M[i,j] is performance of model trained on
           slice i, evaluated on slice j.
        train_keys: Row labels (training slices).
        eval_keys: Column labels (evaluation slices).
    """
    train_keys = sorted(results.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
    eval_keys_set: set[str] = set()
    for ter in results.values():
        eval_keys_set.update(ter.results.keys())
    eval_keys = sorted(eval_keys_set, key=lambda x: int(x) if str(x).isdigit() else x)

    K_train = len(train_keys)
    K_eval = len(eval_keys)
    M = np.full((K_train, K_eval), np.nan)

    for i, train_key in enumerate(train_keys):
        ter = results[train_key]
        for j, eval_key in enumerate(eval_keys):
            if eval_key in ter.results:
                try:
                    M[i, j] = _extract_metric(ter.results[eval_key], metric)
                except AttributeError:
                    pass

    return M, train_keys, eval_keys
