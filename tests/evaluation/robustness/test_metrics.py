from __future__ import annotations

import numpy as np
import pytest

from drift_happens.evaluation.metrics import ClassificationMetrics
from drift_happens.evaluation.robustness.metrics import (
    WeightingScheme,
    analyze_drift_robustness,
    build_matrix_from_results,
    compute_cvar,
    compute_utility_matrix,
)
from drift_happens.pipeline.models import TrainerEvaluationResults


def _classification(accuracy: float) -> ClassificationMetrics:
    total = 100
    correct = round(total * accuracy)
    return ClassificationMetrics(
        confusion_matrix=np.array([[correct, 0], [total - correct, 0]])
    )


def test_compute_utility_matrix_validates_square_matrix_and_times() -> None:
    with pytest.raises(ValueError, match="Expected square matrix"):
        compute_utility_matrix(np.ones((2, 3)))
    with pytest.raises(ValueError, match="times has 1 elements"):
        compute_utility_matrix(np.eye(2), times=np.array([2000]))


def test_analyze_drift_robustness_higher_is_better_small_matrix() -> None:
    result = analyze_drift_robustness(
        np.array([[0.9, 0.8, 0.7], [np.nan, 0.85, 0.6], [np.nan, np.nan, 0.95]]),
        return_matrices=True,
    )

    assert result.strength == pytest.approx((0.9 + 0.85) / 2)
    assert result.num_future_pairs == 3
    assert result.robustness_abs == pytest.approx(1 - np.mean([0.1, 0.2, 0.25]))
    assert result.robustness_rel == pytest.approx(
        np.mean([0.8 / 0.9, 0.7 / 0.9, 0.6 / 0.85]), rel=1e-6
    )
    assert result.utility_matrix is not None
    assert result.retention_matrix is not None


def test_analyze_drift_robustness_lower_is_better_uses_exponential_transform() -> None:
    result = analyze_drift_robustness(
        np.array([[0.2, 0.5], [np.nan, 0.4]]),
        metric_type="lower_is_better",
        return_matrices=True,
    )

    assert result.utility_matrix is not None
    assert result.utility_matrix[0, 0] > result.utility_matrix[0, 1]
    assert result.strength == pytest.approx(np.exp(-0.2))


def test_weighting_scheme_applies_decay_and_row_weights() -> None:
    weights = WeightingScheme(beta=0.5, row_weights=np.array([2.0, 1.0]))

    out = weights.compute_weights(
        time_gaps=np.array([1.0, 2.0]), row_indices=np.array([0, 1])
    )

    np.testing.assert_allclose(out, np.array([2 * np.exp(-0.5), np.exp(-1.0)]))


@pytest.mark.parametrize("tau", [0, 1, -0.1])
def test_compute_cvar_rejects_invalid_tau(tau: float) -> None:
    with pytest.raises(ValueError, match="tau must be"):
        compute_cvar(np.array([[np.nan, 0.5]]), tau=tau)


def test_compute_cvar_all_nan_returns_nan() -> None:
    assert np.isnan(compute_cvar(np.full((2, 2), np.nan), tau=0.5))


def test_compute_cvar_happy_path() -> None:
    # retentions=[0.9, 0.7, 0.6], tau=0.5 -> threshold=quantile(0.5)=0.7
    # tail=[0.7, 0.6], cvar=0.65
    R = np.array([[np.nan, 0.9, 0.7], [np.nan, np.nan, 0.6], [np.nan, np.nan, np.nan]])
    assert compute_cvar(R, tau=0.5) == pytest.approx(0.65)


def test_analyze_drift_robustness_small_matrix_all_scalar_metrics() -> None:
    # Hand-computed expected values for the 3x3 matrix used in the higher-is-better
    # small-matrix test; verifies drs_alpha, ldrs, cvar, and both combined metrics.
    M = np.array([[0.9, 0.8, 0.7], [np.nan, 0.85, 0.6], [np.nan, np.nan, 0.95]])

    result = analyze_drift_robustness(M)

    assert result.drs_alpha == pytest.approx(result.robustness_rel, rel=1e-6)
    assert result.ldrs == pytest.approx(0.7873088119400171, rel=1e-6)
    assert result.cvar == pytest.approx(0.7058823446366782, rel=1e-6)
    assert result.combined_harmonic == pytest.approx(
        2
        * result.strength
        * result.robustness_rel
        / (result.strength + result.robustness_rel),
        rel=1e-6,
    )
    assert result.combined_linear == pytest.approx(
        0.5 * result.strength + 0.5 * result.robustness_rel, rel=1e-6
    )


def test_build_matrix_from_results_extracts_metric_and_sorts_keys() -> None:
    results: dict[str | int, TrainerEvaluationResults] = {
        "b": TrainerEvaluationResults(results={"b": _classification(0.8)}),
        "a": TrainerEvaluationResults(results={"a": _classification(0.6)}),
    }

    matrix, train_keys, eval_keys = build_matrix_from_results(results)

    assert train_keys == ["a", "b"]
    assert eval_keys == ["a", "b"]
    np.testing.assert_allclose(matrix, [[0.6, np.nan], [np.nan, 0.8]])


def test_build_matrix_from_results_orders_numeric_keys_numerically() -> None:
    # Mixed-width slice keys must sort by value (2, 3, 10), not lexically
    # (10, 2, 3), so the matrix rows and columns stay in temporal order.
    results: dict[str | int, TrainerEvaluationResults] = {
        slice_key: TrainerEvaluationResults(
            results={eval_key: _classification(0.5) for eval_key in ("2", "10", "3")}
        )
        for slice_key in ("2", "10", "3")
    }

    _, train_keys, eval_keys = build_matrix_from_results(results)

    assert train_keys == ["2", "3", "10"]
    assert eval_keys == ["2", "3", "10"]


def test_nan_diagonal_pairs_are_dropped_consistently() -> None:
    # A NaN on the diagonal (missing in-distribution baseline) must be dropped
    # by every pair-based metric, matching DRS_alpha, instead of poisoning
    # R_abs/R_rel/LDRS to NaN or crashing on a main-loop/gaps length mismatch.
    M = np.array(
        [
            [0.9, 0.8, 0.7, 0.6],
            [np.nan, np.nan, 0.6, 0.5],
            [np.nan, np.nan, 0.95, 0.7],
            [np.nan, np.nan, np.nan, 0.9],
        ]
    )

    result = analyze_drift_robustness(M)

    assert np.isfinite(result.robustness_abs)
    assert np.isfinite(result.robustness_rel)
    assert np.isfinite(result.ldrs)
    assert np.isfinite(result.drs_alpha)
