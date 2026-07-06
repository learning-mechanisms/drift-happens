from __future__ import annotations

import numpy as np
import pytest

from drift_happens.evaluation.robustness.transforms import (
    ExponentialTransform,
    IdentityTransform,
    get_transform,
)


def test_identity_transform_clips_to_unit_interval() -> None:
    out = IdentityTransform()(np.array([-0.5, 0.2, 1.5]))

    np.testing.assert_allclose(out, [0.0, 0.2, 1.0])


def test_exponential_transform_round_trips() -> None:
    transform = ExponentialTransform(lam=0.5)
    raw = np.array([0.0, 1.0, 2.0])

    utility = transform(raw)

    assert np.all((utility > 0) & (utility <= 1))
    np.testing.assert_allclose(transform.inverse(utility), raw)


def test_get_transform_rejects_unknown_metric_type() -> None:
    with pytest.raises(ValueError, match="Unknown metric type"):
        get_transform("unknown")  # type: ignore[arg-type]


def test_get_transform_higher_is_better_returns_identity() -> None:
    assert isinstance(get_transform("higher_is_better"), IdentityTransform)


def test_get_transform_lower_is_better_returns_exponential_with_lam() -> None:
    t = get_transform("lower_is_better", lam=2.0)
    assert isinstance(t, ExponentialTransform)
    assert t.lam == pytest.approx(2.0)


def test_exponential_transform_rejects_non_positive_lam() -> None:
    with pytest.raises(ValueError, match="lambda must be positive"):
        ExponentialTransform(lam=0)
