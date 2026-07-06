from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

CUDA_ALLOWED_MARKER = "cuda_allowed"
# Marker that exempts a test from the CUDA guard below. The guard test keys on the
# exact name the fixture checks.


@pytest.fixture(autouse=True)
def _no_cuda_torch(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """
    Keep tests off CUDA on a deterministic float32 default (MPS is left available).

    Tests marked ``cuda_allowed`` -- the opt-in GPU model-matrix smoke -- are exempt, so
    they can run on CUDA when ``DRIFT_MODEL_MATRIX_SMOKE_DEVICE`` requests it.
    """
    previous_dtype = torch.get_default_dtype()
    if request.node.get_closest_marker(CUDA_ALLOWED_MARKER) is None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(previous_dtype)


@pytest.fixture
def tmp_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect artifact roots for tests that exercise run writers."""
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr("drift_happens.utils.paths.ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr("drift_happens.utils.paths.RUNS_DIR", artifacts / "runs")
    monkeypatch.setattr("drift_happens.utils.paths.SWEEPS_DIR", artifacts / "sweeps")
    monkeypatch.setattr("drift_happens.utils.paths.REPORTS_DIR", artifacts / "reports")
    monkeypatch.setattr(
        "drift_happens.utils.paths.EXPERIMENT_PLANS_DIR",
        artifacts / "experiment_plans",
    )
    return artifacts
