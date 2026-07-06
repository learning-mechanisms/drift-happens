from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from drift_happens.utils.pytorch import (
    device_manual_mps_or_cuda_if_available,
    seed_everything,
)


def test_seed_everything_rng_is_reproducible() -> None:
    seed_everything(123)
    first = (random.random(), np.random.rand(), torch.rand(1))

    seed_everything(123)
    second = (random.random(), np.random.rand(), torch.rand(1))

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])


def test_seed_everything_deterministic_flag_enables_deterministic_algorithms() -> None:
    seed_everything(0, deterministic=True)
    enabled = torch.are_deterministic_algorithms_enabled()
    # restore global state so this test does not affect others
    torch.use_deterministic_algorithms(False)

    assert enabled is True


def test_gpu_device_id_bare_index_is_normalized_to_cuda_selector(monkeypatch) -> None:
    monkeypatch.setenv("GPU_DEVICE_ID", "1")

    assert device_manual_mps_or_cuda_if_available() == "cuda:1"


def test_gpu_device_id_full_selector_is_passed_through(monkeypatch) -> None:
    monkeypatch.setenv("GPU_DEVICE_ID", "cuda:0")

    assert device_manual_mps_or_cuda_if_available() == "cuda:0"


def test_gpu_device_id_rejects_unparsable_selector(monkeypatch) -> None:
    monkeypatch.setenv("GPU_DEVICE_ID", "not-a-device")

    with pytest.raises(ValueError, match="GPU_DEVICE_ID"):
        device_manual_mps_or_cuda_if_available()


def test_device_auto_detects_none_when_no_gpu(monkeypatch) -> None:
    monkeypatch.delenv("GPU_DEVICE_ID", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert device_manual_mps_or_cuda_if_available() is None


def test_device_auto_detects_cuda_when_available(monkeypatch) -> None:
    monkeypatch.delenv("GPU_DEVICE_ID", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert device_manual_mps_or_cuda_if_available() == "cuda"


def test_device_prefers_mps_over_cuda(monkeypatch) -> None:
    monkeypatch.delenv("GPU_DEVICE_ID", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert device_manual_mps_or_cuda_if_available() == "mps"
