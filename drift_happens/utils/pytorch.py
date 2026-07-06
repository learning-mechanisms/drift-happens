import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for one experiment process."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds all CUDA and MPS devices internally

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def device_manual_mps_or_cuda_if_available() -> str | None:
    """
    Return 'mps' if MPS is available, else 'cuda' if CUDA is available, else None.

    The GPU_DEVICE_ID environment variable overrides auto-detection; a bare index such
    as ``0`` is normalized to ``cuda:0``.
    """
    gpu_device_id = os.getenv("GPU_DEVICE_ID")
    if gpu_device_id is not None:
        selector = f"cuda:{gpu_device_id}" if gpu_device_id.isdigit() else gpu_device_id
        try:
            torch.device(selector)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                f"GPU_DEVICE_ID={gpu_device_id!r} is not a valid torch device selector"
            ) from exc
        return selector

    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return None
