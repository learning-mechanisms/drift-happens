"""Runtime configuration for a single experiment process."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from drift_happens.configs.base import BaseConfig

DeviceSelector = Literal["auto", "cpu", "cuda", "mps"]
MixedPrecisionMode = Literal["off", "fp16", "bf16"]
RuntimeBackend = Literal["local"]


class RuntimeConfig(BaseConfig):
    """
    Where and how one experiment process executes.

    Each ``ExperimentConfig`` describes one seed and one process. Multi-device
    throughput belongs in ``SweepConfig`` by launching multiple processes into explicit
    slots.
    """

    backend: RuntimeBackend = "local"
    """
    Execution backend.

    Only local execution is currently supported.
    """

    device: DeviceSelector = "auto"
    """Torch device selector resolved by the worker process."""

    deterministic: bool = False
    """Request deterministic Torch backend behavior where supported."""

    cudnn_benchmark: bool = True
    """Allow cuDNN kernel autotuning when deterministic mode is disabled."""

    mixed_precision: MixedPrecisionMode = "off"
    """
    Mixed precision mode.

    Exposed now, implemented later.
    """

    num_threads: Annotated[int, Field(gt=0)] | None = None
    """Optional Torch CPU thread-count override for the process."""

    @model_validator(mode="after")
    def _supported_precision(self) -> RuntimeConfig:
        if self.mixed_precision != "off":
            raise ValueError(
                "runtime.mixed_precision is reserved for future work; set it to 'off'"
            )
        return self
