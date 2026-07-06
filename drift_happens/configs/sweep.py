"""Sweep configuration for dispatching many experiment jobs into device slots."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from drift_happens.configs.base import BaseConfig

DEFAULT_SWEEP_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
SlotDevice = Literal["cpu", "cuda", "mps"]
SweepAction = Literal["run", "train", "eval"]
SkipCompletedSource = Literal["local", "wandb"]


class JobSpecConfig(BaseConfig):
    """One concrete experiment config file plus the seed to launch."""

    config_path: Path
    seed: int
    label: str
    action: SweepAction = "run"
    overrides: tuple[str, ...] = ()
    """CLI-style dotted overrides applied before launching the job."""

    tags: tuple[str, ...] = ()
    skip_completed: bool | None = None


class DeviceSlotConfig(BaseConfig):
    """
    A process slot the sweep runner can dispatch into.

    CUDA slots use a host ``device_index``. The sweep runner exposes that index to the
    child process via ``CUDA_VISIBLE_DEVICES`` so the job itself still sees one logical
    CUDA device.
    """

    device: SlotDevice = "cpu"
    device_index: Annotated[int, Field(ge=0)] | None = None
    label: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_label(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("label") is not None:
            return data
        device = data.get("device", cls.model_fields["device"].default)
        device_index = data.get("device_index")
        if device == "cuda" and device_index is not None:
            label = f"cuda:{device_index}"
        else:
            label = str(device)
        return {**data, "label": label}

    @model_validator(mode="after")
    def _device_index_matches_device(self) -> DeviceSlotConfig:
        if self.device != "cuda" and self.device_index is not None:
            raise ValueError("device_index is only valid for CUDA slots")
        if self.device == "cuda" and self.device_index is None:
            raise ValueError("CUDA slots require a device_index")
        return self


class SweepConfig(BaseConfig):
    """A set of jobs plus the local device slots they may run on."""

    name: str
    jobs: tuple[JobSpecConfig, ...]
    slots: tuple[DeviceSlotConfig, ...]
    seeds: Annotated[tuple[int, ...], Field(min_length=1)] = DEFAULT_SWEEP_SEEDS
    concurrency: Annotated[int, Field(gt=0)] = 1
    resume: bool | None = None
    """When omitted, child jobs reuse completed work; set DRIFT_ENABLE_AUTO_RESUME=0 to
    force fresh re-runs."""

    skip_completed: bool = True
    skip_source: SkipCompletedSource = "local"

    @model_validator(mode="before")
    @classmethod
    def _expand_seedless_jobs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        raw_jobs = data.get("jobs")
        if not isinstance(raw_jobs, (list, tuple)):
            return data

        seeds = data.get("seeds", DEFAULT_SWEEP_SEEDS)
        if not isinstance(seeds, (list, tuple)):
            # Let pydantic reject a malformed seeds value with a clear field error.
            return data

        expanded_jobs: list[Any] = []
        for raw_job in raw_jobs:
            if isinstance(raw_job, dict) and raw_job.get("seed") is None:
                expanded_jobs.extend({**raw_job, "seed": seed} for seed in seeds)
            else:
                expanded_jobs.append(raw_job)

        return {**data, "jobs": expanded_jobs}

    @model_validator(mode="after")
    def _validate_scheduler_shape(self) -> SweepConfig:
        if not self.jobs:
            raise ValueError("SweepConfig.jobs must contain at least one entry")
        if not self.slots:
            raise ValueError("SweepConfig.slots must contain at least one entry")
        if self.concurrency > len(self.slots):
            raise ValueError(
                f"concurrency={self.concurrency} exceeds slot count {len(self.slots)}"
            )
        seen: set[tuple[str, int]] = set()
        for job in self.jobs:
            key = (job.label, job.seed)
            if key in seen:
                raise ValueError(
                    f"duplicate sweep job (label={job.label!r}, seed={job.seed}); "
                    "each (label, seed) pair must be unique"
                )
            seen.add(key)
        return self
