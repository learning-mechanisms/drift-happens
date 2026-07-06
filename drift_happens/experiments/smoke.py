"""Synthetic smoke presets for runtime contract tests."""

from __future__ import annotations

from drift_happens.configs import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    LoggingConfig,
    RuntimeConfig,
    TrainerConfig,
)
from drift_happens.experiments.common import SMOKE_SEEDS
from drift_happens.experiments.types import PresetEntry

GROUP = "smoke"


def synthetic_classification_cpu() -> ExperimentConfig:
    """Return a tiny CPU-only config that exercises the local run contract."""
    return ExperimentConfig(
        name="smoke-synthetic-classification-cpu",
        seed=0,
        task="train_eval",
        dataset=DatasetConfig(name="synthetic", variant="classification"),
        trainer=TrainerConfig(
            key="linear-classifier",
            family="synthetic",
            model={"architecture": "linear"},
            training={
                "batch_size": 16,
                "learning_rate": 0.1,
                "n_features": 6,
                "n_samples": 64,
                "num_epochs": 2,
            },
        ),
        evaluation=EvaluationConfig(metric="accuracy"),
        runtime=RuntimeConfig(device="cpu", deterministic=True, cudnn_benchmark=False),
        logging=LoggingConfig(stdout=False),
        tags=("preset", "smoke", "synthetic", "cpu"),
        metadata={"purpose": "runtime_contract_smoke_test"},
    )


def presets() -> tuple[PresetEntry, ...]:
    return (
        PresetEntry(
            group=GROUP,
            name="synthetic-classification-cpu",
            factory=synthetic_classification_cpu,
            seeds=SMOKE_SEEDS,
            description="CPU-only synthetic classification smoke preset for the local runtime contract.",
            tags=("smoke", "synthetic", "cpu"),
            comparison_group="smoke/runtime-contract",
            comparison_role="smoke",
            variant_fields=("seed",),
        ),
    )
