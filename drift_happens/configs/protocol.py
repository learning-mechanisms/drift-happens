"""Typed protocol metadata for comparable experiment presets."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from drift_happens.configs.base import BaseConfig

JsonDict = dict[str, JsonValue]
JobGranularity = Literal["seed_matrix"]


class DatasetScopeProtocol(BaseConfig):
    """Dataset scope that every model in a comparison group must share."""

    name: str = Field(min_length=1)
    variant: str = Field(min_length=1)
    input_description: str = Field(min_length=1)
    label_description: str = Field(min_length=1)
    time_column: str = Field(min_length=1)
    params: JsonDict = Field(default_factory=dict)


class SplitProtocol(BaseConfig):
    """Train/validation/test split policy."""

    name: str = Field(min_length=1)
    split_seed: int = 42
    train_size: float = 0.7
    val_size: float = 0.0
    test_size: float = 0.3
    stratify_by: tuple[str, ...] = ()


class TimeSliceProtocol(BaseConfig):
    """Temporal train/eval slice policy."""

    train: Literal["cumulative_from_start"] = "cumulative_from_start"
    eval: str = Field(min_length=1)
    min_time: int | None = None
    max_time: int | None = None
    time_unit: str = Field(min_length=1)


class EvaluationProtocol(BaseConfig):
    """Primary and secondary metric policy."""

    primary_metric: str = Field(min_length=1)
    secondary_metrics: tuple[str, ...] = ()
    task: str = Field(min_length=1)


class SeedProtocol(BaseConfig):
    """Distinguish model seeds from fixed data sampling/splitting seeds."""

    model_seeds: tuple[int, ...] = (0, 1, 2)
    split_seed: int = 42
    sample_seed: int | None = None


class CacheProtocol(BaseConfig):
    """Cache identity policy for a preset."""

    cache_kind: str = Field(min_length=1)
    cache_scope: str = Field(min_length=1)
    producer: str = Field(min_length=1)
    cache_id_fields: tuple[str, ...] = ()


class ModelProtocol(BaseConfig):
    """Model role in a comparison table."""

    identifier: str = Field(min_length=1)
    family: str = Field(min_length=1)
    tier: str | None = None
    paradigm: str | None = None
    trainable_target: int | None = None
    frozen_backbone: bool = False
    input_kind: str = Field(min_length=1)


class ConferenceProtocol(BaseConfig):
    """One model entry in a conference comparison group."""

    comparison_group: str = Field(min_length=1)
    comparison_role: Literal["headline"] = "headline"
    dataset_scope: DatasetScopeProtocol
    split: SplitProtocol
    time_slices: TimeSliceProtocol
    evaluation: EvaluationProtocol
    seeds: SeedProtocol
    cache: CacheProtocol
    model: ModelProtocol

    @model_validator(mode="after")
    def _split_seeds_agree(self) -> ConferenceProtocol:
        if self.split.split_seed != self.seeds.split_seed:
            raise ValueError(
                f"split.split_seed ({self.split.split_seed}) and seeds.split_seed "
                f"({self.seeds.split_seed}) must match"
            )
        return self

    def as_metadata(self) -> JsonDict:
        """Return a JSON-safe payload for ``ExperimentConfig.metadata``."""
        return self.model_dump(mode="json")


class SplitProtocolConfig(BaseConfig):
    """Stable dataset split policy for an experiment family."""

    name: str = ""
    seed: int | None = None
    train_size: float | None = None
    val_size: float | None = None
    test_size: float | None = None


class TimeSliceProtocolConfig(BaseConfig):
    """Stable train/eval time-slice policy for temporal drift matrices."""

    time_col: str = ""
    train_strategy: str = ""
    eval_strategy: str = ""
    min_time: int | None = None


class EvaluationProtocolConfig(BaseConfig):
    """Stable metric and evaluation protocol identity."""

    metric: str | None = None
    protocol: str = ""


class ExperimentProtocolConfig(BaseConfig):
    """Typed protocol metadata used for preset comparability checks."""

    split: SplitProtocolConfig = Field(default_factory=SplitProtocolConfig)
    time_slices: TimeSliceProtocolConfig = Field(
        default_factory=TimeSliceProtocolConfig
    )
    evaluation: EvaluationProtocolConfig = Field(
        default_factory=EvaluationProtocolConfig
    )
    job_granularity: JobGranularity = "seed_matrix"
