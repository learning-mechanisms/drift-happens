"""Shared builders for Python-defined experiment presets."""

from __future__ import annotations

from pydantic import JsonValue

from drift_happens.configs import (
    CacheArtifactKind,
    CacheSpec,
    ConferenceProtocol,
    DatasetConfig,
    EvaluationConfig,
    EvaluationProtocolConfig,
    ExperimentConfig,
    ExperimentProtocolConfig,
    LoggingConfig,
    PreprocessingConfig,
    RuntimeConfig,
    SplitProtocolConfig,
    TimeSliceProtocolConfig,
    TrainerConfig,
)

BENCHMARK_SEEDS: tuple[int, ...] = (0, 1, 2)
SMOKE_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

CONFERENCE_VARIANT_FIELDS: tuple[str, ...] = (
    "name",
    "notes",
    "tags",
    "metadata",
    "preprocessing.cache",
    "preprocessing.steps",
    "trainer.family",
    "trainer.key",
    "trainer.model",
    "trainer.training",
)

TIER_TARGETS: dict[str, int] = {"small": 100_000, "medium": 500_000, "large": 2_000_000}

JsonDict = dict[str, JsonValue]


def make_experiment(
    *,
    group: str,
    name: str,
    dataset_name: str,
    dataset_variant: str,
    trainer_key: str,
    trainer_family: str,
    model: JsonDict,
    training: JsonDict,
    evaluation_metric: str,
    evaluation_params: JsonDict,
    preprocessing: PreprocessingConfig,
    protocol: ExperimentProtocolConfig | None = None,
    metadata: JsonDict | None = None,
    tags: tuple[str, ...],
    notes: str = "",
) -> ExperimentConfig:
    """Build the common one-run config shape used by materialized presets."""
    return ExperimentConfig(
        name=f"{group}-{name}",
        seed=0,
        task="train_eval",
        dataset=DatasetConfig(name=dataset_name, variant=dataset_variant),
        trainer=TrainerConfig(
            key=trainer_key,
            family=trainer_family,
            model=model,
            training=training,
        ),
        evaluation=EvaluationConfig(
            metric=evaluation_metric,
            params=evaluation_params,
        ),
        preprocessing=preprocessing,
        protocol=(
            protocol
            or _protocol_from_conference_metadata(metadata)
            or ExperimentProtocolConfig()
        ),
        runtime=RuntimeConfig(device="auto"),
        logging=LoggingConfig(),
        tags=("preset", group, dataset_name, *tags),
        notes=notes,
        metadata=metadata or {},
    )


def preprocessing_with_cache(
    *,
    steps: tuple[str, ...],
    kind: CacheArtifactKind,
    dataset: str,
    input_version: str,
    producer: str,
    output: str,
    params: JsonDict,
) -> PreprocessingConfig:
    """Describe reusable preprocessing outputs without resolving local paths."""
    return PreprocessingConfig(
        steps=steps,
        cache=CacheSpec(
            kind=kind,
            dataset=dataset,
            input_version=input_version,
            producer=producer,
            output=output,
            params=params,
        ),
    )


def _protocol_from_conference_metadata(
    metadata: JsonDict | None,
) -> ExperimentProtocolConfig | None:
    if metadata is None or "dataset_scope" not in metadata:
        return None

    # Validate the metadata back through the typed protocol (the inverse of
    # ConferenceProtocol.as_metadata) so a schema change fails loudly.
    protocol = ConferenceProtocol.model_validate(metadata)

    # The evaluation protocol identity equals the time-slice eval strategy by design.
    eval_strategy = protocol.time_slices.eval
    return ExperimentProtocolConfig(
        split=SplitProtocolConfig(
            name=protocol.split.name,
            seed=protocol.split.split_seed,
            train_size=protocol.split.train_size,
            val_size=protocol.split.val_size,
            test_size=protocol.split.test_size,
        ),
        time_slices=TimeSliceProtocolConfig(
            time_col=protocol.dataset_scope.time_column,
            train_strategy=protocol.time_slices.train,
            eval_strategy=eval_strategy,
            min_time=protocol.time_slices.min_time,
        ),
        evaluation=EvaluationProtocolConfig(
            metric=protocol.evaluation.primary_metric,
            protocol=eval_strategy,
        ),
    )
