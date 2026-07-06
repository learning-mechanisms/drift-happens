"""Yearbook experiment presets."""

from __future__ import annotations

from drift_happens.configs import ExperimentConfig
from drift_happens.configs.protocol import (
    CacheProtocol,
    ConferenceProtocol,
    ModelProtocol,
    SeedProtocol,
)
from drift_happens.dataset.yearbook.scope import (
    YEARBOOK_IMAGE_SOURCE,
    YEARBOOK_SCOPE_VARIANT,
    yearbook_dataset_scope_protocol,
    yearbook_evaluation_protocol,
    yearbook_split_protocol,
    yearbook_time_slice_protocol,
)
from drift_happens.experiments.common import (
    CONFERENCE_VARIANT_FIELDS,
    SMOKE_SEEDS,
    TIER_TARGETS,
    make_experiment,
    preprocessing_with_cache,
)
from drift_happens.experiments.image_models import (
    IMAGE_FROZEN_MODELS,
    IMAGE_SCRATCH_MODELS,
    ImageFrozenModel,
    ImageScratchModel,
)
from drift_happens.experiments.types import PresetEntry
from drift_happens.pipeline._shared.conference_defaults import (
    conference_image_training,
)

GROUP = "yearbook"
CONFERENCE_GROUP = "yearbook-conference"
YEARBOOK_BENCHMARK_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def smoke_mlp_s() -> ExperimentConfig:
    # The smoke preset is the conference mlp_s experiment under a smoke
    # identity; the conference group tag keeps it on the conference trainer
    # matrix.
    row = next(row for row in IMAGE_SCRATCH_MODELS if row.model_id == "mlp_s")
    config = _yearbook_conference_scratch(row)
    return config.model_copy(
        update={
            "name": f"{GROUP}-smoke-mlp-s",
            "tags": ("preset", GROUP, "yearbook", "smoke", CONFERENCE_GROUP),
            "metadata": {},
            "notes": "Conference mlp_s run at smoke seeds.",
        }
    )


def _conference_protocol(
    *,
    model_id: str,
    family: str,
    cache_kind: str,
    cache_scope: str,
    producer: str,
    input_kind: str,
    tier: str | None = None,
    paradigm: str | None = None,
    frozen_backbone: bool = False,
) -> dict:
    return ConferenceProtocol(
        comparison_group="yearbook/main-v2",
        dataset_scope=yearbook_dataset_scope_protocol(),
        split=yearbook_split_protocol(),
        time_slices=yearbook_time_slice_protocol(),
        evaluation=yearbook_evaluation_protocol(),
        seeds=SeedProtocol(model_seeds=YEARBOOK_BENCHMARK_SEEDS, split_seed=42),
        cache=CacheProtocol(
            cache_kind=cache_kind,
            cache_scope=cache_scope,
            producer=producer,
            cache_id_fields=(
                "dataset_variant",
                "image_source",
                "image_size",
                "input_channels",
                "label_mapping",
                "producer",
                "output",
            ),
        ),
        model=ModelProtocol(
            identifier=model_id,
            family=family,
            tier=tier,
            paradigm=paradigm,
            trainable_target=TIER_TARGETS.get(tier or ""),
            frozen_backbone=frozen_backbone,
            input_kind=input_kind,
        ),
    ).as_metadata()


def _yearbook_conference_scratch(row: ImageScratchModel) -> ExperimentConfig:
    model_id = row.model_id
    family = row.family
    tier = row.tier
    return make_experiment(
        group=CONFERENCE_GROUP,
        name=model_id,
        dataset_name="yearbook",
        dataset_variant=YEARBOOK_SCOPE_VARIANT,
        trainer_key=model_id,
        trainer_family=f"image-{family}",
        model={
            "architecture": family,
            "preset": model_id,
            "input_channels": 3,
            "image_size": [32, 32],
        },
        training=conference_image_training(frozen=False),
        evaluation_metric="accuracy",
        evaluation_params={
            "eval_time_slices": "simple_year_slices",
            "secondary_metrics": ["balanced_accuracy"],
        },
        preprocessing=preprocessing_with_cache(
            steps=("load_downscaled_faces", "tensorize_images"),
            kind="tensor_dataset",
            dataset="yearbook",
            input_version=f"{YEARBOOK_IMAGE_SOURCE}:32x32:rgb:v2",
            producer="yearbook.convert_to_tensor_dataset",
            output="image_tensor_and_gender_label",
            params={
                "dataset_variant": YEARBOOK_SCOPE_VARIANT,
                "image_source": YEARBOOK_IMAGE_SOURCE,
                "image_size": [32, 32],
                "input_channels": 3,
                "label_mapping": {"M": 1},
                "split_seed": 42,
            },
        ),
        metadata=_conference_protocol(
            model_id=model_id,
            family=family,
            tier=tier,
            cache_kind="tensor_dataset",
            cache_scope="yearbook/faces_32x32",
            producer="yearbook.convert_to_tensor_dataset",
            input_kind="image_tensor",
        ),
        tags=("conference", "headline", "yearbook", "image", "scratch", family, tier),
        notes="Conference headline Yearbook scratch model.",
    )


def _yearbook_conference_frozen(row: ImageFrozenModel) -> ExperimentConfig:
    model_id = row.model_id
    producer = row.producer
    family = row.family
    return make_experiment(
        group=CONFERENCE_GROUP,
        name=model_id,
        dataset_name="yearbook",
        dataset_variant=YEARBOOK_SCOPE_VARIANT,
        trainer_key=model_id,
        trainer_family="image-transfer",
        model={
            "architecture": family,
            "preset": model_id,
            "fine_tune": False,
            "needs_backend_fw_pass": False,
        },
        training=conference_image_training(frozen=True),
        evaluation_metric="accuracy",
        evaluation_params={
            "eval_time_slices": "simple_year_slices",
            "secondary_metrics": ["balanced_accuracy"],
        },
        preprocessing=preprocessing_with_cache(
            steps=(
                "load_downscaled_faces",
                "tensorize_images",
                "precompute_backbone_embeddings",
            ),
            kind="embedding_dataset",
            dataset="yearbook",
            input_version=f"{YEARBOOK_IMAGE_SOURCE}:32x32:rgb:v2",
            producer=producer,
            output="pooled_backbone_embedding_and_gender_label",
            params={
                "adapter_version": 1,
                "dataset_variant": YEARBOOK_SCOPE_VARIANT,
                "dtype": "float32",
                "image_source": YEARBOOK_IMAGE_SOURCE,
                "image_size": [32, 32],
                "input_channels": 3,
                "normalization": "producer_default",
                "pooling_strategy": "producer_default",
                "split_seed": 42,
            },
        ),
        metadata=_conference_protocol(
            model_id=model_id,
            family=family,
            paradigm="frozen_pretrained",
            cache_kind="embedding_dataset",
            cache_scope="yearbook/frozen_backbone_embeddings",
            producer=producer,
            input_kind="pooled_image_embedding",
            frozen_backbone=True,
        ),
        tags=(
            "conference",
            "headline",
            "yearbook",
            "image",
            "frozen-pretrained",
            "embedding-cache",
            family,
        ),
        notes="Conference headline Yearbook frozen-backbone linear-head model.",
    )


def _entry_from_factory(
    *,
    group: str,
    name: str,
    factory,
    description: str,
    tags: tuple[str, ...],
) -> PresetEntry:
    return PresetEntry(
        group=group,
        name=name,
        factory=factory,
        seeds=YEARBOOK_BENCHMARK_SEEDS,
        description=description,
        tags=tags,
        comparison_group="yearbook/main-v2",
        variant_fields=CONFERENCE_VARIANT_FIELDS,
    )


def presets() -> tuple[PresetEntry, ...]:
    smoke_entries = (
        PresetEntry(
            group=GROUP,
            name="smoke-mlp-s",
            factory=smoke_mlp_s,
            seeds=SMOKE_SEEDS,
            description="Yearbook conference mlp_s smoke preset.",
            tags=("smoke", "yearbook", "image"),
            comparison_group="yearbook/cumulative-time-slices",
            comparison_role="headline",
            variant_fields=(
                "name",
                "tags",
                "notes",
                "trainer.key",
                "trainer.family",
                "trainer.model",
                "trainer.training",
            ),
        ),
    )
    scratch_entries = tuple(
        _entry_from_factory(
            group=CONFERENCE_GROUP,
            name=row.model_id,
            factory=lambda row=row: _yearbook_conference_scratch(row),
            description=f"Yearbook conference scratch {row.family} {row.tier} preset.",
            tags=(
                "conference",
                "headline",
                "yearbook",
                "image",
                "scratch",
                row.family,
                row.tier,
            ),
        )
        for row in IMAGE_SCRATCH_MODELS
    )
    frozen_entries = tuple(
        _entry_from_factory(
            group=CONFERENCE_GROUP,
            name=row.model_id,
            factory=lambda row=row: _yearbook_conference_frozen(row),
            description=f"Yearbook conference frozen {row.family} preset.",
            tags=(
                "conference",
                "headline",
                "yearbook",
                "image",
                "frozen-pretrained",
                "embedding-cache",
                row.family,
            ),
        )
        for row in IMAGE_FROZEN_MODELS
    )
    return (*smoke_entries, *scratch_entries, *frozen_entries)
