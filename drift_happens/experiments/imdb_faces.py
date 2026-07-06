"""IMDB faces experiment presets."""

from __future__ import annotations

from drift_happens.configs import ExperimentConfig
from drift_happens.configs.protocol import (
    CacheProtocol,
    ConferenceProtocol,
    ModelProtocol,
    SeedProtocol,
)
from drift_happens.dataset.imdb_faces.scope import (
    IMDB_FACES_IMAGE_SOURCE,
    IMDB_FACES_SCOPE_VARIANT,
    imdb_faces_dataset_scope_protocol,
    imdb_faces_evaluation_protocol,
    imdb_faces_split_protocol,
    imdb_faces_time_slice_protocol,
)
from drift_happens.experiments.common import (
    BENCHMARK_SEEDS,
    CONFERENCE_VARIANT_FIELDS,
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

CONFERENCE_GROUP = "imdb-faces-conference"


def _scratch_tags(row: ImageScratchModel) -> tuple[str, ...]:
    return (
        "conference",
        "headline",
        "imdb-faces",
        "image",
        "scratch",
        row.family,
        row.tier,
    )


def _frozen_tags(row: ImageFrozenModel) -> tuple[str, ...]:
    return (
        "conference",
        "headline",
        "imdb-faces",
        "image",
        "frozen-pretrained",
        "embedding-cache",
        row.family,
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
        comparison_group="imdb-faces/main-v1",
        dataset_scope=imdb_faces_dataset_scope_protocol(),
        split=imdb_faces_split_protocol(),
        time_slices=imdb_faces_time_slice_protocol(),
        evaluation=imdb_faces_evaluation_protocol(),
        seeds=SeedProtocol(model_seeds=BENCHMARK_SEEDS, split_seed=42),
        cache=CacheProtocol(
            cache_kind=cache_kind,
            cache_scope=cache_scope,
            producer=producer,
            cache_id_fields=(
                "dataset_variant",
                "image_source",
                "image_size",
                "input_channels",
                "label_col",
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


def _imdb_conference_scratch(row: ImageScratchModel) -> ExperimentConfig:
    model_id = row.model_id
    family = row.family
    tier = row.tier
    return make_experiment(
        group=CONFERENCE_GROUP,
        name=model_id,
        dataset_name="imdb_faces",
        dataset_variant=IMDB_FACES_SCOPE_VARIANT,
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
            "eval_time_slices": "simple_photo_taken_slices",
            "secondary_metrics": ["balanced_accuracy"],
        },
        preprocessing=preprocessing_with_cache(
            steps=("load_preprocessed_faces", "tensorize_images"),
            kind="tensor_dataset",
            dataset="imdb_faces",
            input_version=f"{IMDB_FACES_IMAGE_SOURCE}:32x32:rgb:v1",
            producer="imdb_faces.convert_to_tensor_dataset",
            output="image_tensor_and_gender_label",
            params={
                "dataset_variant": IMDB_FACES_SCOPE_VARIANT,
                "image_source": IMDB_FACES_IMAGE_SOURCE,
                "image_size": [32, 32],
                "input_channels": 3,
                "label_col": "gender",
                "split_seed": 42,
            },
        ),
        metadata=_conference_protocol(
            model_id=model_id,
            family=family,
            tier=tier,
            cache_kind="tensor_dataset",
            cache_scope="imdb_faces/faces_32x32",
            producer="imdb_faces.convert_to_tensor_dataset",
            input_kind="image_tensor",
        ),
        tags=_scratch_tags(row),
        notes="Conference headline IMDB faces scratch model.",
    )


def _imdb_conference_frozen(row: ImageFrozenModel) -> ExperimentConfig:
    model_id = row.model_id
    producer = row.producer
    family = row.family
    return make_experiment(
        group=CONFERENCE_GROUP,
        name=model_id,
        dataset_name="imdb_faces",
        dataset_variant=IMDB_FACES_SCOPE_VARIANT,
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
            "eval_time_slices": "simple_photo_taken_slices",
            "secondary_metrics": ["balanced_accuracy"],
        },
        preprocessing=preprocessing_with_cache(
            steps=(
                "load_preprocessed_faces",
                "tensorize_images",
                "precompute_backbone_embeddings",
            ),
            kind="embedding_dataset",
            dataset="imdb_faces",
            input_version=f"{IMDB_FACES_IMAGE_SOURCE}:32x32:rgb:v1",
            producer=producer,
            output="pooled_backbone_embedding_and_gender_label",
            params={
                "adapter_version": 1,
                "dataset_variant": IMDB_FACES_SCOPE_VARIANT,
                "dtype": "float32",
                "image_source": IMDB_FACES_IMAGE_SOURCE,
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
            cache_scope="imdb_faces/frozen_backbone_embeddings",
            producer=producer,
            input_kind="pooled_image_embedding",
            frozen_backbone=True,
        ),
        tags=_frozen_tags(row),
        notes="Conference headline IMDB faces frozen-backbone linear-head model.",
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
        seeds=BENCHMARK_SEEDS,
        description=description,
        tags=tags,
        comparison_group="imdb-faces/main-v1",
        variant_fields=CONFERENCE_VARIANT_FIELDS,
    )


def presets() -> tuple[PresetEntry, ...]:
    scratch_entries = tuple(
        _entry_from_factory(
            group=CONFERENCE_GROUP,
            name=row.model_id,
            factory=lambda row=row: _imdb_conference_scratch(row),
            description=f"IMDB faces conference scratch {row.family} {row.tier} preset.",
            tags=_scratch_tags(row),
        )
        for row in IMAGE_SCRATCH_MODELS
    )
    frozen_entries = tuple(
        _entry_from_factory(
            group=CONFERENCE_GROUP,
            name=row.model_id,
            factory=lambda row=row: _imdb_conference_frozen(row),
            description=f"IMDB faces conference frozen {row.family} preset.",
            tags=_frozen_tags(row),
        )
        for row in IMAGE_FROZEN_MODELS
    )
    return (*scratch_entries, *frozen_entries)
