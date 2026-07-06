"""Amazon Reviews 2023 experiment presets."""

from __future__ import annotations

from collections.abc import Callable

from drift_happens.configs import ExperimentConfig
from drift_happens.configs.protocol import (
    CacheProtocol,
    ConferenceProtocol,
    DatasetScopeProtocol,
    EvaluationProtocol,
    ModelProtocol,
    SeedProtocol,
    SplitProtocol,
    TimeSliceProtocol,
)
from drift_happens.dataset.amazon_reviews_23.scope import (
    AMAZON_REVIEWS_23_MAX_HALF_YEAR,
    AMAZON_REVIEWS_23_MIN_HALF_YEAR,
    AMAZON_REVIEWS_23_SAMPLE_SIZE,
    AMAZON_REVIEWS_23_SCOPE_VARIANT,
)
from drift_happens.experiments.common import (
    BENCHMARK_SEEDS,
    CONFERENCE_VARIANT_FIELDS,
    SMOKE_SEEDS,
    TIER_TARGETS,
    make_experiment,
    preprocessing_with_cache,
)
from drift_happens.experiments.types import PresetEntry
from drift_happens.model.dataset.text.architectures import (
    CONFERENCE_AMAZON_REVIEWS_MAX_SEQ_LEN,
    CONFERENCE_SEQUENCE_TEXT_ARCHITECTURES,
    TEXT_SCRATCH_FAMILIES,
)
from drift_happens.model.text.frozen_backbone import (
    FROZEN_TEXT_BACKBONE_IDS,
    FROZEN_TEXT_BACKBONE_PRODUCERS,
)
from drift_happens.pipeline._shared.conference_defaults import (
    conference_text_training,
)

GROUP = "amazon-reviews-23"
CONFERENCE_GROUP = "amazon-reviews-23-conference"


def smoke_minilm_l6_frozen() -> ExperimentConfig:
    # The smoke preset is the conference minilm_l6_frozen experiment under a
    # smoke identity; the conference group tag keeps it on the conference
    # trainer matrix and data path.
    config = _amazon_conference_frozen_model("minilm_l6_frozen")
    return config.model_copy(
        update={
            "name": f"{GROUP}-smoke-minilm-l6-frozen",
            "tags": ("preset", GROUP, "amazon_reviews_23", "smoke", CONFERENCE_GROUP),
            "metadata": {},
            "notes": "Conference minilm_l6_frozen run at smoke seeds.",
        }
    )


def _dataset_scope() -> DatasetScopeProtocol:
    return DatasetScopeProtocol(
        name="amazon_reviews_23",
        variant=AMAZON_REVIEWS_23_SCOPE_VARIANT,
        input_description="Amazon Reviews 2023 review text only",
        label_description="1-5 integer rating as scalar regression target",
        time_column="half_year",
        params={
            "max_half_year": AMAZON_REVIEWS_23_MAX_HALF_YEAR,
            "min_half_year": AMAZON_REVIEWS_23_MIN_HALF_YEAR,
            "sample_seed": 42,
            "sample_size": AMAZON_REVIEWS_23_SAMPLE_SIZE,
            "text_col": "text",
        },
    )


def _conference_protocol(
    *,
    model_id: str,
    family: str,
    cache_kind: str,
    producer: str,
    output: str,
    input_kind: str,
    tier: str | None = None,
    paradigm: str | None = None,
    frozen_backbone: bool = False,
) -> dict:
    return ConferenceProtocol(
        comparison_group="amazon-reviews-23/main-v2",
        dataset_scope=_dataset_scope(),
        split=SplitProtocol(
            name="stratified_temporal_train_test_val",
            split_seed=42,
            train_size=0.7,
            val_size=0.0,
            test_size=0.3,
            stratify_by=("half_year",),
        ),
        time_slices=TimeSliceProtocol(
            eval="simple_half_year_slices",
            min_time=AMAZON_REVIEWS_23_MIN_HALF_YEAR,
            max_time=AMAZON_REVIEWS_23_MAX_HALF_YEAR,
            time_unit="half_year",
        ),
        evaluation=EvaluationProtocol(
            primary_metric="balanced_mse",
            secondary_metrics=("rmse", "mae", "mse"),
            task="rating_regression",
        ),
        seeds=SeedProtocol(model_seeds=BENCHMARK_SEEDS, split_seed=42, sample_seed=42),
        cache=CacheProtocol(
            cache_kind=cache_kind,
            cache_scope=f"amazon_reviews_23/{AMAZON_REVIEWS_23_SCOPE_VARIANT}",
            producer=producer,
            cache_id_fields=(
                "dataset_variant",
                "content_hash",
                "label_schema_hash",
                "producer",
                "producer_revision",
                "output",
                "max_length",
                "text_col",
                "pooling_strategy",
                "dtype",
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


def _amazon_conference_sequence_model(model_id: str) -> ExperimentConfig:
    family, tier = TEXT_SCRATCH_FAMILIES[model_id]
    return make_experiment(
        group=CONFERENCE_GROUP,
        name=model_id,
        dataset_name="amazon_reviews_23",
        dataset_variant=AMAZON_REVIEWS_23_SCOPE_VARIANT,
        trainer_key=model_id,
        trainer_family="text-feature-regression",
        model={
            "architecture": model_id,
            "input_dim": 768,
            "input_kind": "sequence_embedding",
            "output_dim": 1,
            "producer": "roberta-base",
        },
        training={
            **conference_text_training(),
            "loss": "weighted_mse",
            "rating_weighting": "inverse_frequency",
        },
        evaluation_metric="balanced_mse",
        evaluation_params={
            "eval_time_slices": "simple_half_year_slices",
            "secondary_metrics": ["rmse", "mae", "mse"],
        },
        preprocessing=preprocessing_with_cache(
            steps=(
                "load_amazon_reviews_23_sample300k",
                "tokenize_text",
                "cache_roberta_last_hidden_state",
                "tensorize_ratings",
            ),
            kind="sequence_embedding_dataset",
            dataset="amazon_reviews_23",
            input_version=f"{AMAZON_REVIEWS_23_SCOPE_VARIANT}:v1",
            producer="roberta-base",
            output="last_hidden_state_attention_mask_rating_targets",
            params={
                "dataset_variant": AMAZON_REVIEWS_23_SCOPE_VARIANT,
                "dtype": "float16",
                "max_length": CONFERENCE_AMAZON_REVIEWS_MAX_SEQ_LEN,
                "pooling_strategy": None,
                "sample_seed": 42,
                "sample_size": AMAZON_REVIEWS_23_SAMPLE_SIZE,
                "text_col": "text",
            },
        ),
        metadata=_conference_protocol(
            model_id=model_id,
            family=family,
            tier=tier,
            cache_kind="sequence_embedding_dataset",
            producer="roberta-base",
            output="last_hidden_state",
            input_kind="sequence_embedding",
        ),
        tags=(
            "conference",
            "headline",
            "amazon-reviews-23",
            "text",
            "scratch",
            "sequence-cache",
            family,
            tier,
        ),
        notes="Conference headline Amazon scratch body over frozen RoBERTa sequence embeddings.",
    )


def _amazon_conference_frozen_model(model_id: str) -> ExperimentConfig:
    producer = FROZEN_TEXT_BACKBONE_PRODUCERS[model_id]
    family = model_id.removesuffix("_frozen")
    return make_experiment(
        group=CONFERENCE_GROUP,
        name=model_id,
        dataset_name="amazon_reviews_23",
        dataset_variant=AMAZON_REVIEWS_23_SCOPE_VARIANT,
        trainer_key=model_id,
        trainer_family="text-frozen-head-regression",
        model={
            "architecture": model_id,
            "fine_tune": False,
            "input_kind": "pooled_embedding",
            "output_dim": 1,
            "producer": producer,
        },
        training={
            **conference_text_training(),
            "loss": "weighted_mse",
            "rating_weighting": "inverse_frequency",
        },
        evaluation_metric="balanced_mse",
        evaluation_params={
            "eval_time_slices": "simple_half_year_slices",
            "secondary_metrics": ["rmse", "mae", "mse"],
        },
        preprocessing=preprocessing_with_cache(
            steps=(
                "load_amazon_reviews_23_sample300k",
                "tokenize_text",
                "cache_frozen_text_pooled_embedding",
                "tensorize_ratings",
            ),
            kind="pooled_embedding_dataset",
            dataset="amazon_reviews_23",
            input_version=f"{AMAZON_REVIEWS_23_SCOPE_VARIANT}:v1",
            producer=producer,
            output="pooled_embedding_rating_targets",
            params={
                "dataset_variant": AMAZON_REVIEWS_23_SCOPE_VARIANT,
                "dtype": "float16",
                "max_length": CONFERENCE_AMAZON_REVIEWS_MAX_SEQ_LEN,
                "pooling_strategy": "masked_mean",
                "sample_seed": 42,
                "sample_size": AMAZON_REVIEWS_23_SAMPLE_SIZE,
                "text_col": "text",
            },
        ),
        metadata=_conference_protocol(
            model_id=model_id,
            family=family,
            paradigm="frozen_pretrained",
            cache_kind="pooled_embedding_dataset",
            producer=producer,
            output="pooled_embedding",
            input_kind="pooled_embedding",
            frozen_backbone=True,
        ),
        tags=(
            "conference",
            "headline",
            "amazon-reviews-23",
            "text",
            "frozen-pretrained",
            "embedding-cache",
            family,
        ),
        notes="Conference headline Amazon frozen text-backbone linear-head model.",
    )


def _sequence_model_factory(model_id: str) -> Callable[[], ExperimentConfig]:
    def factory() -> ExperimentConfig:
        return _amazon_conference_sequence_model(model_id)

    return factory


def _frozen_model_factory(model_id: str) -> Callable[[], ExperimentConfig]:
    def factory() -> ExperimentConfig:
        return _amazon_conference_frozen_model(model_id)

    return factory


def presets() -> tuple[PresetEntry, ...]:
    smoke_entries = (
        PresetEntry(
            group=GROUP,
            name="smoke-minilm-l6-frozen",
            factory=smoke_minilm_l6_frozen,
            seeds=SMOKE_SEEDS,
            description="Amazon Reviews conference minilm_l6_frozen smoke preset.",
            tags=("smoke", "amazon-reviews-23", "text", "embedding-cache"),
            comparison_group="amazon-reviews-23/cumulative-time-slices",
            comparison_role="headline",
            variant_fields=(
                "name",
                "tags",
                "notes",
                "trainer.key",
                "trainer.family",
                "trainer.model",
                "trainer.training",
                "preprocessing.cache.producer",
            ),
        ),
    )
    sequence_entries = tuple(
        PresetEntry(
            group=CONFERENCE_GROUP,
            name=model_id,
            factory=_sequence_model_factory(model_id),
            seeds=BENCHMARK_SEEDS,
            description=f"Amazon Reviews 2023 conference scratch {model_id} over cached RoBERTa sequence embeddings.",
            tags=(
                "conference",
                "headline",
                "amazon-reviews-23",
                "text",
                "scratch",
                "sequence-cache",
                TEXT_SCRATCH_FAMILIES[model_id][0],
                TEXT_SCRATCH_FAMILIES[model_id][1],
            ),
            comparison_group="amazon-reviews-23/main-v2",
            variant_fields=CONFERENCE_VARIANT_FIELDS,
        )
        for model_id in CONFERENCE_SEQUENCE_TEXT_ARCHITECTURES
    )
    frozen_entries = tuple(
        PresetEntry(
            group=CONFERENCE_GROUP,
            name=model_id,
            factory=_frozen_model_factory(model_id),
            seeds=BENCHMARK_SEEDS,
            description=f"Amazon Reviews 2023 conference frozen {model_id} linear-head preset.",
            tags=(
                "conference",
                "headline",
                "amazon-reviews-23",
                "text",
                "frozen-pretrained",
                "embedding-cache",
            ),
            comparison_group="amazon-reviews-23/main-v2",
            variant_fields=CONFERENCE_VARIANT_FIELDS,
        )
        for model_id in FROZEN_TEXT_BACKBONE_IDS
        if model_id != "minilm_l6_frozen"
    )
    return (*smoke_entries, *sequence_entries, *frozen_entries)
