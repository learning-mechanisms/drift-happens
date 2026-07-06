"""ArXiv experiment presets."""

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
from drift_happens.dataset.arxiv.scope import (
    ARXIV_MAX_YEAR,
    ARXIV_MIN_YEAR,
    ARXIV_SCOPE_VARIANT,
    ARXIV_TARGET_LABELS,
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
    CONFERENCE_ARXIV_MAX_SEQ_LEN,
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

GROUP = "arxiv"
CONFERENCE_GROUP = "arxiv-conference"


def smoke_minilm_l6_frozen() -> ExperimentConfig:
    # The smoke preset is the conference minilm_l6_frozen experiment under a
    # smoke identity; the conference group tag keeps it on the conference
    # trainer matrix and data path.
    config = _arxiv_conference_frozen_model("minilm_l6_frozen")
    return config.model_copy(
        update={
            "name": f"{GROUP}-smoke-minilm-l6-frozen",
            "tags": ("preset", GROUP, "arxiv", "smoke", CONFERENCE_GROUP),
            "metadata": {},
            "notes": "Conference minilm_l6_frozen run at smoke seeds.",
        }
    )


def _dataset_scope() -> DatasetScopeProtocol:
    return DatasetScopeProtocol(
        name="arxiv",
        variant=ARXIV_SCOPE_VARIANT,
        input_description="title + abstract text from first-version arXiv metadata",
        label_description=f"{len(ARXIV_TARGET_LABELS)} arXiv leaf-category multi-label targets",
        time_column="year",
        params={
            "labels": list(ARXIV_TARGET_LABELS),
            "max_year": ARXIV_MAX_YEAR,
            "min_year": ARXIV_MIN_YEAR,
            "text_col": "title_abstract",
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
        comparison_group="arxiv/main-v2",
        dataset_scope=_dataset_scope(),
        split=SplitProtocol(
            name="stratified_temporal_train_test_val",
            split_seed=42,
            train_size=0.7,
            val_size=0.0,
            test_size=0.3,
            stratify_by=("year",),
        ),
        time_slices=TimeSliceProtocol(
            eval="simple_year_slices",
            min_time=ARXIV_MIN_YEAR,
            max_time=ARXIV_MAX_YEAR,
            time_unit="year",
        ),
        evaluation=EvaluationProtocol(
            primary_metric="auc_macro",
            secondary_metrics=(),
            task="multi_label_classification",
        ),
        seeds=SeedProtocol(model_seeds=BENCHMARK_SEEDS, split_seed=42),
        cache=CacheProtocol(
            cache_kind=cache_kind,
            cache_scope=f"arxiv/{ARXIV_SCOPE_VARIANT}",
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


def _arxiv_conference_sequence_model(model_id: str) -> ExperimentConfig:
    family, tier = TEXT_SCRATCH_FAMILIES[model_id]
    return make_experiment(
        group=CONFERENCE_GROUP,
        name=model_id,
        dataset_name="arxiv",
        dataset_variant=ARXIV_SCOPE_VARIANT,
        trainer_key=model_id,
        trainer_family="text-feature",
        model={
            "architecture": model_id,
            "input_dim": 768,
            "input_kind": "sequence_embedding",
            "output_dim": len(ARXIV_TARGET_LABELS),
            "producer": "roberta-base",
        },
        training=conference_text_training(),
        evaluation_metric="auc_macro",
        evaluation_params={
            "eval_time_slices": "simple_year_slices",
            "secondary_metrics": [],
        },
        preprocessing=preprocessing_with_cache(
            steps=(
                "load_arxiv_top7_leaf_title_abstract",
                "tokenize_text",
                "cache_roberta_last_hidden_state",
                "tensorize_multilabels",
            ),
            kind="sequence_embedding_dataset",
            dataset="arxiv",
            input_version=f"{ARXIV_SCOPE_VARIANT}:v1",
            producer="roberta-base",
            output="last_hidden_state_attention_mask_multilabel_targets",
            params={
                "dataset_variant": ARXIV_SCOPE_VARIANT,
                "dtype": "float16",
                "label_schema": list(ARXIV_TARGET_LABELS),
                "max_length": CONFERENCE_ARXIV_MAX_SEQ_LEN,
                "pooling_strategy": None,
                "text_col": "title_abstract",
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
            "arxiv",
            "text",
            "scratch",
            "sequence-cache",
            family,
            tier,
        ),
        notes="Conference headline arXiv scratch body over frozen RoBERTa sequence embeddings.",
    )


def _arxiv_conference_frozen_model(model_id: str) -> ExperimentConfig:
    producer = FROZEN_TEXT_BACKBONE_PRODUCERS[model_id]
    family = model_id.removesuffix("_frozen")
    return make_experiment(
        group=CONFERENCE_GROUP,
        name=model_id,
        dataset_name="arxiv",
        dataset_variant=ARXIV_SCOPE_VARIANT,
        trainer_key=model_id,
        trainer_family="text-frozen-head",
        model={
            "architecture": model_id,
            "input_kind": "pooled_embedding",
            "output_dim": len(ARXIV_TARGET_LABELS),
            "producer": producer,
            "fine_tune": False,
        },
        training=conference_text_training(),
        evaluation_metric="auc_macro",
        evaluation_params={
            "eval_time_slices": "simple_year_slices",
            "secondary_metrics": [],
        },
        preprocessing=preprocessing_with_cache(
            steps=(
                "load_arxiv_top7_leaf_title_abstract",
                "tokenize_text",
                "cache_frozen_text_pooled_embedding",
                "tensorize_multilabels",
            ),
            kind="pooled_embedding_dataset",
            dataset="arxiv",
            input_version=f"{ARXIV_SCOPE_VARIANT}:v1",
            producer=producer,
            output="pooled_embedding_multilabel_targets",
            params={
                "dataset_variant": ARXIV_SCOPE_VARIANT,
                "dtype": "float16",
                "label_schema": list(ARXIV_TARGET_LABELS),
                "max_length": CONFERENCE_ARXIV_MAX_SEQ_LEN,
                "pooling_strategy": "masked_mean",
                "text_col": "title_abstract",
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
            "arxiv",
            "text",
            "frozen-pretrained",
            "embedding-cache",
            family,
        ),
        notes="Conference headline arXiv frozen text-backbone linear-head model.",
    )


def _sequence_model_factory(model_id: str) -> Callable[[], ExperimentConfig]:
    def factory() -> ExperimentConfig:
        return _arxiv_conference_sequence_model(model_id)

    return factory


def _frozen_model_factory(model_id: str) -> Callable[[], ExperimentConfig]:
    def factory() -> ExperimentConfig:
        return _arxiv_conference_frozen_model(model_id)

    return factory


def presets() -> tuple[PresetEntry, ...]:
    smoke_entries = (
        PresetEntry(
            group=GROUP,
            name="smoke-minilm-l6-frozen",
            factory=smoke_minilm_l6_frozen,
            seeds=SMOKE_SEEDS,
            description="arXiv conference minilm_l6_frozen smoke preset.",
            tags=("smoke", "arxiv", "text", "embedding-cache"),
            comparison_group="arxiv/cumulative-time-slices",
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
            description=f"arXiv conference scratch {model_id} over cached RoBERTa sequence embeddings.",
            tags=(
                "conference",
                "headline",
                "arxiv",
                "text",
                "scratch",
                "sequence-cache",
                TEXT_SCRATCH_FAMILIES[model_id][0],
                TEXT_SCRATCH_FAMILIES[model_id][1],
            ),
            comparison_group="arxiv/main-v2",
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
            description=f"arXiv conference frozen {model_id} linear-head preset.",
            tags=(
                "conference",
                "headline",
                "arxiv",
                "text",
                "frozen-pretrained",
                "embedding-cache",
            ),
            comparison_group="arxiv/main-v2",
            variant_fields=CONFERENCE_VARIANT_FIELDS,
        )
        for model_id in FROZEN_TEXT_BACKBONE_IDS
    )
    return (*smoke_entries, *sequence_entries, *frozen_entries)
