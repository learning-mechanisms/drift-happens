import gc
from pathlib import Path
from typing import Any

import polars as pl
import torch

from drift_happens.configs import ExperimentConfig, RunIdentity
from drift_happens.const import ARTIFACTS_DIR
from drift_happens.dataset.amazon_reviews_23.load import load_amazon_reviews_23
from drift_happens.dataset.amazon_reviews_23.scope import (
    AMAZON_REVIEWS_23_MIN_HALF_YEAR,
    AMAZON_REVIEWS_23_SCOPE_VARIANT,
)
from drift_happens.dataset.cache import (
    ChunkedTensorDataset,
    content_fingerprint,
)
from drift_happens.model.dataset.text.architectures import (
    CONFERENCE_AMAZON_REVIEWS_MAX_SEQ_LEN,
    CONFERENCE_TEXT_MODEL_ARCHITECTURES,
)
from drift_happens.model.text.backbone_cache import (
    TextBackboneCacheRequest,
    cache_text_backbone_outputs,
)
from drift_happens.pipeline._shared.runner import run_per_model
from drift_happens.pipeline._shared.text_cache import (
    conference_text_cache_plan,
    single_text_cache_plan,
)
from drift_happens.pipeline.amazon_reviews_23.context import (
    AmazonReviewsPipelineContext,
)
from drift_happens.pipeline.amazon_reviews_23.trainers import (
    amazon_reviews_conference_trainer_configs,
    build_trainers_from_configs,
)
from drift_happens.pipeline.evaluation import eval_models_on_time_slices
from drift_happens.pipeline.training import train_models_on_time_slices
from drift_happens.sample.splits import (
    DatasetTimeSplitConfig,
    create_cumulative_from_start_time_slices,
    create_simple_time_slices,
    create_stratified_temporal_train_test_val_splits,
)
from drift_happens.utils.env import resolve_huggingface_revision
from drift_happens.utils.log import get_logger

logger = get_logger()
# --------------------------------------- SETUP -------------------------------------- #


def _conference_text_dataset(
    df: pl.DataFrame,
    *,
    model_key: str,
    max_seq_len: int,
    cache_dir: Path,
) -> tuple[ChunkedTensorDataset, torch.Tensor]:
    labels = torch.from_numpy(df.get_column("rating").cast(pl.Int64).to_numpy()).long()

    plan = conference_text_cache_plan(model_key)

    cache_root = cache_dir / plan.producer.replace("/", "_") / plan.kind
    cache_root.mkdir(parents=True, exist_ok=True)

    texts = df.get_column("text").fill_null("").cast(pl.Utf8).to_list()
    request = TextBackboneCacheRequest(
        kind=plan.kind,
        cache_id=AMAZON_REVIEWS_23_SCOPE_VARIANT,
        dataset="amazon_reviews_23",
        dataset_variant=AMAZON_REVIEWS_23_SCOPE_VARIANT,
        input_version=f"{AMAZON_REVIEWS_23_SCOPE_VARIANT}:v1",
        producer=plan.producer,
        producer_revision=resolve_huggingface_revision(plan.producer),
        max_length=max_seq_len,
        text_col="text",
        output=plan.output,
        pooling_strategy=plan.pooling_strategy,
        content_hash=content_fingerprint(texts, labels),
        label_schema_hash="\n".join(str(rating) for rating in range(1, 6)),
    )
    manifest = cache_text_backbone_outputs(
        texts=texts,
        labels=labels,
        cache_root=cache_root,
        request=request,
    )
    return (
        ChunkedTensorDataset(cache_root, manifest),
        labels,
    )


def setup(trainer_keys: list[str]) -> AmazonReviewsPipelineContext:
    # Duplicate keys would train the same model twice into one artifacts dir.
    trainer_keys = list(dict.fromkeys(trainer_keys))
    missing = sorted(set(trainer_keys) - set(CONFERENCE_TEXT_MODEL_ARCHITECTURES))
    if missing:
        available = ", ".join(CONFERENCE_TEXT_MODEL_ARCHITECTURES)
        raise KeyError(
            f"unknown conference trainer key(s): {', '.join(missing)}. "
            f"Available keys: {available}"
        )
    # One feature cache is built per invocation, so all keys must share it.
    single_text_cache_plan(trainer_keys)

    logger.info("Loading Amazon Reviews 2023 canonical scope...")
    df = load_amazon_reviews_23()
    df = df.drop("row_id").with_row_index("row_id")
    cache_dir = (
        ARTIFACTS_DIR / "cache" / "amazon_reviews_23" / AMAZON_REVIEWS_23_SCOPE_VARIANT
    )
    logger.info("Caching frozen backbone embeddings...")
    tensor_dataset, _ = _conference_text_dataset(
        df,
        model_key=trainer_keys[0],
        max_seq_len=CONFERENCE_AMAZON_REVIEWS_MAX_SEQ_LEN,
        cache_dir=cache_dir,
    )
    df_pandas = df.select("row_id", "half_year", "rating").to_pandas()
    df_pandas.set_index("row_id", inplace=True)
    del df
    gc.collect()

    # Stratified global temporal splits first: the loss weights below must only
    # see the training split, otherwise the held-out label balance leaks into
    # the loss.
    dataset_splits = create_stratified_temporal_train_test_val_splits(
        df=df_pandas,
        time_col="half_year",
        train_size=0.7,
        val_size=0.0,
        test_size=0.3,
        seed=42,
    )

    # Class weights for WeightedMSELoss from training-split rating frequencies
    rating_counts = (
        dataset_splits.train_df["rating"].value_counts().reindex(range(1, 6))
    )
    if rating_counts.isna().any():
        missing_ratings = rating_counts[rating_counts.isna()].index.tolist()
        raise ValueError(
            f"training split has no rows for rating(s) {missing_ratings}; "
            "cannot compute class weights"
        )
    total_counts = rating_counts.sum()
    class_weights = torch.tensor(
        [total_counts / rating_counts[i] for i in range(1, 6)], dtype=torch.float32
    )

    trainer_configs = amazon_reviews_conference_trainer_configs(
        class_weights=class_weights,
        print_mode=False,
    )

    logger.info("Creating cumulative time slices for training...")
    train_time_slices = create_cumulative_from_start_time_slices(
        df=df_pandas, time_col="half_year", min_time=AMAZON_REVIEWS_23_MIN_HALF_YEAR
    )

    artifacts_dir = (
        ARTIFACTS_DIR
        / "experiments"
        / "amazon_reviews_23"
        / AMAZON_REVIEWS_23_SCOPE_VARIANT
    )

    logger.info("Setup complete.")
    return AmazonReviewsPipelineContext(
        df=df_pandas,
        tensor_dataset=tensor_dataset,
        dataset_splits=dataset_splits,
        trainer_keys=trainer_keys,
        train_time_slices=train_time_slices,
        artifacts_dir=artifacts_dir,
        trainer_configs=trainer_configs,
        class_weights=class_weights,
    )


# ------------------------------------- TRAINING ------------------------------------- #


def train_single_model(
    ctx: AmazonReviewsPipelineContext,
    key: str,
    *,
    resume: bool = True,
    run_identity: RunIdentity | None = None,
    experiment_config: ExperimentConfig | None = None,
    metric_sink: Any | None = None,
    device: str | torch.device | None = None,
) -> None:
    trainer = build_trainers_from_configs(
        {key: ctx.trainer_configs[key]}, device=device
    )[key]

    train_models_on_time_slices(
        tensor_dataset=ctx.tensor_dataset,
        dataset_splits=ctx.dataset_splits,
        training_time_slices=ctx.train_time_slices,
        trainer_key=key,
        trainer_config=ctx.trainer_configs[key],
        trainer=trainer,
        time_col="half_year",
        artifacts_dir=ctx.artifacts_dir,
        resume=resume,
        run_identity=run_identity,
        experiment_config=experiment_config,
        metric_sink=metric_sink,
    )


def train(
    trainer_keys: list[str],
    n_workers: int = 4,
    fail_fast: bool = False,
) -> None:
    ctx = setup(trainer_keys=trainer_keys)
    ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_per_model(
        ctx, ctx.trainer_keys, train_single_model, n_workers, fail_fast=fail_fast
    )


# ------------------------------------ EVALUATION ------------------------------------ #


def eval_single_model(
    ctx: AmazonReviewsPipelineContext,
    key: str,
    eval_time_slices: dict[Any, DatasetTimeSplitConfig],
    *,
    resume: bool = True,
    run_identity: RunIdentity | None = None,
    experiment_config: ExperimentConfig | None = None,
    metric_sink: Any | None = None,
    model_artifacts_dir: Path | None = None,
    device: str | torch.device | None = None,
) -> None:
    trainer = build_trainers_from_configs(
        {key: ctx.trainer_configs[key]}, device=device
    )[key]

    eval_models_on_time_slices(
        tensor_dataset=ctx.tensor_dataset,
        dataset_splits=ctx.dataset_splits,
        training_time_slices=ctx.train_time_slices,
        eval_time_slices=eval_time_slices,
        trainer_key=key,
        trainer_config=ctx.trainer_configs[key],
        trainer=trainer,
        time_col="half_year",
        artifacts_dir=ctx.artifacts_dir,
        resume=resume,
        run_identity=run_identity,
        experiment_config=experiment_config,
        metric_sink=metric_sink,
        model_artifacts_dir=model_artifacts_dir,
    )


def eval(
    trainer_keys: list[str],
    n_workers: int = 8,
    fail_fast: bool = False,
) -> None:
    ctx = setup(trainer_keys=trainer_keys)
    ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
    eval_time_slices = create_simple_time_slices(
        df=ctx.df, time_col="half_year", min_time=AMAZON_REVIEWS_23_MIN_HALF_YEAR
    )

    run_per_model(
        ctx,
        ctx.trainer_keys,
        eval_single_model,
        n_workers,
        extra_args=(eval_time_slices,),
        fail_fast=fail_fast,
    )
