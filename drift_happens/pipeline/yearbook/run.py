from pathlib import Path
from typing import Any

import torch

from drift_happens.configs import ExperimentConfig, RunIdentity
from drift_happens.const import ARTIFACTS_DIR
from drift_happens.dataset.yearbook.const import YB_TENSOR_DATASET_CACHE, YB_UNPACK_DIR
from drift_happens.dataset.yearbook.scope import (
    YEARBOOK_IMAGE_SOURCE,
    YEARBOOK_SCOPE_VARIANT,
)
from drift_happens.dataset.yearbook.transform import (
    convert_to_tensor_dataset,
    load_downscaled_images_into_df,
)
from drift_happens.pipeline._shared.runner import run_per_model
from drift_happens.pipeline.evaluation import eval_models_on_time_slices
from drift_happens.pipeline.image.run import (
    embed_dataset_if_needed,
    reuse_policy_from_config,
)
from drift_happens.pipeline.training import train_models_on_time_slices
from drift_happens.pipeline.yearbook.context import YearbookPipelineContext
from drift_happens.pipeline.yearbook.trainers import (
    build_trainers_from_configs,
    yearbook_conference_trainer_configs,
)
from drift_happens.sample.splits import (
    DatasetTimeSplitConfig,
    create_cumulative_from_start_time_slices,
    create_simple_time_slices,
    create_stratified_temporal_train_test_val_splits,
)
from drift_happens.utils.log import get_logger

logger = get_logger()
# --------------------------------------- SETUP -------------------------------------- #


def setup(trainer_keys: list[str]) -> YearbookPipelineContext:
    trainer_keys = list(dict.fromkeys(trainer_keys))  # drop duplicate keys
    trainer_configs = yearbook_conference_trainer_configs()
    missing = sorted(set(trainer_keys) - set(trainer_configs))
    if missing:
        raise KeyError(
            f"unknown conference trainer key(s): {', '.join(missing)}. "
            f"Available keys: {', '.join(trainer_configs)}"
        )

    logger.info("Loading dataset...")
    df = load_downscaled_images_into_df(YB_UNPACK_DIR / YEARBOOK_IMAGE_SOURCE)

    logger.info("Converting to tensor dataset...")
    tensor_dataset = convert_to_tensor_dataset(df)
    df = df[["year", "gender"]].copy()  # discard other columns to save memory

    dataset_splits = create_stratified_temporal_train_test_val_splits(
        df=df,
        time_col="year",
        label_col="gender",
        train_size=0.7,
        val_size=0.0,
        test_size=0.3,
        seed=42,
    )

    artifacts_dir = ARTIFACTS_DIR / "experiments" / "yearbook" / YEARBOOK_SCOPE_VARIANT

    train_time_slices = create_cumulative_from_start_time_slices(df=df, time_col="year")

    return YearbookPipelineContext(
        df=df,
        tensor_dataset=tensor_dataset,
        dataset_splits=dataset_splits,
        trainer_configs=trainer_configs,
        trainer_keys=trainer_keys,
        train_time_slices=train_time_slices,
        artifacts_dir=artifacts_dir,
    )


# ------------------------------------- TRAINING ------------------------------------- #


def train_single_model(
    ctx: YearbookPipelineContext,
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
    tensor_dataset = embed_dataset_if_needed(
        ctx,
        trainer,
        key,
        dataset_cache_dir=YB_TENSOR_DATASET_CACHE,
        dataset_id="yearbook",
        reuse_policy=reuse_policy_from_config(experiment_config),
    )

    train_models_on_time_slices(
        tensor_dataset=tensor_dataset,
        dataset_splits=ctx.dataset_splits,
        training_time_slices=ctx.train_time_slices,
        trainer_key=key,
        trainer_config=ctx.trainer_configs[key],
        trainer=trainer,
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
    ctx = setup(trainer_keys)
    ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_per_model(
        ctx, ctx.trainer_keys, train_single_model, n_workers, fail_fast=fail_fast
    )


# ------------------------------------ EVALUATION ------------------------------------ #


def eval_single_model(
    ctx: YearbookPipelineContext,
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
    tensor_dataset = embed_dataset_if_needed(
        ctx,
        trainer,
        key,
        dataset_cache_dir=YB_TENSOR_DATASET_CACHE,
        dataset_id="yearbook",
        reuse_policy=reuse_policy_from_config(experiment_config),
    )

    eval_models_on_time_slices(
        tensor_dataset=tensor_dataset,
        dataset_splits=ctx.dataset_splits,
        training_time_slices=ctx.train_time_slices,
        eval_time_slices=eval_time_slices,
        trainer_key=key,
        trainer_config=ctx.trainer_configs[key],
        trainer=trainer,
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
    ctx = setup(trainer_keys)
    ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
    eval_time_slices = create_simple_time_slices(ctx.df, time_col="year")

    run_per_model(
        ctx,
        ctx.trainer_keys,
        eval_single_model,
        n_workers,
        extra_args=(eval_time_slices,),
        fail_fast=fail_fast,
    )
