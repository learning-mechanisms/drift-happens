from pathlib import Path
from typing import Any

import polars as pl
import torch

from drift_happens.configs import ExperimentConfig, RunIdentity
from drift_happens.const import ARTIFACTS_DIR
from drift_happens.dataset.imdb_faces.const import IMDB_TENSOR_DATASET_CACHE
from drift_happens.dataset.imdb_faces.load import load_preprocessed_df
from drift_happens.dataset.imdb_faces.scope import IMDB_FACES_SCOPE_VARIANT
from drift_happens.dataset.imdb_faces.transform import (
    convert_to_tensor_dataset,
)
from drift_happens.pipeline._shared.runner import run_per_model
from drift_happens.pipeline.evaluation import eval_models_on_time_slices
from drift_happens.pipeline.image.run import (
    embed_dataset_if_needed,
    reuse_policy_from_config,
)
from drift_happens.pipeline.imdb_faces.context import (
    ImdbPipelineContext,
)
from drift_happens.pipeline.imdb_faces.trainers import (
    build_trainers_from_configs,
    imdb_faces_conference_trainer_configs,
)
from drift_happens.pipeline.training import train_models_on_time_slices
from drift_happens.sample.splits import (
    DatasetTimeSplitConfig,
    create_cumulative_from_start_time_slices,
    create_instance_based_train_val_test_split,
    create_simple_time_slices,
)

# --------------------------------------- SETUP -------------------------------------- #


def setup(trainer_keys: list[str]) -> ImdbPipelineContext:
    trainer_configs = imdb_faces_conference_trainer_configs()
    missing = sorted(set(trainer_keys) - set(trainer_configs))
    if missing:
        raise KeyError(
            f"unknown conference trainer key(s): {', '.join(missing)}. "
            f"Available keys: {', '.join(trainer_configs)}"
        )

    df_polars = load_preprocessed_df().filter(pl.col("gender").is_not_null())
    tensor_dataset_gender = convert_to_tensor_dataset(df_polars)
    df_gender = df_polars[["photo_taken", "gender", "celeb_id"]].to_pandas()

    dataset_splits = create_instance_based_train_val_test_split(
        df=df_gender,
        instance_col="celeb_id",
        train_size=0.7,
        val_size=0.0,
        test_size=0.3,
        seed=42,
    )

    train_time_slices = create_cumulative_from_start_time_slices(
        df=df_gender, time_col="photo_taken"
    )

    artifacts_dir = (
        ARTIFACTS_DIR / "experiments" / "imdb_faces" / IMDB_FACES_SCOPE_VARIANT
    )

    return ImdbPipelineContext(
        df=df_gender,
        tensor_dataset=tensor_dataset_gender,
        dataset_splits=dataset_splits,
        trainer_configs=trainer_configs,
        trainer_keys=trainer_keys,
        train_time_slices=train_time_slices,
        artifacts_dir=artifacts_dir,
    )


# ------------------------------------- TRAINING ------------------------------------- #


def _prepare_trainer_and_dataset(
    ctx: ImdbPipelineContext,
    key: str,
    device: str | torch.device | None,
    experiment_config: ExperimentConfig | None,
):
    trainer = build_trainers_from_configs(
        {key: ctx.trainer_configs[key]}, device=device
    )[key]
    tensor_dataset = embed_dataset_if_needed(
        ctx,
        trainer,
        key,
        dataset_cache_dir=IMDB_TENSOR_DATASET_CACHE,
        dataset_id="imdb_faces",
        reuse_policy=reuse_policy_from_config(experiment_config),
    )
    return trainer, tensor_dataset


def train_single_model(
    ctx: ImdbPipelineContext,
    key: str,
    *,
    resume: bool = True,
    run_identity: RunIdentity | None = None,
    experiment_config: ExperimentConfig | None = None,
    metric_sink: Any | None = None,
    device: str | torch.device | None = None,
) -> None:
    trainer, tensor_dataset = _prepare_trainer_and_dataset(
        ctx, key, device, experiment_config
    )

    train_models_on_time_slices(
        tensor_dataset=tensor_dataset,
        dataset_splits=ctx.dataset_splits,
        training_time_slices=ctx.train_time_slices,
        trainer_key=key,
        trainer_config=ctx.trainer_configs[key],
        trainer=trainer,
        time_col="photo_taken",
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
    ctx: ImdbPipelineContext,
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
    trainer, tensor_dataset = _prepare_trainer_and_dataset(
        ctx, key, device, experiment_config
    )
    eval_models_on_time_slices(
        tensor_dataset=tensor_dataset,
        dataset_splits=ctx.dataset_splits,
        training_time_slices=ctx.train_time_slices,
        eval_time_slices=eval_time_slices,
        trainer_key=key,
        trainer_config=ctx.trainer_configs[key],
        trainer=trainer,
        time_col="photo_taken",
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
    eval_time_slices = create_simple_time_slices(ctx.df, time_col="photo_taken")
    run_per_model(
        ctx,
        ctx.trainer_keys,
        eval_single_model,
        n_workers,
        extra_args=(eval_time_slices,),
        fail_fast=fail_fast,
    )
