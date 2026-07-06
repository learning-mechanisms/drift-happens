from pathlib import Path
from typing import Any

import pandas as pd
import torch

from drift_happens.configs import ExperimentConfig, RunIdentity
from drift_happens.const import ARTIFACTS_DIR
from drift_happens.dataset.arxiv.load import load_arxiv
from drift_happens.dataset.arxiv.scope import (
    ARXIV_SCOPE_VARIANT,
    ARXIV_TARGET_LABELS,
    label_schema_hash_input,
)
from drift_happens.dataset.cache import (
    ChunkedTensorDataset,
    content_fingerprint,
)
from drift_happens.model.dataset.text.architectures import (
    CONFERENCE_ARXIV_MAX_SEQ_LEN,
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
from drift_happens.pipeline.arxiv.context import ArxivPipelineContext
from drift_happens.pipeline.arxiv.trainers import (
    arxiv_conference_trainer_configs,
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
    df: pd.DataFrame,
    *,
    model_key: str,
    max_seq_len: int,
    cache_dir: Path,
) -> tuple[ChunkedTensorDataset, dict[str, int], torch.Tensor]:
    category_to_idx = {label: index for index, label in enumerate(ARXIV_TARGET_LABELS)}
    labels = torch.zeros((len(df), len(category_to_idx)), dtype=torch.float32)
    for row, subjects in enumerate(df["top_subjects"]):
        for subject in subjects:
            labels[row, category_to_idx[subject]] = 1.0

    plan = conference_text_cache_plan(model_key)

    cache_root = cache_dir / plan.producer.replace("/", "_") / plan.kind
    cache_root.mkdir(parents=True, exist_ok=True)

    texts = df["title_abstract"].astype(str).tolist()
    request = TextBackboneCacheRequest(
        kind=plan.kind,
        cache_id=ARXIV_SCOPE_VARIANT,
        dataset="arxiv",
        dataset_variant=ARXIV_SCOPE_VARIANT,
        input_version=f"{ARXIV_SCOPE_VARIANT}:v1",
        producer=plan.producer,
        producer_revision=resolve_huggingface_revision(plan.producer),
        max_length=max_seq_len,
        text_col="title_abstract",
        output=plan.output,
        pooling_strategy=plan.pooling_strategy,
        content_hash=content_fingerprint(texts, labels),
        label_schema_hash=label_schema_hash_input(),
    )
    manifest = cache_text_backbone_outputs(
        texts=texts,
        labels=labels,
        cache_root=cache_root,
        request=request,
    )
    return (
        ChunkedTensorDataset(cache_root, manifest),
        category_to_idx,
        labels,
    )


def setup(trainer_keys: list[str]) -> ArxivPipelineContext:
    missing = sorted(set(trainer_keys) - set(CONFERENCE_TEXT_MODEL_ARCHITECTURES))
    if missing:
        available = ", ".join(CONFERENCE_TEXT_MODEL_ARCHITECTURES)
        raise KeyError(
            f"unknown conference trainer key(s): {', '.join(missing)}. "
            f"Available keys: {available}"
        )
    # One feature cache is built per invocation, so all keys must share it.
    single_text_cache_plan(trainer_keys)

    logger.info("Loading arXiv top-7 leaf-label scope...")
    df = load_arxiv()
    cache_dir = ARTIFACTS_DIR / "cache" / "arxiv" / ARXIV_SCOPE_VARIANT
    logger.info("Caching frozen backbone embeddings...")
    tensor_dataset, category_to_idx, labels = _conference_text_dataset(
        df,
        model_key=trainer_keys[0],
        max_seq_len=CONFERENCE_ARXIV_MAX_SEQ_LEN,
        cache_dir=cache_dir,
    )

    df = df[["year"]].copy()  # discard other columns to save memory

    # Per-year temporal splits (multilabel, no label stratification): the loss
    # weights below must only see the training split, otherwise the held-out
    # label balance leaks into the loss.
    dataset_splits = create_stratified_temporal_train_test_val_splits(
        df=df, time_col="year", train_size=0.7, val_size=0.0, test_size=0.3, seed=42
    )

    # pos_weight for BCEWithLogitsLoss from training-split label frequencies
    # pos_weight[c] = num_negatives[c] / num_positives[c]
    train_labels = labels[
        torch.as_tensor(dataset_splits.train_df.index.to_numpy(), dtype=torch.long)
    ]
    num_positives = train_labels.sum(dim=0)
    num_negatives = train_labels.shape[0] - num_positives
    pos_weight = torch.where(
        num_positives > 0,
        num_negatives / num_positives,
        torch.ones_like(num_positives),
    )

    trainer_configs = arxiv_conference_trainer_configs(
        category_to_idx=category_to_idx,
        pos_weight=pos_weight,
        print_mode=False,
    )

    logger.info("Creating cumulative time slices for training...")
    train_time_slices = create_cumulative_from_start_time_slices(
        df=df, time_col="year", min_time=2000
    )

    artifacts_dir = ARTIFACTS_DIR / "experiments" / "arxiv" / ARXIV_SCOPE_VARIANT

    logger.info("Setup complete.")
    return ArxivPipelineContext(
        df=df,
        tensor_dataset=tensor_dataset,
        dataset_splits=dataset_splits,
        trainer_keys=trainer_keys,
        train_time_slices=train_time_slices,
        artifacts_dir=artifacts_dir,
        trainer_configs=trainer_configs,
        category_to_idx=category_to_idx,
        pos_weight=pos_weight,
    )


# ------------------------------------- TRAINING ------------------------------------- #


def train_single_model(
    ctx: ArxivPipelineContext,
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
    ctx: ArxivPipelineContext,
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
    # Build trainers only for this key in this process
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
        artifacts_dir=ctx.artifacts_dir,
        resume=resume,
        run_identity=run_identity,
        experiment_config=experiment_config,
        metric_sink=metric_sink,
        model_artifacts_dir=model_artifacts_dir,
    )


def eval(
    trainer_keys: list[str],
    n_workers: int = 4,
    fail_fast: bool = False,
) -> None:
    ctx = setup(trainer_keys=trainer_keys)
    ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
    eval_time_slices = create_simple_time_slices(
        df=ctx.df, time_col="year", min_time=2000
    )

    run_per_model(
        ctx,
        ctx.trainer_keys,
        eval_single_model,
        n_workers,
        extra_args=(eval_time_slices,),
        fail_fast=fail_fast,
    )
