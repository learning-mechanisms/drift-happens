import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, TypeAdapter
from tqdm import tqdm

from drift_happens.configs import ExperimentConfig, RunIdentity
from drift_happens.dataset.cache import ChunkedTensorDataset
from drift_happens.evaluation.metrics import (
    ClassificationMetrics,
    ClassificationMetricsUnion,
    MultiLabelROCAUCTracker,
    RegressiveClassificationMetrics,
)
from drift_happens.model.trainer.pytorch import PytorchTrainer
from drift_happens.pipeline.models import TrainerEvaluationResults
from drift_happens.runtime.metrics import MetricRecord, MetricSink
from drift_happens.runtime.progress import (
    sweep_progress_requested,
    write_sweep_progress_event,
)
from drift_happens.runtime.stages import (
    WorkUnitCompletion,
    read_json_object,
    work_unit_completion_matches,
    write_json_atomic,
)
from drift_happens.sample.splits import (
    DatasetSplit,
    DatasetTimeSplitConfig,
)
from drift_happens.sample.time.filter import (
    filter_dataset_by_time_slice,
    iter_evaluation_sets,
)
from drift_happens.utils.log import get_logger

logger = get_logger()

configs_dict = TypeAdapter(dict[str | int, DatasetTimeSplitConfig])


def eval_models_on_time_slices(
    tensor_dataset: Any,
    dataset_splits: DatasetSplit,
    training_time_slices: dict[Any, DatasetTimeSplitConfig],
    eval_time_slices: dict[Any, DatasetTimeSplitConfig],
    trainer_key: str,
    trainer_config: BaseModel,
    trainer: PytorchTrainer,
    time_col: str = "year",
    *,
    artifacts_dir: Path,
    tqdm_start_pos: int = 1,
    resume: bool = True,
    run_identity: RunIdentity | None = None,
    experiment_config: ExperimentConfig | None = None,
    metric_sink: MetricSink | None = None,
    model_artifacts_dir: Path | None = None,
    save_predictions: bool = False,
) -> None:
    """Evaluate one trained model across all train-slice/eval-slice cells, with per-cell
    resume."""
    logger.info(f"Starting evaluation for model '{trainer_key}'")

    eval_config_dir = artifacts_dir / trainer_key
    eval_config_dir.mkdir(parents=True, exist_ok=True)
    model_config_dir = (model_artifacts_dir or artifacts_dir) / trainer_key
    _write_text_with_parent(
        eval_config_dir / "config.json", trainer_config.model_dump_json()
    )

    # List of model keys that failed at any time slice
    error_keys: list[str] = []
    error_causes: list[BaseException] = []

    # ----------------- Inner loop 1: iterate this model's train slices ---------------- #
    model_slice_iter = training_time_slices.items()
    total_cells = len(training_time_slices) * len(eval_time_slices)
    write_sweep_progress_event(
        "eval_cells_started",
        trainer_key=trainer_key,
        total_cells=total_cells,
    )
    for train_slice_key, train_slice_config in tqdm(
        model_slice_iter,
        desc=f"Models for {trainer_key}",
        unit="slice",
        leave=False,
        position=tqdm_start_pos + 1,
        colour="blue",
        disable=sweep_progress_requested(),
    ):
        write_sweep_progress_event(
            "eval_train_slice_started",
            trainer_key=trainer_key,
            train_slice=str(train_slice_key),
            total_cells=total_cells,
        )
        try:
            # --------------------------- Setup Trainer -------------------------- #
            model_slice_dir = model_config_dir / f"train_slice_{train_slice_key}"
            eval_slice_dir = eval_config_dir / f"train_slice_{train_slice_key}"
            eval_slice_dir.mkdir(parents=True, exist_ok=True)
            trainer.load_model(model_slice_dir / "trained_model.pt")

            # eval_slice_key -> ClassificationMetrics
            model_eval_results = TrainerEvaluationResults()
            eval_configs: dict[Any, DatasetTimeSplitConfig] = {}

            # --------- Inner loop 2: evaluate model on all eval time slices --------- #
            # Build evaluation sets iterator for each training time slice.
            # Future out of distribution train slices will also be evaluated.
            eval_slice_iter = iter_evaluation_sets(
                dataset_split=dataset_splits,
                time_col=time_col,
                split_configs=eval_time_slices,
                training_time_split=train_slice_config,
            )

            # ------------------------- Determine thresholds ------------------------- #
            df_train_slice = filter_dataset_by_time_slice(
                df=dataset_splits.train_df,
                time_col=time_col,
                split_config=train_slice_config,
            )

            if trainer.task_type == "multilabel":
                # The optimal-threshold search needs the cumulative train slice's
                # scores and labels, but only those small per-row vectors — not
                # the heavy embeddings — must be resident, so stream the slice
                # chunk-wise instead of gathering it whole (~1.1 TB at scale).
                train_probs, y_train = _predict_proba_and_labels(
                    trainer, tensor_dataset, df_train_slice.index.tolist()
                )
                thresholds = trainer.find_optimal_threshold(train_probs, y_train)
                # Class count from the training-slice probs; empty eval slices
                # have 1-D probs and cannot provide it themselves.
                multilabel_num_classes = int(train_probs.shape[1])

            # Class count inferred from the softmax width; remembered so empty eval
            # slices reuse the same count instead of producing a 0x0 matrix.
            seen_num_classes: int | None = None

            for eval_slice_key, eval_slice_config, eval_df in tqdm(
                eval_slice_iter,
                desc=f"Time slices for {trainer_key}",
                unit="slice",
                leave=False,
                position=tqdm_start_pos + 2,
                colour="blue",
                disable=sweep_progress_requested(),
            ):
                write_sweep_progress_event(
                    "eval_cell_started",
                    trainer_key=trainer_key,
                    train_slice=str(train_slice_key),
                    eval_slice=str(eval_slice_key),
                    total_cells=total_cells,
                )
                cell_path = _cell_result_path(eval_slice_dir, eval_slice_key)
                cell_completion_path = _cell_completion_path(
                    eval_slice_dir,
                    eval_slice_key,
                )
                if resume and _eval_cell_complete(
                    cell_path,
                    cell_completion_path,
                    trainer_key=trainer_key,
                    train_slice=train_slice_key,
                    eval_slice=eval_slice_key,
                    identity=run_identity,
                ):
                    # The append-only ledger already holds this cell's rows from
                    # the attempt that computed it; logging again would
                    # duplicate them.
                    loaded_metrics = _load_cell_metrics(cell_path)
                    model_eval_results.results[str(eval_slice_key)] = loaded_metrics
                    eval_configs[eval_slice_key] = eval_slice_config
                    # Restore the remembered class count so an empty slice after
                    # the resume point matches the uninterrupted run.
                    if (
                        isinstance(loaded_metrics, ClassificationMetrics)
                        and loaded_metrics.confusion_matrix.shape[0] > 0
                    ):
                        seen_num_classes = int(loaded_metrics.confusion_matrix.shape[0])
                    write_sweep_progress_event(
                        "eval_cell_skipped",
                        trainer_key=trainer_key,
                        train_slice=str(train_slice_key),
                        eval_slice=str(eval_slice_key),
                        total_cells=total_cells,
                    )
                    continue
                # ----------- Predict (streaming the heavy embeddings) ----------- #
                # ``_predict_proba_and_labels`` runs the model chunk-wise so the
                # eval slice's embeddings never all sit in RAM; the returned probs
                # and labels are in row order, numerically equivalent (up to float
                # reassociation) to gathering the slice and calling
                # ``predict_proba`` on it.
                probs, y_eval = _predict_proba_and_labels(
                    trainer, tensor_dataset, eval_df.index.tolist()
                )

                if len(eval_df) != 0:
                    if trainer.task_type == "multilabel":
                        # For every class use the threshold found on training set
                        preds = torch.zeros_like(y_eval)
                        for class_idx in range(probs.shape[1]):
                            thresh = thresholds[class_idx]
                            preds[:, class_idx] = (probs[:, class_idx] >= thresh).long()
                    else:
                        # Derive predictions from the streamed probs rather than a
                        # second forward pass; this equals ``trainer.predict``.
                        preds = _predictions_from_probs(trainer, probs)
                else:
                    preds = torch.tensor([], dtype=torch.long)
                    probs = torch.tensor([], dtype=torch.float32)

                metrics: ClassificationMetricsUnion
                if trainer.task_type == "regression":
                    metrics = RegressiveClassificationMetrics.from_predictions(
                        y_true=np.array(y_eval),
                        predicted=np.array(preds),
                    )

                elif trainer.task_type == "multilabel":
                    metrics = MultiLabelROCAUCTracker.from_predictions(
                        num_classes=multilabel_num_classes,
                        y_true=y_eval,
                        y_pred=preds,
                        y_prob=probs,
                    )
                else:
                    num_classes = _classification_num_classes(probs)
                    if num_classes is None:
                        num_classes = seen_num_classes
                    else:
                        seen_num_classes = num_classes
                    metrics = ClassificationMetrics.from_predictions(
                        y_true=y_eval,
                        y_pred=preds,
                        y_prob=probs,
                        num_classes=num_classes,
                    )

                model_eval_results.results[str(eval_slice_key)] = metrics
                eval_configs[eval_slice_key] = eval_slice_config
                # Raw per-row predictions are only needed for post-hoc analysis
                # (threshold re-tuning, calibration); skip the heavy dump by default.
                if save_predictions:
                    _write_prediction_cell(
                        eval_slice_dir=eval_slice_dir,
                        eval_slice_key=eval_slice_key,
                        trainer_key=trainer_key,
                        trainer_config=trainer_config,
                        seed=None,
                        train_slice_key=train_slice_key,
                        row_ids=eval_df.index.tolist(),
                        y_true=y_eval,
                        y_pred=preds,
                        probabilities=probs,
                        metrics=metrics,
                    )
                _write_text_with_parent(cell_path, metrics.model_dump_json())
                _log_eval_cell(
                    metric_sink,
                    experiment_config,
                    train_slice_key,
                    eval_slice_key,
                    metrics,
                )
                # Write the completion marker last: its presence is the resume
                # gate, so the logged rows must already be durable.
                _write_eval_cell_completion(
                    cell_completion_path,
                    trainer_key=trainer_key,
                    train_slice=train_slice_key,
                    eval_slice=eval_slice_key,
                    identity=run_identity,
                    metrics=metrics,
                )
                write_sweep_progress_event(
                    "eval_cell_finished",
                    trainer_key=trainer_key,
                    train_slice=str(train_slice_key),
                    eval_slice=str(eval_slice_key),
                    total_cells=total_cells,
                )

        except Exception as e:
            write_sweep_progress_event(
                "eval_train_slice_failed",
                trainer_key=trainer_key,
                train_slice=str(train_slice_key),
                total_cells=total_cells,
                error=f"{type(e).__name__}: {e}",
            )
            logger.error(
                f"failure in model '{trainer_key}', train slice '{train_slice_key}'",
                exc_info=e,
            )
            error_keys.append(f"{trainer_key} (train slice {train_slice_key})")
            error_causes.append(e)
            continue

        # ------------------------ Save evaluation results ----------------------- #

        _write_text_with_parent(
            eval_slice_dir / "evaluation_results_on_all_slices.json",
            model_eval_results.model_dump_json(),
        )

        _write_text_with_parent(
            eval_slice_dir / "evaluation_time_slice_configs.json",
            configs_dict.dump_json(eval_configs).decode("utf-8"),
        )
        _write_train_slice_eval_completion(
            eval_slice_dir,
            trainer_key=trainer_key,
            train_slice=train_slice_key,
            identity=run_identity,
        )

    # ------------------------------------ Summary ----------------------------------- #
    if len(error_keys) > 0:
        logger.info("=" * 23)
        logger.info("  FAILED CONFIGURATIONS")
        for k in error_keys:
            logger.info(f" - {k}")
        logger.info(f"\nTotal failed: {len(error_keys)}")
        raise RuntimeError(
            "evaluation failed for "
            + ", ".join(error_keys[:10])
            + ("..." if len(error_keys) > 10 else "")
        ) from error_causes[0]
    else:
        logger.info("\nAll configurations completed successfully!\n")


def _classification_num_classes(probs: torch.Tensor) -> int | None:
    """
    Infer the class count from a softmax probability tensor.

    Returns ``None`` for empty slices (1-D empty tensor), letting the metric constructor
    fall back to its empty-input handling.
    """
    if probs.ndim == 2:
        return int(probs.shape[1])
    return None


def _predictions_from_probs(
    trainer: PytorchTrainer, probs: torch.Tensor
) -> torch.Tensor:
    """
    Predictions for the non-multilabel tasks from already-computed probs.

    Mirrors ``PytorchTrainer.predict`` (regression returns the probs, single-label
    classification takes the argmax) without a second forward pass over the slice.
    """
    if probs.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    if trainer.task_type == "regression":
        return probs
    return probs.argmax(dim=1)


def _predict_proba_and_labels(
    trainer: PytorchTrainer, tensor_dataset: Any, indices: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Probabilities and labels for a slice, streaming the heavy embeddings.

    For a ``ChunkedTensorDataset`` the slice is run through the model one cache chunk at
    a time, so its embeddings never all sit in RAM; the per-chunk probs and labels
    (small) are scattered back into row order. The model runs in eval mode, so the
    result is numerically equivalent to gathering the whole slice and calling
    ``predict_proba`` on it, up to floating-point reassociation from the different batch
    grouping.
    """
    if not isinstance(tensor_dataset, ChunkedTensorDataset):
        # Pooled/materialized caches already fit in RAM; gather as before.
        inputs, labels = _slice_inputs_and_labels(tensor_dataset, indices)
        return trainer.predict_proba(inputs), labels

    probs_by_position: list[tuple[int, torch.Tensor]] = []
    label_rows: list[tuple[int, torch.Tensor]] = []
    for positions, columns in tensor_dataset.iter_chunk_gathers(indices):
        *chunk_inputs, chunk_labels = columns
        inputs = chunk_inputs[0] if len(chunk_inputs) == 1 else tuple(chunk_inputs)
        chunk_probs = trainer.predict_proba(inputs)
        for offset, position in enumerate(positions):
            probs_by_position.append((position, chunk_probs[offset]))
            label_rows.append((position, chunk_labels[offset]))

    if not probs_by_position:
        # Empty slice: hand back shape-correct empties via the gather path.
        _, labels = _slice_inputs_and_labels(tensor_dataset, indices)
        return torch.empty(0), labels

    probs_by_position.sort(key=lambda item: item[0])
    label_rows.sort(key=lambda item: item[0])
    probs = torch.stack([row for _, row in probs_by_position], dim=0)
    labels = torch.stack([row for _, row in label_rows], dim=0)
    return probs, labels


def _slice_inputs_and_labels(
    tensor_dataset: Any, indices: list[int]
) -> tuple[torch.Tensor | tuple[torch.Tensor, ...], torch.Tensor]:
    if isinstance(tensor_dataset, ChunkedTensorDataset):
        tensors = tensor_dataset.gather(indices)
    elif hasattr(tensor_dataset, "tensors"):
        tensors = tuple(tensor[indices] for tensor in tensor_dataset.tensors)
    else:
        raise TypeError(
            "evaluation requires a TensorDataset-like object exposing `.tensors` "
            f"or a ChunkedTensorDataset; got {type(tensor_dataset).__name__}"
        )
    if len(tensors) < 2:
        raise ValueError("evaluation expects at least one input tensor and labels")
    inputs = tuple(tensors[:-1])
    labels = tensors[-1]
    if len(inputs) == 1:
        return inputs[0], labels
    return inputs, labels


def _write_prediction_cell(
    *,
    eval_slice_dir: Path,
    eval_slice_key: Any,
    trainer_key: str,
    trainer_config: BaseModel,
    seed: int | None,
    train_slice_key: Any,
    row_ids: list[Any],
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    probabilities: torch.Tensor,
    metrics: ClassificationMetricsUnion,
) -> None:
    prediction_dir = eval_slice_dir / "predictions" / f"eval_slice_{eval_slice_key}"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    config_json = trainer_config.model_dump_json()
    config_hash = sha256(config_json.encode("utf-8")).hexdigest()
    payload = {
        "config_hash": config_hash,
        "probabilities": probabilities.detach().cpu(),
        "row_id": row_ids,
        "seed": seed,
        "snapshot_sha256": None,
        "trainer_key": trainer_key,
        "train_slice": train_slice_key,
        "eval_slice": eval_slice_key,
        "y_pred": y_pred.detach().cpu(),
        "y_true": y_true.detach().cpu(),
    }
    torch.save(payload, prediction_dir / "predictions.pt")
    _write_text_with_parent(prediction_dir / "metrics.json", metrics.model_dump_json())
    _write_text_with_parent(
        prediction_dir / "completion.json",
        json.dumps(
            {
                "config_hash": config_hash,
                "eval_slice": eval_slice_key,
                "trainer_key": trainer_key,
                "train_slice": train_slice_key,
            },
            sort_keys=True,
        ),
    )


metrics_adapter: TypeAdapter[ClassificationMetricsUnion] = TypeAdapter(
    ClassificationMetricsUnion
)


def _cell_result_path(eval_slice_dir: Path, eval_slice_key: object) -> Path:
    return eval_slice_dir / f"eval_slice={eval_slice_key}.json"


def _cell_completion_path(eval_slice_dir: Path, eval_slice_key: object) -> Path:
    return eval_slice_dir / f"eval_slice={eval_slice_key}.completion.json"


def _write_text_with_parent(path: Path, text: str) -> None:
    """Write text after ensuring the destination directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _eval_cell_complete(
    cell_path: Path,
    completion_path: Path,
    *,
    trainer_key: str,
    train_slice: object,
    eval_slice: object,
    identity: RunIdentity | None,
) -> bool:
    if not cell_path.exists():
        return False
    completion = read_json_object(completion_path)
    return work_unit_completion_matches(
        completion,
        identity=identity,
        trainer_key=trainer_key,
        train_slice=train_slice,
        eval_slice=eval_slice,
    )


def _load_cell_metrics(path: Path) -> ClassificationMetricsUnion:
    return metrics_adapter.validate_json(path.read_text())


def _write_eval_cell_completion(
    path: Path,
    *,
    trainer_key: str,
    train_slice: object,
    eval_slice: object,
    identity: RunIdentity | None,
    metrics: ClassificationMetricsUnion,
) -> None:
    completion = WorkUnitCompletion(
        kind="eval_cell",
        stage="eval",
        exit_status="ok",
        seed=None,
        source_identity=identity.source_identity if identity else None,
        config_hash=identity.config_hash if identity else None,
        snapshot_sha256=identity.snapshot_sha256 if identity else None,
        trainer_key=trainer_key,
        train_slice=str(train_slice),
        eval_slice=str(eval_slice),
        ended_at=datetime.now(UTC),
        metrics=_metric_values(metrics),
    )
    write_json_atomic(path, completion.model_dump(mode="json"))


def _write_train_slice_eval_completion(
    eval_slice_dir: Path,
    *,
    trainer_key: str,
    train_slice: object,
    identity: RunIdentity | None,
) -> None:
    completion = WorkUnitCompletion(
        kind="train_slice",
        stage="eval",
        exit_status="ok",
        seed=None,
        source_identity=identity.source_identity if identity else None,
        config_hash=identity.config_hash if identity else None,
        snapshot_sha256=identity.snapshot_sha256 if identity else None,
        trainer_key=trainer_key,
        train_slice=str(train_slice),
        ended_at=datetime.now(UTC),
    )
    write_json_atomic(
        eval_slice_dir / "completion.json",
        completion.model_dump(mode="json"),
    )


def _log_eval_cell(
    metric_sink: MetricSink | None,
    cfg: ExperimentConfig | None,
    train_slice: object,
    eval_slice: object,
    metrics: ClassificationMetricsUnion,
) -> None:
    if metric_sink is None or cfg is None:
        return
    values = _metric_values(metrics)
    for name, value in values.items():
        metric_sink.log(
            MetricRecord.from_config(
                cfg,
                phase="eval",
                metric=f"eval/{name}",
                value=value,
                train_slice=str(train_slice),
                eval_slice=str(eval_slice),
            )
        )


def _metric_values(metrics: ClassificationMetricsUnion) -> dict[str, float]:
    return metrics.scalar_metrics()
