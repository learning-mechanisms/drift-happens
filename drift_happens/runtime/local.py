"""Local staged experiment runtime."""

from __future__ import annotations

import logging
import random
import shutil
import traceback
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import numpy as np
import torch
from structlog.contextvars import bind_contextvars, clear_contextvars

from drift_happens.configs import ExecutionInfo, ExperimentConfig, RunIdentity
from drift_happens.configs.trainers import validate_registered_trainer_config
from drift_happens.runtime.adapters import run_adapter_stage
from drift_happens.runtime.base import RunExitStatus, RunResult, StageResult, TaskResult
from drift_happens.runtime.locks import stage_lock
from drift_happens.runtime.metrics import (
    CompositeMetricSink,
    JsonlMetricSink,
    MetricRecord,
    MetricSink,
    NoopMetricSink,
)
from drift_happens.runtime.run_store import RunStore, resolve_run_store
from drift_happens.runtime.stage_status import inspect_stage_status
from drift_happens.runtime.stages import (
    RunStage,
    StageCompletion,
    read_json_object,
    stage_completion_matches,
    stage_completion_path,
    stage_dir,
    stage_metadata_path,
    write_json_atomic,
)
from drift_happens.runtime.wandb import WandbMetricSink
from drift_happens.utils.env import resolve_resume_setting
from drift_happens.utils.git import GitState, read_git_state
from drift_happens.utils.lockfile import pixi_lock_sha256
from drift_happens.utils.log import configure_logging, get_logger, shutdown_logging
from drift_happens.utils.snapshot import (
    build_metadata,
    finalise_metadata,
    write_metadata,
)


class _RuntimeState(TypedDict):
    deterministic: bool
    num_threads: int
    cudnn_benchmark: NotRequired[bool]


def worker_main(
    cfg: ExperimentConfig,
    *,
    seed: int | None = None,
    allow_overwrite: bool = False,
    execution: ExecutionInfo | None = None,
    runs_root: Path | None = None,
    source_path: Path | None = None,
    resume: bool | None = None,
) -> RunResult:
    """
    Compatibility entry point for one resolved experiment.

    ``cfg.task='train_eval'`` executes train and eval as separate staged calls in the
    current process. The CLI ``experiment run`` uses subprocesses by default.
    """
    if seed is not None and seed != cfg.seed:
        cfg = cfg.model_copy(update={"seed": seed})
    resume = resolve_resume_setting(resume)

    if cfg.task == "train":
        return _run_result_from_stage(
            run_stage(
                cfg,
                stage="train",
                allow_overwrite=allow_overwrite,
                execution=execution,
                runs_root=runs_root,
                source_path=source_path,
                resume=resume,
            )
        )
    if cfg.task == "eval":
        return _run_result_from_stage(
            run_stage(
                cfg,
                stage="eval",
                allow_overwrite=allow_overwrite,
                execution=execution,
                runs_root=runs_root,
                source_path=source_path,
                resume=resume,
            )
        )

    train = run_stage(
        cfg,
        stage="train",
        allow_overwrite=allow_overwrite,
        execution=execution,
        runs_root=runs_root,
        source_path=source_path,
        resume=resume,
    )
    eval_result = run_stage(
        cfg,
        stage="eval",
        execution=execution,
        runs_root=runs_root,
        source_path=source_path,
        resume=resume,
    )
    return RunResult(
        run_dir=eval_result.run_dir,
        exit_status="ok"
        if train.exit_status == eval_result.exit_status == "ok"
        else "error",
        iterations=train.iterations + eval_result.iterations,
        metrics={**train.metrics, **eval_result.metrics},
    )


def run_stage(
    cfg: ExperimentConfig,
    *,
    stage: RunStage,
    seed: int | None = None,
    allow_overwrite: bool = False,
    execution: ExecutionInfo | None = None,
    runs_root: Path | None = None,
    source_path: Path | None = None,
    resume: bool | None = None,
    allowed_wandb_run_ids: tuple[str, ...] | None = None,
) -> StageResult:
    """Execute one train/eval stage in the current process."""
    if seed is not None and seed != cfg.seed:
        cfg = cfg.model_copy(update={"seed": seed})
    resume = resolve_resume_setting(resume)

    started_at = datetime.now(UTC)
    git = read_git_state()
    device = _resolve_device(cfg.runtime.device)
    validate_registered_trainer_config(cfg.trainer)
    store = resolve_run_store(cfg, source_path=source_path, runs_root=runs_root)

    if resume and not allow_overwrite and _stage_marker_ok(store, stage=stage):
        fast = _completed_stage_result(
            store,
            stage=stage,
            git=git,
            started_at=started_at,
            execution=execution,
            device=device,
        )
        if fast is not None:
            return fast

    if stage == "eval":
        train_status = inspect_stage_status(
            run_dir=store.run_dir,
            identity=store.identity,
            seed=cfg.seed,
            stage="train",
        )
        if train_status.status != "ok":
            raise RuntimeError(
                "eval stage requires a completed train stage for the same "
                f"config/seed; train status is {train_status.status!r}"
            )

    with stage_lock(
        store.run_dir,
        stage,
        metadata=_stage_lock_metadata(store.identity, seed=cfg.seed),
    ):
        if resume and not allow_overwrite and _stage_marker_ok(store, stage=stage):
            locked = _completed_stage_result(
                store,
                stage=stage,
                git=git,
                started_at=started_at,
                execution=execution,
                device=device,
            )
            if locked is not None:
                return locked
        # Destructive work happens only with the lock held: clearing before
        # acquisition would delete a live runner's outputs, and overwriting via
        # ensure_base must never remove the .locks/ directory it relies on.
        if allow_overwrite and stage == "eval":
            _clear_stage_outputs(store.run_dir, "eval")
            allow_overwrite = False
        # Train-stage cleanup also wipes eval outputs (stages/eval, results, the
        # eval ledger), so it must hold the eval lock too; a live eval runner then
        # fails this acquisition instead of losing its outputs. Nested lock order
        # is train then eval everywhere.
        clears_eval_outputs = stage == "train" and (allow_overwrite or not resume)
        eval_guard = (
            stage_lock(
                store.run_dir,
                "eval",
                metadata=_stage_lock_metadata(store.identity, seed=cfg.seed),
            )
            if clears_eval_outputs
            else nullcontext()
        )
        with eval_guard:
            if not resume:
                _clear_stage_outputs(store.run_dir, stage)
            store.ensure_base(allow_overwrite=allow_overwrite)
        return _run_stage_locked(
            cfg,
            stage=stage,
            store=store,
            started_at=started_at,
            execution=execution,
            git=git,
            device=device,
            resume=resume,
            allowed_wandb_run_ids=allowed_wandb_run_ids,
        )


def _run_stage_locked(
    cfg: ExperimentConfig,
    *,
    stage: RunStage,
    store: RunStore,
    started_at: datetime,
    execution: ExecutionInfo | None,
    git,
    device: torch.device,
    resume: bool,
    allowed_wandb_run_ids: tuple[str, ...] | None,
) -> StageResult:
    attempt_dir = store.attempt_dir(stage=stage, started_at=started_at, git=git)
    previous_state = _apply_runtime_state(cfg)
    task_result: TaskResult | None = None
    exit_status: RunExitStatus = "error"
    error_message: str | None = None
    metric_sink: MetricSink = NoopMetricSink()
    stage_meta = build_metadata(
        seed=cfg.seed,
        started_at=started_at,
        execution=execution
        or ExecutionInfo(
            backend=cfg.runtime.backend,
            device_request=cfg.runtime.device,
        ),
        source_git=git,
        effective_device=str(device),
        run_identity=store.identity,
    )

    try:
        configure_logging(
            level=_log_level(cfg),
            console=cfg.logging.stdout,
            plain_log_file=store.run_dir / "logs" / f"{stage}.console.log"
            if cfg.logging.plain_log_file
            else None,
            json_log_file=store.run_dir / "logs" / "events.jsonl"
            if cfg.logging.json_log_file
            else None,
        )
        clear_contextvars()
        bind_contextvars(
            run_dir=str(store.run_dir),
            stage=stage,
            experiment=cfg.name,
            dataset=cfg.dataset.name,
            trainer=cfg.trainer.key,
            seed=cfg.seed,
        )
        write_metadata(stage_metadata_path(store.run_dir, stage), stage_meta)
        app_log = get_logger(__name__).bind(
            run_dir=str(store.run_dir),
            stage=stage,
            experiment=cfg.name,
            dataset=cfg.dataset.name,
            trainer=cfg.trainer.key,
            seed=cfg.seed,
        )
        app_log.info("stage_started", device=str(device), git_commit=git.commit)
        metric_sink = _build_metric_sink(
            cfg,
            run_dir=store.run_dir,
            identity=store.stage_run_identity(stage),
            stage=stage,
            resume=resume,
            allowed_wandb_run_ids=allowed_wandb_run_ids,
        )
        task_result = _run_stage_task(
            cfg,
            stage=stage,
            run_dir=store.run_dir,
            device=device,
            metric_sink=metric_sink,
            resume=resume,
            identity=store.identity,
        )
        exit_status = "ok"
        app_log.info(
            "stage_finished",
            stage=stage,
            exit_status=exit_status,
            last_completed_iteration=task_result.iterations,
        )
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}\n" + traceback.format_exc()
        get_logger(__name__).exception("stage_failed", stage=stage, error=str(exc))
        raise
    finally:
        ended_at = datetime.now(UTC)
        final_stage_meta = finalise_metadata(
            stage_meta,
            exit_status=exit_status,
            error_message=error_message,
            last_completed_iteration=task_result.iterations if task_result else None,
            ended_at=ended_at,
        )
        write_metadata(stage_metadata_path(store.run_dir, stage), final_stage_meta)
        # Persist this attempt's environment snapshot in its own attempt directory. The
        # per-stage and run-level metadata.json are rebuilt from the current environment on the
        # next attempt, so without this the original attempt's lockfile/git/host would be lost;
        # the attempt directory is unique per attempt and never overwritten.
        write_metadata(attempt_dir / "metadata.json", final_stage_meta)
        _write_stage_completion(
            store,
            stage=stage,
            exit_status=exit_status,
            started_at=started_at,
            ended_at=ended_at,
            attempt_dir=attempt_dir,
            task_result=task_result,
            error_message=error_message,
            lockfile_sha256=pixi_lock_sha256(),
            git_commit=git.commit,
        )
        _update_run_metadata(
            store,
            started_at=started_at,
            ended_at=ended_at,
            execution=execution,
            git=git,
            device=device,
            last_completed_iteration=task_result.iterations if task_result else None,
            error_message=error_message,
        )
        close_error: Exception | None = None
        try:
            _log_completion_metric(
                cfg,
                metric_sink=metric_sink,
                stage=stage,
                exit_status=exit_status,
                run_identity=store.identity,
                run_complete=_run_markers_complete(store),
            )
        except Exception as exc:
            close_error = exc
        try:
            metric_sink.close(exit_code=0 if exit_status == "ok" else 1)
        except Exception as exc:
            if close_error is None:
                close_error = exc
        shutdown_logging()
        _copy_attempt_logs(store.run_dir, attempt_dir, stage)
        clear_contextvars()
        _restore_runtime_state(previous_state)
        if close_error is not None and error_message is None:
            raise close_error

    return StageResult(
        run_dir=store.run_dir,
        stage=stage,
        exit_status=exit_status,
        iterations=task_result.iterations if task_result else 0,
        metrics=task_result.metrics if task_result else {},
    )


def _stage_lock_metadata(
    identity: RunIdentity,
    *,
    seed: int,
) -> dict[str, str | int | None]:
    return {
        "seed": seed,
        "source_identity": identity.source_identity,
        "config_hash": identity.config_hash,
        "completion_hash": identity.completion_hash,
        "snapshot_sha256": identity.snapshot_sha256,
    }


def _run_stage_task(
    cfg: ExperimentConfig,
    *,
    stage: RunStage,
    run_dir: Path,
    device: torch.device,
    metric_sink: MetricSink,
    resume: bool,
    identity: RunIdentity,
) -> TaskResult:
    return run_adapter_stage(
        cfg,
        stage=stage,
        run_dir=run_dir,
        device=device,
        metric_sink=metric_sink,
        resume=resume,
        identity=identity,
    )


def _build_metric_sink(
    cfg: ExperimentConfig,
    *,
    run_dir: Path,
    identity: RunIdentity,
    stage: RunStage,
    resume: bool,
    allowed_wandb_run_ids: tuple[str, ...] | None = None,
) -> MetricSink:
    sinks: list[MetricSink] = []
    if cfg.logging.metrics_jsonl:
        sinks.append(JsonlMetricSink(run_dir=run_dir, identity=identity))
    wandb_cfg = cfg.logging.wandb
    if wandb_cfg is not None and wandb_cfg.mode != "disabled":
        sinks.append(
            WandbMetricSink(
                cfg=cfg,
                wandb_cfg=wandb_cfg.model_copy(update={"job_type": stage}),
                run_dir=run_dir,
                identity=identity,
                stage=stage,
                resume=resume,
                allowed_resume_run_ids=allowed_wandb_run_ids,
            )
        )
    if not sinks:
        return NoopMetricSink()
    if len(sinks) == 1:
        return sinks[0]
    return CompositeMetricSink(sinks=sinks)


def _write_stage_completion(
    store: RunStore,
    *,
    stage: RunStage,
    exit_status: RunExitStatus,
    started_at: datetime,
    ended_at: datetime,
    attempt_dir: Path,
    task_result: TaskResult | None,
    error_message: str | None,
    lockfile_sha256: str | None = None,
    git_commit: str | None = None,
) -> None:
    completion = StageCompletion(
        stage=stage,
        exit_status=exit_status,
        seed=store.cfg.seed,
        source_identity=store.identity.source_identity,
        config_hash=store.identity.config_hash,
        completion_hash=store.identity.completion_hash,
        snapshot_sha256=store.identity.snapshot_sha256,
        trainer_key=store.cfg.trainer.key,
        dataset_name=store.cfg.dataset.name,
        run_dir=str(store.run_dir),
        attempt_dir=str(attempt_dir),
        started_at=started_at,
        ended_at=ended_at,
        iterations=task_result.iterations if task_result else None,
        metrics=task_result.metrics if task_result else {},
        error_message=error_message,
        lockfile_sha256=lockfile_sha256,
        git_commit=git_commit,
    )
    write_json_atomic(
        stage_completion_path(store.run_dir, stage),
        completion.model_dump(mode="json"),
    )


def _update_run_metadata(
    store: RunStore,
    *,
    started_at: datetime,
    ended_at: datetime,
    execution: ExecutionInfo | None,
    git,
    device: torch.device,
    last_completed_iteration: int | None,
    error_message: str | None,
) -> None:
    train_done = _stage_marker_ok(store, "train")
    eval_done = _stage_marker_ok(store, "eval")
    run_status = "ok" if train_done and eval_done else "partial"
    if error_message is not None:
        run_status = "error"
    meta = build_metadata(
        seed=store.cfg.seed,
        started_at=started_at,
        execution=execution
        or ExecutionInfo(
            backend=store.cfg.runtime.backend,
            device_request=store.cfg.runtime.device,
        ),
        source_git=git,
        effective_device=str(device),
        run_identity=store.identity,
    )
    final_meta = finalise_metadata(
        meta,
        exit_status=run_status,
        error_message=error_message,
        last_completed_iteration=last_completed_iteration,
        ended_at=ended_at,
    )
    write_metadata(store.run_dir / "metadata.json", final_meta)


def _log_completion_metric(
    cfg: ExperimentConfig,
    *,
    metric_sink: MetricSink,
    stage: RunStage,
    exit_status: RunExitStatus,
    run_identity: RunIdentity,
    run_complete: bool,
) -> None:
    run_exit_status = "error" if exit_status != "ok" else "partial"
    if run_complete:
        run_exit_status = "ok"
    context = {
        "run/config_hash": run_identity.config_hash,
        "run/exit_status": run_exit_status,
        "run/snapshot_sha256": run_identity.snapshot_sha256,
        "run/source_identity": run_identity.source_identity,
        "run/stage": stage,
        "stage/exit_status": exit_status,
    }
    metric_sink.log(
        MetricRecord.from_config(
            cfg,
            phase="summary",
            metric="stage/complete",
            value=1.0 if exit_status == "ok" else 0.0,
            context=context,
        )
    )
    if stage == "eval":
        metric_sink.log(
            MetricRecord.from_config(
                cfg,
                phase="summary",
                metric="run/complete",
                value=1.0 if run_complete else 0.0,
                context=context,
            )
        )


def _warn_environment_drift(
    completion: dict[str, Any], git: GitState, stage: RunStage
) -> None:
    """
    Warn when a resumed stage was completed under a different environment.

    The pixi.lock hash and git commit recorded in the completion marker are compared
    against the current environment, so resuming under a changed lockfile or checkout
    surfaces the environment mixing instead of skipping silently. Resume still
    proceeds: the completed work is an immutable artifact, so the operator decides
    whether to re-run.
    """
    recorded_lock = completion.get("lockfile_sha256")
    recorded_commit = completion.get("git_commit")
    current_lock = pixi_lock_sha256()
    # Only the recorded side is guarded: a marker that never recorded an
    # environment cannot be compared, but a recorded lockfile that is
    # now missing or changed is a genuine drift worth surfacing.
    lock_drift = recorded_lock is not None and recorded_lock != current_lock
    commit_drift = recorded_commit is not None and recorded_commit != git.commit
    if lock_drift or commit_drift:
        get_logger(__name__).warning(
            "resume_environment_drift",
            stage=stage,
            recorded_lockfile_sha256=recorded_lock if lock_drift else None,
            current_lockfile_sha256=current_lock if lock_drift else None,
            recorded_git_commit=recorded_commit if commit_drift else None,
            current_git_commit=git.commit if commit_drift else None,
        )


def _completed_stage_result(
    store: RunStore,
    *,
    stage: RunStage,
    git: GitState,
    started_at: datetime,
    execution: ExecutionInfo | None,
    device: torch.device,
) -> StageResult | None:
    """Return a StageResult from the completion marker, or None if the marker is
    absent/invalid."""
    completion = read_json_object(stage_completion_path(store.run_dir, stage))
    if not completion:
        return None
    _warn_environment_drift(completion, git, stage)
    _repair_run_metadata_if_complete(
        store,
        started_at=started_at,
        execution=execution,
        git=git,
        device=device,
    )
    return StageResult(
        run_dir=store.run_dir,
        stage=stage,
        exit_status="ok",
        iterations=int(completion.get("iterations") or 0),
        metrics=_float_metrics(completion.get("metrics")),
    )


def _stage_marker_ok(store: RunStore, stage: RunStage) -> bool:
    completion = read_json_object(stage_completion_path(store.run_dir, stage))
    return bool(
        completion
        and stage_completion_matches(
            completion,
            identity=store.identity,
            seed=store.cfg.seed,
            stage=stage,
        )
        and completion.get("exit_status") == "ok"
    )


def _run_markers_complete(store: RunStore) -> bool:
    return _stage_marker_ok(store, "train") and _stage_marker_ok(store, "eval")


def _repair_run_metadata_if_complete(
    store: RunStore,
    *,
    started_at: datetime,
    execution: ExecutionInfo | None,
    git,
    device: torch.device,
) -> None:
    if not _run_markers_complete(store):
        return
    _update_run_metadata(
        store,
        started_at=started_at,
        ended_at=datetime.now(UTC),
        execution=execution,
        git=git,
        device=device,
        last_completed_iteration=_completed_iteration(store, "eval"),
        error_message=None,
    )


def _completed_iteration(store: RunStore, stage: RunStage) -> int | None:
    completion = read_json_object(stage_completion_path(store.run_dir, stage))
    raw = completion.get("iterations")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    return None


def _clear_stage_outputs(run_dir: Path, stage: RunStage) -> None:
    shutil.rmtree(stage_dir(run_dir, stage), ignore_errors=True)
    metrics_path = run_dir / "metrics" / f"{stage}.jsonl"
    metrics_path.unlink(missing_ok=True)
    if stage == "train":
        shutil.rmtree(stage_dir(run_dir, "eval"), ignore_errors=True)
        shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
        (run_dir / "training_history.json").unlink(missing_ok=True)
        (run_dir / "metrics" / "eval.jsonl").unlink(missing_ok=True)
        shutil.rmtree(run_dir / "results", ignore_errors=True)
    elif stage == "eval":
        shutil.rmtree(run_dir / "results", ignore_errors=True)


def _copy_attempt_logs(run_dir: Path, attempt_dir: Path, stage: RunStage) -> None:
    plain = run_dir / "logs" / f"{stage}.console.log"
    if plain.exists():
        shutil.copy2(plain, attempt_dir / "console.log")


def _run_result_from_stage(result: StageResult) -> RunResult:
    return RunResult(
        run_dir=result.run_dir,
        exit_status=result.exit_status,
        iterations=result.iterations,
        metrics=result.metrics,
    )


def _float_metrics(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def _resolve_device(selector: str) -> torch.device:
    if selector == "cpu":
        return torch.device("cpu")
    if selector == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "runtime.device='cuda' requested, but CUDA is unavailable"
            )
        return torch.device("cuda")
    if selector == "mps":
        if not _mps_available():
            raise RuntimeError("runtime.device='mps' requested, but MPS is unavailable")
        return torch.device("mps")
    if selector == "auto":
        # Prefer mps over cuda to match device_manual_mps_or_cuda_if_available,
        # so the recorded effective_device is the device the trainers run on.
        if _mps_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    raise ValueError(f"unsupported runtime.device={selector!r}")


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    is_available = getattr(backend, "is_available", None) if backend else None
    return bool(is_available() if callable(is_available) else False)


def _apply_runtime_state(cfg: ExperimentConfig) -> _RuntimeState:
    previous: _RuntimeState = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "num_threads": torch.get_num_threads(),
    }
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    torch.use_deterministic_algorithms(cfg.runtime.deterministic)
    if cfg.runtime.num_threads is not None:
        torch.set_num_threads(int(cfg.runtime.num_threads))
    if hasattr(torch.backends, "cudnn"):
        previous["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = (
            cfg.runtime.cudnn_benchmark and not cfg.runtime.deterministic
        )
    return previous


def _restore_runtime_state(previous: _RuntimeState) -> None:
    torch.use_deterministic_algorithms(previous["deterministic"])
    torch.set_num_threads(previous["num_threads"])
    if "cudnn_benchmark" in previous and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = bool(previous["cudnn_benchmark"])


def _log_level(cfg: ExperimentConfig) -> int:
    return getattr(logging, cfg.logging.level.upper())
