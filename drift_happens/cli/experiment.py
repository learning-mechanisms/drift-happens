"""CLI commands for experiment presets, staged runs, plans, and sweeps."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import typer

from drift_happens.configs import ExperimentConfig
from drift_happens.configs.logging_cfg import WandbMode
from drift_happens.configs.sweep import DeviceSlotConfig, SlotDevice
from drift_happens.utils.env import (
    RESUME_CHECKPOINTS_ENV,
    RUN_RESUME_ENV,
    apply_resume_checkpoints_override,
    resolve_resume_setting,
    with_wandb_from_env,
)
from drift_happens.utils.paths import (
    EXPERIMENT_PLANS_DIR,
    SNAPSHOTS_DIR,
    relative_to_project,
)

if TYPE_CHECKING:
    from drift_happens.runtime.run_store import RunStore
    from drift_happens.runtime.stages import RunStage

# Mirrors experiments.materialize.PRESETS_ROOT. Importing the experiments or runtime
# packages at module level would pull the preset registry and torch into every CLI
# startup, so those modules are imported inside the command bodies instead.
PRESETS_ROOT = SNAPSHOTS_DIR / "presets"
WANDB_PREFLIGHT_MAX_FAILED_ATTEMPTS = 20

app = typer.Typer(
    help="List, materialize, run, and manage experiments.",
    no_args_is_help=True,
)
seeds_app = typer.Typer(
    help="Inspect and summarize multi-seed experiment status.",
    no_args_is_help=True,
)
stages_app = typer.Typer(
    help="Inspect train/eval stage status.",
    no_args_is_help=True,
)
plans_app = typer.Typer(
    help="Generate staged multi-seed sweep plans.",
    no_args_is_help=True,
)
locks_app = typer.Typer(
    help="Inspect and repair stale stage locks.",
    no_args_is_help=True,
)


@app.command("list")
def list_presets(
    group: str | None = typer.Option(None, "--group", help="Only show one group."),
) -> None:
    """List registered experiment presets without building datasets or models."""
    from drift_happens.experiments.registry import preset_groups

    groups = preset_groups()
    shown = 0
    for group_name, names in groups.items():
        if group is not None and group != group_name:
            continue
        for name in names:
            typer.echo(f"{group_name}/{name}")
            shown += 1
    if shown == 0:
        raise typer.BadParameter(f"no presets matched group={group!r}")


@app.command("materialize")
def materialize(
    write: bool = typer.Option(
        False,
        "--write",
        help="Write deterministic snapshots from the Python registry.",
    ),
    check: bool = typer.Option(
        False,
        "--check",
        help="Fail if materialized snapshots are missing, stale, or orphaned.",
    ),
    out_dir: Path = typer.Option(
        PRESETS_ROOT,
        "--out-dir",
        help="Preset snapshot root.",
    ),
) -> None:
    """Write or check materialized preset snapshots."""
    from drift_happens.experiments.materialize import (
        check_materialized_snapshots,
        selected_presets,
        write_materialized_snapshots,
    )

    if not write and not check:
        for entry in selected_presets():
            typer.echo(
                relative_to_project(out_dir / entry.group / f"{entry.name}.json")
            )
        typer.echo(relative_to_project(out_dir / "index.json"))
        return

    if write:
        written = write_materialized_snapshots(out_dir)
        typer.echo(
            f"materialized {len(written)} file(s) under {relative_to_project(out_dir)}"
        )

    if check:
        diff = check_materialized_snapshots(out_dir)
        if not diff.ok:
            typer.echo(diff.format(), err=True)
            raise typer.Exit(code=1)
        typer.echo(diff.format())


@plans_app.command("list")
def list_plans() -> None:
    """List standard staged experiment plans."""
    from drift_happens.experiments.plans import list_plan_stages

    for stage in list_plan_stages():
        typer.echo(f"{stage.name}\t{len(stage.jobs)} job(s)")


@plans_app.command("materialize")
def materialize_plans(
    write: bool = typer.Option(False, "--write"),
    check: bool = typer.Option(False, "--check"),
    out_dir: Path = typer.Option(
        EXPERIMENT_PLANS_DIR,
        "--out-dir",
        help="Plan output directory.",
    ),
    device: SlotDevice = typer.Option("cpu", "--device"),
    gpu_indices: str = typer.Option("0", "--gpu-indices"),
    jobs_per_device: int = typer.Option(1, "--jobs-per-device"),
    concurrency: int = typer.Option(1, "--concurrency"),
    seeds: str | None = typer.Option(
        None,
        "--seeds",
        help="Comma-separated seed filter for generated plan jobs.",
    ),
) -> None:
    """Write or check staged sweep YAMLs."""
    from drift_happens.experiments.plans import (
        check_plan_files,
        expected_plan_files,
        write_plan_files,
    )

    parsed_seed_filter = _parse_optional_int_csv(seeds, param_name="--seeds")
    if not write and not check:
        for path in expected_plan_files(
            out_dir=out_dir,
            device=device,
            gpu_indices=_parse_int_csv(gpu_indices),
            jobs_per_device=jobs_per_device,
            concurrency=concurrency,
            seed_filter=parsed_seed_filter,
        ):
            typer.echo(relative_to_project(path))
        return

    parsed_gpu_indices = _parse_int_csv(gpu_indices)
    if write:
        written = write_plan_files(
            out_dir=out_dir,
            device=device,
            gpu_indices=parsed_gpu_indices,
            jobs_per_device=jobs_per_device,
            concurrency=concurrency,
            seed_filter=parsed_seed_filter,
        )
        typer.echo(
            f"wrote {len(written)} plan file(s) under {relative_to_project(out_dir)}"
        )
    if check:
        diff = check_plan_files(
            out_dir=out_dir,
            device=device,
            gpu_indices=parsed_gpu_indices,
            jobs_per_device=jobs_per_device,
            concurrency=concurrency,
            seed_filter=parsed_seed_filter,
        )
        if not diff.ok:
            typer.echo(diff.format(), err=True)
            raise typer.Exit(code=1)
        typer.echo(diff.format())


@plans_app.command("status")
def plan_status(
    sweep_config: Path = typer.Argument(..., help="SweepConfig YAML/JSON path."),
    source: Literal["local", "wandb"] = typer.Option("wandb", "--source"),
    wandb_project: str | None = typer.Option(None, "--wandb-project"),
    wandb_entity: str | None = typer.Option(None, "--wandb-entity"),
    wandb_mode: WandbMode | None = typer.Option(None, "--wandb-mode"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Summarize seed completion for every job in a sweep plan."""
    from drift_happens.experiments.source import load_experiment_source
    from drift_happens.runtime.completion_filter import local_seed_statuses_by_identity
    from drift_happens.runtime.run_store import resolve_run_store
    from drift_happens.runtime.sweep import load_sweep_config
    from drift_happens.utils.wandb_completion import WandbCompletionIndex

    sweep = load_sweep_config(sweep_config)
    device_overrides = {_plan_status_device_override(slot) for slot in sweep.slots}
    if len(device_overrides) != 1:
        raise typer.BadParameter(
            "plan status requires all sweep slots to target the same runtime device"
        )
    device_override = next(iter(device_overrides))
    wandb_indexes: dict[tuple[str, str | None], WandbCompletionIndex] = {}
    summaries: dict[str, dict[str, int]] = {}

    for job in sweep.jobs:
        experiment_source = load_experiment_source(
            job.config_path,
            overrides=(*job.overrides, device_override),
        )
        cfg = experiment_source.config.model_copy(update={"seed": job.seed})
        cfg = with_wandb_from_env(
            cfg,
            project=wandb_project,
            entity=wandb_entity,
            mode=wandb_mode,
        )
        store = resolve_run_store(
            cfg,
            source_path=experiment_source.path,
            runs_root=runs_root,
        )
        if source == "wandb":
            wandb_cfg = cfg.logging.wandb
            if wandb_cfg is None or wandb_cfg.mode == "disabled":
                raise typer.BadParameter(
                    "W&B status requires cfg.logging.wandb, WANDB_PROJECT, "
                    "or --wandb-project"
                )
            index_key = (wandb_cfg.project, wandb_cfg.entity)
            index = wandb_indexes.setdefault(
                index_key,
                WandbCompletionIndex(
                    project=wandb_cfg.project,
                    entity=wandb_cfg.entity,
                ),
            )
            train = (
                "ok"
                if index.is_stage_complete(
                    group=store.identity.wandb_group,
                    seed=job.seed,
                    stage="train",
                    config_hash=store.identity.config_hash,
                    snapshot_sha256=store.identity.snapshot_sha256,
                    completion_hash=store.identity.completion_hash,
                )
                else "missing"
            )
            eval_status = (
                "ok"
                if index.is_stage_complete(
                    group=store.identity.wandb_group,
                    seed=job.seed,
                    stage="eval",
                    config_hash=store.identity.config_hash,
                    snapshot_sha256=store.identity.snapshot_sha256,
                    completion_hash=store.identity.completion_hash,
                )
                else "missing"
            )
            run = (
                "ok"
                if index.is_run_complete(
                    group=store.identity.wandb_group,
                    seed=job.seed,
                    config_hash=store.identity.config_hash,
                    snapshot_sha256=store.identity.snapshot_sha256,
                    completion_hash=store.identity.completion_hash,
                )
                else "missing"
            )
            if run != "ok" and train == "ok" and eval_status == "ok":
                run = "partial"
        else:
            row = local_seed_statuses_by_identity(
                {job.seed: store.identity},
                runs_root=runs_root,
            )[0]
            train = row.train
            eval_status = row.eval
            run = row.status

        summary = summaries.setdefault(
            job.label,
            {
                "total": 0,
                "run_ok": 0,
                "partial": 0,
                "missing": 0,
                "train_ok": 0,
                "eval_ok": 0,
            },
        )
        summary["total"] += 1
        if run == "ok":
            summary["run_ok"] += 1
        elif run == "partial":
            summary["partial"] += 1
        else:
            summary["missing"] += 1
        if train == "ok":
            summary["train_ok"] += 1
        if eval_status == "ok":
            summary["eval_ok"] += 1

    totals = {
        "total": 0,
        "run_ok": 0,
        "partial": 0,
        "missing": 0,
        "train_ok": 0,
        "eval_ok": 0,
    }
    rows: list[dict[str, str | int]] = []
    for label, summary in summaries.items():
        _add_plan_status_totals(totals, summary)
        rows.append({"label": label, **summary})
    rows.append({"label": "TOTAL", **totals})
    typer.echo(_format_plan_status_table(rows))


@locks_app.command("repair")
def repair_locks(
    sweep_config: Path = typer.Argument(..., help="SweepConfig YAML/JSON path."),
    stage: Literal["train", "eval"] = typer.Option("train", "--stage"),
    apply: bool = typer.Option(
        False,
        "--apply/--dry-run",
        help="Remove reclaimable locks. Defaults to dry-run.",
    ),
    allow_legacy_foreign: bool = typer.Option(
        False,
        "--allow-legacy-foreign",
        help=(
            "Allow removing foreign-host locks that predate heartbeat metadata when "
            "W&B shows the stage is not running."
        ),
    ),
    stale_after_seconds: float = typer.Option(
        3600.0,
        "--stale-after-seconds",
        help="Heartbeat age required before a foreign-host lock is stale.",
    ),
    wandb_project: str | None = typer.Option(None, "--wandb-project"),
    wandb_entity: str | None = typer.Option(None, "--wandb-entity"),
    wandb_mode: WandbMode | None = typer.Option(None, "--wandb-mode"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Dry-run or remove stale stage locks for jobs in a sweep plan."""
    from drift_happens.experiments.source import load_experiment_source
    from drift_happens.runtime.lock_repair import (
        WandbLockState,
        apply_lock_repair,
        classify_stage_lock,
        slurm_job_is_running,
    )
    from drift_happens.runtime.locks import read_lock_owner
    from drift_happens.runtime.run_store import resolve_run_store
    from drift_happens.runtime.sweep import load_sweep_config
    from drift_happens.utils.wandb_completion import WandbCompletionIndex

    sweep = load_sweep_config(sweep_config)
    device_overrides = {_plan_status_device_override(slot) for slot in sweep.slots}
    if len(device_overrides) != 1:
        raise typer.BadParameter(
            "lock repair requires all sweep slots to target the same runtime device"
        )
    device_override = next(iter(device_overrides))
    wandb_indexes: dict[tuple[str, str | None], WandbCompletionIndex] = {}
    rows: list[dict[str, str | int | float | None]] = []

    for job in sweep.jobs:
        experiment_source = load_experiment_source(
            job.config_path,
            overrides=(*job.overrides, device_override),
        )
        cfg = experiment_source.config.model_copy(update={"seed": job.seed})
        cfg = with_wandb_from_env(
            cfg,
            project=wandb_project,
            entity=wandb_entity,
            mode=wandb_mode,
        )
        store = resolve_run_store(
            cfg,
            source_path=experiment_source.path,
            runs_root=runs_root,
        )
        lock_path = store.run_dir / ".locks" / f"{stage}.lock"
        owner = read_lock_owner(lock_path)
        wandb_state: WandbLockState = "unavailable"
        wandb_cfg = cfg.logging.wandb
        if wandb_cfg is not None and wandb_cfg.mode != "disabled":
            index_key = (wandb_cfg.project, wandb_cfg.entity)
            index = wandb_indexes.setdefault(
                index_key,
                WandbCompletionIndex(
                    project=wandb_cfg.project,
                    entity=wandb_cfg.entity,
                ),
            )
            wandb_state = index.preflight_status(
                group=store.identity.wandb_group,
                seed=job.seed,
                config_hash=store.identity.config_hash,
                snapshot_sha256=store.identity.snapshot_sha256,
                completion_hash=store.identity.completion_hash,
                max_failed_attempts=WANDB_PREFLIGHT_MAX_FAILED_ATTEMPTS,
                stage=stage,
            ).state
        slurm_running = (
            slurm_job_is_running(owner.slurm_job_id)
            if owner is not None and owner.slurm_job_id is not None
            else None
        )
        decision = classify_stage_lock(
            lock_path=lock_path,
            run_dir=store.run_dir,
            stage=stage,
            wandb_state=wandb_state,
            slurm_running=slurm_running,
            stale_after_seconds=stale_after_seconds,
            allow_legacy_foreign=allow_legacy_foreign,
        )
        if apply:
            decision = apply_lock_repair(decision)
        rows.append(
            {
                "action": decision.action,
                "removed": int(decision.removed),
                "label": job.label,
                "seed": job.seed,
                "owner": decision.owner_label,
                "wandb": decision.wandb_state,
                "slurm_running": _optional_bool(decision.slurm_running),
                "heartbeat_age_s": _optional_float(decision.heartbeat_age_seconds),
                "reason": decision.reason,
                "lock": str(relative_to_project(decision.lock_path)),
            }
        )

    typer.echo(_format_lock_repair_table(rows))


@app.command("train")
def train_experiment(
    config_or_snapshot: Path = typer.Argument(
        ...,
        help="ExperimentConfig YAML/JSON or materialized preset snapshot.",
    ),
    seed: int | None = typer.Option(None, "--seed"),
    set_overrides: list[str] | None = typer.Option(None, "--set"),
    wandb_project: str | None = typer.Option(None, "--wandb-project"),
    wandb_entity: str | None = typer.Option(None, "--wandb-entity"),
    wandb_mode: WandbMode | None = typer.Option(None, "--wandb-mode"),
    wandb_tags: str | None = typer.Option(None, "--wandb-tags"),
    wandb_upload_artifacts: bool | None = typer.Option(
        None,
        "--wandb-upload-artifacts/--no-wandb-upload-artifacts",
    ),
    wandb_upload_checkpoints: bool | None = typer.Option(
        None,
        "--wandb-upload-checkpoints/--no-wandb-upload-checkpoints",
    ),
    skip_completed: bool = typer.Option(False, "--skip-completed"),
    resume: bool | None = typer.Option(
        None,
        "--resume/--no-resume",
        help=(
            "Reuse existing completed work units (default). Pass --no-resume to "
            f"clear and recompute owned outputs. Set {RUN_RESUME_ENV}=0 to force "
            "fresh re-runs when no flag is given."
        ),
    ),
    allow_overwrite: bool = typer.Option(False, "--allow-overwrite"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
    resume_checkpoints: bool | None = typer.Option(
        None,
        "--resume-checkpoints/--no-resume-checkpoints",
        help=(
            "Continue an unfinished slice from its mid-training epoch checkpoint "
            f"(default off; finished slices are reused regardless). {RESUME_CHECKPOINTS_ENV}=1 "
            "enables it when no flag is given."
        ),
    ),
) -> None:
    """Run only the train stage for one resolved config and seed."""
    apply_resume_checkpoints_override(resume_checkpoints)
    _execute_stage_command(
        "train",
        config_or_snapshot=config_or_snapshot,
        seed=seed,
        set_overrides=set_overrides,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        wandb_tags=wandb_tags,
        wandb_upload_artifacts=wandb_upload_artifacts,
        wandb_upload_checkpoints=wandb_upload_checkpoints,
        skip_completed=skip_completed,
        resume=resume,
        allow_overwrite=allow_overwrite,
        runs_root=runs_root,
    )


@app.command("eval")
def eval_experiment(
    config_or_snapshot: Path = typer.Argument(
        ...,
        help="ExperimentConfig YAML/JSON or materialized preset snapshot.",
    ),
    seed: int | None = typer.Option(None, "--seed"),
    set_overrides: list[str] | None = typer.Option(None, "--set"),
    wandb_project: str | None = typer.Option(None, "--wandb-project"),
    wandb_entity: str | None = typer.Option(None, "--wandb-entity"),
    wandb_mode: WandbMode | None = typer.Option(None, "--wandb-mode"),
    wandb_tags: str | None = typer.Option(None, "--wandb-tags"),
    wandb_upload_artifacts: bool | None = typer.Option(
        None,
        "--wandb-upload-artifacts/--no-wandb-upload-artifacts",
    ),
    wandb_upload_checkpoints: bool | None = typer.Option(
        None,
        "--wandb-upload-checkpoints/--no-wandb-upload-checkpoints",
    ),
    skip_completed: bool = typer.Option(False, "--skip-completed"),
    resume: bool | None = typer.Option(
        None,
        "--resume/--no-resume",
        help=(
            "Reuse existing completed work units (default). Pass --no-resume to "
            f"clear and recompute owned outputs. Set {RUN_RESUME_ENV}=0 to force "
            "fresh re-runs when no flag is given."
        ),
    ),
    allow_overwrite: bool = typer.Option(False, "--allow-overwrite"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Run only the eval stage for one resolved config and seed."""
    _execute_stage_command(
        "eval",
        config_or_snapshot=config_or_snapshot,
        seed=seed,
        set_overrides=set_overrides,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        wandb_tags=wandb_tags,
        wandb_upload_artifacts=wandb_upload_artifacts,
        wandb_upload_checkpoints=wandb_upload_checkpoints,
        skip_completed=skip_completed,
        resume=resume,
        allow_overwrite=allow_overwrite,
        runs_root=runs_root,
    )


@app.command("run")
def run_experiment(
    config_or_snapshot: Path = typer.Argument(
        ...,
        help="ExperimentConfig YAML/JSON or materialized preset snapshot.",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Seed override for this staged run.",
    ),
    set_overrides: list[str] | None = typer.Option(
        None,
        "--set",
        help="Dotted override, for example trainer.training.num_epochs=2.",
    ),
    wandb_project: str | None = typer.Option(
        None,
        "--wandb-project",
        help="Enable or override W&B project for this command.",
    ),
    wandb_entity: str | None = typer.Option(None, "--wandb-entity"),
    wandb_mode: WandbMode | None = typer.Option(None, "--wandb-mode"),
    wandb_tags: str | None = typer.Option(
        None,
        "--wandb-tags",
        help="Comma-separated extra W&B tags.",
    ),
    wandb_upload_artifacts: bool | None = typer.Option(
        None,
        "--wandb-upload-artifacts/--no-wandb-upload-artifacts",
        help="Override curated W&B run artifact upload.",
    ),
    wandb_upload_checkpoints: bool | None = typer.Option(
        None,
        "--wandb-upload-checkpoints/--no-wandb-upload-checkpoints",
        help="Override checkpoint upload to W&B artifacts.",
    ),
    skip_completed: bool = typer.Option(
        False,
        "--skip-completed",
        help="Skip if a matching local or W&B run is already complete.",
    ),
    resume: bool | None = typer.Option(
        None,
        "--resume/--no-resume",
        help=(
            "Reuse existing completed work units (default). Pass --no-resume to "
            f"clear and recompute owned outputs. Set {RUN_RESUME_ENV}=0 to force "
            "fresh re-runs when no flag is given."
        ),
    ),
    allow_overwrite: bool = typer.Option(False, "--allow-overwrite"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
    in_process: bool = typer.Option(False, "--in-process"),
    resume_checkpoints: bool | None = typer.Option(
        None,
        "--resume-checkpoints/--no-resume-checkpoints",
        help=(
            "Continue an unfinished slice from its mid-training epoch checkpoint "
            f"(default off; finished slices are reused regardless). {RESUME_CHECKPOINTS_ENV}=1 "
            "enables it when no flag is given."
        ),
    ),
) -> None:
    """Run train and eval as separate stages for one resolved config and seed."""
    from drift_happens.runtime.run_store import resolve_run_store

    apply_resume_checkpoints_override(resume_checkpoints)

    cfg, source_path = _load_cfg(
        config_or_snapshot,
        seed=seed,
        set_overrides=set_overrides,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        wandb_tags=wandb_tags,
        wandb_upload_artifacts=wandb_upload_artifacts,
        wandb_upload_checkpoints=wandb_upload_checkpoints,
    )
    resolved_resume = resolve_resume_setting(resume)
    store = resolve_run_store(cfg, source_path=source_path, runs_root=runs_root)
    preflight_status = _wandb_preflight_status(
        cfg,
        store,
        stage=None,
        max_failed_attempts=WANDB_PREFLIGHT_MAX_FAILED_ATTEMPTS,
    )
    if _wandb_status_skips(
        preflight_status,
        cfg=cfg,
        stage=None,
    ):
        return
    if skip_completed and _seed_complete(cfg, store, runs_root=runs_root):
        typer.echo(f"skipped complete seed {cfg.seed} for {source_path}")
        return
    if in_process:
        from drift_happens.runtime.local import run_stage
        from drift_happens.runtime.stages import STAGE_CONTENTION_EXIT_CODE

        try:
            run_stage(
                cfg,
                stage="train",
                allow_overwrite=allow_overwrite,
                runs_root=runs_root,
                source_path=source_path,
                resume=resolved_resume,
            )
            eval_result = run_stage(
                cfg,
                stage="eval",
                runs_root=runs_root,
                source_path=source_path,
                resume=resolved_resume,
            )
        except RuntimeError as exc:
            if _is_stage_contention(exc):
                typer.echo(f"skipped seed {cfg.seed}: stage is running")
                raise typer.Exit(code=STAGE_CONTENTION_EXIT_CODE) from exc
            raise
        typer.echo(relative_to_project(eval_result.run_dir))
        return

    for stage in ("train", "eval"):
        code = _spawn_stage_process(
            stage,
            config_or_snapshot=config_or_snapshot,
            seed=seed,
            set_overrides=set_overrides or [],
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_mode=wandb_mode,
            wandb_tags=wandb_tags,
            wandb_upload_artifacts=wandb_upload_artifacts,
            wandb_upload_checkpoints=wandb_upload_checkpoints,
            skip_completed=skip_completed,
            resume=resolved_resume,
            allow_overwrite=allow_overwrite if stage == "train" else False,
            runs_root=runs_root,
        )
        if code != 0:
            raise typer.Exit(code=code)
    typer.echo(relative_to_project(store.run_dir))


@app.command("sweep")
def run_sweep(
    sweep_config: Path = typer.Argument(..., help="SweepConfig YAML/JSON path."),
    skip_completed: bool | None = typer.Option(
        None,
        "--skip-completed/--no-skip-completed",
        help="Override the sweep file skip_completed setting.",
    ),
    skip_source: Literal["local", "wandb"] | None = typer.Option(
        None,
        "--skip-source",
        help="Completion source used when skipping completed jobs.",
    ),
    resume: bool | None = typer.Option(
        None,
        "--resume/--no-resume",
        help=(
            "Reuse existing completed work units in child jobs (default). "
            f"Pass --no-resume to clear owned outputs. Set {RUN_RESUME_ENV}=0 "
            "to force fresh re-runs when no flag is given."
        ),
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    sweep_root: Path | None = typer.Option(None, "--sweep-root"),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show a parent sweep progress bar.",
    ),
) -> None:
    """Run a local staged sweep file."""
    from drift_happens.runtime.sweep import SweepRunner, load_sweep_config

    sweep = load_sweep_config(sweep_config)
    result = SweepRunner(
        sweep,
        sweep_root=sweep_root,
        skip_completed=skip_completed,
        skip_source=skip_source,
        resume=resume,
        dry_run=dry_run,
        show_progress=progress,
    ).run()
    typer.echo(relative_to_project(result.sweep_dir))
    if not result.all_ok:
        raise typer.Exit(code=1)


@app.command("aggregate")
def aggregate_results(
    runs_root: Path | None = typer.Option(None, "--runs-root"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Destination Parquet path."
    ),
) -> None:
    """Compact all run metric ledgers into one long-format results.parquet."""
    from drift_happens.runtime.aggregate import write_results_parquet

    path = write_results_parquet(runs_root=runs_root, output=output)
    typer.echo(relative_to_project(path))


@stages_app.command("status")
def stage_status(
    config_or_snapshot: Path = typer.Argument(...),
    seed: int | None = typer.Option(None, "--seed"),
    set_overrides: list[str] | None = typer.Option(None, "--set"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Show train/eval/run status for one config and seed."""
    from drift_happens.runtime.run_store import resolve_run_store
    from drift_happens.runtime.stage_status import inspect_run_status

    cfg, source_path = _load_cfg(
        config_or_snapshot,
        seed=seed,
        set_overrides=set_overrides,
    )
    store = resolve_run_store(cfg, source_path=source_path, runs_root=runs_root)
    row = inspect_run_status(
        run_dir=store.run_dir,
        identity=store.identity,
        seed=cfg.seed,
    )
    run_dir = relative_to_project(row.run_dir) if row.run_dir else ""
    typer.echo("seed\ttrain\teval\trun\trun_dir")
    typer.echo(f"{row.seed}\t{row.train}\t{row.eval}\t{row.run}\t{run_dir}")


@seeds_app.command("status")
def seed_status(
    config_or_snapshot: Path | None = typer.Argument(
        None,
        help="ExperimentConfig YAML/JSON or materialized preset snapshot.",
    ),
    group: str | None = typer.Option(None, "--group"),
    name: str | None = typer.Option(None, "--name"),
    source: Literal["local", "wandb"] = typer.Option("local", "--source"),
    wandb_project: str | None = typer.Option(None, "--wandb-project"),
    wandb_entity: str | None = typer.Option(None, "--wandb-entity"),
    wandb_mode: WandbMode | None = typer.Option(None, "--wandb-mode"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Show seed-level and stage-level status for declared seeds."""
    from drift_happens.experiments.source import load_experiment_source
    from drift_happens.runtime.completion_filter import (
        local_seed_statuses_by_identity,
        wandb_seed_statuses_by_identity,
    )
    from drift_happens.runtime.run_store import resolve_run_store

    path = _resolve_status_path(config_or_snapshot, group=group, name=name)
    experiment_source = load_experiment_source(path)
    cfg = with_wandb_from_env(
        experiment_source.config,
        project=wandb_project,
        entity=wandb_entity,
        mode=wandb_mode,
    )
    identities = {
        declared_seed: resolve_run_store(
            cfg.model_copy(update={"seed": declared_seed}),
            source_path=experiment_source.path,
            runs_root=runs_root,
        ).identity
        for declared_seed in experiment_source.seeds
    }
    if source == "wandb":
        wandb_cfg = cfg.logging.wandb
        if wandb_cfg is None or wandb_cfg.mode == "disabled":
            raise typer.BadParameter(
                "W&B status requires cfg.logging.wandb, WANDB_PROJECT, "
                "or --wandb-project"
            )
        typer.echo("seed\ttrain\teval\trun")
        for row in wandb_seed_statuses_by_identity(identities, wandb_cfg=wandb_cfg):
            typer.echo(f"{row.seed}\t{row.train}\t{row.eval}\t{row.status}")
        return

    typer.echo("seed\ttrain\teval\trun\trun_dir")
    for row in local_seed_statuses_by_identity(identities, runs_root=runs_root):
        run_dir = relative_to_project(row.run_dir) if row.run_dir is not None else ""
        typer.echo(f"{row.seed}\t{row.train}\t{row.eval}\t{row.status}\t{run_dir}")


@seeds_app.command("summarize")
def summarize_seed_runs(
    config_or_snapshot: Path | None = typer.Argument(
        None,
        help="ExperimentConfig YAML/JSON or materialized preset snapshot.",
    ),
    group: str | None = typer.Option(None, "--group"),
    name: str | None = typer.Option(None, "--name"),
    metric: str | None = typer.Option(None, "--metric"),
    out_dir: Path | None = typer.Option(None, "--out-dir"),
    write_csv: bool = typer.Option(False, "--csv/--no-csv"),
    write_markdown: bool = typer.Option(False, "--markdown/--no-markdown"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Aggregate completed local seed run summaries."""
    from drift_happens.experiments.seed_summary import summarize_seeds
    from drift_happens.experiments.source import load_experiment_source

    path = _resolve_status_path(config_or_snapshot, group=group, name=name)
    experiment_source = load_experiment_source(path)
    report = summarize_seeds(
        experiment_source.config,
        source_path=path.resolve(),
        seeds=experiment_source.seeds,
        out_dir=out_dir,
        runs_root=runs_root,
        metric=metric,
        write_csv=write_csv,
        write_markdown=write_markdown,
    )
    typer.echo(relative_to_project(report))


def _execute_stage_command(
    stage: RunStage,
    *,
    config_or_snapshot: Path,
    seed: int | None,
    set_overrides: list[str] | None,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_mode: WandbMode | None,
    wandb_tags: str | None,
    wandb_upload_artifacts: bool | None,
    wandb_upload_checkpoints: bool | None,
    skip_completed: bool,
    resume: bool | None,
    allow_overwrite: bool,
    runs_root: Path | None,
) -> None:
    from drift_happens.runtime.local import run_stage
    from drift_happens.runtime.run_store import resolve_run_store
    from drift_happens.runtime.stage_status import inspect_run_status
    from drift_happens.runtime.stages import STAGE_CONTENTION_EXIT_CODE

    cfg, source_path = _load_cfg(
        config_or_snapshot,
        seed=seed,
        set_overrides=set_overrides,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        wandb_tags=wandb_tags,
        wandb_upload_artifacts=wandb_upload_artifacts,
        wandb_upload_checkpoints=wandb_upload_checkpoints,
    )
    resolved_resume = resolve_resume_setting(resume)
    store = resolve_run_store(cfg, source_path=source_path, runs_root=runs_root)
    preflight_status = _wandb_preflight_status(
        cfg,
        store,
        stage=stage,
        max_failed_attempts=WANDB_PREFLIGHT_MAX_FAILED_ATTEMPTS,
    )
    if _wandb_status_skips(
        preflight_status,
        cfg=cfg,
        stage=stage,
    ):
        return
    if skip_completed:
        status = inspect_run_status(
            run_dir=store.run_dir,
            identity=store.identity,
            seed=cfg.seed,
        )
        if (stage == "train" and status.train == "ok") or (
            stage == "eval" and status.eval == "ok"
        ):
            typer.echo(f"skipped complete {stage} stage for seed {cfg.seed}")
            return
    try:
        result = run_stage(
            cfg,
            stage=stage,
            allow_overwrite=allow_overwrite,
            runs_root=runs_root,
            source_path=source_path,
            resume=resolved_resume,
            allowed_wandb_run_ids=_preflight_run_ids(preflight_status),
        )
    except RuntimeError as exc:
        if _is_stage_contention(exc):
            typer.echo(f"skipped {stage} stage for seed {cfg.seed}: stage is running")
            raise typer.Exit(code=STAGE_CONTENTION_EXIT_CODE) from exc
        raise
    typer.echo(relative_to_project(result.run_dir))


def _wandb_preflight_skips(
    cfg: ExperimentConfig,
    store: RunStore,
    *,
    stage: RunStage | None,
    max_failed_attempts: int,
) -> bool:
    """Return True when W&B says this config/seed should not start now."""
    return _wandb_status_skips(
        _wandb_preflight_status(
            cfg,
            store,
            stage=stage,
            max_failed_attempts=max_failed_attempts,
        ),
        cfg=cfg,
        stage=stage,
    )


def _wandb_preflight_status(
    cfg: ExperimentConfig,
    store: RunStore,
    *,
    stage: RunStage | None,
    max_failed_attempts: int,
):
    """Return W&B preflight status, or None when W&B is not active."""
    wandb_cfg = cfg.logging.wandb
    if wandb_cfg is None or wandb_cfg.mode != "online":
        return None

    from drift_happens.utils.wandb_completion import WandbCompletionIndex

    return WandbCompletionIndex(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
    ).preflight_status(
        group=store.identity.wandb_group,
        seed=cfg.seed,
        config_hash=store.identity.config_hash,
        snapshot_sha256=store.identity.snapshot_sha256,
        completion_hash=store.identity.completion_hash,
        max_failed_attempts=max_failed_attempts,
        stage=stage,
    )


def _wandb_status_skips(
    status,
    *,
    cfg: ExperimentConfig,
    stage: RunStage | None,
) -> bool:
    """Print W&B preflight decision and return whether work should be skipped."""
    if status is None:
        return False
    label = f"{stage} stage" if stage is not None else "seed"
    if status.state == "missing":
        return False
    if status.state == "retry":
        typer.echo(
            f"retrying {label} for seed {cfg.seed} after "
            f"{status.failed_attempts}/{status.max_failed_attempts} failed "
            "W&B attempt(s)"
        )
        return False
    if status.state == "running":
        typer.echo(f"skipped {label} for seed {cfg.seed}: matching W&B run in progress")
        return True
    if status.state == "complete":
        typer.echo(f"skipped complete {label} for seed {cfg.seed} from W&B")
        return True
    typer.echo(
        f"skipped {label} for seed {cfg.seed}: "
        f"{status.failed_attempts}/{status.max_failed_attempts} failed W&B "
        "attempt(s)"
    )
    return True


def _preflight_run_ids(status) -> tuple[str, ...] | None:
    if status is None:
        return None
    return tuple(status.run_ids)


def _is_stage_contention(exc: RuntimeError) -> bool:
    """Return True for local concurrency errors that should be treated as skips."""
    message = str(exc)
    return "lock already held:" in message or "train status is 'running'" in message


def _load_cfg(
    config_or_snapshot: Path,
    *,
    seed: int | None = None,
    set_overrides: list[str] | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_mode: WandbMode | None = None,
    wandb_tags: str | None = None,
    wandb_upload_artifacts: bool | None = None,
    wandb_upload_checkpoints: bool | None = None,
) -> tuple[ExperimentConfig, Path]:
    from drift_happens.experiments.source import load_experiment_source

    source = load_experiment_source(
        config_or_snapshot,
        overrides=tuple(set_overrides or ()),
    )
    cfg = source.config
    if seed is not None:
        cfg = cfg.model_copy(update={"seed": seed})
    cfg = with_wandb_from_env(
        cfg,
        project=wandb_project,
        entity=wandb_entity,
        mode=wandb_mode,
        tags=_parse_tags(wandb_tags),
        upload_artifacts=wandb_upload_artifacts,
        upload_checkpoints=wandb_upload_checkpoints,
    )
    return cfg, source.path


def _spawn_stage_process(
    stage: str,
    *,
    config_or_snapshot: Path,
    seed: int | None,
    set_overrides: list[str],
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_mode: WandbMode | None,
    wandb_tags: str | None,
    wandb_upload_artifacts: bool | None,
    wandb_upload_checkpoints: bool | None,
    skip_completed: bool,
    resume: bool,
    allow_overwrite: bool,
    runs_root: Path | None,
) -> int:
    args = [
        sys.executable,
        "-m",
        "drift_happens.cli.main",
        "experiment",
        stage,
        str(config_or_snapshot),
    ]
    if seed is not None:
        args.extend(["--seed", str(seed)])
    for override in set_overrides:
        args.extend(["--set", override])
    if wandb_project is not None:
        args.extend(["--wandb-project", wandb_project])
    if wandb_entity is not None:
        args.extend(["--wandb-entity", wandb_entity])
    if wandb_mode is not None:
        args.extend(["--wandb-mode", wandb_mode])
    if wandb_tags is not None:
        args.extend(["--wandb-tags", wandb_tags])
    if wandb_upload_artifacts is not None:
        args.append(
            "--wandb-upload-artifacts"
            if wandb_upload_artifacts
            else "--no-wandb-upload-artifacts"
        )
    if wandb_upload_checkpoints is not None:
        args.append(
            "--wandb-upload-checkpoints"
            if wandb_upload_checkpoints
            else "--no-wandb-upload-checkpoints"
        )
    if skip_completed:
        args.append("--skip-completed")
    args.append("--resume" if resume else "--no-resume")
    if allow_overwrite:
        args.append("--allow-overwrite")
    if runs_root is not None:
        args.extend(["--runs-root", str(runs_root)])
    return subprocess.run(args, check=False).returncode


def _resolve_status_path(
    path: Path | None,
    *,
    group: str | None,
    name: str | None,
) -> Path:
    if path is not None:
        return path
    if not group or not name:
        raise typer.BadParameter("provide a snapshot path or both --group and --name")
    return PRESETS_ROOT / group / f"{name}.json"


def _seed_complete(
    cfg: ExperimentConfig,
    store: RunStore,
    *,
    runs_root: Path | None,
) -> bool:
    from drift_happens.runtime.completion_filter import (
        local_seed_complete,
        wandb_seed_complete,
    )

    if local_seed_complete(store.identity, seed=cfg.seed, runs_root=runs_root):
        return True
    wandb_cfg = cfg.logging.wandb
    if wandb_cfg is None or wandb_cfg.mode == "disabled":
        return False
    return wandb_seed_complete(store.identity, seed=cfg.seed, wandb_cfg=wandb_cfg)


def _parse_tags(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return tuple(values)


def _parse_optional_int_csv(
    raw: str | None, *, param_name: str
) -> tuple[int, ...] | None:
    if raw is None:
        return None
    try:
        values = _parse_int_csv(raw)
    except ValueError as exc:
        raise typer.BadParameter(
            f"{param_name} must be a comma-separated list of integers"
        ) from exc
    if not values:
        raise typer.BadParameter(f"{param_name} must include at least one integer")
    return values


def _plan_status_device_override(slot: DeviceSlotConfig) -> str:
    if slot.device == "cuda":
        return "runtime.device=cuda"
    if slot.device == "mps":
        return "runtime.device=mps"
    return "runtime.device=cpu"


def _add_plan_status_totals(
    totals: dict[str, int],
    summary: dict[str, int],
) -> None:
    for key in totals:
        totals[key] += summary[key]


def _format_plan_status_table(rows: list[dict[str, str | int]]) -> str:
    import polars as pl

    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=160,
        fmt_str_lengths=80,
    ):
        return str(pl.DataFrame(rows))


def _format_lock_repair_table(
    rows: list[dict[str, str | int | float | None]],
) -> str:
    import polars as pl

    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=220,
        fmt_str_lengths=100,
    ):
        return str(pl.DataFrame(rows))


def _optional_bool(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def _optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 1)


app.add_typer(stages_app, name="stages")
app.add_typer(seeds_app, name="seeds")
app.add_typer(plans_app, name="plans")
app.add_typer(locks_app, name="locks")
