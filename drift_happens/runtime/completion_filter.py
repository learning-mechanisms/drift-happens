"""Local completion helpers for experiment seeds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from drift_happens.configs import RunIdentity, WandbConfig
from drift_happens.runtime.stage_status import (
    StageStatus as SeedStatus,
)
from drift_happens.runtime.stage_status import (
    local_run_complete,
    local_run_statuses_by_identity,
)
from drift_happens.utils.wandb_completion import WandbApiFactory, WandbCompletionIndex


@dataclass(frozen=True, slots=True)
class SeedStatusRow:
    """Completion status for one seed replica."""

    seed: int
    status: SeedStatus
    train: SeedStatus
    eval: SeedStatus
    run_dir: Path | None = None


def local_seed_complete(
    identity: RunIdentity,
    *,
    seed: int,
    runs_root: Path | None = None,
) -> bool:
    """Return whether a matching local staged run completed successfully."""
    return local_run_complete(identity, seed=seed, runs_root=runs_root)


def local_seed_statuses_by_identity(
    identities: dict[int, RunIdentity],
    *,
    runs_root: Path | None = None,
) -> tuple[SeedStatusRow, ...]:
    """Return seed-level rows backed by train/eval stage status."""
    rows = local_run_statuses_by_identity(identities, runs_root=runs_root)
    return tuple(
        SeedStatusRow(
            seed=row.seed,
            status=row.run,
            train=row.train,
            eval=row.eval,
            run_dir=row.run_dir,
        )
        for row in rows
    )


def wandb_seed_complete(
    identity: RunIdentity,
    *,
    seed: int,
    wandb_cfg: WandbConfig,
) -> bool:
    """Return whether W&B has a finished matching complete seed run."""
    index = WandbCompletionIndex(project=wandb_cfg.project, entity=wandb_cfg.entity)
    return index.is_run_complete(
        group=identity.wandb_group,
        seed=seed,
        config_hash=identity.config_hash,
        snapshot_sha256=identity.snapshot_sha256,
        completion_hash=identity.completion_hash,
    )


def wandb_seed_statuses_by_identity(
    identities: dict[int, RunIdentity],
    *,
    wandb_cfg: WandbConfig,
    api_factory: WandbApiFactory | None = None,
) -> tuple[SeedStatusRow, ...]:
    """Return stage-aware seed rows backed by W&B completion predicates."""
    index = WandbCompletionIndex(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        **({"api_factory": api_factory} if api_factory is not None else {}),
    )
    rows: list[SeedStatusRow] = []
    for seed, identity in identities.items():
        train: SeedStatus = (
            "ok"
            if index.is_stage_complete(
                group=identity.wandb_group,
                seed=seed,
                stage="train",
                config_hash=identity.config_hash,
                snapshot_sha256=identity.snapshot_sha256,
                completion_hash=identity.completion_hash,
            )
            else "missing"
        )
        eval_status: SeedStatus = (
            "ok"
            if index.is_stage_complete(
                group=identity.wandb_group,
                seed=seed,
                stage="eval",
                config_hash=identity.config_hash,
                snapshot_sha256=identity.snapshot_sha256,
                completion_hash=identity.completion_hash,
            )
            else "missing"
        )
        run: SeedStatus = (
            "ok"
            if index.is_run_complete(
                group=identity.wandb_group,
                seed=seed,
                config_hash=identity.config_hash,
                snapshot_sha256=identity.snapshot_sha256,
                completion_hash=identity.completion_hash,
            )
            else "missing"
        )
        if run != "ok" and train == "ok" and eval_status == "ok":
            run = "partial"
        rows.append(
            SeedStatusRow(
                seed=seed,
                status=run,
                train=train,
                eval=eval_status,
            )
        )
    return tuple(rows)
