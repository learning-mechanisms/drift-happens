"""Stable local run-directory resolution and manifests."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from drift_happens.configs import ExperimentConfig, RunIdentity
from drift_happens.runtime.stages import RunStage, write_json_atomic
from drift_happens.utils import paths
from drift_happens.utils.git import GitState
from drift_happens.utils.ids import slugify, utc_timestamp
from drift_happens.utils.snapshot import write_snapshot
from drift_happens.utils.wandb_identity import (
    completion_hash,
    config_hash,
    snapshot_sha256,
    source_wandb_identity,
    wandb_group,
    wandb_run_name,
)


@dataclass(frozen=True, slots=True)
class RunStore:
    """Resolved storage for one config/seed run."""

    cfg: ExperimentConfig
    run_dir: Path
    identity: RunIdentity
    source_path: Path | None = None

    def ensure_base(self, *, allow_overwrite: bool = False) -> None:
        """Create the canonical run dir and write stable base artifacts."""
        if allow_overwrite and self.run_dir.exists():
            # Clear the contents but keep .locks/: the caller holds the stage
            # lock inside that directory, and removing it would hand the same
            # run to a second concurrent runner.
            for child in self.run_dir.iterdir():
                if child.name == ".locks":
                    continue
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        (self.run_dir / "metrics").mkdir(exist_ok=True)
        (self.run_dir / "results").mkdir(exist_ok=True)
        write_snapshot(self.run_dir / "snapshot.json", self.cfg)
        self._copy_input_config()
        write_json_atomic(self.run_dir / "run_manifest.json", self.manifest_payload())

    def attempt_dir(
        self,
        *,
        stage: RunStage,
        started_at: datetime,
        git: GitState,
    ) -> Path:
        """Create and return a timestamped attempt directory for stage logs."""
        suffix = utc_timestamp(started_at)
        if git.short_commit:
            suffix = f"{suffix}__{slugify(git.short_commit)}"
        parent = self.run_dir / "attempts" / stage
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / suffix
        counter = 1
        while True:
            try:
                path.mkdir()
                return path
            except FileExistsError:
                path = parent / f"{suffix}__{counter:03d}"
                counter += 1

    def stage_run_identity(self, stage: RunStage) -> RunIdentity:
        """Return a W&B identity copy with a stage-specific run name."""
        wandb_cfg = self.cfg.logging.wandb
        base = (
            wandb_cfg.run_name
            if wandb_cfg is not None and wandb_cfg.run_name is not None
            else self.identity.wandb_group
        )
        return self.identity.model_copy(
            update={"wandb_run_name": f"{base}__seed={self.cfg.seed}__{stage}"}
        )

    def manifest_payload(self) -> dict[str, object]:
        """Return a compact stable identity manifest for the canonical run."""
        return {
            "schema_version": 1,
            "run_dir": str(self.run_dir),
            "source_path": str(self.source_path) if self.source_path else None,
            "experiment": self.cfg.name,
            "dataset": self.cfg.dataset.name,
            "dataset_variant": self.cfg.dataset.variant,
            "trainer": self.cfg.trainer.key,
            "trainer_family": self.cfg.trainer.family,
            "seed": self.cfg.seed,
            "identity": self.identity.model_dump(mode="json"),
        }

    def _copy_input_config(self) -> None:
        if self.source_path is None:
            return
        suffix = self.source_path.suffix.lower()
        if suffix not in {".json", ".yaml", ".yml"}:
            return
        target = self.run_dir / f"config.input{suffix}"
        if target.exists() and target.read_bytes() == self.source_path.read_bytes():
            return
        shutil.copy2(self.source_path, target)


def resolve_run_store(
    cfg: ExperimentConfig,
    *,
    source_path: Path | None = None,
    runs_root: Path | None = None,
) -> RunStore:
    """Resolve the canonical run directory for a config and seed."""
    source = Path(source_path).resolve() if source_path is not None else None
    cfg_hash = config_hash(cfg)
    snap_hash = snapshot_sha256(source, cfg)
    source_identity = source_wandb_identity(source) if source is not None else None
    group = wandb_group(cfg, source_identity=source_identity, cfg_hash=cfg_hash)
    root = runs_root or paths.RUNS_DIR
    run_leaf = _run_leaf(
        source_identity=source_identity, group=group, cfg_hash=cfg_hash
    )
    run_dir = (
        root
        / slugify(cfg.dataset.name)
        / slugify(cfg.trainer.key)
        / slugify(cfg.name)
        / f"seed={cfg.seed}"
        / run_leaf
    )
    identity = RunIdentity(
        source_identity=source_identity or group,
        config_hash=cfg_hash,
        completion_hash=completion_hash(cfg),
        snapshot_sha256=snap_hash,
        wandb_group=group,
        wandb_run_name=wandb_run_name(
            cfg,
            seed=cfg.seed,
            run_dir=run_dir,
            source_identity=source_identity,
            cfg_hash=cfg_hash,
        ),
    )
    return RunStore(cfg=cfg, run_dir=run_dir, identity=identity, source_path=source)


def _run_leaf(
    *,
    source_identity: str | None,
    group: str,
    cfg_hash: str,
) -> str:
    label = source_identity or group
    return slugify(f"{label}__cfg={cfg_hash[:12]}")
