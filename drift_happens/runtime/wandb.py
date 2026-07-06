"""Optional Weights & Biases metric sink and artifact publisher."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from drift_happens.configs import ExperimentConfig, RunIdentity, WandbConfig
from drift_happens.runtime.metrics import MetricRecord, MetricSink
from drift_happens.runtime.stages import RunStage
from drift_happens.utils.ids import slugify

WANDB_STEP_METRIC = "step/global"
WANDB_ARTIFACT_NAME_MAXLEN = 128
_RESUME_METADATA_FILE = "drift_happens_resume.yaml"


@dataclass(slots=True)
class WandbMetricSink(MetricSink):
    """Lazy-imported W&B sink for scalar metrics and curated run artifacts."""

    cfg: ExperimentConfig
    wandb_cfg: WandbConfig
    run_dir: Path
    identity: RunIdentity
    stage: RunStage | None = None
    resume: bool = True
    allowed_resume_run_ids: tuple[str, ...] | None = None
    _run: Any = field(default=None, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        import wandb

        # Only continue a prior local run when resume is requested. A fresh
        # rerun (``--no-resume``) clears the stage ledger but leaves
        # ``run_dir/wandb`` in place, so reusing the discovered id here would
        # silently write new metrics into the old run.
        resumed_run_id = (
            _resumable_wandb_run_id(
                self.run_dir,
                cfg=self.cfg,
                identity=self.identity,
                stage=self.stage,
                allowed_run_ids=self.allowed_resume_run_ids,
            )
            if self.resume
            else None
        )
        if resumed_run_id is not None:
            self._step = _existing_metric_count(self.run_dir, stage=self.stage)

        init_kwargs: dict[str, Any] = {
            "project": self.wandb_cfg.project,
            "entity": self.wandb_cfg.entity,
            "group": self.identity.wandb_group,
            "tags": list((*self.cfg.tags, *self.wandb_cfg.tags)),
            "mode": self.wandb_cfg.mode,
            "name": self.identity.wandb_run_name,
            "config": _wandb_config_payload(
                self.cfg,
                self.identity,
                stage=self.stage,
            ),
            "notes": self.cfg.notes or None,
            "job_type": self.wandb_cfg.job_type,
            "resume": self.wandb_cfg.resume,
            "dir": str(self.run_dir),
            "reinit": "finish_previous",
        }
        if resumed_run_id is not None:
            init_kwargs["id"] = resumed_run_id
            if init_kwargs["resume"] == "never":
                init_kwargs["resume"] = "allow"
        if hasattr(wandb, "Settings"):
            init_kwargs["settings"] = wandb.Settings(console="wrap")
        self._run = wandb.init(
            **init_kwargs,
        )
        _write_wandb_resume_metadata(
            self._run,
            run_dir=self.run_dir,
            cfg=self.cfg,
            identity=self.identity,
            stage=self.stage,
        )
        wandb.define_metric(WANDB_STEP_METRIC)
        wandb.define_metric("*", step_metric=WANDB_STEP_METRIC)

    def log(self, record: MetricRecord) -> None:
        if self._run is None:
            return
        self._step += 1
        payload: dict[str, Any] = {
            WANDB_STEP_METRIC: self._step,
            record.metric: record.value,
            "run/seed": record.seed,
        }
        if record.step is not None:
            payload["step/local"] = record.step
        if record.epoch is not None:
            payload["epoch"] = record.epoch
        if record.train_slice is not None:
            payload["train_slice"] = record.train_slice
        if record.eval_slice is not None:
            payload["eval_slice"] = record.eval_slice
        payload.update(_scalar_context(record.context))
        self._run.log(payload)

    def close(self, exit_code: int | None = None) -> None:
        if self._run is None:
            return
        try:
            if self.wandb_cfg.upload_artifacts and self.wandb_cfg.mode != "disabled":
                self._log_run_artifact()
        finally:
            self._run.finish(exit_code=exit_code)
            self._run = None

    def _log_run_artifact(self) -> None:
        if self._run is None:
            return
        import wandb

        artifact = wandb.Artifact(
            name=_run_artifact_name(self.cfg, self.identity),
            type="run",
            metadata={
                "run_dir": str(self.run_dir),
                **self.identity.model_dump(mode="json"),
            },
        )
        added = False
        for path in curated_run_artifact_files(
            self.run_dir,
            upload_checkpoints=self.wandb_cfg.upload_checkpoints,
        ):
            artifact.add_file(str(path), name=str(path.relative_to(self.run_dir)))
            added = True
        if added:
            self._run.log_artifact(
                artifact,
                aliases=list(self.wandb_cfg.artifact_aliases) or None,
            )


def curated_run_artifact_files(
    run_dir: Path,
    *,
    upload_checkpoints: bool = False,
) -> tuple[Path, ...]:
    """Return curated files safe to attach to a W&B run artifact."""
    candidates: list[Path] = [
        run_dir / "snapshot.json",
        run_dir / "metadata.json",
        run_dir / "run_manifest.json",
        run_dir / "config.input.json",
        run_dir / "config.input.yaml",
        run_dir / "config.input.yml",
        run_dir / "training_history.json",
        run_dir / "logs" / "train.console.log",
        run_dir / "logs" / "eval.console.log",
        run_dir / "logs" / "events.jsonl",
        run_dir / "results" / "summary.json",
    ]
    candidates.extend(sorted((run_dir / "metrics").glob("*.jsonl")))
    candidates.extend(sorted((run_dir / "stages").glob("*/*.json")))
    candidates.extend(sorted((run_dir / "stages").glob("*/*/*.json")))
    candidates.extend(sorted((run_dir / "results").glob("drift_matrix.*")))
    if upload_checkpoints:
        candidates.extend(sorted((run_dir / "checkpoints").glob("*")))
        candidates.extend(
            sorted((run_dir / "stages" / "train" / "checkpoints").glob("*"))
        )
        candidates.extend(
            sorted(
                (run_dir / "stages" / "train").glob("*/train_slice_*/trained_model.pt")
            )
        )
    root = run_dir.resolve()
    out: list[Path] = []
    for path in candidates:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            continue
        out.append(path)
    return tuple(dict.fromkeys(out))


def _wandb_config_payload(
    cfg: ExperimentConfig,
    identity: RunIdentity,
    *,
    stage: RunStage | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "config": cfg.model_dump(mode="json"),
        "seed": cfg.seed,
        "run/config_hash": identity.config_hash,
        "run/snapshot_sha256": identity.snapshot_sha256,
        "run/source_identity": identity.source_identity,
        "run/wandb_group": identity.wandb_group,
    }
    if identity.completion_hash is not None:
        payload["run/completion_hash"] = identity.completion_hash
    if stage is not None:
        payload["run/stage"] = stage
    return payload


def _resumable_wandb_run_id(
    run_dir: Path,
    *,
    cfg: ExperimentConfig,
    identity: RunIdentity,
    stage: RunStage | None,
    allowed_run_ids: tuple[str, ...] | None = None,
) -> str | None:
    """Return a local W&B run id that can safely be resumed."""
    allowed = set(allowed_run_ids) if allowed_run_ids is not None else None
    candidates = tuple(
        (child, run_id)
        for child, run_id in _local_wandb_runs(run_dir)
        if allowed is None or run_id in allowed
    )
    if not candidates:
        return None

    matching: list[tuple[str, str]] = []
    unresolved: list[tuple[str, str]] = []
    stage_matching_unresolved: list[tuple[str, str]] = []
    for child, run_id in candidates:
        run_order = _wandb_run_order(child)
        # Prefer our own resume marker, then W&B's own config dump. Parsing the
        # launch args from ``wandb-metadata.json`` is a last resort for legacy
        # runs that predate both files.
        config = _read_wandb_config(child / "files" / _RESUME_METADATA_FILE)
        if config is None:
            config = _read_wandb_config(child / "files" / "config.yaml")
        if config is None:
            unresolved.append((run_order, run_id))
            if _wandb_metadata_matches(child, cfg=cfg, stage=stage):
                stage_matching_unresolved.append((run_order, run_id))
            continue
        if _wandb_config_matches(config, cfg=cfg, identity=identity, stage=stage):
            matching.append((run_order, run_id))

    # Fold duplicates back into the earliest same-stage run so repeated reruns
    # converge on the original W&B run even if a later retry created a duplicate
    # with a stronger resume marker.
    same_stage_runs = [*matching, *stage_matching_unresolved]
    if same_stage_runs:
        return _earliest_run_id(same_stage_runs)
    if len(candidates) == 1 and unresolved:
        return unresolved[0][1]
    return None


# W&B names local run directories ``run-<ts>-<id>`` online and
# ``offline-run-<ts>-<id>`` offline; both layouts must be resumable.
_WANDB_RUN_DIR_PREFIXES = ("run-", "offline-run-")


def _local_wandb_runs(run_dir: Path) -> tuple[tuple[Path, str], ...]:
    wandb_root = run_dir / "wandb"
    if not wandb_root.exists():
        return ()
    runs_by_id: dict[str, Path] = {}
    for child in sorted(wandb_root.iterdir()):
        if child.is_symlink():
            continue
        if not child.is_dir() or not child.name.startswith(_WANDB_RUN_DIR_PREFIXES):
            continue
        run_id = _run_id_from_wandb_dir(child)
        if run_id is not None:
            runs_by_id.setdefault(run_id, child)
    return tuple((child, run_id) for run_id, child in runs_by_id.items())


def _run_id_from_wandb_dir(path: Path) -> str | None:
    for wandb_file in sorted(path.glob("run-*.wandb")):
        run_id = wandb_file.stem.removeprefix("run-")
        if run_id:
            return run_id
    if path.name.startswith(_WANDB_RUN_DIR_PREFIXES):
        run_id = path.name.rsplit("-", 1)[-1]
        if run_id:
            return run_id
    return None


def _read_wandb_config(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return None
    return payload if isinstance(payload, dict) else None


def _wandb_config_matches(
    config: dict[str, Any],
    *,
    cfg: ExperimentConfig,
    identity: RunIdentity,
    stage: RunStage | None,
) -> bool:
    if _wandb_config_value(config, "seed") != cfg.seed:
        return False
    if _wandb_config_value(config, "run/config_hash") != identity.config_hash:
        return False
    if _wandb_config_value(config, "run/snapshot_sha256") != identity.snapshot_sha256:
        return False
    if identity.completion_hash is not None and (
        _wandb_config_value(config, "run/completion_hash") != identity.completion_hash
    ):
        return False
    if stage is not None and _wandb_config_value(config, "run/stage") != stage:
        return False
    return True


def _wandb_config_value(config: dict[str, Any], key: str) -> Any:
    raw = config.get(key)
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"]
    return raw


def _write_wandb_resume_metadata(
    run: Any,
    *,
    run_dir: Path,
    cfg: ExperimentConfig,
    identity: RunIdentity,
    stage: RunStage | None,
) -> None:
    files_dir = _wandb_run_files_dir(run, run_dir=run_dir)
    if files_dir is None:
        return
    files_dir.mkdir(parents=True, exist_ok=True)
    (files_dir / _RESUME_METADATA_FILE).write_text(
        yaml.safe_dump(
            _wandb_config_payload(cfg, identity, stage=stage),
            sort_keys=True,
        )
    )


def _wandb_run_files_dir(run: Any, *, run_dir: Path) -> Path | None:
    raw_dir = getattr(run, "dir", None)
    if isinstance(raw_dir, str | Path):
        path = Path(raw_dir)
        return path if path.name == "files" else path / "files"

    run_id = getattr(run, "id", None)
    if not isinstance(run_id, str) or not run_id:
        return None
    for child, child_run_id in _local_wandb_runs(run_dir):
        if child_run_id == run_id:
            return child / "files"
    return None


def _wandb_metadata_matches(
    child: Path,
    *,
    cfg: ExperimentConfig,
    stage: RunStage | None,
) -> bool:
    metadata = _read_wandb_config(child / "files" / "wandb-metadata.json")
    if metadata is None:
        return False
    args = metadata.get("args")
    if not isinstance(args, list):
        return False
    normalized_args = [str(arg) for arg in args]
    if _metadata_seed(normalized_args) != cfg.seed:
        return False
    if stage is None:
        return True
    return _metadata_stage(normalized_args) == stage


def _metadata_seed(args: list[str]) -> int | None:
    for index, arg in enumerate(args):
        if arg == "--seed" and index + 1 < len(args):
            try:
                return int(args[index + 1])
            except ValueError:
                return None
    return None


def _metadata_stage(args: list[str]) -> str | None:
    for index, arg in enumerate(args):
        if arg != "experiment" or index + 1 >= len(args):
            continue
        action = args[index + 1]
        if action in {"train", "eval"}:
            return action
    return None


def _earliest_run_id(runs: list[tuple[str, str]]) -> str:
    return sorted(runs)[0][1]


def _wandb_run_order(path: Path) -> str:
    return path.name


def _existing_metric_count(run_dir: Path, *, stage: RunStage | None) -> int:
    # Resume the ``step/global`` counter from the stage's own metric series.
    # Cross-phase ``summary`` records also advance the step but live in a
    # shared ``summary.jsonl``, so they are intentionally not counted here; the
    # custom step metric keeps those on separate series, so the slight
    # undercount does not collide with the resumed train/eval points.
    if stage is None:
        return 0
    path = run_dir / "metrics" / f"{stage}.jsonl"
    try:
        with path.open() as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


_SEP = "__"
_ARTIFACT_PREFIX = "drift-run"


def _run_artifact_name(cfg: ExperimentConfig, identity: RunIdentity) -> str:
    """Return a W&B artifact name within the service length limit."""
    suffix = (
        f"seed-{cfg.seed}{_SEP}cfg-{identity.config_hash[:12]}"
        f"{_SEP}snap-{identity.snapshot_sha256[:12]}"
    )
    budget = (
        WANDB_ARTIFACT_NAME_MAXLEN - len(_ARTIFACT_PREFIX) - len(suffix) - 2 * len(_SEP)
    )
    source = slugify(identity.source_identity)[: max(0, budget)].strip("-_.")
    parts = [_ARTIFACT_PREFIX]
    if source:
        parts.append(source)
    parts.append(suffix)
    name = _SEP.join(parts)
    if len(name) > WANDB_ARTIFACT_NAME_MAXLEN:
        raise ValueError(f"W&B artifact name exceeds {WANDB_ARTIFACT_NAME_MAXLEN}")
    return name


def _scalar_context(context: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in context.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[str(key)] = value
    return payload
