"""Stable W&B identity helpers for one-seed experiment runs."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from drift_happens.configs import ExperimentConfig, RunIdentity
from drift_happens.utils.ids import slugify
from drift_happens.utils.snapshot import SNAPSHOT_KIND

_COMPLETION_HASH_VERSION = "experiment-completion/v1"


def source_wandb_identity(source_path: Path) -> str | None:
    """Return the preset identity encoded by a materialized snapshot path."""
    path = Path(source_path)
    if path.suffix.lower() != ".json":
        return None
    try:
        payload: Any = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("kind") != SNAPSHOT_KIND:
        return None
    group = payload.get("group")
    name = payload.get("name")
    if isinstance(group, str) and isinstance(name, str):
        return _join_identity(group, name)
    return None


def config_hash(cfg: ExperimentConfig) -> str:
    """Hash the full resolved config minus logging.wandb (all other fields, including
    notes/tags/metadata, affect the hash)."""
    semantic_cfg = cfg.model_copy(
        update={"logging": cfg.logging.model_copy(update={"wandb": None})}
    )
    return sha256(semantic_cfg.to_snapshot_json().encode("utf-8")).hexdigest()


def completion_hash(cfg: ExperimentConfig) -> str:
    """Hash fields that affect one seed's train/eval completion."""
    payload = _completion_payload(cfg)
    body = json.dumps(
        {"version": _COMPLETION_HASH_VERSION, "config": payload},
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(body.encode("utf-8")).hexdigest()


def completion_hash_from_config_payload(payload: Any) -> str | None:
    """Return the completion hash for a W&B-stored config payload, if valid."""
    if not isinstance(payload, dict):
        return None
    try:
        cfg = ExperimentConfig.model_validate(payload)
    except ValueError:
        return None
    return completion_hash(cfg)


def snapshot_sha256(source_path: Path | None, cfg: ExperimentConfig) -> str:
    """Hash the input snapshot/config file when available, else the config."""
    if source_path is None:
        return config_hash(cfg)
    try:
        return sha256(Path(source_path).read_bytes()).hexdigest()
    except FileNotFoundError:
        return config_hash(cfg)


def wandb_group(
    cfg: ExperimentConfig,
    *,
    source_identity: str | None = None,
    cfg_hash: str | None = None,
) -> str:
    """Return the effective W&B group for comparable seed replicas."""
    if cfg.logging.wandb is not None and cfg.logging.wandb.group is not None:
        return cfg.logging.wandb.group
    if source_identity is not None:
        return source_identity
    return default_wandb_group(cfg, cfg_hash=cfg_hash)


def wandb_run_name(
    cfg: ExperimentConfig,
    *,
    seed: int,
    run_dir: Path,
    source_identity: str | None = None,
    cfg_hash: str | None = None,
) -> str:
    """Return a W&B run name that includes group, seed, and local run leaf."""
    base = (
        cfg.logging.wandb.run_name
        if cfg.logging.wandb is not None and cfg.logging.wandb.run_name is not None
        else wandb_group(cfg, source_identity=source_identity, cfg_hash=cfg_hash)
    )
    return f"{base}__seed={seed}__{run_dir.name}"


def build_run_identity(
    cfg: ExperimentConfig,
    *,
    run_dir: Path,
    source_path: Path | None = None,
) -> RunIdentity:
    """Build the stable identity record stored in metadata and W&B config."""
    source_identity = source_wandb_identity(source_path) if source_path else None
    cfg_hash = config_hash(cfg)
    group = wandb_group(cfg, source_identity=source_identity, cfg_hash=cfg_hash)
    return RunIdentity(
        source_identity=source_identity or group,
        config_hash=cfg_hash,
        completion_hash=completion_hash(cfg),
        snapshot_sha256=snapshot_sha256(source_path, cfg),
        wandb_group=group,
        wandb_run_name=wandb_run_name(
            cfg,
            seed=cfg.seed,
            run_dir=run_dir,
            source_identity=source_identity,
            cfg_hash=cfg_hash,
        ),
    )


def default_wandb_group(
    cfg: ExperimentConfig,
    *,
    cfg_hash: str | None = None,
) -> str:
    """Return a deterministic fallback group for ad-hoc configs."""
    variant = cfg.dataset.variant or "default"
    family = cfg.trainer.family or cfg.trainer.key
    digest = (cfg_hash or config_hash(cfg))[:12]
    return _join_identity(
        cfg.dataset.name,
        variant,
        family,
        cfg.trainer.key,
        cfg.name,
        digest,
    )


def _join_identity(*parts: str) -> str:
    return "__".join(slugify(part) for part in parts if part)


def _completion_payload(cfg: ExperimentConfig) -> dict[str, Any]:
    payload = cfg.model_dump(mode="json")
    for key in ("logging", "metadata", "name", "notes", "protocol"):
        payload.pop(key, None)

    runtime = payload.get("runtime")
    if isinstance(runtime, dict):
        for key in ("backend", "device", "num_threads"):
            runtime.pop(key, None)

    preprocessing = payload.get("preprocessing")
    if isinstance(preprocessing, dict):
        cache = preprocessing.get("cache")
        if isinstance(cache, dict):
            for key in ("cache_id", "reuse_policy"):
                cache.pop(key, None)

    return payload
