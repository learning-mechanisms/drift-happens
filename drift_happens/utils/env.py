from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

from drift_happens.configs import ExperimentConfig, WandbConfig
from drift_happens.configs.logging_cfg import WandbMode
from drift_happens.utils.log import get_logger

_LOADED_ENVFILE = False
RUN_RESUME_ENV = "DRIFT_ENABLE_AUTO_RESUME"
RESUME_CHECKPOINTS_ENV = "DRIFT_RESUME_CHECKPOINTS"


def load_envfile(path: Path | None = None) -> bool:
    """Load environment variables from a .env file."""
    global _LOADED_ENVFILE
    if _LOADED_ENVFILE:
        return False
    _LOADED_ENVFILE = True
    return bool(load_dotenv(path or Path(__file__).parent.parent.parent / ".env"))


def with_wandb_from_env(
    cfg: ExperimentConfig,
    *,
    project: str | None = None,
    entity: str | None = None,
    mode: WandbMode | None = None,
    tags: tuple[str, ...] = (),
    upload_artifacts: bool | None = None,
    upload_checkpoints: bool | None = None,
) -> ExperimentConfig:
    """
    Apply W&B CLI/env settings at the command boundary.

    Environment variables only populate W&B when the config has no explicit W&B block.
    CLI arguments passed to this function are treated as explicit overrides.
    """
    existing = cfg.logging.wandb
    if existing is not None:
        updates: dict[str, object] = {}
        if project is not None:
            updates["project"] = project
        if entity is not None:
            updates["entity"] = entity
        if mode is not None:
            updates["mode"] = mode
        if tags:
            updates["tags"] = tags
        if upload_artifacts is not None:
            updates["upload_artifacts"] = upload_artifacts
        if upload_checkpoints is not None:
            updates["upload_checkpoints"] = upload_checkpoints
        if not updates:
            return cfg
        return _replace_wandb(cfg, existing.model_copy(update=updates))

    resolved_project = project or os.getenv("WANDB_PROJECT")
    if not resolved_project:
        return cfg

    kwargs: dict[str, Any] = {"project": resolved_project}
    resolved_entity = entity or os.getenv("WANDB_ENTITY") or None
    if resolved_entity:
        kwargs["entity"] = resolved_entity
    resolved_mode = mode or _env_mode()
    if resolved_mode:
        kwargs["mode"] = resolved_mode
    resolved_tags = tags or _env_tags()
    if resolved_tags:
        kwargs["tags"] = resolved_tags
    resolved_upload_artifacts = (
        upload_artifacts
        if upload_artifacts is not None
        else _env_bool("WANDB_UPLOAD_ARTIFACTS")
    )
    if resolved_upload_artifacts is not None:
        kwargs["upload_artifacts"] = resolved_upload_artifacts
    resolved_upload_checkpoints = (
        upload_checkpoints
        if upload_checkpoints is not None
        else _env_bool("WANDB_UPLOAD_CHECKPOINTS")
    )
    if resolved_upload_checkpoints is not None:
        kwargs["upload_checkpoints"] = resolved_upload_checkpoints
    wandb_cfg = WandbConfig(**kwargs)
    return _replace_wandb(cfg, wandb_cfg)


def _replace_wandb(cfg: ExperimentConfig, wandb: WandbConfig) -> ExperimentConfig:
    return cfg.model_copy(
        update={"logging": cfg.logging.model_copy(update={"wandb": wandb})}
    )


def _env_mode() -> WandbMode | None:
    raw = os.getenv("WANDB_MODE")
    if not raw:
        return None
    if raw in {"online", "offline", "disabled"}:
        return cast(WandbMode, raw)
    raise ValueError(f"WANDB_MODE must be online, offline, or disabled; got {raw!r}")


def _env_tags() -> tuple[str, ...]:
    raw = os.getenv("WANDB_TAGS")
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def resolve_resume_setting(resume: bool | None) -> bool:
    """
    Resolve the resume setting for a run.

    Reusing completed work units is the default: a plain re-run skips finished
    units instead of clearing them. An explicit ``--resume``/``--no-resume``
    wins; otherwise ``DRIFT_ENABLE_AUTO_RESUME`` can override the default (set it
    to ``0`` to force fresh re-runs that clear and recompute owned outputs).
    """
    if resume is not None:
        return resume
    env = _env_bool(RUN_RESUME_ENV)
    if env is not None:
        return env
    return True


def resume_checkpoints_enabled() -> bool:
    """
    Return whether an unfinished slice may resume from its epoch checkpoint.

    This is off by default: a slice that did not finish retrains from epoch 0
    rather than continuing from a mid-training checkpoint. Set
    ``DRIFT_RESUME_CHECKPOINTS=1`` (or pass ``--resume-checkpoints``) to opt in.
    Completed slices are unaffected; they are reused regardless of this setting.
    """
    return bool(_env_bool(RESUME_CHECKPOINTS_ENV))


def apply_resume_checkpoints_override(resume_checkpoints: bool | None) -> None:
    """
    Persist an explicit ``--resume-checkpoints`` choice into the environment.

    Spawned stage subprocesses and sweep workers inherit the parent environment, so
    writing the env var here propagates the choice to them. A ``None`` flag (omitted)
    leaves any existing ``DRIFT_RESUME_CHECKPOINTS`` value in place.
    """
    if resume_checkpoints is None:
        return
    os.environ[RESUME_CHECKPOINTS_ENV] = "1" if resume_checkpoints else "0"


def huggingface_token_from_env() -> str | None:
    """Return the configured Hugging Face token without loading additional files."""
    return os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")


def ensure_huggingface_auth_from_env(*, required: bool) -> str | None:
    """
    Log in to Hugging Face from command-boundary environment when requested.

    Model constructors should not call this helper; runtime or CLI code can call it
    immediately before creating a gated model.
    """
    token = huggingface_token_from_env()
    if not token:
        if required:
            raise RuntimeError(
                "Hugging Face credentials are required; set HUGGINGFACE_TOKEN "
                "or HF_TOKEN"
            )
        return None
    from huggingface_hub import login

    login(token=token)
    return token


def resolve_huggingface_revision(repo_id: str) -> str | None:
    """
    Resolve the commit sha a Hugging Face model repo currently points at.

    Folding the resolved revision into a feature cache's identity lets a changed model
    revision invalidate a stale cache instead of serving it under the same name. Returns
    None (with a warning) when the revision cannot be resolved -- e.g. offline -- so the
    cache falls back to its unpinned identity rather than failing the run.
    """
    from huggingface_hub import model_info

    try:
        return model_info(repo_id).sha
    except Exception as error:
        get_logger().warning(
            f"Could not resolve the Hugging Face revision for {repo_id!r} ({error}); "
            "the feature cache will not pin the model revision."
        )
        return None
