from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from drift_happens.configs import WandbConfig
from drift_happens.experiments.registry import preset
from drift_happens.utils.env import (
    RESUME_CHECKPOINTS_ENV,
    RUN_RESUME_ENV,
    apply_resume_checkpoints_override,
    ensure_huggingface_auth_from_env,
    huggingface_token_from_env,
    resolve_huggingface_revision,
    resolve_resume_setting,
    resume_checkpoints_enabled,
    with_wandb_from_env,
)


def test_resolve_huggingface_revision_returns_the_commit_sha(monkeypatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub.model_info",
        lambda repo_id, *args, **kwargs: SimpleNamespace(sha="deadbeefcafe"),
    )
    assert resolve_huggingface_revision("roberta-base") == "deadbeefcafe"


def test_resolve_huggingface_revision_returns_none_when_unresolvable(
    monkeypatch,
) -> None:
    def _raise(repo_id, *args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr("huggingface_hub.model_info", _raise)
    assert resolve_huggingface_revision("roberta-base") is None


def test_wandb_env_populates_config_when_absent(monkeypatch) -> None:
    for var in ("WANDB_ENTITY", "WANDB_UPLOAD_ARTIFACTS", "WANDB_UPLOAD_CHECKPOINTS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("WANDB_PROJECT", "drift-happens")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_TAGS", "screen,seed0")
    cfg = preset("smoke", "synthetic-classification-cpu").build()

    updated = with_wandb_from_env(cfg)

    assert updated.logging.wandb is not None
    assert updated.logging.wandb.project == "drift-happens"
    assert updated.logging.wandb.mode == "offline"
    assert updated.logging.wandb.tags == ("screen", "seed0")


def test_wandb_env_does_not_overwrite_config_level_wandb(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "from-env")
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    cfg = cfg.model_copy(
        update={
            "logging": cfg.logging.model_copy(
                update={"wandb": WandbConfig(project="from-config")}
            )
        }
    )

    updated = with_wandb_from_env(cfg)

    assert updated.logging.wandb is not None
    assert updated.logging.wandb.project == "from-config"


def test_huggingface_token_hf_token_is_read_when_huggingface_token_absent(
    monkeypatch,
) -> None:
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf-token")

    assert huggingface_token_from_env() == "hf-token"


def test_huggingface_token_huggingface_token_takes_precedence(monkeypatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf-primary")
    monkeypatch.setenv("HF_TOKEN", "hf-fallback")

    assert huggingface_token_from_env() == "hf-primary"


def test_huggingface_token_huggingface_token_alone_is_sufficient(monkeypatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf-primary")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    assert huggingface_token_from_env() == "hf-primary"


def test_huggingface_auth_errors_only_when_required(monkeypatch) -> None:
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    assert ensure_huggingface_auth_from_env(required=False) is None
    with pytest.raises(RuntimeError, match="Hugging Face credentials"):
        ensure_huggingface_auth_from_env(required=True)


def test_wandb_env_rejects_invalid_boolean(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "drift-happens")
    monkeypatch.setenv("WANDB_UPLOAD_ARTIFACTS", "maybe")
    cfg = preset("smoke", "synthetic-classification-cpu").build()

    with pytest.raises(ValueError, match="WANDB_UPLOAD_ARTIFACTS"):
        with_wandb_from_env(cfg)


def test_wandb_cli_overrides_preserve_existing_config() -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    cfg = cfg.model_copy(
        update={
            "logging": cfg.logging.model_copy(
                update={"wandb": WandbConfig(project="from-config", tags=("old",))}
            )
        }
    )

    updated = with_wandb_from_env(cfg, mode="offline", tags=("new",))

    assert updated.logging.wandb is not None
    assert updated.logging.wandb.project == "from-config"
    assert updated.logging.wandb.mode == "offline"
    assert updated.logging.wandb.tags == ("new",)


def test_wandb_env_rejects_invalid_mode(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "drift-happens")
    monkeypatch.setenv("WANDB_MODE", "invalid")
    cfg = preset("smoke", "synthetic-classification-cpu").build()

    with pytest.raises(ValueError, match="WANDB_MODE"):
        with_wandb_from_env(cfg)


def test_resume_default_is_enabled_unless_env_disables_it(monkeypatch) -> None:
    monkeypatch.delenv(RUN_RESUME_ENV, raising=False)

    assert resolve_resume_setting(None) is True

    monkeypatch.setenv(RUN_RESUME_ENV, "false")

    assert resolve_resume_setting(None) is False


def test_resume_explicit_option_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv(RUN_RESUME_ENV, "true")

    assert resolve_resume_setting(False) is False

    monkeypatch.setenv(RUN_RESUME_ENV, "false")

    assert resolve_resume_setting(True) is True


def test_resume_env_rejects_invalid_boolean(monkeypatch) -> None:
    monkeypatch.setenv(RUN_RESUME_ENV, "maybe")

    with pytest.raises(ValueError, match=RUN_RESUME_ENV):
        resolve_resume_setting(None)


def test_resume_checkpoints_default_off_unless_env_enables(monkeypatch) -> None:
    monkeypatch.delenv(RESUME_CHECKPOINTS_ENV, raising=False)

    assert resume_checkpoints_enabled() is False

    monkeypatch.setenv(RESUME_CHECKPOINTS_ENV, "1")

    assert resume_checkpoints_enabled() is True


def test_apply_resume_checkpoints_override_propagates_explicit_choice(
    monkeypatch,
) -> None:
    monkeypatch.delenv(RESUME_CHECKPOINTS_ENV, raising=False)

    # An omitted flag leaves the environment untouched for subprocesses to inherit.
    apply_resume_checkpoints_override(None)
    assert RESUME_CHECKPOINTS_ENV not in os.environ

    apply_resume_checkpoints_override(True)
    assert os.environ[RESUME_CHECKPOINTS_ENV] == "1"

    apply_resume_checkpoints_override(False)
    assert os.environ[RESUME_CHECKPOINTS_ENV] == "0"
