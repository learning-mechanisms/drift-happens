from __future__ import annotations

from drift_happens.configs import LoggingConfig, WandbConfig


def test_wandb_config_defaults_keep_checkpoint_upload_disabled() -> None:
    cfg = WandbConfig(project="drift-happens")

    assert cfg.mode == "online"
    assert cfg.job_type == "train_eval"
    assert cfg.upload_artifacts is True
    assert cfg.upload_checkpoints is False


def test_logging_config_keeps_wandb_disabled_by_default() -> None:
    cfg = LoggingConfig()

    assert cfg.wandb is None
