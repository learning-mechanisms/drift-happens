from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from drift_happens.configs import WandbConfig
from drift_happens.experiments.registry import preset
from drift_happens.utils.wandb_identity import (
    build_run_identity,
    completion_hash,
    config_hash,
    snapshot_sha256,
    source_wandb_identity,
    wandb_group,
    wandb_run_name,
)


def test_materialized_snapshot_identity_maps_to_group_and_name() -> None:
    path = Path("configs/snapshots/presets/yearbook/smoke-mlp-s.json")

    assert source_wandb_identity(path) == "yearbook__smoke-mlp-s"


def test_seed_replicas_share_group_but_run_names_include_seed_and_leaf() -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    path = Path("configs/snapshots/presets/smoke/synthetic-classification-cpu.json")
    run_dir = Path("artifacts/runs/x/y/z/seed=1/leaf")
    seeded = cfg.model_copy(update={"seed": 1})

    group = wandb_group(seeded, source_identity=source_wandb_identity(path))
    name = wandb_run_name(
        seeded,
        seed=seeded.seed,
        run_dir=run_dir,
        source_identity=source_wandb_identity(path),
    )

    assert group == "smoke__synthetic-classification-cpu"
    assert name == "smoke__synthetic-classification-cpu__seed=1__leaf"


def test_config_hash_changes_when_meaningful_config_changes() -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    changed = cfg.model_copy(update={"seed": cfg.seed + 1})

    assert config_hash(cfg) != config_hash(changed)


def test_config_hash_ignores_wandb_mirror_settings() -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    mirrored = cfg.model_copy(
        update={
            "logging": cfg.logging.model_copy(
                update={"wandb": WandbConfig(project="other", mode="offline")}
            )
        }
    )

    assert config_hash(cfg) == config_hash(mirrored)


def test_completion_hash_ignores_runset_and_execution_bookkeeping() -> None:
    cfg = preset("yearbook-conference", "mlp_s").build()
    cache = cfg.preprocessing.cache
    assert cache is not None

    metadata = dict(cfg.metadata)
    seed_metadata = dict(metadata["seeds"])
    seed_metadata["model_seeds"] = [0, 1, 2, 3, 4]
    metadata["seeds"] = seed_metadata
    changed = cfg.model_copy(
        update={
            "logging": cfg.logging.model_copy(update={"stdout": False}),
            "metadata": metadata,
            "name": f"{cfg.name}-rerun",
            "notes": "expanded seed plan",
            "preprocessing": cfg.preprocessing.model_copy(
                update={"cache": cache.model_copy(update={"reuse_policy": "refresh"})}
            ),
            "runtime": cfg.runtime.model_copy(
                update={"device": "cuda", "num_threads": 2}
            ),
        }
    )

    assert config_hash(cfg) != config_hash(changed)
    assert completion_hash(cfg) == completion_hash(changed)


def test_config_level_wandb_group_takes_precedence() -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    cfg = cfg.model_copy(
        update={
            "logging": cfg.logging.model_copy(
                update={"wandb": WandbConfig(project="p", group="manual")}
            )
        }
    )

    identity = build_run_identity(cfg, run_dir=Path("leaf"), source_path=None)

    assert identity.wandb_group == "manual"


def test_snapshot_sha256_hashes_file_contents(tmp_path: Path) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    src = tmp_path / "snap.json"
    src.write_bytes(b"snapshot-bytes")

    assert snapshot_sha256(src, cfg) == sha256(b"snapshot-bytes").hexdigest()
    assert snapshot_sha256(None, cfg) == config_hash(cfg)


def test_snapshot_sha256_propagates_unreadable_source(tmp_path: Path) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    src = tmp_path / "noperm.json"
    src.write_bytes(b"x")
    src.chmod(0o000)
    try:
        with pytest.raises(OSError):
            snapshot_sha256(src, cfg)
    finally:
        src.chmod(0o600)
