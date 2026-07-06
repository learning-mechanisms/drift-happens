from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from drift_happens.configs import RunIdentity, WandbConfig
from drift_happens.experiments.registry import preset
from drift_happens.runtime.metrics import MetricRecord
from drift_happens.runtime.wandb import (
    WANDB_ARTIFACT_NAME_MAXLEN,
    WANDB_STEP_METRIC,
    WandbMetricSink,
    curated_run_artifact_files,
)
from drift_happens.utils.wandb_identity import build_run_identity


class FakeArtifact:
    def __init__(self, name, type, metadata=None):
        self.name = name
        self.type = type
        self.metadata = metadata or {}
        self.files = []

    def add_file(self, path, name=None):
        self.files.append((path, name))


class FakeRun:
    def __init__(self):
        self.id = None
        self.dir = None
        self.logs = []
        self.artifacts = []
        self.finished = False
        self.exit_code = None

    def log(self, payload):
        self.logs.append(payload)

    def log_artifact(self, artifact, aliases=None):
        self.artifacts.append((artifact, aliases))

    def finish(self, exit_code=None, quiet=None):
        self.finished = True
        self.exit_code = exit_code


def _install_fake_wandb(monkeypatch) -> tuple[FakeRun, list, list]:
    """Return (fake_run, init_calls, metric_calls) with wandb patched."""
    fake_run = FakeRun()
    init_calls: list = []
    metric_calls: list[tuple[tuple, dict]] = []

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        fake_run.id = kwargs.get("id", "generated")
        base_dir = kwargs.get("dir")
        if isinstance(base_dir, str):
            fake_run.dir = str(
                Path(base_dir)
                / "wandb"
                / f"run-20260101_000000-{fake_run.id}"
                / "files"
            )
        return fake_run

    fake_wandb = SimpleNamespace(
        Artifact=FakeArtifact,
        Settings=lambda **kwargs: SimpleNamespace(**kwargs),
        define_metric=lambda *args, **kwargs: metric_calls.append((args, kwargs)),
        init=fake_init,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    return fake_run, init_calls, metric_calls


def test_wandb_sink_is_stage_aware(tmp_path: Path, monkeypatch) -> None:
    fake_run, init_calls, metric_calls = _install_fake_wandb(monkeypatch)
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    identity = RunIdentity(
        source_identity="smoke__synthetic",
        config_hash="abc",
        snapshot_sha256="def",
        wandb_group="smoke__synthetic",
        wandb_run_name="smoke__synthetic__seed=0__train",
    )

    sink = WandbMetricSink(
        cfg=cfg,
        wandb_cfg=WandbConfig(
            project="drift-happens",
            mode="offline",
            job_type="train",
        ),
        run_dir=tmp_path,
        identity=identity,
        stage="train",
    )
    sink.log(
        MetricRecord.from_config(
            cfg,
            phase="train",
            metric="train/loss",
            value=1.25,
            context={"stage/exit_status": "ok"},
        )
    )
    sink.close()

    assert init_calls[0]["project"] == "drift-happens"
    assert init_calls[0]["group"] == "smoke__synthetic"
    assert init_calls[0]["name"] == "smoke__synthetic__seed=0__train"
    assert init_calls[0]["job_type"] == "train"
    assert init_calls[0]["config"]["run/stage"] == "train"
    assert init_calls[0]["settings"].console == "wrap"
    assert metric_calls == [
        ((WANDB_STEP_METRIC,), {}),
        (("*",), {"step_metric": WANDB_STEP_METRIC}),
    ]
    assert fake_run.logs[0]["train/loss"] == 1.25
    assert fake_run.logs[0]["stage/exit_status"] == "ok"
    assert fake_run.finished
    assert fake_run.exit_code is None


def test_wandb_metric_sink_marks_failed_close(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_run, _, _ = _install_fake_wandb(monkeypatch)
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    identity = build_run_identity(cfg, run_dir=tmp_path, source_path=None)

    sink = WandbMetricSink(
        cfg=cfg,
        wandb_cfg=WandbConfig(project="proj", mode="offline"),
        run_dir=tmp_path,
        identity=identity,
        stage="eval",
    )
    sink.close(exit_code=1)

    assert fake_run.finished is True
    assert fake_run.exit_code == 1


def test_wandb_metric_sink_logs_scalars_and_curated_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_run, init_calls, _ = _install_fake_wandb(monkeypatch)

    (tmp_path / "snapshot.json").write_text("{}\n")
    (tmp_path / "metadata.json").write_text("{}\n")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "final.pt").write_text("large")

    cfg = preset("smoke", "synthetic-classification-cpu").build()
    wandb_cfg = WandbConfig(project="proj", mode="offline")
    cfg = cfg.model_copy(
        update={"logging": cfg.logging.model_copy(update={"wandb": wandb_cfg})}
    )
    identity = build_run_identity(cfg, run_dir=tmp_path, source_path=None)

    sink = WandbMetricSink(
        cfg=cfg,
        wandb_cfg=wandb_cfg,
        run_dir=tmp_path,
        identity=identity,
    )
    sink.log(
        MetricRecord.from_config(
            cfg,
            phase="train",
            metric="train/loss",
            value=0.25,
            # step=5 differs from the internal counter (1) so the assertion
            # distinguishes step/global (counter) from step/local (record.step).
            step=5,
        )
    )
    sink.close()

    assert init_calls[0]["project"] == "proj"
    assert init_calls[0]["group"] == identity.wandb_group
    assert fake_run.logs[0][WANDB_STEP_METRIC] == 1
    assert fake_run.logs[0]["step/local"] == 5
    assert fake_run.logs[0]["train/loss"] == 0.25
    artifact, aliases = fake_run.artifacts[0]
    assert aliases == ["latest"]
    assert {name for _, name in artifact.files} == {"snapshot.json", "metadata.json"}
    assert fake_run.finished is True


def test_wandb_metric_sink_resumes_single_local_wandb_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_run, init_calls, _ = _install_fake_wandb(monkeypatch)
    run_id = "abc123"
    wandb_dir = tmp_path / "wandb" / f"run-20260101_000000-{run_id}"
    wandb_dir.mkdir(parents=True)
    (wandb_dir / f"run-{run_id}.wandb").write_bytes(b"")
    (tmp_path / "wandb" / "latest-run").symlink_to(
        wandb_dir.name,
        target_is_directory=True,
    )
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "train.jsonl").write_text("{}\n{}\n")

    cfg = preset("smoke", "synthetic-classification-cpu").build()
    identity = RunIdentity(
        source_identity="smoke__synthetic",
        config_hash="abc",
        snapshot_sha256="def",
        wandb_group="smoke__synthetic",
        wandb_run_name="smoke__synthetic__seed=0__train",
    )

    sink = WandbMetricSink(
        cfg=cfg,
        wandb_cfg=WandbConfig(project="proj", mode="offline"),
        run_dir=tmp_path,
        identity=identity,
        stage="train",
    )
    sink.log(
        MetricRecord.from_config(
            cfg,
            phase="train",
            metric="train/loss",
            value=0.25,
        )
    )

    assert init_calls[0]["id"] == run_id
    assert init_calls[0]["resume"] == "allow"
    assert fake_run.logs[0][WANDB_STEP_METRIC] == 3


def test_wandb_metric_sink_uses_metadata_to_recover_original_duplicate_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, init_calls, _ = _install_fake_wandb(monkeypatch)
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    identity = RunIdentity(
        source_identity="smoke__synthetic",
        config_hash="abc",
        snapshot_sha256="def",
        wandb_group="smoke__synthetic",
        wandb_run_name="smoke__synthetic__seed=0__train",
    )
    _write_wandb_metadata_run(
        tmp_path,
        run_id="first123",
        timestamp="20260101_000000",
        seed=cfg.seed,
        action="train",
    )
    _write_wandb_metadata_run(
        tmp_path,
        run_id="duplicate456",
        timestamp="20260101_010000",
        seed=cfg.seed,
        action="train",
    )

    WandbMetricSink(
        cfg=cfg,
        wandb_cfg=WandbConfig(project="proj", mode="offline"),
        run_dir=tmp_path,
        identity=identity,
        stage="train",
    )

    assert init_calls[0]["id"] == "first123"
    assert init_calls[0]["resume"] == "allow"


def test_wandb_metric_sink_prefers_original_run_over_marked_duplicate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, init_calls, _ = _install_fake_wandb(monkeypatch)
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    identity = RunIdentity(
        source_identity="smoke__synthetic",
        config_hash="abc",
        snapshot_sha256="def",
        completion_hash="ghi",
        wandb_group="smoke__synthetic",
        wandb_run_name="smoke__synthetic__seed=0__train",
    )
    _write_wandb_metadata_run(
        tmp_path,
        run_id="original123",
        timestamp="20260101_000000",
        seed=cfg.seed,
        action="train",
    )
    _write_wandb_config_run(
        tmp_path,
        run_id="duplicate456",
        seed=cfg.seed,
        stage="train",
        identity=identity,
        timestamp="20260101_010000",
    )

    WandbMetricSink(
        cfg=cfg,
        wandb_cfg=WandbConfig(project="proj", mode="offline"),
        run_dir=tmp_path,
        identity=identity,
        stage="train",
    )

    assert init_calls[0]["id"] == "original123"
    assert init_calls[0]["resume"] == "allow"


def test_wandb_metric_sink_uses_metadata_to_match_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, init_calls, _ = _install_fake_wandb(monkeypatch)
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    identity = RunIdentity(
        source_identity="smoke__synthetic",
        config_hash="abc",
        snapshot_sha256="def",
        wandb_group="smoke__synthetic",
        wandb_run_name="smoke__synthetic__seed=0__eval",
    )
    _write_wandb_metadata_run(
        tmp_path,
        run_id="train123",
        timestamp="20260101_000000",
        seed=cfg.seed,
        action="train",
    )
    _write_wandb_metadata_run(
        tmp_path,
        run_id="eval456",
        timestamp="20260101_010000",
        seed=cfg.seed,
        action="eval",
    )

    WandbMetricSink(
        cfg=cfg,
        wandb_cfg=WandbConfig(project="proj", mode="offline"),
        run_dir=tmp_path,
        identity=identity,
        stage="eval",
    )

    assert init_calls[0]["id"] == "eval456"
    assert init_calls[0]["resume"] == "allow"


def test_wandb_metric_sink_resumes_stage_matching_local_wandb_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, init_calls, _ = _install_fake_wandb(monkeypatch)
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    identity = RunIdentity(
        source_identity="smoke__synthetic",
        config_hash="abc",
        snapshot_sha256="def",
        completion_hash="ghi",
        wandb_group="smoke__synthetic",
        wandb_run_name="smoke__synthetic__seed=0__eval",
    )
    _write_wandb_config_run(
        tmp_path,
        run_id="train123",
        seed=cfg.seed,
        stage="train",
        identity=identity,
    )
    _write_wandb_config_run(
        tmp_path,
        run_id="eval456",
        seed=cfg.seed,
        stage="eval",
        identity=identity,
    )

    WandbMetricSink(
        cfg=cfg,
        wandb_cfg=WandbConfig(project="proj", mode="offline"),
        run_dir=tmp_path,
        identity=identity,
        stage="eval",
    )

    assert init_calls[0]["id"] == "eval456"
    assert init_calls[0]["resume"] == "allow"


def test_wandb_metric_sink_resumes_offline_local_wandb_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, init_calls, _ = _install_fake_wandb(monkeypatch)
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    identity = RunIdentity(
        source_identity="smoke__synthetic",
        config_hash="abc",
        snapshot_sha256="def",
        completion_hash="ghi",
        wandb_group="smoke__synthetic",
        wandb_run_name="smoke__synthetic__seed=0__eval",
    )
    _write_wandb_config_run(
        tmp_path,
        run_id="eval456",
        seed=cfg.seed,
        stage="eval",
        identity=identity,
        dir_prefix="offline-run-",
    )

    WandbMetricSink(
        cfg=cfg,
        wandb_cfg=WandbConfig(project="proj", mode="offline"),
        run_dir=tmp_path,
        identity=identity,
        stage="eval",
    )

    assert init_calls[0]["id"] == "eval456"
    assert init_calls[0]["resume"] == "allow"


def test_wandb_metric_sink_skips_resume_when_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, init_calls, _ = _install_fake_wandb(monkeypatch)
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    identity = RunIdentity(
        source_identity="smoke__synthetic",
        config_hash="abc",
        snapshot_sha256="def",
        completion_hash="ghi",
        wandb_group="smoke__synthetic",
        wandb_run_name="smoke__synthetic__seed=0__train",
    )
    _write_wandb_config_run(
        tmp_path,
        run_id="train123",
        seed=cfg.seed,
        stage="train",
        identity=identity,
    )

    WandbMetricSink(
        cfg=cfg,
        wandb_cfg=WandbConfig(project="proj", mode="offline"),
        run_dir=tmp_path,
        identity=identity,
        stage="train",
        resume=False,
    )

    assert "id" not in init_calls[0]
    assert init_calls[0]["resume"] == "never"


def test_wandb_metric_sink_ignores_local_run_id_not_seen_remotely(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, init_calls, _ = _install_fake_wandb(monkeypatch)
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    identity = RunIdentity(
        source_identity="smoke__synthetic",
        config_hash="abc",
        snapshot_sha256="def",
        completion_hash="ghi",
        wandb_group="smoke__synthetic",
        wandb_run_name="smoke__synthetic__seed=0__eval",
    )
    _write_wandb_config_run(
        tmp_path,
        run_id="deleted123",
        seed=cfg.seed,
        stage="eval",
        identity=identity,
    )

    WandbMetricSink(
        cfg=cfg,
        wandb_cfg=WandbConfig(project="proj", mode="offline"),
        run_dir=tmp_path,
        identity=identity,
        stage="eval",
        allowed_resume_run_ids=(),
    )

    assert "id" not in init_calls[0]
    assert init_calls[0]["resume"] == "never"


def test_wandb_metric_sink_uses_bounded_artifact_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_run, _, _ = _install_fake_wandb(monkeypatch)

    (tmp_path / "snapshot.json").write_text("{}\n")

    cfg = preset("smoke", "synthetic-classification-cpu").build()
    identity = RunIdentity(
        source_identity=(
            "synthetic__linear-classifier__smoke-synthetic-classification-cpu"
            "__seed-0__smoke__synthetic-classification-cpu"
        ),
        config_hash="a" * 64,
        snapshot_sha256="b" * 64,
        wandb_group="group",
        wandb_run_name="run",
    )

    sink = WandbMetricSink(
        cfg=cfg,
        wandb_cfg=WandbConfig(project="proj", mode="offline"),
        run_dir=tmp_path,
        identity=identity,
    )
    sink.close()

    artifact, _ = fake_run.artifacts[0]
    assert len(artifact.name) <= WANDB_ARTIFACT_NAME_MAXLEN
    assert artifact.name.startswith("drift-run__")
    assert artifact.name.endswith("__seed-0__cfg-aaaaaaaaaaaa__snap-bbbbbbbbbbbb")


def test_upload_artifacts_false_suppresses_artifact_upload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_run, _, _ = _install_fake_wandb(monkeypatch)
    (tmp_path / "snapshot.json").write_text("{}\n")

    cfg = preset("smoke", "synthetic-classification-cpu").build()
    identity = build_run_identity(cfg, run_dir=tmp_path, source_path=None)
    sink = WandbMetricSink(
        cfg=cfg,
        wandb_cfg=WandbConfig(project="proj", mode="offline", upload_artifacts=False),
        run_dir=tmp_path,
        identity=identity,
    )
    sink.close()

    assert fake_run.artifacts == []


def test_curated_artifacts_include_dataset_pipeline_checkpoint(tmp_path: Path) -> None:
    slice_dir = tmp_path / "stages" / "train" / "resnet" / "train_slice_2010"
    slice_dir.mkdir(parents=True)
    checkpoint = slice_dir / "trained_model.pt"
    checkpoint.write_bytes(b"weights")
    # An eval prediction under stages/train must not be swept in.
    preds = slice_dir / "predictions" / "eval_slice_2011"
    preds.mkdir(parents=True)
    (preds / "predictions.pt").write_bytes(b"preds")

    assert checkpoint not in curated_run_artifact_files(
        tmp_path, upload_checkpoints=False
    )
    with_checkpoints = curated_run_artifact_files(tmp_path, upload_checkpoints=True)
    assert checkpoint in with_checkpoints
    assert preds / "predictions.pt" not in with_checkpoints


def test_curated_artifacts_include_drift_matrix_results(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    matrix_json = results_dir / "drift_matrix.json"
    matrix_csv = results_dir / "drift_matrix.csv"
    matrix_json.write_text('{"2000": {"2000": {"accuracy": 0.8}}}\n')
    matrix_csv.write_text(",2000\n2000,0.8\n")

    curated = curated_run_artifact_files(tmp_path)

    assert matrix_json in curated
    assert matrix_csv in curated


def _write_wandb_config_run(
    root: Path,
    *,
    run_id: str,
    seed: int,
    stage: str,
    identity: RunIdentity,
    dir_prefix: str = "run-",
    timestamp: str = "20260101_000000",
) -> None:
    wandb_dir = root / "wandb" / f"{dir_prefix}{timestamp}-{run_id}"
    files_dir = wandb_dir / "files"
    files_dir.mkdir(parents=True)
    (wandb_dir / f"run-{run_id}.wandb").write_bytes(b"")
    (files_dir / "config.yaml").write_text(
        "\n".join(
            [
                "run/completion_hash:",
                f"    value: {identity.completion_hash}",
                "run/config_hash:",
                f"    value: {identity.config_hash}",
                "run/snapshot_sha256:",
                f"    value: {identity.snapshot_sha256}",
                "run/stage:",
                f"    value: {stage}",
                "seed:",
                f"    value: {seed}",
                "",
            ]
        )
    )


def _write_wandb_metadata_run(
    root: Path,
    *,
    run_id: str,
    timestamp: str,
    seed: int,
    action: str,
) -> None:
    wandb_dir = root / "wandb" / f"run-{timestamp}-{run_id}"
    files_dir = wandb_dir / "files"
    files_dir.mkdir(parents=True)
    (wandb_dir / f"run-{run_id}.wandb").write_bytes(b"")
    (files_dir / "wandb-metadata.json").write_text(
        "\n".join(
            [
                "{",
                '  "args": [',
                '    "experiment",',
                f'    "{action}",',
                '    "configs/snapshots/presets/smoke/synthetic-classification-cpu.json",',
                '    "--seed",',
                f'    "{seed}"',
                "  ]",
                "}",
                "",
            ]
        )
    )
