import json

from typer.testing import CliRunner

from drift_happens.cli.main import app
from drift_happens.runtime.base import StageResult
from drift_happens.runtime.stages import STAGE_CONTENTION_EXIT_CODE
from drift_happens.utils.env import RUN_RESUME_ENV
from drift_happens.utils.wandb_completion import WandbPreflightStatus

runner = CliRunner()
SNAPSHOT = "configs/snapshots/presets/smoke/synthetic-classification-cpu.json"


def test_experiment_run_in_process_runs_both_stages(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_run_stage(cfg, *, stage, runs_root=None, **kwargs):
        calls.append((stage, kwargs["resume"]))
        return StageResult(
            run_dir=tmp_path / "runs" / stage,
            stage=stage,
            exit_status="ok",
            iterations=1,
        )

    monkeypatch.setattr("drift_happens.runtime.local.run_stage", fake_run_stage)
    monkeypatch.delenv(RUN_RESUME_ENV, raising=False)

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            SNAPSHOT,
            "--in-process",
            "--runs-root",
            str(tmp_path),
            "--wandb-mode",
            "disabled",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [("train", True), ("eval", True)]
    assert "runs/eval" in result.output


def test_experiment_run_in_process_resume_env_can_disable(
    tmp_path, monkeypatch
) -> None:
    calls = []

    def fake_run_stage(cfg, *, stage, runs_root=None, **kwargs):
        calls.append((stage, kwargs["resume"]))
        return StageResult(
            run_dir=tmp_path / "runs" / stage,
            stage=stage,
            exit_status="ok",
            iterations=1,
        )

    monkeypatch.setattr("drift_happens.runtime.local.run_stage", fake_run_stage)
    monkeypatch.setenv(RUN_RESUME_ENV, "false")

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            SNAPSHOT,
            "--in-process",
            "--runs-root",
            str(tmp_path),
            "--wandb-mode",
            "disabled",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [("train", False), ("eval", False)]


def test_experiment_stage_disallows_local_wandb_resume_when_remote_missing(
    tmp_path,
    monkeypatch,
) -> None:
    calls = []

    def fake_run_stage(cfg, *, stage, runs_root=None, **kwargs):
        calls.append(kwargs["allowed_wandb_run_ids"])
        return StageResult(
            run_dir=tmp_path / "runs" / stage,
            stage=stage,
            exit_status="ok",
            iterations=1,
        )

    monkeypatch.setattr("drift_happens.runtime.local.run_stage", fake_run_stage)
    monkeypatch.setattr(
        "drift_happens.utils.wandb_completion.WandbCompletionIndex.preflight_status",
        lambda *args, **kwargs: WandbPreflightStatus(state="missing"),
    )

    result = runner.invoke(
        app,
        [
            "experiment",
            "eval",
            SNAPSHOT,
            "--runs-root",
            str(tmp_path),
            "--wandb-project",
            "project",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [()]


def test_experiment_run_in_process_propagates_stage_failure(
    tmp_path, monkeypatch
) -> None:
    calls = []

    def fake_run_stage(cfg, *, stage, runs_root=None, **kwargs):
        calls.append(stage)
        raise RuntimeError("stage blew up")

    monkeypatch.setattr("drift_happens.runtime.local.run_stage", fake_run_stage)

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            SNAPSHOT,
            "--in-process",
            "--runs-root",
            str(tmp_path),
            "--wandb-mode",
            "disabled",
        ],
    )

    assert result.exit_code != 0
    assert calls == ["train"]


def test_experiment_sweep_resume_flag_overrides_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(RUN_RESUME_ENV, "0")
    sweep_config = tmp_path / "sweep.yaml"
    sweep_config.write_text(
        "\n".join(
            [
                "name: cli-resume",
                "skip_completed: false",
                "jobs:",
                f"  - config_path: {SNAPSHOT}",
                "    seed: 0",
                "    label: smoke",
                "slots:",
                "  - device: cpu",
            ]
        )
    )

    result = runner.invoke(
        app,
        [
            "experiment",
            "sweep",
            str(sweep_config),
            "--resume",
            "--dry-run",
            "--sweep-root",
            str(tmp_path / "sweeps"),
        ],
    )

    assert result.exit_code == 0, result.output
    sweep_dirs = list((tmp_path / "sweeps").iterdir())
    assert len(sweep_dirs) == 1
    manifest = json.loads((sweep_dirs[0] / "manifest.json").read_text())
    results = json.loads((sweep_dirs[0] / "results.json").read_text())

    assert manifest["resume"] is True
    assert results[0]["command"][-1] == "--resume"


def test_experiment_run_in_process_reports_stage_contention(
    tmp_path, monkeypatch
) -> None:
    def fake_run_stage(cfg, *, stage, runs_root=None, **kwargs):
        raise RuntimeError("lock already held: train.lock")

    monkeypatch.setattr("drift_happens.runtime.local.run_stage", fake_run_stage)

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            SNAPSHOT,
            "--in-process",
            "--runs-root",
            str(tmp_path),
            "--wandb-mode",
            "disabled",
        ],
    )

    assert result.exit_code == STAGE_CONTENTION_EXIT_CODE
    assert "stage is running" in result.output
