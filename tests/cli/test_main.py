"""Tests for the drift CLI command surface."""

import re
import subprocess
import sys

import pytest
import yaml
from typer.testing import CliRunner

from drift_happens.cli.experiment import _parse_int_csv, _parse_tags
from drift_happens.cli.main import app
from drift_happens.configs import RunIdentity
from drift_happens.runtime.base import StageResult
from drift_happens.runtime.completion_filter import SeedStatusRow
from drift_happens.runtime.stages import STAGE_CONTENTION_EXIT_CODE, write_json_atomic
from drift_happens.utils.wandb_completion import WandbPreflightStatus

runner = CliRunner()
SNAPSHOT = "configs/snapshots/presets/smoke/synthetic-classification-cpu.json"
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _plain_help_lines(output: str) -> list[str]:
    return [_ANSI_ESCAPE_RE.sub("", line) for line in output.splitlines()]


def test_root_help_lists_main_command_groups() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    lines = _plain_help_lines(result.output)
    # Each name must appear as a command-table row, not just in surrounding prose.
    for name in (
        "datasets-setup",
        "dataset",
        "eval",
        "experiment",
        "artifacts",
    ):
        assert any(re.match(rf"^\s*│\s+{re.escape(name)}\s", line) for line in lines), (
            f"command {name!r} missing from Commands table"
        )


@pytest.mark.parametrize("group", ["datasets-setup", "dataset"])
def test_dataset_group_help_lists_dataset_commands(group: str) -> None:
    result = runner.invoke(app, [group, "--help"])

    assert result.exit_code == 0
    lines = _plain_help_lines(result.output)
    for name in ("yearbook", "arxiv", "amazon-reviews-23", "imdb-faces"):
        assert any(re.match(rf"^\s*│\s+{re.escape(name)}\s", line) for line in lines), (
            f"dataset command {name!r} missing from {group!r} help table"
        )


def test_datasets_setup_dataset_help_lists_setup_steps() -> None:
    arxiv = runner.invoke(app, ["datasets-setup", "arxiv", "--help"])
    amazon = runner.invoke(app, ["datasets-setup", "amazon-reviews-23", "--help"])

    assert arxiv.exit_code == 0
    assert "download" in arxiv.output
    assert "prepare" in arxiv.output
    assert "full" in arxiv.output
    assert amazon.exit_code == 0
    assert "build-from-cache" in amazon.output
    assert "download-reviews" in amazon.output
    assert "merge-review-categories" in amazon.output
    assert "full" in amazon.output


def test_eval_help_lists_robustness_command() -> None:
    result = runner.invoke(app, ["eval", "--help"])

    assert result.exit_code == 0
    assert "robustness" in result.output


def test_experiment_help_lists_registry_commands() -> None:
    result = runner.invoke(app, ["experiment", "--help"])

    assert result.exit_code == 0
    lines = _plain_help_lines(result.output)
    for name in (
        "list",
        "materialize",
        "train",
        "eval",
        "run",
        "sweep",
        "stages",
        "seeds",
        "plans",
    ):
        assert any(re.match(rf"^\s*│\s+{re.escape(name)}\s", line) for line in lines), (
            f"subcommand {name!r} missing from experiment Commands table"
        )


def test_experiment_list_shows_materialized_groups() -> None:
    result = runner.invoke(app, ["experiment", "list"])

    assert result.exit_code == 0
    assert "yearbook/smoke-mlp-s" in result.output


def test_experiment_run_launches_split_synthetic_stages(tmp_path) -> None:
    # Stages run in spawned subprocesses; only --runs-root isolates writes.
    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            SNAPSHOT,
            "--seed",
            "1",
            "--set",
            "trainer.training.num_epochs=1",
            "--wandb-mode",
            "disabled",
            "--runs-root",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "smoke-synthetic-classification-cpu" in result.output
    status = runner.invoke(
        app,
        [
            "experiment",
            "stages",
            "status",
            SNAPSHOT,
            "--seed",
            "1",
            "--set",
            "trainer.training.num_epochs=1",
            "--runs-root",
            str(tmp_path / "runs"),
        ],
    )
    assert status.exit_code == 0, status.output
    assert "1\tok\tok\tok" in status.output


def test_experiment_seed_status_reports_missing(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "experiment",
            "seeds",
            "status",
            SNAPSHOT,
            "--runs-root",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "0\tmissing\tmissing\tmissing" in result.output


def test_artifacts_help_lists_commands() -> None:
    result = runner.invoke(app, ["artifacts", "--help"])

    assert result.exit_code == 0
    assert "ls" in result.output
    assert "gc" in result.output
    assert "bundle" in result.output


def test_experiment_train_then_eval_commands(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_run_stage(cfg, *, stage, runs_root=None, **kwargs):
        calls.append(stage)
        return StageResult(
            run_dir=tmp_path / "runs" / stage,
            stage=stage,
            exit_status="ok",
            iterations=1,
        )

    monkeypatch.setattr("drift_happens.runtime.local.run_stage", fake_run_stage)
    monkeypatch.setattr(
        "drift_happens.cli.experiment._wandb_preflight_skips",
        lambda *a, **k: False,
    )

    train = runner.invoke(
        app,
        [
            "experiment",
            "train",
            SNAPSHOT,
            "--runs-root",
            str(tmp_path),
            "--wandb-mode",
            "disabled",
        ],
    )
    eval_result = runner.invoke(
        app,
        [
            "experiment",
            "eval",
            SNAPSHOT,
            "--runs-root",
            str(tmp_path),
            "--wandb-mode",
            "disabled",
        ],
    )

    assert train.exit_code == 0, train.output
    assert eval_result.exit_code == 0, eval_result.output
    assert calls == ["train", "eval"]


def test_experiment_stage_command_skips_local_contention(tmp_path, monkeypatch) -> None:
    def fake_run_stage(*args, **kwargs):
        raise RuntimeError("lock already held: train.lock")

    monkeypatch.setattr("drift_happens.runtime.local.run_stage", fake_run_stage)
    monkeypatch.setattr(
        "drift_happens.cli.experiment._wandb_preflight_skips",
        lambda *a, **k: False,
    )

    result = runner.invoke(
        app,
        [
            "experiment",
            "train",
            SNAPSHOT,
            "--runs-root",
            str(tmp_path),
            "--wandb-mode",
            "disabled",
        ],
    )

    assert result.exit_code == STAGE_CONTENTION_EXIT_CODE, result.output
    assert "stage is running" in result.output


def test_experiment_run_in_process_skips_local_contention(
    tmp_path, monkeypatch
) -> None:
    def fake_run_stage(*args, **kwargs):
        raise RuntimeError("lock already held: train.lock")

    monkeypatch.setattr("drift_happens.runtime.local.run_stage", fake_run_stage)

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            SNAPSHOT,
            "--in-process",
            "--wandb-mode",
            "disabled",
            "--runs-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == STAGE_CONTENTION_EXIT_CODE, result.output
    assert "stage is running" in result.output


def test_experiment_run_skip_completed_reports_skip(monkeypatch) -> None:
    monkeypatch.setattr(
        "drift_happens.runtime.completion_filter.local_seed_complete",
        lambda *a, **k: True,
    )

    result = runner.invoke(app, ["experiment", "run", SNAPSHOT, "--skip-completed"])

    assert result.exit_code == 0, result.output
    assert "skipped complete seed" in result.output


def test_experiment_run_wandb_preflight_skips_running_seed(monkeypatch) -> None:
    def fail_spawn(*args, **kwargs):
        raise AssertionError("stage subprocess should not be spawned")

    monkeypatch.setattr("drift_happens.cli.experiment._spawn_stage_process", fail_spawn)
    monkeypatch.setattr(
        "drift_happens.utils.wandb_completion.WandbCompletionIndex.preflight_status",
        lambda *args, **kwargs: WandbPreflightStatus(state="running"),
    )

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            SNAPSHOT,
            "--wandb-project",
            "project",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "matching W&B run in progress" in result.output


def test_experiment_materialize_check_success_with_tmp_root(tmp_path) -> None:
    write = runner.invoke(
        app, ["experiment", "materialize", "--write", "--out-dir", str(tmp_path)]
    )
    check = runner.invoke(
        app, ["experiment", "materialize", "--check", "--out-dir", str(tmp_path)]
    )

    assert write.exit_code == 0, write.output
    assert check.exit_code == 0, check.output


def test_experiment_plans_check_reports_stale_with_tmp_root(tmp_path) -> None:
    write = runner.invoke(
        app,
        ["experiment", "plans", "materialize", "--write", "--out-dir", str(tmp_path)],
    )
    assert write.exit_code == 0, write.output
    plan_file = next(tmp_path.rglob("*.yaml"))
    plan_file.write_text(plan_file.read_text() + "\n# stale\n")

    result = runner.invoke(
        app,
        ["experiment", "plans", "materialize", "--check", "--out-dir", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "stale" in result.output.lower()


def test_experiment_plans_materialize_filters_selected_seeds(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "experiment",
            "plans",
            "materialize",
            "--write",
            "--out-dir",
            str(tmp_path),
            "--device",
            "cuda",
            "--gpu-indices",
            "0,1",
            "--concurrency",
            "2",
            "--seeds",
            "1,3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "p80_seed0_all_presets.yaml").exists()
    plan_paths = sorted(tmp_path.glob("*.yaml"))
    assert plan_paths
    observed_seeds: set[int] = set()
    for path in plan_paths:
        payload = yaml.safe_load(path.read_text())
        job_seeds = {job["seed"] for job in payload["jobs"]}
        assert job_seeds <= {1, 3}
        assert set(payload["seeds"]) == job_seeds
        observed_seeds.update(job_seeds)

    assert observed_seeds == {1, 3}


def test_experiment_plans_status_summarizes_plan_jobs(tmp_path, monkeypatch) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(
            {
                "name": "host-plan",
                "jobs": [
                    {
                        "action": "run",
                        "config_path": SNAPSHOT,
                        "label": "smoke/synthetic-classification-cpu",
                        "seed": 0,
                    },
                    {
                        "action": "run",
                        "config_path": SNAPSHOT,
                        "label": "smoke/synthetic-classification-cpu",
                        "seed": 1,
                    },
                ],
                "slots": [{"device": "cpu", "label": "cpu:0"}],
                "seeds": [0, 1],
                "concurrency": 1,
            },
            sort_keys=False,
        )
    )

    def fake_statuses(identities, **kwargs):
        seed = next(iter(identities))
        if seed == 0:
            return (SeedStatusRow(seed=seed, status="ok", train="ok", eval="ok"),)
        return (SeedStatusRow(seed=seed, status="partial", train="ok", eval="ok"),)

    monkeypatch.setattr(
        "drift_happens.runtime.completion_filter.local_seed_statuses_by_identity",
        fake_statuses,
    )

    result = runner.invoke(
        app,
        ["experiment", "plans", "status", str(plan_path), "--source", "local"],
    )

    assert result.exit_code == 0, result.output
    assert "shape: (2, 7)" in result.output
    assert "label" in result.output
    assert "run_ok" in result.output
    assert "smoke/synthetic-classification-cpu" in result.output
    assert "TOTAL" in result.output


def test_artifacts_ls_outputs_canonical_runs(tmp_path) -> None:
    identity = RunIdentity(
        source_identity="src",
        config_hash="cfg",
        snapshot_sha256="snap",
        wandb_group="group",
        wandb_run_name="run",
    )
    run_dir = (
        tmp_path / "runs" / "dataset" / "trainer" / "experiment" / "seed=0" / "run"
    )
    run_dir.mkdir(parents=True)
    write_json_atomic(
        run_dir / "run_manifest.json",
        {
            "dataset": "synthetic",
            "experiment": "unit",
            "identity": identity.model_dump(mode="json"),
            "seed": 0,
            "trainer": "fake",
        },
    )

    result = runner.invoke(
        app, ["artifacts", "ls", "--kind", "runs", "--root", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "synthetic\tfake\tunit" in result.output


def test_parse_tags_and_parse_int_csv_helpers() -> None:
    assert _parse_tags(" a, b ,,") == ("a", "b")
    assert _parse_tags(None) == ()
    assert _parse_int_csv("1, 2,,3") == (1, 2, 3)


def test_cli_experiment_module_import_stays_torch_free() -> None:
    # Guard the lazy-import pattern: only command bodies may pull torch and the
    # preset registry, not the module import itself.
    code = (
        "import sys; import drift_happens.cli.experiment; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )

    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_cli_presets_root_mirrors_materialize_constant() -> None:
    from drift_happens.cli import experiment
    from drift_happens.experiments.materialize import PRESETS_ROOT

    assert experiment.PRESETS_ROOT == PRESETS_ROOT
