from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from drift_happens.configs import DeviceSlotConfig, JobSpecConfig, SweepConfig
from drift_happens.runtime.progress import SWEEP_PROGRESS_FILE_ENV
from drift_happens.runtime.stages import STAGE_CONTENTION_EXIT_CODE
from drift_happens.runtime.sweep import (
    SweepRunner,
    _error_summary,
    _job_args,
    _job_env,
    _JobProgressReader,
)
from drift_happens.utils.env import RUN_RESUME_ENV

_SMOKE_CONFIG_PATH = "configs/snapshots/presets/smoke/synthetic-classification-cpu.json"

# Tail size from _error_summary's lines[-12:] implementation.
_ERROR_TAIL = 12


def _smoke_sweep(**extra) -> SweepConfig:
    """Single-job synthetic smoke sweep; per-test keys override via extra."""
    base: dict = {
        "name": "synthetic-smoke",
        "jobs": [
            {
                "config_path": _SMOKE_CONFIG_PATH,
                "label": "synthetic",
                "seed": 0,
            }
        ],
        "slots": [{"device": "cpu"}],
    }
    base.update(extra)
    return SweepConfig.model_validate(base)


@pytest.mark.integration
def test_cpu_only_synthetic_sweep_runs_end_to_end(tmp_artifacts, monkeypatch) -> None:
    monkeypatch.setenv("DRIFT_ARTIFACTS_DIR", str(tmp_artifacts))
    sweep = SweepConfig.model_validate(
        {
            "name": "synthetic-smoke",
            "jobs": [
                {
                    "config_path": _SMOKE_CONFIG_PATH,
                    "label": "synthetic",
                    "overrides": ["trainer.training.num_epochs=1"],
                    "seed": 0,
                }
            ],
            "slots": [{"device": "cpu"}],
        }
    )

    result = SweepRunner(sweep, sweep_root=tmp_artifacts / "sweeps").run()

    assert result.all_ok
    results = json.loads((result.sweep_dir / "results.json").read_text())
    assert results[0]["status"] == "ok"
    assert (result.sweep_dir / "logs" / "events.jsonl").exists()


def test_cuda_slot_env_injects_visible_device() -> None:
    env = _job_env(DeviceSlotConfig(device="cuda", device_index=2))

    assert env["CUDA_VISIBLE_DEVICES"] == "2"


def test_job_env_injects_progress_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(SWEEP_PROGRESS_FILE_ENV, "stale.jsonl")
    progress_path = tmp_path / "progress.jsonl"

    env = _job_env(DeviceSlotConfig(device="cpu"), progress_path=progress_path)
    without_progress = _job_env(DeviceSlotConfig(device="cpu"))

    assert env[SWEEP_PROGRESS_FILE_ENV] == str(progress_path)
    assert SWEEP_PROGRESS_FILE_ENV not in without_progress


def test_job_args_use_declared_action_and_resume_flag() -> None:
    # Build the job directly; the sweep-level "resume" field is unrelated to
    # _job_args' resume kwarg, which the runner always passes explicitly.
    job = JobSpecConfig(
        action="train",
        config_path=_SMOKE_CONFIG_PATH,
        label="synthetic",
        seed=0,
    )

    args = _job_args(
        job,
        slot=DeviceSlotConfig(device="cpu"),
        # The runner itself always calls _job_args with skip_completed=False;
        # True is exercised here to confirm the flag is forwarded when set.
        skip_completed=True,
        resume=False,
    )

    assert "train" in args
    assert "--skip-completed" in args
    assert "--no-resume" in args

    resume_args = _job_args(
        job,
        slot=DeviceSlotConfig(device="cpu"),
        skip_completed=False,
        resume=True,
    )

    assert "--resume" in resume_args
    assert "--no-resume" not in resume_args


def test_sweep_resume_defaults_to_enabled(tmp_artifacts, monkeypatch) -> None:
    monkeypatch.delenv(RUN_RESUME_ENV, raising=False)
    sweep = _smoke_sweep(skip_completed=False)

    result = SweepRunner(
        sweep,
        sweep_root=tmp_artifacts / "sweeps",
        dry_run=True,
    ).run()
    manifest = json.loads((result.sweep_dir / "manifest.json").read_text())

    assert result.results[0].command[-1] == "--resume"
    assert manifest["resume"] is True


def test_sweep_resume_env_can_disable(tmp_artifacts, monkeypatch) -> None:
    monkeypatch.setenv(RUN_RESUME_ENV, "0")
    sweep = _smoke_sweep(skip_completed=False)

    result = SweepRunner(
        sweep,
        sweep_root=tmp_artifacts / "sweeps",
        dry_run=True,
    ).run()
    manifest = json.loads((result.sweep_dir / "manifest.json").read_text())

    assert result.results[0].command[-1] == "--no-resume"
    assert manifest["resume"] is False


def test_sweep_resume_argument_overrides_env(tmp_artifacts, monkeypatch) -> None:
    monkeypatch.setenv(RUN_RESUME_ENV, "0")
    sweep = _smoke_sweep(skip_completed=False)

    result = SweepRunner(
        sweep,
        sweep_root=tmp_artifacts / "sweeps",
        resume=True,
        dry_run=True,
    ).run()
    manifest = json.loads((result.sweep_dir / "manifest.json").read_text())

    assert result.results[0].command[-1] == "--resume"
    assert manifest["resume"] is True


def test_job_progress_reader_updates_nested_train_bar(tmp_path, monkeypatch) -> None:
    class FakeBar:
        def __init__(self, **kwargs) -> None:
            self.total = kwargs["total"]
            self.desc = kwargs["desc"]
            self.unit = kwargs["unit"]
            self.n = 0
            self.postfixes: list[str] = []
            self.closed = False

        def update(self, value: int = 1) -> None:
            self.n += value

        def set_postfix_str(self, value: str, *, refresh: bool = True) -> None:
            self.postfixes.append(value)

        def close(self) -> None:
            self.closed = True

    bars: list[FakeBar] = []

    def fake_tqdm(**kwargs) -> FakeBar:
        bar = FakeBar(**kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr("drift_happens.runtime.sweep.tqdm", fake_tqdm)
    progress_path = tmp_path / "progress.jsonl"
    records = [
        {"event": "train_slices_started", "total_slices": 2},
        {
            "event": "train_slice_started",
            "total_slices": 2,
            "train_slice": "2000",
        },
        {
            "event": "train_slice_finished",
            "total_slices": 2,
            "train_slice": "2000",
        },
        {
            "event": "train_slice_skipped",
            "total_slices": 2,
            "train_slice": "2001",
        },
    ]
    progress_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    reader = _JobProgressReader(progress_path, label="fake seed=0", position=1)
    reader.refresh()
    reader.close("ok")

    assert len(bars) == 1
    assert bars[0].desc == "Train fake seed=0"
    assert bars[0].unit == "slice"
    assert bars[0].total == 2
    assert bars[0].n == 2
    assert bars[0].postfixes[-2:] == ["skipped 2001", "ok"]
    assert bars[0].closed


def test_job_progress_reader_switches_to_eval_bar(tmp_path, monkeypatch) -> None:
    class FakeBar:
        def __init__(self, **kwargs) -> None:
            self.total = kwargs["total"]
            self.desc = kwargs["desc"]
            self.unit = kwargs["unit"]
            self.n = 0
            self.postfixes: list[str] = []
            self.closed = False

        def update(self, value: int = 1) -> None:
            self.n += value

        def set_postfix_str(self, value: str, *, refresh: bool = True) -> None:
            self.postfixes.append(value)

        def close(self) -> None:
            self.closed = True

    bars: list[FakeBar] = []

    def fake_tqdm(**kwargs) -> FakeBar:
        bar = FakeBar(**kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr("drift_happens.runtime.sweep.tqdm", fake_tqdm)
    progress_path = tmp_path / "progress.jsonl"
    records = [
        {"event": "train_slices_started", "total_slices": 1},
        {
            "event": "train_slice_finished",
            "total_slices": 1,
            "train_slice": "2000",
        },
        {"event": "eval_cells_started", "total_cells": 2},
        {
            "event": "eval_cell_started",
            "total_cells": 2,
            "train_slice": "2000",
            "eval_slice": "2000",
        },
        {
            "event": "eval_cell_finished",
            "total_cells": 2,
            "train_slice": "2000",
            "eval_slice": "2000",
        },
        {
            "event": "eval_cell_skipped",
            "total_cells": 2,
            "train_slice": "2000",
            "eval_slice": "2001",
        },
    ]
    progress_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    reader = _JobProgressReader(progress_path, label="fake seed=0", position=1)
    reader.refresh()
    reader.close("ok")

    assert len(bars) == 2
    assert bars[0].desc == "Train fake seed=0"
    assert bars[0].closed
    assert bars[1].desc == "Eval fake seed=0"
    assert bars[1].unit == "cell"
    assert bars[1].total == 2
    assert bars[1].n == 2
    assert bars[1].postfixes[-3:] == [
        "done 2000->2000",
        "skipped 2000->2001",
        "ok",
    ]


def test_sweep_dry_run_records_planned_jobs(tmp_artifacts) -> None:
    sweep = _smoke_sweep()

    result = SweepRunner(
        sweep,
        sweep_root=tmp_artifacts / "sweeps",
        dry_run=True,
    ).run()

    assert result.all_ok
    results = json.loads((result.sweep_dir / "results.json").read_text())
    assert results[0]["status"] == "planned"


def test_sweep_skip_completed_local_returns_skipped_result(
    tmp_artifacts, monkeypatch
) -> None:
    monkeypatch.setattr(
        "drift_happens.runtime.sweep.local_seed_complete", lambda *a, **k: True
    )
    sweep = _smoke_sweep()

    result = SweepRunner(sweep, sweep_root=tmp_artifacts / "sweeps").run()

    assert result.all_ok
    assert result.results[0].status == "skipped"
    assert result.results[0].run_dir is not None


def test_sweep_progress_updates_for_skipped_jobs(
    tmp_artifacts, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "drift_happens.runtime.sweep.local_seed_complete", lambda *a, **k: True
    )
    sweep = _smoke_sweep()

    result = SweepRunner(
        sweep,
        sweep_root=tmp_artifacts / "sweeps",
        show_progress=True,
    ).run()

    captured = capsys.readouterr()
    assert result.all_ok
    assert "Sweep" in captured.err
    assert "1/1" in captured.err
    assert "skipped synthetic seed=0" in captured.err


def test_sweep_tees_worker_output_to_log_and_parent_stderr(
    tmp_artifacts,
    monkeypatch,
    capsys,
) -> None:
    def fake_spawn(job, slot, log_path, *, progress_path, skip_completed, resume):
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "print('worker stdout'); "
                    "print('worker stderr', file=sys.stderr)"
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if proc.stdout is not None:
            os.set_blocking(proc.stdout.fileno(), False)
        return proc

    monkeypatch.setattr("drift_happens.runtime.sweep._spawn_job", fake_spawn)
    sweep = _smoke_sweep(skip_completed=False)

    result = SweepRunner(sweep, sweep_root=tmp_artifacts / "sweeps").run()

    captured = capsys.readouterr()
    log_text = (result.sweep_dir / "logs" / "synthetic__seed=0.log").read_text()
    assert result.all_ok
    assert "[cpu synthetic seed=0] worker stdout" in captured.err
    assert "[cpu synthetic seed=0] worker stderr" in captured.err
    assert "worker stdout" in log_text
    assert "worker stderr" in log_text


def test_sweep_skip_completed_wandb_requires_wandb_config(
    tmp_artifacts, monkeypatch
) -> None:
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    sweep = _smoke_sweep(skip_source="wandb")

    with pytest.raises(RuntimeError, match="skip-source wandb requires"):
        SweepRunner(sweep, sweep_root=tmp_artifacts / "sweeps").run()


def test_sweep_reports_stage_contention_as_blocked(tmp_artifacts, monkeypatch) -> None:
    def fake_spawn(job, slot, log_path, *, progress_path, skip_completed, resume):
        return subprocess.Popen(
            [sys.executable, "-c", f"raise SystemExit({STAGE_CONTENTION_EXIT_CODE})"]
        )

    monkeypatch.setattr("drift_happens.runtime.sweep._spawn_job", fake_spawn)
    sweep = _smoke_sweep(skip_completed=False)

    result = SweepRunner(sweep, sweep_root=tmp_artifacts / "sweeps").run()

    assert not result.all_ok
    assert result.results[0].status == "blocked"
    results = json.loads((result.sweep_dir / "results.json").read_text())
    assert results[0]["status"] == "blocked"


def test_sweep_dispatch_error_reaps_children_and_writes_partial_results(
    tmp_artifacts, monkeypatch
) -> None:
    spawned: list[subprocess.Popen[bytes]] = []

    def fake_spawn(job, slot, log_path, *, progress_path, skip_completed, resume):
        # A few seconds is enough to outlive the dispatch loop; 60 s would
        # stall the suite on the regression path this test is designed to catch.
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
        spawned.append(proc)
        return proc

    monkeypatch.setattr("drift_happens.runtime.sweep._spawn_job", fake_spawn)
    sweep = SweepConfig.model_validate(
        {
            "name": "abort-smoke",
            "concurrency": 2,
            "jobs": [
                {
                    "config_path": _SMOKE_CONFIG_PATH,
                    "label": "running",
                    "seed": 0,
                    "skip_completed": False,
                },
                {
                    "config_path": "configs/does-not-exist.json",
                    "label": "broken",
                    "seed": 0,
                    "skip_completed": True,
                },
            ],
            "slots": [{"device": "cpu"}, {"device": "cpu"}],
        }
    )
    runner = SweepRunner(sweep, sweep_root=tmp_artifacts / "sweeps")

    try:
        # The broken job raises during its skip check while the first job is running.
        with pytest.raises(FileNotFoundError):
            runner.run()
    finally:
        for proc in spawned:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    assert spawned and spawned[0].poll() is not None
    results = json.loads((runner.sweep_dir / "results.json").read_text())
    assert [(entry["label"], entry["status"]) for entry in results] == [
        ("running", "failed")
    ]
    assert "interrupted" in results[0]["error_summary"]
    events = (runner.sweep_dir / "logs" / "events.jsonl").read_text()
    assert "job_interrupted" in events


def test_same_second_same_name_sweeps_get_distinct_dirs(
    tmp_artifacts, monkeypatch
) -> None:
    monkeypatch.setattr(
        "drift_happens.runtime.sweep.utc_timestamp",
        lambda *args, **kwargs: "20260611T120000Z",
    )
    sweep = _smoke_sweep()

    first = SweepRunner(sweep, sweep_root=tmp_artifacts / "sweeps", dry_run=True)
    second = SweepRunner(sweep, sweep_root=tmp_artifacts / "sweeps", dry_run=True)

    assert first.sweep_dir != second.sweep_dir
    assert second.sweep_dir.name == f"{first.sweep_dir.name}-1"
    assert first.run().all_ok and second.run().all_ok


def test_error_summary_reads_log_tail(tmp_path) -> None:
    total = _ERROR_TAIL + 3  # 3 lines that should be excluded from the tail
    log_path = tmp_path / "job.log"
    log_path.write_text("\n".join(f"line {idx}" for idx in range(total)))

    summary = _error_summary(log_path)

    assert summary == "\n".join(
        f"line {idx}" for idx in range(total - _ERROR_TAIL, total)
    )
