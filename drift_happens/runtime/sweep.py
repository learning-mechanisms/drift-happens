"""Local sweep runner that dispatches one experiment process per slot."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Literal

import yaml
from tqdm import tqdm

from drift_happens.configs import DeviceSlotConfig, JobSpecConfig, SweepConfig
from drift_happens.experiments.source import load_experiment_source
from drift_happens.runtime.completion_filter import (
    local_seed_complete,
    wandb_seed_complete,
)
from drift_happens.runtime.progress import SWEEP_PROGRESS_FILE_ENV
from drift_happens.runtime.run_store import resolve_run_store
from drift_happens.runtime.stages import STAGE_CONTENTION_EXIT_CODE
from drift_happens.utils.env import resolve_resume_setting, with_wandb_from_env
from drift_happens.utils.ids import slugify, utc_timestamp
from drift_happens.utils.paths import SWEEPS_DIR

JOB_STATUS_OK = "ok"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_BLOCKED = "blocked"
JOB_STATUS_SKIPPED = "skipped"
JOB_STATUS_PLANNED = "planned"
SkipSource = Literal["local", "wandb"]

_POLL_INTERVAL_S = 0.2
_TERMINATE_GRACE_S = 5.0
_ERROR_SUMMARY_TAIL_LINES = 12
_MAX_DIR_SUFFIXES = 1000


@dataclass(frozen=True, slots=True)
class JobResult:
    """Persisted outcome for one sweep job subprocess."""

    label: str
    seed: int
    status: str
    exit_code: int
    started_at: str
    ended_at: str
    log_path: str
    slot_label: str
    action: str = "run"
    command: tuple[str, ...] = ()
    run_dir: str | None = None
    error_summary: str | None = None


@dataclass(frozen=True, slots=True)
class SweepResult:
    """Outcome of a local sweep."""

    sweep_dir: Path
    results: tuple[JobResult, ...]
    all_ok: bool


@dataclass(slots=True)
class _JobProgressReader:
    """Read child-emitted progress events and render one nested worker bar."""

    path: Path
    label: str
    position: int
    offset: int = 0
    bar: Any | None = None
    phase: str | None = None
    completed_units: set[str] = field(default_factory=set)

    def refresh(self) -> None:
        """Read and apply any new progress events."""
        if not self.path.exists():
            return
        with self.path.open() as handle:
            handle.seek(self.offset)
            for line in handle:
                if not line.strip():
                    continue
                self._handle_event(json.loads(line))
            self.offset = handle.tell()

    def close(self, status: str) -> None:
        """Finalize the nested bar if it was shown."""
        self.refresh()
        if self.bar is None:
            return
        self.bar.set_postfix_str(status, refresh=True)
        self.bar.close()

    def _handle_event(self, event: dict[str, object]) -> None:
        event_name = str(event.get("event", ""))
        total_slices = _event_total_units(event, "total_slices")
        total_cells = _event_total_units(event, "total_cells")
        if event_name == "train_slices_started":
            self._ensure_bar(phase="Train", total=total_slices, unit="slice")
            return
        if event_name == "train_slice_started":
            self._ensure_bar(phase="Train", total=total_slices, unit="slice")
            train_slice = event.get("train_slice")
            if train_slice is not None and self.bar is not None:
                self.bar.set_postfix_str(f"running {train_slice}", refresh=True)
            return
        if event_name in {"train_slice_finished", "train_slice_skipped"}:
            self._ensure_bar(phase="Train", total=total_slices, unit="slice")
            train_slice = event.get("train_slice")
            if train_slice is None:
                return
            key = str(train_slice)
            self._advance_once(key)
            if self.bar is not None:
                status = "skipped" if event_name.endswith("skipped") else "done"
                self.bar.set_postfix_str(f"{status} {key}", refresh=True)
            return
        if event_name == "train_slice_failed":
            self._ensure_bar(phase="Train", total=total_slices, unit="slice")
            if self.bar is not None:
                train_slice = event.get("train_slice", "?")
                self.bar.set_postfix_str(f"failed {train_slice}", refresh=True)
            return

        if event_name == "eval_cells_started":
            self._ensure_bar(phase="Eval", total=total_cells, unit="cell")
            return
        if event_name == "eval_train_slice_started":
            self._ensure_bar(phase="Eval", total=total_cells, unit="cell")
            train_slice = event.get("train_slice")
            if train_slice is not None and self.bar is not None:
                self.bar.set_postfix_str(f"loading {train_slice}", refresh=True)
            return
        if event_name == "eval_cell_started":
            self._ensure_bar(phase="Eval", total=total_cells, unit="cell")
            key = _eval_cell_key(event)
            if self.bar is not None:
                self.bar.set_postfix_str(f"running {key}", refresh=True)
            return
        if event_name in {"eval_cell_finished", "eval_cell_skipped"}:
            self._ensure_bar(phase="Eval", total=total_cells, unit="cell")
            key = _eval_cell_key(event)
            self._advance_once(key)
            if self.bar is not None:
                status = "skipped" if event_name.endswith("skipped") else "done"
                self.bar.set_postfix_str(f"{status} {key}", refresh=True)
            return
        if event_name in {"eval_cell_failed", "eval_train_slice_failed"}:
            self._ensure_bar(phase="Eval", total=total_cells, unit="cell")
            if self.bar is not None:
                key = _eval_cell_key(event)
                self.bar.set_postfix_str(f"failed {key}", refresh=True)

    def _ensure_bar(self, *, phase: str, total: int | None, unit: str) -> None:
        if self.bar is not None and self.phase != phase:
            self.bar.close()
            self.bar = None
            self.completed_units.clear()
        if self.bar is not None:
            if total is not None:
                self.bar.total = total
            return
        self.phase = phase
        self.bar = tqdm(
            total=total,
            desc=f"{phase} {self.label}",
            unit=unit,
            position=self.position,
            leave=False,
            colour="blue",
            dynamic_ncols=True,
        )

    def _advance_once(self, key: str) -> None:
        if key in self.completed_units:
            return
        self.completed_units.add(key)
        if self.bar is not None:
            self.bar.update(1)


@dataclass(slots=True)
class _ActiveJob:
    """One child process currently owned by a sweep slot."""

    proc: subprocess.Popen[bytes]
    job: JobSpecConfig
    slot: DeviceSlotConfig
    started_at: str
    log_path: Path
    command: list[str]
    output: _JobOutput
    progress: _JobProgressReader | None = None


@dataclass(slots=True)
class _JobOutput:
    """Tee one worker's stdout to its log file and the parent terminal."""

    handle: BinaryIO
    prefix: str
    buffer: bytearray = field(default_factory=bytearray)

    def drain(self, proc: subprocess.Popen[bytes]) -> None:
        if proc.stdout is None:
            return
        while True:
            try:
                chunk = os.read(proc.stdout.fileno(), 65536)
            except BlockingIOError:
                break
            except OSError:
                break
            if not chunk:
                break
            self.handle.write(chunk)
            self.handle.flush()
            self._emit_terminal_lines(chunk)

    def close(self) -> None:
        if self.buffer:
            self._emit_terminal_line(bytes(self.buffer))
            self.buffer.clear()
        if not self.handle.closed:
            self.handle.close()

    def _emit_terminal_lines(self, chunk: bytes) -> None:
        self.buffer.extend(chunk)
        while True:
            newline = self.buffer.find(b"\n")
            carriage = self.buffer.find(b"\r")
            endings = [idx for idx in (newline, carriage) if idx >= 0]
            if not endings:
                return
            end = min(endings)
            line = bytes(self.buffer[:end])
            del self.buffer[: end + 1]
            self._emit_terminal_line(line)

    def _emit_terminal_line(self, raw_line: bytes) -> None:
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        if line:
            tqdm.write(f"[{self.prefix}] {line}", file=sys.stderr)


def load_sweep_config(path: Path) -> SweepConfig:
    """Load a sweep YAML or JSON file."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"sweep config {path} is not a mapping")
    return SweepConfig.model_validate(data)


class SweepRunner:
    """Spawn at most ``concurrency`` child run processes across configured slots."""

    def __init__(
        self,
        sweep: SweepConfig,
        *,
        sweep_root: Path | None = None,
        skip_completed: bool | None = None,
        skip_source: SkipSource | None = None,
        resume: bool | None = None,
        dry_run: bool = False,
        show_progress: bool = False,
    ) -> None:
        self.sweep = sweep
        self.skip_completed = (
            sweep.skip_completed if skip_completed is None else skip_completed
        )
        self.skip_source = skip_source or sweep.skip_source
        self.resume = resolve_resume_setting(sweep.resume if resume is None else resume)
        self.dry_run = dry_run
        self.show_progress = show_progress
        self.sweep_dir = _create_sweep_dir(
            (sweep_root or SWEEPS_DIR) / f"{utc_timestamp()}__{slugify(sweep.name)}"
        )
        self.log_dir = self.sweep_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> SweepResult:
        """Run all jobs and write manifest/results files."""
        self._write_manifest()
        if self.dry_run:
            planned_results = [self._planned_result(job) for job in self.sweep.jobs]
            self._write_results(planned_results)
            return SweepResult(
                sweep_dir=self.sweep_dir,
                results=tuple(planned_results),
                all_ok=True,
            )

        free_slots = list(self.sweep.slots[: self.sweep.concurrency])
        progress_positions = {
            id(slot): position for position, slot in enumerate(free_slots, start=1)
        }
        queue = list(self.sweep.jobs)
        active: list[_ActiveJob] = []
        results: list[JobResult] = []
        progress_bar = _progress_bar(
            total=len(self.sweep.jobs),
            enabled=self.show_progress,
        )

        try:
            self._dispatch_and_collect(
                queue,
                free_slots,
                active,
                results,
                progress_positions=progress_positions,
                progress_bar=progress_bar,
            )
        except BaseException:
            # A bad job config (or Ctrl-C) must not orphan running children or
            # leave the sweep without a results file.
            self._abort_active_jobs(active, results, progress_bar=progress_bar)
            self._write_results(results)
            raise
        finally:
            if progress_bar is not None:
                progress_bar.close()

        self._write_results(results)
        return SweepResult(
            sweep_dir=self.sweep_dir,
            results=tuple(results),
            all_ok=all(
                result.status in {JOB_STATUS_OK, JOB_STATUS_SKIPPED, JOB_STATUS_PLANNED}
                for result in results
            ),
        )

    def _dispatch_and_collect(
        self,
        queue: list[JobSpecConfig],
        free_slots: list[DeviceSlotConfig],
        active: list[_ActiveJob],
        results: list[JobResult],
        *,
        progress_positions: dict[int, int],
        progress_bar: Any | None,
    ) -> None:
        while queue or active:
            while queue and free_slots:
                job = queue.pop(0)
                slot = free_slots.pop(0)
                _set_progress_message(
                    progress_bar,
                    f"checking {job.label} seed={job.seed}",
                )
                skipped = self._skip_result_if_complete(job, slot)
                if skipped is not None:
                    results.append(skipped)
                    _advance_progress(
                        progress_bar,
                        f"skipped {job.label} seed={job.seed}",
                    )
                    free_slots.append(slot)
                    continue
                started_at = datetime.now(UTC).isoformat()
                log_path = self.log_dir / _job_log_name(job)
                command = _job_args(
                    job,
                    slot=slot,
                    skip_completed=False,
                    resume=self.resume,
                )
                self._write_event(
                    "job_started",
                    label=job.label,
                    seed=job.seed,
                    slot=slot.label or slot.device,
                    command=command,
                )
                output = _JobOutput(
                    handle=log_path.open("ab"),
                    prefix=_job_output_prefix(job, slot),
                )
                try:
                    proc = _spawn_job(
                        job,
                        slot,
                        log_path,
                        progress_path=self._job_progress_path(job)
                        if self.show_progress
                        else None,
                        skip_completed=False,
                        resume=self.resume,
                    )
                except BaseException:
                    output.close()
                    raise
                progress = self._job_progress_reader(
                    job,
                    position=progress_positions[id(slot)],
                )
                active.append(
                    _ActiveJob(
                        proc=proc,
                        job=job,
                        slot=slot,
                        started_at=started_at,
                        log_path=log_path,
                        command=command,
                        output=output,
                        progress=progress,
                    )
                )
                _set_progress_message(
                    progress_bar,
                    f"running={len(active)} queued={len(queue)}",
                )

            if not active:
                break

            time.sleep(_POLL_INTERVAL_S)
            still_active = []
            for active_job in active:
                active_job.output.drain(active_job.proc)
                if active_job.progress is not None:
                    active_job.progress.refresh()
                return_code = active_job.proc.poll()
                if return_code is None:
                    still_active.append(active_job)
                    continue
                active_job.output.drain(active_job.proc)
                active_job.output.close()
                ended_at = datetime.now(UTC).isoformat()
                if return_code == 0:
                    status = JOB_STATUS_OK
                elif return_code == STAGE_CONTENTION_EXIT_CODE:
                    status = JOB_STATUS_BLOCKED
                else:
                    status = JOB_STATUS_FAILED
                job = active_job.job
                slot = active_job.slot
                self._write_event(
                    "job_finished",
                    label=job.label,
                    seed=job.seed,
                    status=status,
                    return_code=return_code,
                )
                results.append(
                    JobResult(
                        action=job.action,
                        command=tuple(active_job.command),
                        label=job.label,
                        seed=job.seed,
                        status=status,
                        exit_code=return_code,
                        started_at=active_job.started_at,
                        ended_at=ended_at,
                        log_path=str(active_job.log_path),
                        slot_label=slot.label or slot.device,
                        error_summary=_error_summary(active_job.log_path)
                        if return_code != 0
                        else None,
                    )
                )
                if active_job.progress is not None:
                    active_job.progress.close(status)
                _advance_progress(
                    progress_bar,
                    f"{status} {job.label} seed={job.seed}",
                )
                free_slots.append(slot)
            active[:] = still_active

    def _abort_active_jobs(
        self,
        active: list[_ActiveJob],
        results: list[JobResult],
        *,
        progress_bar: Any | None,
    ) -> None:
        """Terminate, reap, and record every still-running child."""
        for active_job in active:
            if active_job.proc.poll() is None:
                active_job.proc.terminate()
        for active_job in active:
            try:
                active_job.proc.wait(timeout=_TERMINATE_GRACE_S)
            except subprocess.TimeoutExpired:
                active_job.proc.kill()
                active_job.proc.wait()
            active_job.output.drain(active_job.proc)
            active_job.output.close()
            ended_at = datetime.now(UTC).isoformat()
            exit_code = (
                active_job.proc.returncode
                if active_job.proc.returncode is not None
                else -1
            )
            job = active_job.job
            slot = active_job.slot
            self._write_event(
                "job_interrupted",
                label=job.label,
                seed=job.seed,
                return_code=exit_code,
            )
            results.append(
                JobResult(
                    action=job.action,
                    command=tuple(active_job.command),
                    label=job.label,
                    seed=job.seed,
                    status=JOB_STATUS_FAILED,
                    exit_code=exit_code,
                    started_at=active_job.started_at,
                    ended_at=ended_at,
                    log_path=str(active_job.log_path),
                    slot_label=slot.label or slot.device,
                    error_summary="interrupted: sweep aborted while the job was running",
                )
            )
            if active_job.progress is not None:
                active_job.progress.close(JOB_STATUS_FAILED)
            _advance_progress(
                progress_bar,
                f"failed {job.label} seed={job.seed}",
            )
        active.clear()

    def _write_manifest(self) -> None:
        payload = {
            "concurrency": self.sweep.concurrency,
            "jobs": [job.model_dump(mode="json") for job in self.sweep.jobs],
            "name": self.sweep.name,
            "seeds": list(self.sweep.seeds),
            "resume": self.resume,
            "skip_completed": self.skip_completed,
            "skip_source": self.skip_source,
            "slots": [slot.model_dump(mode="json") for slot in self.sweep.slots],
            "dry_run": self.dry_run,
        }
        (self.sweep_dir / "manifest.json").write_text(_json(payload))

    def _write_results(self, results: list[JobResult]) -> None:
        payload = [asdict(result) for result in results]
        (self.sweep_dir / "results.json").write_text(_json(payload))

    def _write_event(self, event: str, **payload: object) -> None:
        record = {
            "event": event,
            "timestamp": datetime.now(UTC).isoformat(),
            **payload,
        }
        with (self.log_dir / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _job_progress_reader(
        self,
        job: JobSpecConfig,
        *,
        position: int,
    ) -> _JobProgressReader | None:
        if not self.show_progress:
            return None
        return _JobProgressReader(
            path=self._job_progress_path(job),
            label=f"{job.label} seed={job.seed}",
            position=position,
        )

    def _job_progress_path(self, job: JobSpecConfig) -> Path:
        return self.log_dir / f"{slugify(job.label)}__seed={job.seed}.progress.jsonl"

    def _planned_result(self, job: JobSpecConfig) -> JobResult:
        command = _job_args(
            job,
            slot=self.sweep.slots[0],
            skip_completed=False,
            resume=self.resume,
        )
        now = datetime.now(UTC).isoformat()
        return JobResult(
            action=job.action,
            command=tuple(command),
            label=job.label,
            seed=job.seed,
            status=JOB_STATUS_PLANNED,
            exit_code=0,
            started_at=now,
            ended_at=now,
            log_path="",
            slot_label="",
        )

    def _skip_result_if_complete(
        self,
        job: JobSpecConfig,
        slot: DeviceSlotConfig,
    ) -> JobResult | None:
        job_skip = (
            self.skip_completed if job.skip_completed is None else job.skip_completed
        )
        if not job_skip:
            return None
        source = load_experiment_source(
            job.config_path,
            overrides=(*job.overrides, _device_override(slot)),
        )
        cfg = source.config.model_copy(update={"seed": job.seed})
        cfg = with_wandb_from_env(cfg)
        store = resolve_run_store(cfg, source_path=source.path)
        if self.skip_source == "local":
            complete = local_seed_complete(store.identity, seed=job.seed)
        else:
            wandb_cfg = cfg.logging.wandb
            if wandb_cfg is None or wandb_cfg.mode == "disabled":
                raise RuntimeError(
                    "--skip-source wandb requires cfg.logging.wandb, WANDB_PROJECT, "
                    "or --wandb-project in the child environment"
                )
            complete = wandb_seed_complete(
                store.identity,
                seed=job.seed,
                wandb_cfg=wandb_cfg,
            )
        if not complete:
            return None
        now = datetime.now(UTC).isoformat()
        command = _job_args(
            job,
            slot=slot,
            skip_completed=False,
            resume=self.resume,
        )
        self._write_event("job_skipped", label=job.label, seed=job.seed)
        return JobResult(
            action=job.action,
            command=tuple(command),
            label=job.label,
            seed=job.seed,
            status=JOB_STATUS_SKIPPED,
            exit_code=0,
            started_at=now,
            ended_at=now,
            log_path="",
            run_dir=str(store.run_dir),
            slot_label=slot.label or slot.device,
        )


def _progress_bar(*, total: int, enabled: bool) -> Any | None:
    if not enabled:
        return None
    return tqdm(
        total=total,
        desc="Sweep",
        unit="job",
        position=0,
        colour="green",
        dynamic_ncols=True,
    )


def _set_progress_message(progress_bar: Any | None, message: str) -> None:
    if progress_bar is None:
        return
    progress_bar.set_postfix_str(message, refresh=True)


def _advance_progress(progress_bar: Any | None, message: str) -> None:
    if progress_bar is None:
        return
    progress_bar.set_postfix_str(message, refresh=False)
    progress_bar.update()


def _event_total_units(event: dict[str, object], key: str) -> int | None:
    raw = event.get(key)
    if isinstance(raw, int):
        return raw
    return None


def _eval_cell_key(event: dict[str, object]) -> str:
    train_slice = event.get("train_slice", "?")
    eval_slice = event.get("eval_slice")
    if eval_slice is None:
        return str(train_slice)
    return f"{train_slice}->{eval_slice}"


def _create_sweep_dir(base: Path) -> Path:
    """
    Create ``base`` or a ``-N`` suffixed sibling.

    The timestamp in the sweep dir name is second-granular, so two sweeps with the same
    name started in the same second must not share (and overwrite) one directory.
    """
    for suffix in range(_MAX_DIR_SUFFIXES):
        candidate = base if suffix == 0 else base.with_name(f"{base.name}-{suffix}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"could not create a unique sweep directory near {base}")


def _spawn_job(
    job: JobSpecConfig,
    slot: DeviceSlotConfig,
    log_path: Path,
    *,
    progress_path: Path | None,
    skip_completed: bool,
    resume: bool,
) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = _job_env(slot, progress_path=progress_path)
    args = _job_args(job, slot=slot, skip_completed=skip_completed, resume=resume)
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    if proc.stdout is not None:
        os.set_blocking(proc.stdout.fileno(), False)
    return proc


def _job_args(
    job: JobSpecConfig,
    *,
    slot: DeviceSlotConfig,
    skip_completed: bool,
    resume: bool,
) -> list[str]:
    args = [
        sys.executable,
        "-m",
        "drift_happens.cli.main",
        "experiment",
        job.action,
        str(job.config_path),
        "--seed",
        str(job.seed),
    ]
    for override in (*job.overrides, _device_override(slot)):
        args.extend(["--set", override])
    if skip_completed:
        args.append("--skip-completed")
    args.append("--resume" if resume else "--no-resume")
    return args


def _job_env(
    slot: DeviceSlotConfig,
    *,
    progress_path: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if slot.device == "cuda" and slot.device_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(slot.device_index)
    elif slot.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    if progress_path is None:
        env.pop(SWEEP_PROGRESS_FILE_ENV, None)
    else:
        env[SWEEP_PROGRESS_FILE_ENV] = str(progress_path)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _device_override(slot: DeviceSlotConfig) -> str:
    if slot.device == "cuda":
        return "runtime.device=cuda"
    if slot.device == "mps":
        return "runtime.device=mps"
    return "runtime.device=cpu"


def _job_log_name(job: JobSpecConfig) -> str:
    return f"{slugify(job.label)}__seed={job.seed}.log"


def _job_output_prefix(job: JobSpecConfig, slot: DeviceSlotConfig) -> str:
    return f"{slot.label or slot.device} {job.label} seed={job.seed}"


def _json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _error_summary(log_path: Path) -> str | None:
    try:
        lines = log_path.read_text(errors="ignore").splitlines()
    except OSError:
        return None
    tail = [line.strip() for line in lines[-_ERROR_SUMMARY_TAIL_LINES:] if line.strip()]
    return "\n".join(tail) if tail else None
