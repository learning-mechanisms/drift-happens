from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from drift_happens.configs import RunIdentity
from drift_happens.experiments.registry import preset
from drift_happens.runtime.local import run_stage
from drift_happens.runtime.stages import write_json_atomic
from drift_happens.utils.artifacts import (
    DeletionPlanItem,
    apply_gc,
    list_artifacts,
    plan_gc,
    plan_wandb_deleted_run_gc,
)


def test_list_artifacts_reads_canonical_run_status(tmp_artifacts: Path) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    run_stage(cfg, stage="train", runs_root=tmp_artifacts / "runs")

    rows = list_artifacts(root=tmp_artifacts, kind="runs")

    assert len(rows) == 1
    assert rows[0].dataset == "synthetic"
    assert rows[0].train == "ok"
    assert rows[0].eval == "missing"
    assert rows[0].status == "partial"


def test_gc_dry_plan_and_apply_delete_only_old_attempts(tmp_artifacts: Path) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    result = run_stage(cfg, stage="train", runs_root=tmp_artifacts / "runs")
    stage_attempts = result.run_dir / "attempts" / "train"
    for index in range(4):
        (stage_attempts / f"2026-01-01T00-00-0{index}Z__abc").mkdir(
            parents=True,
            exist_ok=True,
        )

    items = plan_gc(root=tmp_artifacts, keep_attempts=2)
    deleted = apply_gc(items, root=tmp_artifacts)

    assert len(deleted) >= 2
    assert result.run_dir.exists()


def test_list_artifacts_tolerates_malformed_run_manifest(tmp_artifacts: Path) -> None:
    run_dir = tmp_artifacts / "runs" / "bad"
    run_dir.mkdir(parents=True)
    write_json_atomic(run_dir / "run_manifest.json", {"seed": 0, "identity": "bad"})

    rows = list_artifacts(root=tmp_artifacts, kind="runs")

    assert len(rows) == 1
    assert rows[0].status == "partial"


def test_list_artifacts_marks_sweep_failed_when_any_job_failed(
    tmp_artifacts: Path,
) -> None:
    sweep_dir = tmp_artifacts / "sweeps" / "sweep"
    sweep_dir.mkdir(parents=True)
    write_json_atomic(sweep_dir / "manifest.json", {"name": "unit"})
    write_json_atomic(
        sweep_dir / "results.json", [{"status": "ok"}, {"status": "failed"}]
    )

    rows = list_artifacts(root=tmp_artifacts, kind="sweeps")

    assert rows[0].experiment == "unit"
    assert rows[0].status == "failed"


def test_gc_keeps_newest_sweep_attempts_per_name(tmp_artifacts: Path) -> None:
    sweeps = tmp_artifacts / "sweeps"
    for index in range(4):
        (sweeps / f"2026-01-01T00-00-0{index}Z__demo").mkdir(parents=True)
    (sweeps / "2026-01-01T00-00-00Z__other").mkdir(parents=True)

    items = plan_gc(root=tmp_artifacts, keep_attempts=2)

    assert {item.path.name for item in items} == {
        "2026-01-01T00-00-00Z__demo",
        "2026-01-01T00-00-01Z__demo",
    }


def test_apply_gc_unlinks_symlink_without_deleting_its_target(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    target = root / "runs" / "real"
    target.mkdir(parents=True)
    (target / "weights.bin").write_bytes(b"payload")
    link = root / "runs" / "link"
    link.symlink_to(target, target_is_directory=True)

    apply_gc((DeletionPlanItem(path=link, reason="old attempt"),), root=root)

    assert not link.is_symlink()
    assert target.is_dir()
    assert (target / "weights.bin").read_bytes() == b"payload"


def test_plan_wandb_deleted_run_gc_only_selects_remote_missing_runs(
    tmp_artifacts: Path,
) -> None:
    deleted = _write_local_wandb_run(tmp_artifacts, group="deleted", run_id="gone")
    kept_by_id = _write_local_wandb_run(tmp_artifacts, group="kept-by-id", run_id="id1")
    kept_by_identity = _write_local_wandb_run(
        tmp_artifacts,
        group="kept-by-identity",
        run_id="old-local",
    )
    _write_local_wandb_run(tmp_artifacts, group="no-wandb", run_id=None)
    api = _FakeWandbApi(
        {
            "kept-by-id": (
                _wandb_run(
                    "id1",
                    group="kept-by-id",
                    config_hash="other",
                    completion_hash="other-completion",
                ),
            ),
            "kept-by-identity": (_wandb_run("replacement", group="kept-by-identity"),),
        }
    )

    items = plan_wandb_deleted_run_gc(
        project="project",
        entity="entity",
        root=tmp_artifacts,
        api_factory=lambda: api,
    )

    assert [item.path for item in items] == [deleted]
    assert items[0].reason == "deleted wandb run ids=gone"
    assert kept_by_id.exists()
    assert kept_by_identity.exists()


def test_apply_gc_can_delete_wandb_deleted_run_plan(tmp_artifacts: Path) -> None:
    run_dir = _write_local_wandb_run(tmp_artifacts, group="deleted", run_id="gone")
    items = plan_wandb_deleted_run_gc(
        project="project",
        entity="entity",
        root=tmp_artifacts,
        api_factory=lambda: _FakeWandbApi({}),
    )

    deleted = apply_gc(items, root=tmp_artifacts)

    assert deleted == (run_dir,)
    assert not run_dir.exists()


class _FakeWandbApi:
    def __init__(self, runs_by_group: dict[str, tuple[SimpleNamespace, ...]]) -> None:
        self._runs_by_group = runs_by_group

    def runs(self, path: str, filters: dict[str, str]) -> tuple[SimpleNamespace, ...]:
        assert path == "entity/project"
        return self._runs_by_group.get(filters["group"], ())


def _identity(group: str) -> RunIdentity:
    return RunIdentity(
        source_identity=group,
        config_hash="cfg",
        completion_hash="completion",
        snapshot_sha256="snap",
        wandb_group=group,
        wandb_run_name=f"{group}__seed=0__eval",
    )


def _write_local_wandb_run(
    root: Path,
    *,
    group: str,
    run_id: str | None,
) -> Path:
    identity = _identity(group)
    run_dir = root / "runs" / "dataset" / group / "experiment" / "seed=0" / "run"
    run_dir.mkdir(parents=True)
    write_json_atomic(
        run_dir / "run_manifest.json",
        {
            "dataset": "dataset",
            "experiment": "experiment",
            "identity": identity.model_dump(mode="json"),
            "seed": 0,
            "trainer": group,
        },
    )
    if run_id is not None:
        wandb_dir = run_dir / "wandb" / f"run-20260101_000000-{run_id}"
        wandb_dir.mkdir(parents=True)
        (wandb_dir / f"run-{run_id}.wandb").write_bytes(b"")
    return run_dir


def _wandb_run(
    run_id: str,
    *,
    group: str,
    config_hash: str = "cfg",
    completion_hash: str = "completion",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=run_id,
        state="finished",
        config={
            "seed": 0,
            "run/config_hash": config_hash,
            "run/completion_hash": completion_hash,
            "run/snapshot_sha256": "snap",
            "run/wandb_group": group,
        },
        summary={},
    )
