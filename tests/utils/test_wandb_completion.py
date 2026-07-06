from __future__ import annotations

from types import SimpleNamespace

from drift_happens.configs import RunIdentity, WandbConfig
from drift_happens.experiments.registry import preset
from drift_happens.runtime.completion_filter import wandb_seed_statuses_by_identity
from drift_happens.utils.wandb_completion import WandbCompletionIndex
from drift_happens.utils.wandb_identity import completion_hash


class FakeApi:
    def __init__(self, runs):
        self._runs = runs

    def runs(self, path, filters):
        assert path == "entity/project"
        assert filters == {"group": "group"}
        return self._runs


_IDENTITY = RunIdentity(
    source_identity="source",
    config_hash="cfg",
    snapshot_sha256="snap",
    wandb_group="group",
    wandb_run_name="group__seed=0__eval",
)

_WANDB_CFG = WandbConfig(project="project", entity="entity")


def _make_stage_run(
    stage: str, *, seed: int = 0, run_complete: bool = False
) -> SimpleNamespace:
    summary: dict = {"stage/complete": True, "stage/exit_status": "ok"}
    if run_complete:
        summary["run/complete"] = True
        summary["run/exit_status"] = "ok"
    return SimpleNamespace(
        id=stage,
        state="finished",
        config={
            "seed": seed,
            "run/config_hash": "cfg",
            "run/snapshot_sha256": "snap",
            "run/stage": stage,
        },
        summary=summary,
    )


def _make_raw_run(
    run_id: str,
    *,
    state: str,
    stage: str = "eval",
    seed: int = 0,
    config: dict | None = None,
    summary: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=run_id,
        state=state,
        config={
            "seed": seed,
            "run/config_hash": "cfg",
            "run/snapshot_sha256": "snap",
            "run/stage": stage,
            **(config or {}),
        },
        summary=summary or {},
    )


def test_wandb_seed_statuses_are_stage_aware() -> None:
    runs = [
        _make_stage_run("train"),
        _make_stage_run("eval", run_complete=True),
    ]

    row = wandb_seed_statuses_by_identity(
        {0: _IDENTITY},
        wandb_cfg=_WANDB_CFG,
        api_factory=lambda: FakeApi(runs),
    )[0]

    assert (row.train, row.eval, row.status) == ("ok", "ok", "ok")


def test_wandb_seed_statuses_partial_when_stages_ok_but_run_not_complete() -> None:
    # both stages finished but run/complete not set → status must be "partial"
    runs = [
        _make_stage_run("train"),
        _make_stage_run("eval"),
    ]

    row = wandb_seed_statuses_by_identity(
        {0: _IDENTITY},
        wandb_cfg=_WANDB_CFG,
        api_factory=lambda: FakeApi(runs),
    )[0]

    assert (row.train, row.eval, row.status) == ("ok", "ok", "partial")


def test_wandb_seed_statuses_missing_when_only_train_complete() -> None:
    # only train stage finished; eval and run/complete absent → eval and status "missing"
    runs = [_make_stage_run("train")]

    row = wandb_seed_statuses_by_identity(
        {0: _IDENTITY},
        wandb_cfg=_WANDB_CFG,
        api_factory=lambda: FakeApi(runs),
    )[0]

    assert (row.train, row.eval, row.status) == ("ok", "missing", "missing")


def test_wandb_seed_statuses_missing_when_no_runs() -> None:
    row = wandb_seed_statuses_by_identity(
        {0: _IDENTITY},
        wandb_cfg=_WANDB_CFG,
        api_factory=lambda: FakeApi([]),
    )[0]

    assert (row.train, row.eval, row.status) == ("missing", "missing", "missing")


def test_wandb_preflight_skips_running_matching_stage() -> None:
    index = WandbCompletionIndex(
        project="project",
        entity="entity",
        api_factory=lambda: FakeApi([_make_raw_run("run-1", state="running")]),
    )

    status = index.preflight_status(
        group="group",
        seed=0,
        stage="eval",
        config_hash="cfg",
        snapshot_sha256="snap",
    )

    assert status.state == "running"
    assert status.should_run is False
    assert status.run_ids == ("run-1",)


def test_wandb_preflight_retries_failed_stage_below_budget() -> None:
    index = WandbCompletionIndex(
        project="project",
        entity="entity",
        api_factory=lambda: FakeApi(
            [
                _make_raw_run(
                    "failed-1",
                    state="finished",
                    summary={"stage/exit_status": "error"},
                )
            ]
        ),
    )

    status = index.preflight_status(
        group="group",
        seed=0,
        stage="eval",
        config_hash="cfg",
        snapshot_sha256="snap",
        max_failed_attempts=3,
    )

    assert status.state == "retry"
    assert status.should_run is True
    assert status.failed_attempts == 1


def test_wandb_preflight_exhausts_failed_attempt_budget() -> None:
    index = WandbCompletionIndex(
        project="project",
        entity="entity",
        api_factory=lambda: FakeApi(
            [
                _make_raw_run("failed-1", state="failed"),
                _make_raw_run("failed-2", state="crashed"),
                _make_raw_run(
                    "failed-3",
                    state="finished",
                    summary={"run/exit_status": "error"},
                ),
            ]
        ),
    )

    status = index.preflight_status(
        group="group",
        seed=0,
        config_hash="cfg",
        snapshot_sha256="snap",
        max_failed_attempts=3,
    )

    assert status.state == "retry_exhausted"
    assert status.should_run is False
    assert status.failed_attempts == 3


def test_wandb_preflight_skips_complete_seed() -> None:
    index = WandbCompletionIndex(
        project="project",
        entity="entity",
        api_factory=lambda: FakeApi([_make_stage_run("eval", run_complete=True)]),
    )

    status = index.preflight_status(
        group="group",
        seed=0,
        config_hash="cfg",
        snapshot_sha256="snap",
    )

    assert status.state == "complete"
    assert status.should_run is False


def test_wandb_matching_run_ids_return_matching_identity_without_state_filter() -> None:
    index = WandbCompletionIndex(
        project="project",
        entity="entity",
        api_factory=lambda: FakeApi(
            [
                _make_raw_run("running-match", state="running"),
                _make_raw_run(
                    "wrong-config",
                    state="finished",
                    config={"run/config_hash": "other"},
                ),
            ]
        ),
    )

    run_ids = index.matching_run_ids(
        group="group",
        seed=0,
        config_hash="cfg",
        snapshot_sha256="snap",
    )

    assert run_ids == ("running-match",)


def test_wandb_run_ids_return_group_seed_ids_without_identity_filter() -> None:
    index = WandbCompletionIndex(
        project="project",
        entity="entity",
        api_factory=lambda: FakeApi(
            [
                _make_raw_run("run-1", state="finished"),
                _make_raw_run(
                    "wrong-config",
                    state="finished",
                    config={"run/config_hash": "other"},
                ),
                _make_raw_run("wrong-seed", state="finished", seed=1),
            ]
        ),
    )

    run_ids = index.run_ids(group="group", seed=0)

    assert run_ids == ("run-1", "wrong-config")


def test_wandb_preflight_matches_old_run_by_stored_config_completion_hash() -> None:
    old_cfg = preset("yearbook-conference", "mlp_s").build()
    new_metadata = dict(old_cfg.metadata)
    seed_metadata = dict(new_metadata["seeds"])
    seed_metadata["model_seeds"] = [0, 1, 2, 3, 4]
    new_metadata["seeds"] = seed_metadata
    new_cfg = old_cfg.model_copy(update={"metadata": new_metadata})
    run = _make_raw_run(
        "old-run",
        state="finished",
        config={
            "config": old_cfg.model_dump(mode="json"),
            "run/config_hash": "old-cfg",
            "run/snapshot_sha256": "old-snap",
        },
        summary={"run/complete": True, "run/exit_status": "ok"},
    )
    index = WandbCompletionIndex(
        project="project",
        entity="entity",
        api_factory=lambda: FakeApi([run]),
    )

    status = index.preflight_status(
        group="group",
        seed=0,
        config_hash="new-cfg",
        snapshot_sha256="new-snap",
        completion_hash=completion_hash(new_cfg),
    )

    assert status.state == "complete"
    assert status.run_ids == ("old-run",)
