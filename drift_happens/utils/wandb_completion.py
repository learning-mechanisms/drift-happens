"""W&B completion checks keyed by group, seed, stage, and config identity."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from drift_happens.runtime.stages import RunStage
from drift_happens.utils.wandb_identity import completion_hash_from_config_payload


@dataclass(frozen=True, slots=True)
class WandbRunSnapshot:
    """Minimal W&B run view needed for completion predicates."""

    run_id: str
    state: str
    config: dict[str, Any]
    summary: dict[str, Any]


WandbPreflightState = Literal[
    "missing",
    "retry",
    "retry_exhausted",
    "running",
    "complete",
]


@dataclass(frozen=True, slots=True)
class WandbPreflightStatus:
    """W&B-backed decision inputs for one config/seed or stage."""

    state: WandbPreflightState
    failed_attempts: int = 0
    max_failed_attempts: int = 3
    run_ids: tuple[str, ...] = ()

    @property
    def should_run(self) -> bool:
        """Whether the caller should start work for this config/seed."""
        return self.state in {"missing", "retry"}


WandbApiFactory = Callable[[], Any]


def _default_api_factory() -> Any:
    import wandb

    return wandb.Api()


@dataclass
class WandbCompletionIndex:
    """Cache W&B runs by group and test exact stage/run completion identity."""

    project: str
    entity: str | None = None
    api_factory: WandbApiFactory = field(default=_default_api_factory)
    _api: Any = field(default=None, init=False, repr=False)
    _loaded_groups: set[str] = field(default_factory=set, init=False, repr=False)
    _by_group_seed: dict[tuple[str, int], list[WandbRunSnapshot]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def is_run_complete(
        self,
        *,
        group: str,
        seed: int,
        config_hash: str,
        snapshot_sha256: str,
        completion_hash: str | None = None,
    ) -> bool:
        """Return true iff a finished W&B run matches a complete seed run."""
        return any(
            _matches_common(
                run,
                config_hash=config_hash,
                snapshot_sha256=snapshot_sha256,
                completion_hash=completion_hash,
            )
            and _truthy(_lookup(run, "run/complete"))
            and _lookup(run, "run/exit_status") == "ok"
            for run in self._runs(group, seed)
        )

    def is_stage_complete(
        self,
        *,
        group: str,
        seed: int,
        stage: RunStage,
        config_hash: str,
        snapshot_sha256: str,
        completion_hash: str | None = None,
    ) -> bool:
        """Return true iff a finished W&B run matches a complete stage."""
        return any(
            _matches_common(
                run,
                config_hash=config_hash,
                snapshot_sha256=snapshot_sha256,
                completion_hash=completion_hash,
            )
            and _lookup(run, "run/stage") == stage
            and _truthy(_lookup(run, "stage/complete"))
            and _lookup(run, "stage/exit_status") == "ok"
            for run in self._runs(group, seed)
        )

    def preflight_status(
        self,
        *,
        group: str,
        seed: int,
        config_hash: str,
        snapshot_sha256: str,
        completion_hash: str | None = None,
        max_failed_attempts: int = 3,
        stage: RunStage | None = None,
    ) -> WandbPreflightStatus:
        """Classify whether a matching W&B run should be started or skipped."""
        if max_failed_attempts < 1:
            raise ValueError("max_failed_attempts must be >= 1")
        matching = tuple(
            run
            for run in self._runs(group, seed)
            if _matches_identity(
                run,
                config_hash=config_hash,
                snapshot_sha256=snapshot_sha256,
                completion_hash=completion_hash,
            )
            and (stage is None or _lookup(run, "run/stage") == stage)
        )
        running = tuple(run.run_id for run in matching if _is_running(run))
        if running:
            return WandbPreflightStatus(
                state="running",
                max_failed_attempts=max_failed_attempts,
                run_ids=running,
            )
        complete = tuple(
            run.run_id for run in matching if _is_complete_for_scope(run, stage=stage)
        )
        if complete:
            return WandbPreflightStatus(
                state="complete",
                max_failed_attempts=max_failed_attempts,
                run_ids=complete,
            )
        failed = tuple(run.run_id for run in matching if _is_failed(run))
        if len(failed) >= max_failed_attempts:
            return WandbPreflightStatus(
                state="retry_exhausted",
                failed_attempts=len(failed),
                max_failed_attempts=max_failed_attempts,
                run_ids=failed,
            )
        if failed:
            return WandbPreflightStatus(
                state="retry",
                failed_attempts=len(failed),
                max_failed_attempts=max_failed_attempts,
                run_ids=failed,
            )
        return WandbPreflightStatus(
            state="missing",
            max_failed_attempts=max_failed_attempts,
        )

    def matching_run_ids(
        self,
        *,
        group: str,
        seed: int,
        config_hash: str,
        snapshot_sha256: str,
        completion_hash: str | None = None,
    ) -> tuple[str, ...]:
        """Return remote W&B run ids matching a local run identity."""
        return tuple(
            run.run_id
            for run in self._runs(group, seed)
            if _matches_identity(
                run,
                config_hash=config_hash,
                snapshot_sha256=snapshot_sha256,
                completion_hash=completion_hash,
            )
        )

    def run_ids(
        self,
        *,
        group: str,
        seed: int,
    ) -> tuple[str, ...]:
        """Return remote W&B run ids for a group/seed without identity filtering."""
        return tuple(run.run_id for run in self._runs(group, seed))

    def _runs(self, group: str, seed: int) -> tuple[WandbRunSnapshot, ...]:
        if group not in self._loaded_groups:
            self._load_group(group)
        return tuple(self._by_group_seed.get((group, seed), ()))

    def _load_group(self, group: str) -> None:
        if self._api is None:
            self._api = self.api_factory()
        path = f"{self.entity}/{self.project}" if self.entity else self.project
        runs: Iterable[Any] = self._api.runs(path, filters={"group": group})
        batch: dict[tuple[str, int], list[WandbRunSnapshot]] = {}
        for run in runs:
            config = _mapping_as_dict(getattr(run, "config", None))
            summary = _mapping_as_dict(getattr(run, "summary", None))
            seed = _extract_int(_first_present(config, "seed", "run/seed"))
            if seed is None:
                seed = _extract_int(summary.get("run/seed"))
            if seed is None:
                continue
            snapshot = WandbRunSnapshot(
                run_id=str(getattr(run, "id", "")),
                state=str(getattr(run, "state", "")),
                config=config,
                summary=summary,
            )
            batch.setdefault((group, seed), []).append(snapshot)
        for key, snapshots in batch.items():
            self._by_group_seed.setdefault(key, []).extend(snapshots)
        self._loaded_groups.add(group)


def _matches_common(
    run: WandbRunSnapshot,
    *,
    config_hash: str,
    snapshot_sha256: str,
    completion_hash: str | None,
) -> bool:
    return run.state == "finished" and _matches_identity(
        run,
        config_hash=config_hash,
        snapshot_sha256=snapshot_sha256,
        completion_hash=completion_hash,
    )


def _matches_identity(
    run: WandbRunSnapshot,
    *,
    config_hash: str,
    snapshot_sha256: str,
    completion_hash: str | None,
) -> bool:
    if completion_hash is not None:
        stored_completion_hash = _lookup(run, "run/completion_hash")
        if stored_completion_hash == completion_hash:
            return True
        config_payload_hash = completion_hash_from_config_payload(
            _lookup(run, "config")
        )
        if config_payload_hash == completion_hash:
            return True

    return (
        _lookup(run, "run/config_hash") == config_hash
        and _lookup(run, "run/snapshot_sha256") == snapshot_sha256
    )


_FAILED_STATES = {"crashed", "failed", "killed", "preempted"}
_TERMINAL_STATES = {"finished", *_FAILED_STATES}


def _is_running(run: WandbRunSnapshot) -> bool:
    return run.state not in _TERMINAL_STATES


def _is_failed(run: WandbRunSnapshot) -> bool:
    return (
        run.state in _FAILED_STATES
        or _lookup(run, "run/exit_status") == "error"
        or _lookup(run, "stage/exit_status") == "error"
    )


def _is_complete_for_scope(
    run: WandbRunSnapshot,
    *,
    stage: RunStage | None,
) -> bool:
    if run.state != "finished":
        return False
    if stage is None:
        return (
            _truthy(_lookup(run, "run/complete"))
            and _lookup(run, "run/exit_status") == "ok"
        )
    return (
        _lookup(run, "run/stage") == stage
        and _truthy(_lookup(run, "stage/complete"))
        and _lookup(run, "stage/exit_status") == "ok"
    )


def _lookup(run: WandbRunSnapshot, key: str) -> Any:
    if key in run.summary:
        return run.summary[key]
    return run.config.get(key)


def _mapping_as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return {str(k): v for k, v in value.items()}
    return {}


def _extract_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _truthy(value: Any) -> bool:
    return value == 1


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None
