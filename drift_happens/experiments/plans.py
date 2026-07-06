"""Generate staged multi-seed sweep plans from materialized presets."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from drift_happens.configs import DeviceSlotConfig, JobSpecConfig, SweepConfig
from drift_happens.configs.sweep import SlotDevice
from drift_happens.experiments.materialize import PRESETS_ROOT
from drift_happens.utils.log import get_logger
from drift_happens.utils.paths import EXPERIMENT_PLANS_DIR, relative_to_project

logger = get_logger()

PlanDevice = SlotDevice


@dataclass(frozen=True, slots=True)
class MaterializedPreset:
    """Small view of a materialized preset snapshot used by plan generation."""

    group: str
    name: str
    path: Path
    seeds: tuple[int, ...]
    tags: tuple[str, ...]
    comparison_role: str

    @property
    def label(self) -> str:
        return f"{self.group}/{self.name}"


@dataclass(frozen=True, slots=True)
class PlanStage:
    """Named staged sweep plan."""

    name: str
    description: str
    jobs: tuple[JobSpecConfig, ...]
    seeds: tuple[int, ...]
    skip_completed: bool = True

    def to_sweep(
        self, slots: tuple[DeviceSlotConfig, ...], concurrency: int
    ) -> SweepConfig:
        return SweepConfig(
            name=self.name,
            jobs=self.jobs,
            slots=slots,
            seeds=self.seeds,
            concurrency=concurrency,
            skip_completed=self.skip_completed,
        )


@dataclass(frozen=True, slots=True)
class PlanDiff:
    """Differences between expected and materialized plan YAMLs."""

    missing: tuple[Path, ...]
    stale: tuple[Path, ...]
    orphaned: tuple[Path, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.stale and not self.orphaned

    def format(self) -> str:
        if self.ok:
            return "experiment plans are current"
        sections: list[str] = []
        for label, paths in (
            ("missing", self.missing),
            ("stale", self.stale),
            ("orphaned", self.orphaned),
        ):
            if paths:
                rel_paths = ", ".join(str(relative_to_project(path)) for path in paths)
                sections.append(f"{label}: {rel_paths}")
        return "; ".join(sections)


def list_plan_stages(root: Path = PRESETS_ROOT) -> tuple[PlanStage, ...]:
    """Return the standard staged plans built from materialized presets."""
    return build_plan_stages(materialized_presets(root))


def materialized_presets(root: Path = PRESETS_ROOT) -> tuple[MaterializedPreset, ...]:
    """Read materialized preset snapshot envelopes."""
    presets: list[MaterializedPreset] = []
    if not root.exists():
        return ()
    for path in sorted(root.rglob("*.json")):
        if path.name == "index.json":
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning(f"Skipping unparsable preset snapshot {path}")
            continue
        if (
            not isinstance(payload, dict)
            or payload.get("kind") != "experiment_snapshot/v1"
        ):
            continue
        group = payload.get("group")
        name = payload.get("name")
        if not isinstance(group, str) or not isinstance(name, str):
            continue
        presets.append(
            MaterializedPreset(
                group=group,
                name=name,
                path=path,
                seeds=_seeds(payload),
                tags=_str_tuple(payload.get("tags")),
                comparison_role=str(payload.get("comparison_role") or "headline"),
            )
        )
    return tuple(presets)


def build_plan_stages(
    presets: tuple[MaterializedPreset, ...],
) -> tuple[PlanStage, ...]:
    """Build the standard seed-screening stages."""
    smoke_seed0 = tuple(
        _job(preset, seed=0)
        for preset in presets
        if _is_integration_smoke_preset(preset) and 0 in preset.seeds
    )
    seed0_all = tuple(_job(preset, seed=0) for preset in presets if 0 in preset.seeds)
    remaining = tuple(
        _job(preset, seed=seed)
        for preset in presets
        for seed in preset.seeds
        if seed != 0
    )
    all_seeds = tuple(
        _job(preset, seed=seed) for preset in presets for seed in preset.seeds
    )
    yearbook_headline = tuple(
        _job(preset, seed=seed)
        for preset in presets
        if preset.group == "yearbook" and preset.comparison_role == "headline"
        for seed in preset.seeds
    )
    text_headline = tuple(
        _job(preset, seed=seed)
        for preset in presets
        if preset.group in {"arxiv", "amazon-reviews-23"}
        and preset.comparison_role == "headline"
        for seed in preset.seeds
    )
    amazon_reviews_all = _dataset_jobs(
        presets,
        groups=("amazon-reviews-23", "amazon-reviews-23-conference"),
    )
    arxiv_all = _dataset_jobs(
        presets,
        groups=("arxiv", "arxiv-conference"),
    )
    yearbook_all = _dataset_jobs(
        presets,
        groups=("yearbook", "yearbook-conference"),
    )
    imdb_faces_all = _dataset_jobs(
        presets,
        groups=("imdb-faces-conference",),
    )
    return (
        PlanStage(
            name="p00_smoke_seed0",
            description="Super-fast synthetic CPU integration presets, seed 0 only.",
            jobs=smoke_seed0,
            seeds=(0,),
        ),
        PlanStage(
            name="p10_yearbook_headline_seeds",
            description="Yearbook headline presets and all declared seeds.",
            jobs=yearbook_headline,
            seeds=_unique(job.seed for job in yearbook_headline),
        ),
        PlanStage(
            name="p20_text_headline_seeds",
            description="arXiv and Amazon Reviews headline presets and all declared seeds.",
            jobs=text_headline,
            seeds=_unique(job.seed for job in text_headline),
        ),
        PlanStage(
            name="p40_amazon_reviews_23_all_seeds",
            description="Amazon Reviews 23 presets and all declared seeds.",
            jobs=amazon_reviews_all,
            seeds=_unique(job.seed for job in amazon_reviews_all),
        ),
        PlanStage(
            name="p50_arxiv_all_seeds",
            description="arXiv presets and all declared seeds.",
            jobs=arxiv_all,
            seeds=_unique(job.seed for job in arxiv_all),
        ),
        PlanStage(
            name="p60_yearbook_all_seeds",
            description="Yearbook presets and all declared seeds.",
            jobs=yearbook_all,
            seeds=_unique(job.seed for job in yearbook_all),
        ),
        PlanStage(
            name="p70_imdb_faces_all_seeds",
            description="IMDB faces presets and all declared seeds.",
            jobs=imdb_faces_all,
            seeds=_unique(job.seed for job in imdb_faces_all),
        ),
        PlanStage(
            name="p80_seed0_all_presets",
            description="All materialized presets, seed 0 only.",
            jobs=seed0_all,
            seeds=(0,),
        ),
        PlanStage(
            name="p90_remaining_seeds_all_presets",
            description="All declared preset seeds excluding seed 0.",
            jobs=remaining,
            seeds=_unique(job.seed for job in remaining),
        ),
        PlanStage(
            name="p99_everything_all_seeds",
            description="All declared preset seeds from scratch.",
            jobs=all_seeds,
            seeds=_unique(job.seed for job in all_seeds),
        ),
    )


def build_slots(
    *,
    device: PlanDevice,
    gpu_indices: tuple[int, ...] = (0,),
    jobs_per_device: int = 1,
    concurrency: int = 1,
) -> tuple[DeviceSlotConfig, ...]:
    """Build scheduler slots for CPU, MPS, or CUDA execution."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if jobs_per_device < 1:
        raise ValueError("jobs_per_device must be >= 1")
    if device == "cuda":
        slots: list[DeviceSlotConfig] = []
        for index in gpu_indices or (0,):
            for replica in range(jobs_per_device):
                slots.append(
                    DeviceSlotConfig(
                        device="cuda",
                        device_index=index,
                        label=f"cuda:{index}:{replica}",
                    )
                )
        return tuple(slots)
    return tuple(
        DeviceSlotConfig(device=device, label=f"{device}:{index}")
        for index in range(concurrency)
    )


def expected_plan_files(
    *,
    out_dir: Path = EXPERIMENT_PLANS_DIR,
    presets_root: Path = PRESETS_ROOT,
    device: PlanDevice = "cpu",
    gpu_indices: tuple[int, ...] = (0,),
    jobs_per_device: int = 1,
    concurrency: int = 1,
    seed_filter: tuple[int, ...] | None = None,
) -> dict[Path, str]:
    """Return expected sweep YAML text for all non-empty plan stages."""
    slots = build_slots(
        device=device,
        gpu_indices=gpu_indices,
        jobs_per_device=jobs_per_device,
        concurrency=concurrency,
    )
    stages = _filter_plan_stages_by_seed(
        build_plan_stages(materialized_presets(presets_root)),
        seed_filter=seed_filter,
    )
    return {
        out_dir / f"{stage.name}.yaml": _sweep_yaml(stage, slots, concurrency)
        for stage in stages
        if stage.jobs
    }


def write_plan_files(
    *,
    out_dir: Path = EXPERIMENT_PLANS_DIR,
    presets_root: Path = PRESETS_ROOT,
    device: PlanDevice = "cpu",
    gpu_indices: tuple[int, ...] = (0,),
    jobs_per_device: int = 1,
    concurrency: int = 1,
    seed_filter: tuple[int, ...] | None = None,
) -> tuple[Path, ...]:
    """Write expected staged sweep YAML files; deletes orphan YAMLs in out_dir."""
    expected = expected_plan_files(
        out_dir=out_dir,
        presets_root=presets_root,
        device=device,
        gpu_indices=gpu_indices,
        jobs_per_device=jobs_per_device,
        concurrency=concurrency,
        seed_filter=seed_filter,
    )
    written: list[Path] = []
    for path, text in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        written.append(path)
    for orphan in sorted(set(out_dir.glob("*.yaml")) - set(expected)):
        orphan.unlink()
    return tuple(written)


def check_plan_files(
    *,
    out_dir: Path = EXPERIMENT_PLANS_DIR,
    presets_root: Path = PRESETS_ROOT,
    device: PlanDevice = "cpu",
    gpu_indices: tuple[int, ...] = (0,),
    jobs_per_device: int = 1,
    concurrency: int = 1,
    seed_filter: tuple[int, ...] | None = None,
) -> PlanDiff:
    """Check generated plan YAMLs against materialized files."""
    expected = expected_plan_files(
        out_dir=out_dir,
        presets_root=presets_root,
        device=device,
        gpu_indices=gpu_indices,
        jobs_per_device=jobs_per_device,
        concurrency=concurrency,
        seed_filter=seed_filter,
    )
    missing: list[Path] = []
    stale: list[Path] = []
    for path, text in expected.items():
        if not path.exists():
            missing.append(path)
        elif path.read_text() != text:
            stale.append(path)
    actual = set(out_dir.glob("*.yaml")) if out_dir.exists() else set()
    orphaned = sorted(actual - set(expected))
    return PlanDiff(
        missing=tuple(sorted(missing)),
        stale=tuple(sorted(stale)),
        orphaned=tuple(orphaned),
    )


def _filter_plan_stages_by_seed(
    stages: tuple[PlanStage, ...],
    *,
    seed_filter: tuple[int, ...] | None,
) -> tuple[PlanStage, ...]:
    """Return stages containing only jobs whose seed is in ``seed_filter``."""
    if seed_filter is None:
        return stages
    selected = set(seed_filter)
    available = {job.seed for stage in stages for job in stage.jobs}
    missing = selected - available
    if missing:
        missing_text = ", ".join(str(seed) for seed in sorted(missing))
        raise ValueError(f"seed filter includes unavailable seed(s): {missing_text}")

    filtered: list[PlanStage] = []
    for stage in stages:
        jobs = tuple(job for job in stage.jobs if job.seed in selected)
        if not jobs:
            continue
        filtered.append(
            PlanStage(
                name=stage.name,
                description=stage.description,
                jobs=jobs,
                seeds=_unique(job.seed for job in jobs),
                skip_completed=stage.skip_completed,
            )
        )
    if not filtered:
        seeds = ", ".join(str(seed) for seed in seed_filter)
        raise ValueError(f"no plan jobs matched seed filter: {seeds}")
    return tuple(filtered)


def _sweep_yaml(
    stage: PlanStage,
    slots: tuple[DeviceSlotConfig, ...],
    concurrency: int,
) -> str:
    # Pass concurrency through unclamped so SweepConfig rejects an over-ask.
    sweep = stage.to_sweep(slots=slots, concurrency=concurrency)
    payload = sweep.model_dump(mode="json")
    if payload.get("resume") is None:
        payload.pop("resume", None)
    return yaml.safe_dump(
        payload,
        sort_keys=True,
        default_flow_style=False,
    )


def _job(preset: MaterializedPreset, *, seed: int) -> JobSpecConfig:
    return JobSpecConfig(
        action="run",
        config_path=relative_to_project(preset.path),
        seed=seed,
        label=preset.label,
        tags=preset.tags,
    )


def _dataset_jobs(
    presets: tuple[MaterializedPreset, ...],
    *,
    groups: tuple[str, ...],
) -> tuple[JobSpecConfig, ...]:
    return tuple(
        _job(preset, seed=seed)
        for preset in presets
        if preset.group in groups
        for seed in preset.seeds
    )


def _is_integration_smoke_preset(preset: MaterializedPreset) -> bool:
    tags = set(preset.tags)
    return preset.comparison_role == "smoke" and "synthetic" in tags and "cpu" in tags


def _seeds(payload: dict) -> tuple[int, ...]:
    raw = payload.get("seeds")
    if not isinstance(raw, list | tuple):
        return (0,)
    seeds = tuple(
        item for item in raw if isinstance(item, int) and not isinstance(item, bool)
    )
    return seeds or (0,)


def _str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str))


def _unique(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(set(values)))
