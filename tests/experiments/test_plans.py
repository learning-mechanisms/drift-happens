from __future__ import annotations

import json

import pytest
import yaml

from drift_happens.configs import SweepConfig
from drift_happens.experiments.plans import (
    PlanStage,
    build_plan_stages,
    build_slots,
    expected_plan_files,
    materialized_presets,
)


def _job_keys(stage: PlanStage) -> set[tuple[str, int]]:
    return {(job.label, job.seed) for job in stage.jobs}


def test_p00_is_super_fast_synthetic_integration_plan() -> None:
    stages = {stage.name: stage for stage in build_plan_stages(materialized_presets())}

    p00 = stages["p00_smoke_seed0"]

    assert [(job.label, job.seed) for job in p00.jobs] == [
        ("smoke/synthetic-classification-cpu", 0)
    ]
    assert {"smoke", "synthetic", "cpu"}.issubset(p00.jobs[0].tags)


def test_standard_plan_stages_split_seed0_and_remaining_seeds() -> None:
    stages = {stage.name: stage for stage in build_plan_stages(materialized_presets())}

    assert {job.seed for job in stages["p80_seed0_all_presets"].jobs} == {0}
    assert 0 not in {job.seed for job in stages["p90_remaining_seeds_all_presets"].jobs}
    assert {0, 1, 2, 3, 4}.issubset(
        {job.seed for job in stages["p99_everything_all_seeds"].jobs}
    )
    assert stages["p10_yearbook_headline_seeds"].jobs
    assert stages["p20_text_headline_seeds"].jobs
    assert stages["p40_amazon_reviews_23_all_seeds"].jobs
    assert stages["p50_arxiv_all_seeds"].jobs
    assert stages["p60_yearbook_all_seeds"].jobs
    assert stages["p70_imdb_faces_all_seeds"].jobs
    assert _job_keys(stages["p00_smoke_seed0"]) < _job_keys(
        stages["p80_seed0_all_presets"]
    )


def test_plan_stage_names_sort_from_targeted_to_full_sets() -> None:
    names = [stage.name for stage in build_plan_stages(materialized_presets())]

    assert names == [
        "p00_smoke_seed0",
        "p10_yearbook_headline_seeds",
        "p20_text_headline_seeds",
        "p40_amazon_reviews_23_all_seeds",
        "p50_arxiv_all_seeds",
        "p60_yearbook_all_seeds",
        "p70_imdb_faces_all_seeds",
        "p80_seed0_all_presets",
        "p90_remaining_seeds_all_presets",
        "p99_everything_all_seeds",
    ]


def test_targeted_plan_stages_are_subsets_of_seed0_and_remaining_plans() -> None:
    stages = {stage.name: stage for stage in build_plan_stages(materialized_presets())}
    seed0_jobs = _job_keys(stages["p80_seed0_all_presets"])
    remaining_seed_jobs = _job_keys(stages["p90_remaining_seeds_all_presets"])
    all_jobs = _job_keys(stages["p99_everything_all_seeds"])

    for name in (
        "p10_yearbook_headline_seeds",
        "p20_text_headline_seeds",
        "p40_amazon_reviews_23_all_seeds",
        "p50_arxiv_all_seeds",
        "p60_yearbook_all_seeds",
        "p70_imdb_faces_all_seeds",
    ):
        jobs = _job_keys(stages[name])
        targeted_seed0_jobs = {(label, seed) for label, seed in jobs if seed == 0}
        targeted_remaining_jobs = jobs - targeted_seed0_jobs

        assert jobs <= all_jobs
        assert targeted_seed0_jobs <= seed0_jobs
        assert targeted_remaining_jobs <= remaining_seed_jobs
        assert targeted_remaining_jobs


def test_expected_plan_files_validate_as_sweep_configs(tmp_path) -> None:
    plans = expected_plan_files(out_dir=tmp_path, device="cpu", concurrency=2)

    assert plans
    for text in plans.values():
        SweepConfig.model_validate(yaml.safe_load(text))


def test_expected_plan_files_filter_to_selected_seeds(tmp_path) -> None:
    plans = expected_plan_files(
        out_dir=tmp_path,
        device="cuda",
        gpu_indices=(0, 1),
        concurrency=2,
        seed_filter=(1, 3),
    )

    assert plans
    assert tmp_path / "p80_seed0_all_presets.yaml" not in plans
    observed_seeds: set[int] = set()
    for text in plans.values():
        payload = yaml.safe_load(text)
        SweepConfig.model_validate(payload)
        job_seeds = {job["seed"] for job in payload["jobs"]}
        assert job_seeds <= {1, 3}
        assert set(payload["seeds"]) == job_seeds
        observed_seeds.update(job_seeds)

    assert observed_seeds == {1, 3}


def test_expected_plan_files_reject_unavailable_seed(tmp_path) -> None:
    with pytest.raises(ValueError, match="unavailable seed"):
        expected_plan_files(out_dir=tmp_path, seed_filter=(999,))


def test_materialized_presets_skip_unparsable_snapshot(tmp_path) -> None:
    valid = {
        "kind": "experiment_snapshot/v1",
        "group": "smoke",
        "name": "ok-preset",
        "seeds": [0, 1],
        "tags": ["smoke"],
    }
    (tmp_path / "ok.json").write_text(json.dumps(valid))
    (tmp_path / "broken.json").write_text('{"kind": "experiment_snapshot/v1", "gro')

    presets = materialized_presets(tmp_path)

    assert [(preset.group, preset.name) for preset in presets] == [
        ("smoke", "ok-preset")
    ]


def test_cuda_slots_expand_jobs_per_device() -> None:
    slots = build_slots(
        device="cuda",
        gpu_indices=(0, 2),
        jobs_per_device=2,
        concurrency=4,
    )

    assert [slot.device_index for slot in slots] == [0, 0, 2, 2]
    assert [slot.label for slot in slots] == [
        "cuda:0:0",
        "cuda:0:1",
        "cuda:2:0",
        "cuda:2:1",
    ]


def test_plan_concurrency_exceeding_slots_raises(tmp_path) -> None:
    # A requested concurrency above the slot count must fail loudly instead of
    # being silently clamped (which would serialize a multi-GPU plan).
    with pytest.raises(ValueError, match="concurrency"):
        expected_plan_files(
            out_dir=tmp_path,
            device="cuda",
            gpu_indices=(0,),
            jobs_per_device=1,
            concurrency=8,
        )
