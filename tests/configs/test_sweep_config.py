from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from drift_happens.configs.sweep import DEFAULT_SWEEP_SEEDS, SweepConfig


def test_seedless_jobs_expand_to_default_seed_replicas() -> None:
    sweep = SweepConfig.model_validate(
        {
            "name": "smoke",
            "jobs": [
                {
                    "config_path": "configs/snapshots/yearbook-smoke.json",
                    "label": "yearbook-smoke",
                }
            ],
            "slots": [{"device": "cpu"}],
        }
    )

    assert sweep.seeds == DEFAULT_SWEEP_SEEDS
    assert [job.seed for job in sweep.jobs] == list(DEFAULT_SWEEP_SEEDS)
    assert {job.config_path for job in sweep.jobs} == {
        Path("configs/snapshots/yearbook-smoke.json")
    }
    assert sweep.slots[0].label == "cpu"


def test_custom_seeds_expand_seedless_jobs_and_coerce_strings() -> None:
    # The _expand_seedless_jobs branch for user-supplied seeds must expand each
    # seedless job once per custom seed and coerce string seeds via pydantic's int
    # coercion on the tuple[int, ...] field.
    sweep = SweepConfig.model_validate(
        {
            "name": "custom-seeds",
            "seeds": [7, "8"],
            "jobs": [{"config_path": "c.json", "label": "j"}],
            "slots": [{"device": "cpu"}],
        }
    )

    assert sweep.seeds == (7, 8)
    assert [job.seed for job in sweep.jobs] == [7, 8]


def test_explicit_job_seed_is_not_expanded() -> None:
    sweep = SweepConfig.model_validate(
        _sweep(
            jobs=[
                {
                    "config_path": "configs/snapshots/yearbook-smoke.json",
                    "label": "yearbook-smoke",
                    "seed": 99,
                }
            ]
        )
    )

    assert [job.seed for job in sweep.jobs] == [99]


def _sweep(**overrides: object) -> dict[str, object]:
    return {
        "name": "smoke",
        "jobs": [{"config_path": "c.json", "label": "j"}],
        "slots": [{"device": "cpu"}],
        **overrides,
    }


@pytest.mark.parametrize("bad_seeds", [None, 5])
def test_non_sequence_seeds_raise_validation_error(bad_seeds: object) -> None:
    with pytest.raises(ValidationError):
        SweepConfig.model_validate(_sweep(seeds=bad_seeds))


def test_float_seed_is_rejected_not_truncated() -> None:
    with pytest.raises(ValidationError, match="int"):
        SweepConfig.model_validate(_sweep(seeds=[1.7]))


def test_empty_seeds_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SweepConfig.model_validate(_sweep(seeds=[]))


def test_duplicate_label_seed_jobs_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        SweepConfig.model_validate(
            {
                "name": "smoke",
                "jobs": [
                    {"config_path": "a.json", "label": "j", "seed": 0},
                    {"config_path": "b.json", "label": "j", "seed": 0},
                ],
                "slots": [{"device": "cpu"}],
            }
        )


def test_cuda_slot_gets_stable_default_label() -> None:
    sweep = SweepConfig.model_validate(
        _sweep(
            jobs=[{"config_path": "c.json", "label": "j", "seed": 0}],
            slots=[{"device": "cuda", "device_index": 1}],
        )
    )

    assert sweep.slots[0].label == "cuda:1"


def test_device_index_is_only_valid_for_cuda_slots() -> None:
    with pytest.raises(ValidationError, match="device_index"):
        SweepConfig.model_validate(
            _sweep(
                jobs=[{"config_path": "c.json", "label": "j", "seed": 0}],
                slots=[{"device": "mps", "device_index": 0}],
            )
        )


def test_cuda_slot_requires_device_index() -> None:
    with pytest.raises(ValidationError, match="device_index"):
        SweepConfig.model_validate(
            _sweep(
                jobs=[{"config_path": "c.json", "label": "j", "seed": 0}],
                slots=[{"device": "cuda"}],
            )
        )


def test_concurrency_cannot_exceed_slot_count() -> None:
    with pytest.raises(ValidationError, match="concurrency"):
        SweepConfig.model_validate(
            _sweep(
                jobs=[{"config_path": "c.json", "label": "j", "seed": 0}], concurrency=2
            )
        )


def test_resume_skip_and_job_action_are_supported() -> None:
    sweep = SweepConfig.model_validate(
        {
            "name": "resume",
            "resume": True,
            "skip_completed": True,
            "skip_source": "local",
            "jobs": [
                {
                    "action": "train",
                    "config_path": "configs/snapshots/yearbook-smoke.json",
                    "label": "yearbook-smoke",
                    "seed": 0,
                    "tags": ["smoke"],
                }
            ],
            "slots": [{"device": "cpu"}],
        }
    )

    assert sweep.resume is True
    assert sweep.skip_completed is True
    assert sweep.skip_source == "local"
    assert sweep.jobs[0].action == "train"
    assert sweep.jobs[0].tags == ("smoke",)
