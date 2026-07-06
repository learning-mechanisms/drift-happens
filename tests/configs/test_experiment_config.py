from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from drift_happens.configs import (
    CacheSpec,
    ExperimentConfig,
    RuntimeConfig,
    apply_overrides,
    load_experiment_config,
    load_yaml,
)


def _minimal_config() -> dict:
    return {
        "name": "yearbook-smoke",
        "seed": 7,
        "dataset": {"name": "yearbook"},
        "trainer": {
            "key": "unit-trainer",
            "training": {"batch_size": 64, "learning_rate": 0.001},
        },
    }


def test_experiment_config_forbids_extra_fields() -> None:
    data = _minimal_config()
    data["unknown"] = True

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(data)


def test_experiment_config_is_immutable() -> None:
    cfg = ExperimentConfig.model_validate(_minimal_config())

    with pytest.raises(ValidationError):
        cfg.seed = 9


def _all_keys_sorted(obj: object) -> bool:
    """True when every nested dict in obj has keys in sorted order."""
    if isinstance(obj, dict):
        keys = list(obj.keys())
        if keys != sorted(keys):
            return False
        return all(_all_keys_sorted(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_all_keys_sorted(item) for item in obj)
    return True


def test_snapshot_json_is_deterministic_and_sorted() -> None:
    # Build two configs from dicts with different key insertion orders.
    data_a = _minimal_config()
    data_b = {k: data_a[k] for k in reversed(list(data_a))}

    json_a = ExperimentConfig.model_validate(data_a).to_snapshot_json()
    json_b = ExperimentConfig.model_validate(data_b).to_snapshot_json()

    assert json_a == json_b
    assert _all_keys_sorted(json.loads(json_a))


def test_json_only_parameter_sections_reject_python_objects() -> None:
    data = _minimal_config()
    data["trainer"]["training"]["path"] = Path("not-json")

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(data)


def _cache_spec(**overrides: object) -> CacheSpec:
    """Build a CacheSpec with shared defaults; pass only the differing fields."""
    base: dict[str, object] = {
        "kind": "embedding_dataset",
        "dataset": "yearbook",
        "input_version": "yearbook:v1",
        "producer": "dinov2-small",
        "output": "cls_embedding",
        "params": {"batch_size": 1024, "pin_memory": True},
    }
    return CacheSpec(**{**base, **overrides})


def test_preprocessing_cache_id_is_deterministic_and_content_addressed() -> None:
    cache = _cache_spec()
    # reuse_policy must not perturb cache_id (excluded by cache_id_from_data).
    same = _cache_spec(
        params={"pin_memory": True, "batch_size": 1024}, reuse_policy="refresh"
    )
    different = _cache_spec(params={"batch_size": 512, "pin_memory": True})

    assert cache.cache_id == same.cache_id
    assert cache.cache_id != different.cache_id


def test_preprocessing_cache_id_is_in_snapshot_json() -> None:
    data = _minimal_config()
    data["preprocessing"] = {
        "steps": ["embed"],
        "cache": {
            "kind": "embedding_dataset",
            "dataset": "yearbook",
            "input_version": "yearbook:v1",
            "producer": "dinov2-small",
            "output": "cls_embedding",
            "params": {"batch_size": 1024},
        },
    }

    cfg = ExperimentConfig.model_validate(data)
    payload = json.loads(cfg.to_snapshot_json())

    assert payload["preprocessing"]["cache"]["cache_id"].startswith(
        "embedding_dataset-"
    )


def test_apply_overrides_supports_dotted_json_literals() -> None:
    data = _minimal_config()

    out = apply_overrides(
        data,
        [
            "seed=11",
            "trainer.training.batch_size=128",
            'tags=["smoke", "ci"]',
            "name=override-smoke",
        ],
    )

    assert out["seed"] == 11
    assert out["trainer"]["training"]["batch_size"] == 128
    assert out["tags"] == ["smoke", "ci"]
    assert out["name"] == "override-smoke"
    assert data["seed"] == 7


def test_load_yaml_extends_and_deep_merges(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "name": "base",
                "dataset": {"name": "yearbook"},
                "trainer": {
                    "key": "mlp",
                    "training": {"batch_size": 64, "learning_rate": 0.001},
                },
            }
        )
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        yaml.safe_dump(
            {
                "_extends": "base.yaml",
                "name": "child",
                "trainer": {"training": {"batch_size": 128}},
            }
        )
    )

    data = load_yaml(child)

    assert data["name"] == "child"
    assert data["dataset"]["name"] == "yearbook"
    assert data["trainer"]["training"]["batch_size"] == 128
    assert data["trainer"]["training"]["learning_rate"] == 0.001


def test_load_yaml_rejects_extends_self_cycle(tmp_path: Path) -> None:
    config = tmp_path / "a.yaml"
    config.write_text(yaml.safe_dump({"_extends": "a.yaml", "name": "a"}))

    with pytest.raises(ValueError, match="_extends cycle detected"):
        load_yaml(config)


def test_load_yaml_rejects_extends_two_file_cycle_and_names_the_chain(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.yaml").write_text(
        yaml.safe_dump({"_extends": "b.yaml", "name": "a"})
    )
    (tmp_path / "b.yaml").write_text(
        yaml.safe_dump({"_extends": "a.yaml", "name": "b"})
    )

    with pytest.raises(
        ValueError,
        match=r"_extends cycle detected: .*a\.yaml -> .*b\.yaml -> .*a\.yaml",
    ):
        load_yaml(tmp_path / "a.yaml")


def test_load_experiment_config_applies_overrides(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(_minimal_config()))

    cfg = load_experiment_config(path, overrides=("runtime.device=cpu",))

    assert cfg.runtime.device == "cpu"


def test_runtime_rejects_unimplemented_mixed_precision() -> None:
    with pytest.raises(ValidationError, match="mixed_precision"):
        RuntimeConfig(mixed_precision="fp16")
