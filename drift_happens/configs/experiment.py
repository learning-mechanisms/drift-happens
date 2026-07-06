"""Root ``ExperimentConfig`` plus file loading and override helpers."""

from __future__ import annotations

import copy
import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, JsonValue, model_validator

from drift_happens.configs.base import BaseConfig
from drift_happens.configs.logging_cfg import LoggingConfig
from drift_happens.configs.protocol import ExperimentProtocolConfig
from drift_happens.configs.runtime import RuntimeConfig

ExperimentTask = Literal["train", "eval", "train_eval"]
CacheArtifactKind = Literal[
    "tensor_dataset",
    "tokenized_dataset",
    "embedding_dataset",
    "sequence_embedding_dataset",
    "pooled_embedding_dataset",
    "prediction_table",
]
CacheReusePolicy = Literal["reuse", "refresh", "disabled"]


class DatasetConfig(BaseConfig):
    """Dataset identity for an experiment run."""

    name: str = Field(min_length=1)
    variant: str | None = None
    data_dir: Path | None = None


class TrainerConfig(BaseConfig):
    """
    Trainer identity and JSON-serializable knobs.

    ``model`` and ``training`` intentionally accept JSON values only. This keeps
    snapshots stable while letting later dataset-specific registry code define richer
    typed configs without changing the root contract.
    """

    key: str = Field(min_length=1)
    family: str | None = None
    model: dict[str, JsonValue] = Field(default_factory=dict)
    training: dict[str, JsonValue] = Field(default_factory=dict)


class EvaluationConfig(BaseConfig):
    """Evaluation identity and JSON-serializable knobs."""

    metric: str | None = None
    params: dict[str, JsonValue] = Field(default_factory=dict)


class CacheSpec(BaseConfig):
    """
    Stable identity for reusable preprocessing artifacts.

    This describes cache intent only. Runtime state such as cache path, hit/miss,
    creation time, and file hashes belongs in run metadata or a cache manifest.
    """

    kind: CacheArtifactKind
    dataset: str = Field(min_length=1)
    input_version: str = Field(min_length=1)
    producer: str = Field(min_length=1)
    output: str = Field(min_length=1)
    params: dict[str, JsonValue] = Field(default_factory=dict)
    schema_version: int = 1
    reuse_policy: CacheReusePolicy = "reuse"
    cache_id: str = ""

    @model_validator(mode="after")
    def _fill_or_check_cache_id(self) -> CacheSpec:
        # Hash the normalized field values so a value pydantic coerces is hashed
        # the same way it is later compared.
        expected = self.cache_id_from_data(
            {
                "dataset": self.dataset,
                "input_version": self.input_version,
                "kind": self.kind,
                "output": self.output,
                "params": self.params,
                "producer": self.producer,
                "schema_version": self.schema_version,
            }
        )
        if not self.cache_id:
            object.__setattr__(self, "cache_id", expected)
        elif self.cache_id != expected:
            raise ValueError(
                f"cache_id={self.cache_id!r} does not match expected {expected!r}"
            )
        return self

    @staticmethod
    def cache_id_from_data(data: dict[str, Any]) -> str:
        """Return a deterministic id from fields that affect cache contents."""
        payload = {
            "dataset": data["dataset"],
            "input_version": data["input_version"],
            "kind": data["kind"],
            "output": data["output"],
            "params": data.get("params", {}),
            "producer": data["producer"],
            "schema_version": data.get("schema_version", 1),
        }
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = sha256(body.encode("utf-8")).hexdigest()[:16]
        return f"{payload['kind']}-{digest}"


class PreprocessingConfig(BaseConfig):
    """Preprocessing steps and cache intent for one experiment run."""

    steps: tuple[str, ...] = ()
    cache: CacheSpec | None = None


class ExperimentConfig(BaseConfig):
    """Root configuration for one experiment run: one task, one seed, one process."""

    name: str = Field(min_length=1)
    seed: int = 0
    task: ExperimentTask = "train"
    dataset: DatasetConfig
    trainer: TrainerConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    protocol: ExperimentProtocolConfig = Field(default_factory=ExperimentProtocolConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tags: tuple[str, ...] = ()
    notes: str = ""
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    def to_snapshot_json(self) -> str:
        """Return a deterministic JSON representation suitable for ``snapshot.json``."""
        payload = self.model_dump(mode="json")
        return json.dumps(payload, indent=2, sort_keys=True)


def load_experiment_config(
    path: Path, overrides: list[str] | tuple[str, ...] = ()
) -> ExperimentConfig:
    """Load and validate an ``ExperimentConfig`` from YAML or JSON."""
    data = load_config_data(path)
    if overrides:
        data = apply_overrides(data, list(overrides))
    return ExperimentConfig.model_validate(data)


def load_config_data(path: Path) -> dict[str, Any]:
    """Read a YAML or JSON config, resolving recursive ``_extends`` for YAML."""
    path = Path(path).resolve()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return load_yaml(path)
    if suffix == ".json":
        with path.open() as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"config {path} is not a JSON object")
        return data
    raise ValueError(f"unsupported config format for {path}; expected .yaml or .json")


def load_yaml(path: Path, _visited: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a YAML config and recursively merge a relative ``_extends`` file."""
    path = Path(path).resolve()
    if path in _visited:
        chain = " -> ".join(str(seen) for seen in (*_visited, path))
        raise ValueError(f"_extends cycle detected: {chain}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} is not a YAML mapping")

    extends = data.pop("_extends", None)
    if extends is None:
        return data

    base_path = (path.parent / str(extends)).resolve()
    base = load_yaml(base_path, _visited=(*_visited, path))
    return _deep_merge(base, data)


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """
    Apply ``key.path=<json-literal>`` overrides to a config dictionary.

    The right-hand side is parsed with ``json.loads`` and falls back to a raw string, so
    both ``trainer.training.batch_size=64`` and ``name=smoke`` are valid.
    """
    out = _deep_copy(data)
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"override {spec!r} must be of form key.path=value")
        key, raw = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"override {spec!r} has an empty key")
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        _set_dotted(out, key, value)
    return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = _deep_copy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _deep_copy(data: dict[str, Any]) -> dict[str, Any]:
    # deepcopy preserves YAML-native scalars (dates) and non-string keys that a
    # JSON round-trip would corrupt.
    return copy.deepcopy(data)


def _set_dotted(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: Any = target
    for part in parts[:-1]:
        if not isinstance(cur, dict):
            raise ValueError(f"cannot descend into {part!r}: parent is not a mapping")
        cur = cur.setdefault(part, {})
    if not isinstance(cur, dict):
        raise ValueError(f"cannot set {dotted_key!r}: parent is not a mapping")
    cur[parts[-1]] = value
