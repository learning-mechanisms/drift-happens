from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from drift_happens.configs import ExperimentConfig
from drift_happens.experiments.common import _protocol_from_conference_metadata
from drift_happens.experiments.materialize import validate_comparison_groups
from drift_happens.experiments.registry import iter_presets, preset, preset_groups


def test_registry_entries_are_unique_and_sorted() -> None:
    entries = list(iter_presets())
    keys = [entry.key for entry in entries]

    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))


def test_registry_factories_build_one_run_configs() -> None:
    for entry in iter_presets():
        cfg = entry.build()

        assert isinstance(cfg, ExperimentConfig)
        assert cfg.seed == 0
        assert cfg.name == f"{entry.group}-{entry.name}"
        assert entry.seeds
        assert entry.comparison_role in {"smoke", "headline"}


def test_preset_lookup_matches_registry_entry() -> None:
    for entry in iter_presets():
        assert preset(entry.group, entry.name) is entry

    with pytest.raises(KeyError):
        preset("__no_such_group__", "__no_such_name__")


def test_preset_groups_lists_all_entries() -> None:
    groups = preset_groups()

    for entry in iter_presets():
        assert entry.name in groups[entry.group]


def test_embedding_presets_expose_reusable_cache_identity() -> None:
    embedding_entries = [
        entry for entry in iter_presets() if "embedding-cache" in entry.tags
    ]

    assert embedding_entries
    for entry in embedding_entries:
        cache = entry.build().preprocessing.cache
        assert cache is not None
        assert cache.kind in {"embedding_dataset", "pooled_embedding_dataset"}
        assert cache.cache_id.startswith(f"{cache.kind}-")


def test_comparison_group_invariants_hold_for_registered_presets() -> None:
    # Guard against vacuous pass: validate_comparison_groups silently skips None entries.
    entries_with_group = [e for e in iter_presets() if e.comparison_group is not None]
    assert entries_with_group
    validate_comparison_groups(iter(entries_with_group))


def test_registered_presets_expose_typed_protocol_metadata() -> None:
    protocol_entries = [entry for entry in iter_presets() if entry.group != "smoke"]

    assert protocol_entries
    for entry in protocol_entries:
        cfg = entry.build()
        assert cfg.protocol.job_granularity == "seed_matrix"
        assert cfg.protocol.time_slices.time_col
        assert cfg.protocol.time_slices.train_strategy == "cumulative_from_start"
        assert cfg.protocol.evaluation.metric == cfg.evaluation.metric


def test_declared_eval_strategies_are_dispatched_simple_slices() -> None:
    """
    Pin eval strategies to what the staged runtime actually dispatches.

    runtime/dataset_pipeline.py builds eval slices exclusively with
    ``create_simple_time_slices`` over each dataset's time column, so any other declared
    strategy string would be silent dead metadata.
    """
    implemented = {
        "simple_year_slices",
        "simple_photo_taken_slices",
        "simple_half_year_slices",
    }
    protocol_entries = [entry for entry in iter_presets() if entry.group != "smoke"]

    assert protocol_entries
    for entry in protocol_entries:
        time_slices = entry.build().protocol.time_slices
        assert time_slices.eval_strategy in implemented
        assert time_slices.eval_strategy == f"simple_{time_slices.time_col}_slices"


def test_conference_metadata_derivation_rejects_schema_drift() -> None:
    # The protocol must be derived by validating the metadata back through the
    # typed ConferenceProtocol, so a renamed/retyped key fails loudly instead of
    # silently degrading identity fields to ""/None.
    entry = next(e for e in iter_presets() if e.group == "amazon-reviews-23-conference")
    metadata = entry.build().metadata

    protocol = _protocol_from_conference_metadata(metadata)
    assert protocol is not None
    assert protocol.split.name
    assert protocol.split.seed == 42

    corrupted = copy.deepcopy(metadata)
    split = corrupted["split"]
    assert isinstance(split, dict)
    split["renamed_seed"] = split.pop("split_seed")
    with pytest.raises(ValidationError):
        _protocol_from_conference_metadata(corrupted)
