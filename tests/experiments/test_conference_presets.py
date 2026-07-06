"""Tests for materialized conference preset invariants."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import torch
from pydantic import JsonValue

from drift_happens.experiments.common import (
    BENCHMARK_SEEDS,
    CONFERENCE_VARIANT_FIELDS,
)
from drift_happens.experiments.registry import iter_presets
from drift_happens.experiments.yearbook import YEARBOOK_BENCHMARK_SEEDS
from drift_happens.pipeline.amazon_reviews_23.trainers import (
    amazon_reviews_conference_trainer_configs,
)
from drift_happens.pipeline.arxiv.trainers import arxiv_conference_trainer_configs
from drift_happens.pipeline.imdb_faces.trainers import (
    imdb_faces_conference_trainer_configs,
)
from drift_happens.pipeline.training_config import TrainingConfig
from drift_happens.pipeline.yearbook.trainers import (
    yearbook_conference_trainer_configs,
)

JsonDict = dict[str, JsonValue]

_TEXT_TRAINING_FIELDS: tuple[str, ...] = (
    "batch_size",
    "learning_rate",
    "num_epochs",
    "weight_decay",
    "optimizer",
    "gradient_clip_norm",
)
_IMAGE_TRAINING_FIELDS: tuple[str, ...] = ("batch_size", "learning_rate", "num_epochs")

# The amazon conference presets carry two descriptive training keys that no code reads; they must
# stay truthful to pipeline/amazon_reviews_23/run.py, where the criterion is ``WeightedMSELoss`` and
# the class weights are plain inverse frequency (``total / count``, no square root). Pinned here as an
# independent literal so a builder edit that reintroduces a wrong label fails loudly.
_AMAZON_DESCRIPTIVE_TRAINING = {
    "loss": "weighted_mse",
    "rating_weighting": "inverse_frequency",
}
_CONFERENCE_SEEDS_BY_GROUP = {
    "amazon-reviews-23-conference": BENCHMARK_SEEDS,
    "arxiv-conference": BENCHMARK_SEEDS,
    "yearbook-conference": YEARBOOK_BENCHMARK_SEEDS,
    "imdb-faces-conference": BENCHMARK_SEEDS,
}


def _json_dict(value: JsonValue) -> JsonDict:
    assert isinstance(value, dict)
    return value


def _json_int(value: JsonValue) -> int:
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _json_int_tuple(value: JsonValue) -> tuple[int, ...]:
    assert isinstance(value, list)
    items: list[int] = []
    for item in value:
        assert isinstance(item, int) and not isinstance(item, bool)
        items.append(item)
    return tuple(items)


def _json_str(value: JsonValue) -> str:
    assert isinstance(value, str)
    return value


def test_conference_registry_contains_expected_presets() -> None:
    entries = [entry for entry in iter_presets() if entry.group.endswith("-conference")]
    counts = Counter(entry.group for entry in entries)

    assert len(entries) == 81
    assert counts == {
        "amazon-reviews-23-conference": 19,
        "arxiv-conference": 20,
        "yearbook-conference": 21,
        "imdb-faces-conference": 21,
    }
    for entry in entries:
        assert entry.seeds == _CONFERENCE_SEEDS_BY_GROUP[entry.group]


def test_conference_presets_share_protocol_within_dataset() -> None:
    entries = [entry for entry in iter_presets() if entry.group.endswith("-conference")]

    for group in {entry.group for entry in entries}:
        configs = [entry.build() for entry in entries if entry.group == group]
        variants = {cfg.dataset.variant for cfg in configs}
        split_seeds = {
            _json_int(_json_dict(cfg.metadata["split"])["split_seed"])
            for cfg in configs
        }
        model_seeds = {
            _json_int_tuple(_json_dict(cfg.metadata["seeds"])["model_seeds"])
            for cfg in configs
        }
        comparison_groups = {
            _json_str(cfg.metadata["comparison_group"]) for cfg in configs
        }

        assert len(variants) == 1
        assert split_seeds == {42}
        assert model_seeds == {_CONFERENCE_SEEDS_BY_GROUP[group]}
        assert len(comparison_groups) == 1


def test_conference_presets_stratify_by_matches_pipeline_split() -> None:
    # The amazon and arxiv pipelines split only by the time column (no
    # label_col passed to create_stratified_temporal_train_test_val_splits), so
    # the protocol metadata must not claim a label stratification dimension it
    # never performs. Must stay truthful to pipeline/<dataset>/run.py.
    expected = {
        "amazon-reviews-23-conference": ("half_year",),
        "arxiv-conference": ("year",),
    }
    entries = [entry for entry in iter_presets() if entry.group.endswith("-conference")]

    for group, want in expected.items():
        configs = [entry.build() for entry in entries if entry.group == group]
        assert configs, f"no presets found for {group}"
        stratify = {
            tuple(_json_dict(cfg.metadata["split"])["stratify_by"]) for cfg in configs
        }
        assert stratify == {want}, f"{group} stratify_by mismatch: {stratify}"


def test_conference_preset_training_matches_factory() -> None:
    factories: Mapping[str, tuple[Mapping[str, TrainingConfig], tuple[str, ...]]] = {
        "amazon-reviews-23-conference": (
            amazon_reviews_conference_trainer_configs(class_weights=torch.ones(5)),
            _TEXT_TRAINING_FIELDS,
        ),
        "arxiv-conference": (
            arxiv_conference_trainer_configs(category_to_idx={"a": 0, "b": 1}),
            _TEXT_TRAINING_FIELDS,
        ),
        "yearbook-conference": (
            yearbook_conference_trainer_configs(),
            _IMAGE_TRAINING_FIELDS,
        ),
        "imdb-faces-conference": (
            imdb_faces_conference_trainer_configs(),
            _IMAGE_TRAINING_FIELDS,
        ),
    }

    entries = [entry for entry in iter_presets() if entry.group.endswith("-conference")]
    for entry in entries:
        factory_configs, fields = factories[entry.group]
        training = entry.build().trainer.training
        factory_config = factory_configs[entry.name]
        for field in fields:
            assert training[field] == getattr(factory_config, field), (
                f"{entry.name}: snapshot {field}={training[field]!r} disagrees with "
                f"factory {field}={getattr(factory_config, field)!r}"
            )
        if entry.group == "amazon-reviews-23-conference":
            for key, expected in _AMAZON_DESCRIPTIVE_TRAINING.items():
                assert training[key] == expected, (
                    f"{entry.name}: descriptive training key {key}={training[key]!r} "
                    f"diverged from the pipeline reality ({expected!r})"
                )

    preset_names_by_group: dict[str, set[str]] = {}
    for entry in entries:
        preset_names_by_group.setdefault(entry.group, set()).add(entry.name)
    for group, (factory_configs, _fields) in factories.items():
        expected_names = preset_names_by_group[group]
        if group == "amazon-reviews-23-conference":
            # Amazon MiniLM is kept runnable for its smoke preset, but it is not
            # part of the rendered Amazon conference matrix or paper results.
            expected_names = expected_names | {"minilm_l6_frozen"}
        assert set(factory_configs) == expected_names, (
            f"{group}: factory architecture keys are not 1:1 with preset names: "
            f"{sorted(set(factory_configs) ^ expected_names)}"
        )


def test_amazon_minilm_is_smoke_only() -> None:
    entries = list(iter_presets())

    assert not any(
        entry.group == "amazon-reviews-23-conference"
        and entry.name == "minilm_l6_frozen"
        for entry in entries
    )
    assert any(
        entry.group == "amazon-reviews-23" and entry.name == "smoke-minilm-l6-frozen"
        for entry in entries
    )


def test_conference_preset_entries_declare_comparison_invariants() -> None:
    conference_entries = [
        entry for entry in iter_presets() if entry.group.endswith("-conference")
    ]

    for entry in conference_entries:
        assert entry.comparison_group is not None, (
            f"{entry.group}/{entry.name} has no comparison_group; "
            "validate_comparison_groups would silently skip it"
        )
        assert entry.variant_fields == CONFERENCE_VARIANT_FIELDS, (
            f"{entry.group}/{entry.name} variant_fields drifted from "
            "CONFERENCE_VARIANT_FIELDS"
        )
        cfg = entry.build()
        assert entry.comparison_group == _json_str(cfg.metadata["comparison_group"]), (
            f"{entry.group}/{entry.name}: PresetEntry.comparison_group "
            f"({entry.comparison_group!r}) disagrees with the materialized metadata "
            f"comparison_group ({cfg.metadata['comparison_group']!r})"
        )
