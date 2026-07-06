"""Cross-check hardcoded pipeline split sizes and seed against the declared
protocols."""

from __future__ import annotations

import ast
import inspect
from types import ModuleType

import pytest

from drift_happens.configs.protocol import SplitProtocolConfig
from drift_happens.experiments.registry import iter_presets
from drift_happens.pipeline.amazon_reviews_23 import run as amazon_reviews_23_run
from drift_happens.pipeline.arxiv import run as arxiv_run
from drift_happens.pipeline.imdb_faces import run as imdb_faces_run
from drift_happens.pipeline.yearbook import run as yearbook_run

# The split builder each pipeline setup() calls with literal kwargs.
_SPLIT_CALLS: dict[str, tuple[ModuleType, str]] = {
    "amazon_reviews_23": (
        amazon_reviews_23_run,
        "create_stratified_temporal_train_test_val_splits",
    ),
    "arxiv": (arxiv_run, "create_stratified_temporal_train_test_val_splits"),
    "imdb_faces": (imdb_faces_run, "create_instance_based_train_val_test_split"),
    "yearbook": (yearbook_run, "create_stratified_temporal_train_test_val_splits"),
}


def _pipeline_split_constants(module: ModuleType, func_name: str) -> dict[str, object]:
    """Extract the literal split kwargs without running the heavy setup()."""
    tree = ast.parse(inspect.getsource(module))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == func_name
    ]
    assert len(calls) == 1, f"expected one {func_name} call in {module.__name__}"
    result = {}
    for keyword in calls[0].keywords:
        if keyword.arg not in {"train_size", "val_size", "test_size", "seed"}:
            continue
        try:
            result[keyword.arg] = ast.literal_eval(keyword.value)
        except ValueError as exc:
            raise AssertionError(
                f"{keyword.arg!r} in {module.__name__} is not a literal; "
                "update this guard test if it becomes a named constant"
            ) from exc
    return result


def _declared_split(dataset: str) -> SplitProtocolConfig:
    splits = [
        entry.build().protocol.split
        for entry in iter_presets()
        if entry.group.endswith("-conference") and entry.build().dataset.name == dataset
    ]
    if not splits:
        raise AssertionError(f"no conference preset found for dataset {dataset!r}")
    # All presets in the same group must declare the same split sizes.
    first = splits[0]
    for split in splits[1:]:
        assert (
            split.train_size == first.train_size
            and split.val_size == first.val_size
            and split.test_size == first.test_size
            and split.seed == first.seed
        ), f"conference presets for {dataset!r} disagree on split config: {splits}"
    return first


@pytest.mark.parametrize("dataset", sorted(_SPLIT_CALLS))
def test_pipeline_split_constants_match_declared_protocol(dataset: str) -> None:
    module, func_name = _SPLIT_CALLS[dataset]
    split = _declared_split(dataset)

    assert _pipeline_split_constants(module, func_name) == {
        "train_size": split.train_size,
        "val_size": split.val_size,
        "test_size": split.test_size,
        "seed": split.seed,
    }
