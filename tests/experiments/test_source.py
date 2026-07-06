from __future__ import annotations

import json
from pathlib import Path

import pytest

from drift_happens.experiments.source import _snapshot_seeds, load_experiment_source
from drift_happens.utils.snapshot import SNAPSHOT_KIND


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.parametrize("config", [None, [], "oops", 3])
def test_snapshot_with_non_dict_config_raises_targeted_error(
    tmp_path: Path, config: object
) -> None:
    # A snapshot envelope whose config is not an object must fail with the
    # author's targeted diagnostic rather than being reinterpreted as a plain
    # config and dying with misleading pydantic errors.
    path = _write(
        tmp_path / "snap.json",
        {"kind": SNAPSHOT_KIND, "name": "broken", "config": config},
    )

    with pytest.raises(ValueError, match="non-object config"):
        load_experiment_source(path)


def test_snapshot_missing_config_raises_targeted_error(tmp_path: Path) -> None:
    # A snapshot envelope with no config key at all must hit the same guard,
    # not raise a bare KeyError.
    path = _write(tmp_path / "snap.json", {"kind": SNAPSHOT_KIND, "name": "broken"})

    with pytest.raises(ValueError, match="non-object config"):
        load_experiment_source(path)


@pytest.mark.parametrize(
    "raw",
    [
        {},  # missing seeds key
        {"seeds": None},
        {"seeds": 0},
        {"seeds": "0"},
        {"seeds": []},  # empty
        {"seeds": ["0"]},  # non-integer item
        {"seeds": [1.5]},
        {"seeds": [True]},  # bool is not an honest seed
        {"seeds": [0, "1"]},  # one bad item poisons the set
    ],
)
def test_snapshot_seeds_reject_corrupt_values(raw: dict[str, object]) -> None:
    # A corrupt seeds field must fail loudly naming the file rather than
    # silently falling back to (0,), which would report output for a seed set
    # the snapshot never declared.
    with pytest.raises(ValueError, match="seed"):
        _snapshot_seeds(raw, Path("snap.json"))


def test_snapshot_seeds_accept_integer_list() -> None:
    assert _snapshot_seeds({"seeds": [0, 1, 2]}, Path("snap.json")) == (0, 1, 2)


def test_valid_snapshot_still_loads(tmp_path: Path) -> None:
    # Behaviour-preserving: a well-formed snapshot keeps loading from the
    # snapshot path with its declared seeds.
    source = load_experiment_source(
        Path("configs/snapshots/presets/smoke/synthetic-classification-cpu.json")
    )

    assert source.is_preset_snapshot
    assert source.seeds
