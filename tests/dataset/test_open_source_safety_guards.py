from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_DATASET_ROOT = ROOT / "drift_happens" / "dataset"


def test_dataset_unpack_commands_do_not_use_raw_extractall() -> None:
    modules = sorted(_DATASET_ROOT.rglob("cli.py"))
    assert modules, "no cli.py files found — check _DATASET_ROOT"

    for path in modules:
        source = path.read_text()
        assert ".extractall(" not in source, path
        assert "unpack_archive(" not in source, path


def test_dataset_constant_modules_do_not_create_directories_at_import() -> None:
    modules = sorted(_DATASET_ROOT.rglob("const.py"))
    assert modules, "no const.py files found — check _DATASET_ROOT"

    for path in modules:
        assert ".mkdir(" not in path.read_text(), path
