from __future__ import annotations

import hashlib
from pathlib import Path

from drift_happens.utils.lockfile import pixi_lock_sha256


def test_pixi_lock_sha256_hashes_file_contents(tmp_path: Path) -> None:
    lockfile = tmp_path / "pixi.lock"
    lockfile.write_bytes(b"lock contents")

    assert pixi_lock_sha256(lockfile) == hashlib.sha256(b"lock contents").hexdigest()


def test_pixi_lock_sha256_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert pixi_lock_sha256(tmp_path / "missing.lock") is None
