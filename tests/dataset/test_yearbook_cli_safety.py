from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from drift_happens.dataset.yearbook import cli


def _make_yearbook_tar(path: Path, payload: bytes = b"IMG") -> None:
    # The yearbook archive extracts its contents directly into the unpack dir.
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("faces/data.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


def _point_yearbook_cli_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "yearbook"
    target.mkdir(parents=True)
    unpack_dir = target / "raw"
    tar_file = target / "yearbook.tar.gz"
    monkeypatch.setattr(cli, "YB_UNPACK_DIR", unpack_dir)
    monkeypatch.setattr(cli, "YB_TAR_FILE", tar_file)
    return target, unpack_dir, tar_file


def _no_staging_leftover(parent: Path) -> bool:
    return not any(p.name.startswith(".yb-unpack-") for p in parent.iterdir())


def test_unpack_missing_tar_keeps_existing_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, unpack_dir, tar_file = _point_yearbook_cli_at(tmp_path, monkeypatch)
    unpack_dir.mkdir()
    (unpack_dir / "old.txt").write_text("PRECIOUS")

    with pytest.raises(FileNotFoundError):
        cli.unpack(yes=True)

    # A missing archive must not destroy the existing unpacked dataset or leave
    # a stale staging directory behind.
    assert (unpack_dir / "old.txt").read_text() == "PRECIOUS"
    assert _no_staging_leftover(target)


def test_unpack_replaces_data_from_valid_tar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, unpack_dir, tar_file = _point_yearbook_cli_at(tmp_path, monkeypatch)
    unpack_dir.mkdir()
    (unpack_dir / "old.txt").write_text("OLD")
    _make_yearbook_tar(tar_file)

    cli.unpack(yes=True)

    assert (unpack_dir / "faces" / "data.txt").read_text() == "IMG"
    assert not (unpack_dir / "old.txt").exists()
    assert _no_staging_leftover(target)


def test_download_failure_keeps_existing_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _target, _unpack_dir, tar_file = _point_yearbook_cli_at(tmp_path, monkeypatch)
    tar_file.write_bytes(b"OLD ARCHIVE")

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("download failed")

    monkeypatch.setattr(cli, "download_pcloud_file", _boom)

    with pytest.raises(RuntimeError, match="download failed"):
        cli.download(yes=True)

    assert tar_file.read_bytes() == b"OLD ARCHIVE"
