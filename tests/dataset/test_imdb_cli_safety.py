from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from drift_happens.dataset.imdb_faces import cli, load


def _make_raw_tar(path: Path, payload: bytes = b"IMG") -> None:
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("raw/data.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


def _point_imdb_cli_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "imdb_faces"
    target.mkdir(parents=True)
    unpack_dir = target / "raw"
    tar_file = target / "imdb.tar.gz"
    monkeypatch.setattr(cli, "IMDB_UNPACK_DIR", unpack_dir)
    monkeypatch.setattr(cli, "IMDB_TAR_FILE", tar_file)
    return target, unpack_dir, tar_file


def _no_staging_leftover(parent: Path) -> bool:
    return not any(p.name.startswith(".imdb-unpack-") for p in parent.iterdir())


def test_unpack_missing_tar_keeps_existing_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, unpack_dir, tar_file = _point_imdb_cli_at(tmp_path, monkeypatch)
    unpack_dir.mkdir()
    (unpack_dir / "old.txt").write_text("PRECIOUS")

    with pytest.raises(FileNotFoundError):
        cli.unpack(yes=True)

    # The archive is missing, so the existing unpacked dataset must survive and
    # no staging directory should be left behind.
    assert (unpack_dir / "old.txt").read_text() == "PRECIOUS"
    assert _no_staging_leftover(target)


def test_unpack_replaces_data_from_valid_tar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, unpack_dir, tar_file = _point_imdb_cli_at(tmp_path, monkeypatch)
    unpack_dir.mkdir()
    (unpack_dir / "old.txt").write_text("OLD")
    _make_raw_tar(tar_file)

    cli.unpack(yes=True)

    assert (unpack_dir / "data.txt").read_text() == "IMG"
    assert not (unpack_dir / "old.txt").exists()
    assert _no_staging_leftover(target)


def test_download_failure_keeps_existing_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _target, _unpack_dir, tar_file = _point_imdb_cli_at(tmp_path, monkeypatch)
    tar_file.write_bytes(b"OLD ARCHIVE")

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("download failed")

    monkeypatch.setattr(cli, "download_pcloud_file", _boom)

    with pytest.raises(RuntimeError, match="download failed"):
        cli.download(yes=True)

    # A failed re-download must not have destroyed the existing archive.
    assert tar_file.read_bytes() == b"OLD ARCHIVE"


def test_prepare_failure_keeps_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(load, "IMDB_PREPROCESSED_DIR", tmp_path)
    cache_path = load.get_cache_path(target_size=32)
    cache_path.write_text("OLD CACHE")

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("preprocessing failed")

    monkeypatch.setattr(cli, "load_and_preprocess_images_into_df", _boom)

    with pytest.raises(RuntimeError, match="preprocessing failed"):
        cli.prepare(yes=True)

    # A crash during preprocessing must not have destroyed the existing cache.
    assert cache_path.read_text() == "OLD CACHE"
