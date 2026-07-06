from __future__ import annotations

import io
import tarfile
from collections.abc import Iterable
from pathlib import Path

import pytest

from drift_happens.dataset import utils
from drift_happens.dataset.utils import safe_extract_tar


class FakeResponse:
    def __init__(self, *, text: str = "", chunks: Iterable[bytes] = ()) -> None:
        self.text = text
        self._chunks = tuple(chunks)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Iterable[bytes]:
        return self._chunks


def _make_fake_get(
    chunks: tuple[bytes, ...],
) -> tuple[list[tuple[str, dict[str, object]]], object]:
    """Return (calls, fake_get) sharing the pCloud two-step response pattern."""
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        calls.append((url, kwargs))
        if len(calls) == 1:
            return FakeResponse(text='"downloadlink": "https:\\/\\/files.test\\/x"')
        return FakeResponse(chunks=chunks)

    return calls, fake_get


def test_download_pcloud_file_parses_link_and_streams_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, fake_get = _make_fake_get((b"abc", b"", b"def"))
    monkeypatch.setattr(utils.requests, "get", fake_get)

    target = tmp_path / "dataset.tar.gz"
    utils.download_pcloud_file(
        target, download_link="https://e.pcloud.link/publink/show?code=abc123"
    )

    assert calls[0][0].endswith("code=abc123")
    assert calls[1][0] == "https://files.test/x"
    assert target.read_bytes() == b"abcdef"


def test_download_pcloud_file_sets_timeouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, fake_get = _make_fake_get((b"x",))
    monkeypatch.setattr(utils.requests, "get", fake_get)

    utils.download_pcloud_file(tmp_path / "dataset.tar.gz", file_id="abc123")

    assert calls
    assert all("timeout" in kwargs for _, kwargs in calls)


def test_download_pcloud_file_rejects_missing_download_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        utils.requests,
        "get",
        lambda url, **kwargs: FakeResponse(text="{}"),
    )

    with pytest.raises(ValueError, match="downloadlink"):
        utils.download_pcloud_file(tmp_path / "dataset.tar.gz", file_id="abc123")


def test_download_pcloud_file_rejects_wrong_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, fake_get = _make_fake_get((b"abc",))
    monkeypatch.setattr(utils.requests, "get", fake_get)

    target = tmp_path / "dataset.tar.gz"
    with pytest.raises(ValueError, match="sha256"):
        utils.download_pcloud_file(target, file_id="x", expected_sha256="deadbeef")

    assert not target.exists()
    assert not (tmp_path / "dataset.tar.gz.tmp").exists()


def test_download_pcloud_file_rejects_wrong_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, fake_get = _make_fake_get((b"abc",))
    monkeypatch.setattr(utils.requests, "get", fake_get)

    target = tmp_path / "dataset.tar.gz"
    with pytest.raises(ValueError, match="size mismatch"):
        utils.download_pcloud_file(target, file_id="x", expected_size=999)

    assert not target.exists()
    assert not (tmp_path / "dataset.tar.gz.tmp").exists()


def test_download_pcloud_file_accepts_correct_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    payload = b"abc"
    digest = hashlib.sha256(payload).hexdigest()
    _, fake_get = _make_fake_get((payload,))
    monkeypatch.setattr(utils.requests, "get", fake_get)

    target = tmp_path / "dataset.tar.gz"
    utils.download_pcloud_file(target, file_id="x", expected_sha256=digest)
    assert target.read_bytes() == payload


def test_safe_extract_tar_rejects_path_traversal(tmp_path: Path) -> None:
    tar_path = tmp_path / "bad.tar"
    with tarfile.open(tar_path, "w") as archive:
        payload = b"bad"
        info = tarfile.TarInfo("../bad.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with tarfile.open(tar_path) as archive:
        with pytest.raises(ValueError, match="unsafe tar member"):
            safe_extract_tar(archive, tmp_path / "out")


def test_safe_extract_tar_rejects_symlink(tmp_path: Path) -> None:
    tar_path = tmp_path / "sym.tar"
    with tarfile.open(tar_path, "w") as archive:
        info = tarfile.TarInfo("evil.txt")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)

    with tarfile.open(tar_path) as archive:
        with pytest.raises(ValueError, match="tar links are not allowed"):
            safe_extract_tar(archive, tmp_path / "out")


def test_safe_extract_tar_rejects_hardlink(tmp_path: Path) -> None:
    tar_path = tmp_path / "hard.tar"
    with tarfile.open(tar_path, "w") as archive:
        info = tarfile.TarInfo("evil.txt")
        info.type = tarfile.LNKTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)

    with tarfile.open(tar_path) as archive:
        with pytest.raises(ValueError, match="tar links are not allowed"):
            safe_extract_tar(archive, tmp_path / "out")


def test_safe_extract_tar_extracts_benign_member(tmp_path: Path) -> None:
    tar_path = tmp_path / "good.tar"
    with tarfile.open(tar_path, "w") as archive:
        payload = b"hello"
        info = tarfile.TarInfo("subdir/file.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    out = tmp_path / "out"
    with tarfile.open(tar_path) as archive:
        safe_extract_tar(archive, out)

    assert (out / "subdir" / "file.txt").read_bytes() == payload


def test_safe_extract_tar_rejects_special_files(tmp_path: Path) -> None:
    tar_path = tmp_path / "fifo.tar"
    with tarfile.open(tar_path, "w") as archive:
        info = tarfile.TarInfo("evil.fifo")
        info.type = tarfile.FIFOTYPE
        archive.addfile(info)

    out = tmp_path / "out"
    with tarfile.open(tar_path) as archive:
        with pytest.raises(tarfile.SpecialFileError):
            safe_extract_tar(archive, out)

    assert not (out / "evil.fifo").exists()


def _make_reviews_cache(path: Path, payload: bytes = b"NEW") -> None:
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("reviews/data.parquet")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


def _point_amazon_cli_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from drift_happens.dataset.amazon_reviews_23 import cli

    reviews_dir = tmp_path / "processed" / "reviews"
    cache_file = tmp_path / "processed" / "reviews_cache.tar.gz"
    monkeypatch.setattr(cli, "AR23_PREPROCESSED_REVIEWS_DIR", reviews_dir)
    monkeypatch.setattr(cli, "AR23_PREPROCESSED_REVIEWS_CACHE_FILE", cache_file)
    return cli, reviews_dir, cache_file


def test_build_from_cache_replaces_data_and_drops_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli, reviews_dir, cache_file = _point_amazon_cli_at(tmp_path, monkeypatch)
    reviews_dir.mkdir(parents=True)
    (reviews_dir / "old.parquet").write_text("OLD")
    _make_reviews_cache(cache_file)

    cli.build_from_cache(yes=True)

    assert (reviews_dir / "data.parquet").exists()
    assert not (reviews_dir / "old.parquet").exists()
    assert not cache_file.exists()


def test_build_from_cache_keeps_data_and_drops_corrupt_cache_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli, reviews_dir, cache_file = _point_amazon_cli_at(tmp_path, monkeypatch)
    reviews_dir.mkdir(parents=True)
    (reviews_dir / "old.parquet").write_text("PRECIOUS")
    cache_file.write_bytes(b"not a valid tar")

    with pytest.raises(Exception):
        cli.build_from_cache(yes=True)

    # Existing data is untouched and the corrupt cache is dropped so a rerun
    # re-downloads it.
    assert (reviews_dir / "old.parquet").read_text() == "PRECIOUS"
    assert not cache_file.exists()
