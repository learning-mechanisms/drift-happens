from __future__ import annotations

from pathlib import Path

import pytest

from drift_happens.dataset.arxiv import cli


def _fake_kagglehub_source(tmp_path: Path) -> Path:
    """A stand-in for kagglehub's cache directory holding the dataset files."""
    source = tmp_path / "kagglehub_cache" / "versions" / "1"
    source.mkdir(parents=True)
    (source / "arxiv.json").write_text("DATA")
    return source


def _no_staging_leftover(parent: Path) -> bool:
    return not any(p.name.startswith(".arxiv-cache-") for p in parent.iterdir())


def test_download_copies_files_and_leaves_kagglehub_cache_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "arxiv"
    source = _fake_kagglehub_source(tmp_path)
    monkeypatch.setattr(cli, "ARXIV_CACHE_DIR", cache_dir)
    monkeypatch.setattr(
        cli.kagglehub, "dataset_download", lambda *args, **kwargs: str(source)
    )

    cli.download(yes=True)

    # Files land in the cache directory.
    assert (cache_dir / "arxiv.json").read_text() == "DATA"
    # kagglehub's own cache is copied, not emptied, so its completion marker
    # stays consistent and the next run is not a false cache hit.
    assert (source / "arxiv.json").read_text() == "DATA"
    # The atomic swap leaves no staging directory behind.
    assert _no_staging_leftover(tmp_path)


def test_download_failure_keeps_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "arxiv"
    cache_dir.mkdir()
    (cache_dir / "arxiv.json").write_text("PRECIOUS")
    monkeypatch.setattr(cli, "ARXIV_CACHE_DIR", cache_dir)

    def _boom(*args: object, **kwargs: object) -> str:
        raise RuntimeError("download failed")

    monkeypatch.setattr(cli.kagglehub, "dataset_download", _boom)

    with pytest.raises(RuntimeError, match="download failed"):
        cli.download(yes=True)

    # The existing cache is untouched by a failed download.
    assert (cache_dir / "arxiv.json").read_text() == "PRECIOUS"
    assert _no_staging_leftover(tmp_path)


def test_download_keeps_existing_cache_when_copy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "arxiv"
    cache_dir.mkdir()
    (cache_dir / "arxiv.json").write_text("PRECIOUS")
    source = _fake_kagglehub_source(tmp_path)
    monkeypatch.setattr(cli, "ARXIV_CACHE_DIR", cache_dir)
    monkeypatch.setattr(
        cli.kagglehub, "dataset_download", lambda *args, **kwargs: str(source)
    )

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("copy interrupted")

    monkeypatch.setattr(cli.shutil, "copytree", _boom)

    with pytest.raises(OSError, match="copy interrupted"):
        cli.download(yes=True)

    # A download that succeeds but whose copy fails must not destroy the cache.
    assert (cache_dir / "arxiv.json").read_text() == "PRECIOUS"
    assert _no_staging_leftover(tmp_path)
