from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import pytest

from drift_happens.utils.artifact_bundles import (
    build_bundle,
    download_bundle,
    pack_bundle,
    sha256_file,
    stage_bundle,
)

SALIENCY_TRAINERS = ("cnn_l", "resnet_s", "mlp_l")
SALIENCY_CUTOFFS = (1950, 1970)


def test_public_full_runs_stage_includes_only_curated_conference_files(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    bundle_root = tmp_path / "bundles"
    run_dir = (
        runs_root
        / "yearbook"
        / "mlp_l"
        / "yearbook-conference-mlp_l"
        / "seed=0"
        / "run"
    )
    _write(run_dir / "metadata.json", b"{}")
    _write(run_dir / "logs" / "train.log", b"log")
    _write(run_dir / "metrics" / "train.json", b"{}")
    _write(run_dir / "results" / "drift_matrix.json", b"{}")
    _write(run_dir / "stages" / "train" / "completion.json", b"{}")
    _write(
        run_dir
        / "stages"
        / "train"
        / "mlp_l"
        / "train_slice_1950"
        / "training_history.json",
        b"{}",
    )
    _write(
        run_dir
        / "stages"
        / "train"
        / "mlp_l"
        / "train_slice_1950"
        / "trained_model.pt",
        b"model",
    )
    _write(run_dir / "attempts" / "train" / "attempt.json", b"skip")
    _write(tmp_path / "wandb-cache" / "debug-core.log", b"skip")
    wandb_link = run_dir / "wandb" / "run-abc" / "logs" / "debug-core.log"
    wandb_link.parent.mkdir(parents=True)
    wandb_link.symlink_to(tmp_path / "wandb-cache" / "debug-core.log")
    _write(
        run_dir
        / "stages"
        / "train"
        / "mlp_l"
        / "train_slice_1950"
        / "checkpoints"
        / "epoch.pt",
        b"skip",
    )
    _write(run_dir / ".locks" / "train.lock", b"skip")
    _write(
        runs_root
        / "synthetic"
        / "linear"
        / "smoke"
        / "seed=0"
        / "run"
        / "metadata.json",
        b"skip",
    )
    _write(
        runs_root
        / "yearbook"
        / "mlp_s"
        / "yearbook-smoke-mlp-s"
        / "seed=0"
        / "run"
        / "metadata.json",
        b"skip",
    )

    result = stage_bundle(
        "public-full-runs",
        runs_root=runs_root,
        bundle_root=bundle_root,
    )

    staged = result.staged_dir
    assert (staged / "runs" / "yearbook" / "mlp_l").exists()
    assert (
        staged / "runs" / "yearbook" / "mlp_l" / "yearbook-conference-mlp_l"
    ).exists()
    assert not list(staged.rglob("attempt.json"))
    assert not list(staged.rglob("debug-core.log"))
    assert not list(staged.rglob("epoch.pt"))
    assert not list(staged.rglob("train.lock"))
    assert not (staged / "runs" / "synthetic").exists()
    assert not (
        staged / "runs" / "yearbook" / "mlp_s" / "yearbook-smoke-mlp-s"
    ).exists()
    manifest = json.loads((staged / "manifest.json").read_text())
    assert manifest["bundle"] == "public-full-runs"
    assert manifest["file_count"] == 7


def test_yearbook_saliency_stage_creates_loader_layout(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    bundle_root = tmp_path / "bundles"
    for trainer in SALIENCY_TRAINERS:
        for cutoff in SALIENCY_CUTOFFS:
            _saliency_checkpoint(runs_root, trainer, cutoff).write_bytes(
                f"{trainer}-{cutoff}".encode()
            )

    result = stage_bundle(
        "yearbook-saliency",
        runs_root=runs_root,
        bundle_root=bundle_root,
    )

    for trainer in SALIENCY_TRAINERS:
        for cutoff in SALIENCY_CUTOFFS:
            assert (
                result.staged_dir
                / trainer
                / f"train_slice_{cutoff}"
                / "trained_model.pt"
            ).exists()
    manifest = json.loads((result.staged_dir / "manifest.json").read_text())
    assert manifest["bundle"] == "yearbook-saliency"
    assert manifest["trainers"] == list(SALIENCY_TRAINERS)
    assert manifest["cutoffs"] == list(SALIENCY_CUTOFFS)
    assert manifest["eval_years"] == [1960, 1980, 2000]
    assert manifest["saliency_command"] == "pixi run analysis-saliency"
    assert all("sha256" in file for file in manifest["files"])


def test_yearbook_saliency_stage_fails_when_checkpoint_is_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="missing saliency checkpoint"):
        stage_bundle(
            "yearbook-saliency",
            runs_root=tmp_path / "runs",
            bundle_root=tmp_path / "bundles",
        )


def test_pack_bundle_writes_archive_and_sha256(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    bundle_root = tmp_path / "bundles"
    for trainer in SALIENCY_TRAINERS:
        for cutoff in SALIENCY_CUTOFFS:
            _saliency_checkpoint(runs_root, trainer, cutoff).write_bytes(b"model")

    stage_bundle("yearbook-saliency", runs_root=runs_root, bundle_root=bundle_root)
    result = pack_bundle("yearbook-saliency", bundle_root=bundle_root)

    assert result.archive_path.exists()
    assert result.sha256_path.exists()
    assert result.archive_path.name in result.sha256_path.read_text()
    with tarfile.open(result.archive_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert "manifest.json" in names
    assert "mlp_l/train_slice_1950/trained_model.pt" in names


def test_pack_bundle_is_byte_reproducible(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    for trainer in SALIENCY_TRAINERS:
        for cutoff in SALIENCY_CUTOFFS:
            _saliency_checkpoint(runs_root, trainer, cutoff).write_bytes(b"model")

    first = build_bundle(
        "yearbook-saliency",
        runs_root=runs_root,
        bundle_root=tmp_path / "first",
    )
    second = build_bundle(
        "yearbook-saliency",
        runs_root=runs_root,
        bundle_root=tmp_path / "second",
    )

    assert sha256_file(first.archive_path) == sha256_file(second.archive_path)


def test_download_bundle_extracts_downloaded_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runs"
    source_bundle_root = tmp_path / "source-bundles"
    for trainer in SALIENCY_TRAINERS:
        for cutoff in SALIENCY_CUTOFFS:
            _saliency_checkpoint(runs_root, trainer, cutoff).write_bytes(b"model")
    source = build_bundle(
        "yearbook-saliency",
        runs_root=runs_root,
        bundle_root=source_bundle_root,
    )

    def fake_download(destination: Path, **_: object) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.archive_path, destination)

    monkeypatch.setattr(
        "drift_happens.utils.artifact_bundles.download_pcloud_file",
        fake_download,
    )

    result = download_bundle(
        "yearbook-saliency",
        bundle_root=tmp_path / "downloaded-bundles",
        download_link="https://e.pcloud.link/publink/show?code=fake",
    )

    assert (
        result.staged_dir / "resnet_s" / "train_slice_1970" / "trained_model.pt"
    ).exists()


def test_download_bundle_uses_integrity_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runs"
    source_bundle_root = tmp_path / "source-bundles"
    for trainer in SALIENCY_TRAINERS:
        for cutoff in SALIENCY_CUTOFFS:
            _saliency_checkpoint(runs_root, trainer, cutoff).write_bytes(b"model")
    source = build_bundle(
        "yearbook-saliency",
        runs_root=runs_root,
        bundle_root=source_bundle_root,
    )
    captured: dict[str, object] = {}

    def fake_download(destination: Path, **kwargs: object) -> None:
        captured.update(kwargs)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.archive_path, destination)

    monkeypatch.setattr(
        "drift_happens.utils.artifact_bundles.download_pcloud_file",
        fake_download,
    )

    download_bundle(
        "yearbook-saliency",
        bundle_root=tmp_path / "downloaded-bundles",
        download_link="https://e.pcloud.link/publink/show?code=fake",
        expected_sha256="a" * 64,
        expected_size=123,
    )

    assert captured["expected_sha256"] == "a" * 64
    assert captured["expected_size"] == 123


def test_download_bundle_can_skip_configured_integrity_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runs"
    source_bundle_root = tmp_path / "source-bundles"
    for trainer in SALIENCY_TRAINERS:
        for cutoff in SALIENCY_CUTOFFS:
            _saliency_checkpoint(runs_root, trainer, cutoff).write_bytes(b"model")
    source = build_bundle(
        "yearbook-saliency",
        runs_root=runs_root,
        bundle_root=source_bundle_root,
    )
    captured: dict[str, object] = {}

    def fake_download(destination: Path, **kwargs: object) -> None:
        captured.update(kwargs)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.archive_path, destination)

    monkeypatch.setattr(
        "drift_happens.utils.artifact_bundles.download_pcloud_file",
        fake_download,
    )

    download_bundle(
        "yearbook-saliency",
        bundle_root=tmp_path / "downloaded-bundles",
        download_link="https://e.pcloud.link/publink/show?code=fake",
        skip_integrity_check=True,
    )

    assert captured["expected_sha256"] is None
    assert captured["expected_size"] is None


def test_download_bundle_refreshes_stale_checksum_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runs"
    source_bundle_root = tmp_path / "source-bundles"
    for trainer in SALIENCY_TRAINERS:
        for cutoff in SALIENCY_CUTOFFS:
            _saliency_checkpoint(runs_root, trainer, cutoff).write_bytes(b"model")
    source = build_bundle(
        "yearbook-saliency",
        runs_root=runs_root,
        bundle_root=source_bundle_root,
    )

    def fake_download(destination: Path, **_: object) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.archive_path, destination)

    monkeypatch.setattr(
        "drift_happens.utils.artifact_bundles.download_pcloud_file",
        fake_download,
    )

    download_root = tmp_path / "downloaded-bundles"
    bundle_dir = download_root / "yearbook-saliency"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "yearbook-saliency.tar.gz").write_bytes(b"stale archive")
    stale_sidecar = bundle_dir / "yearbook-saliency.tar.gz.sha256"
    stale_sidecar.write_text("deadbeef  yearbook-saliency.tar.gz\n")

    result = download_bundle(
        "yearbook-saliency",
        bundle_root=download_root,
        download_link="https://e.pcloud.link/publink/show?code=fake",
        overwrite=True,
    )

    expected = sha256_file(result.archive_path)
    assert result.sha256_path.read_text() == f"{expected}  {result.archive_path.name}\n"
    assert "deadbeef" not in result.sha256_path.read_text()


def _saliency_checkpoint(runs_root: Path, trainer: str, cutoff: int) -> Path:
    path = (
        runs_root
        / "yearbook"
        / trainer
        / f"yearbook-conference-{trainer}"
        / "seed=0"
        / f"run-{trainer}"
        / "stages"
        / "train"
        / trainer
        / f"train_slice_{cutoff}"
        / "trained_model.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
