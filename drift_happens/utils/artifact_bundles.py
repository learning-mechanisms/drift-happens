"""Build curated public artifact bundles from local run artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from drift_happens.dataset.utils import download_pcloud_file, safe_extract_tar
from drift_happens.utils.paths import BUNDLES_DIR, RUNS_DIR, relative_to_project

BundleName = Literal["public-full-runs", "yearbook-saliency"]

PUBLIC_FULL_RUNS: BundleName = "public-full-runs"
YEARBOOK_SALIENCY: BundleName = "yearbook-saliency"
BUNDLE_NAMES: tuple[BundleName, ...] = (PUBLIC_FULL_RUNS, YEARBOOK_SALIENCY)

_PUBLIC_FULL_RUNS_DOWNLOAD_LINK = (
    "https://e.pcloud.link/publink/show?code=XZghvcZHNVVTmTJLXm7eYWNk6hiSbgtvviX"
)
_YEARBOOK_SALIENCY_DOWNLOAD_LINK = (
    "https://e.pcloud.link/publink/show?code=XZc3icZyr0QCT2QmfB3gy9qRyiVvFJUyThX"
)
_PUBLIC_DATASETS = frozenset(("amazon_reviews_23", "arxiv", "yearbook"))
_TRAINED_MODEL_NAMES = frozenset(
    ("trained_model.model", "trained_model.pt", "trained_model.pth")
)
_CONFIG_NAMES = frozenset(
    ("config.input.json", "config.input.yaml", "config.input.yml")
)
_ROOT_METADATA_NAMES = frozenset(
    ("metadata.json", "run_manifest.json", "snapshot.json", *_CONFIG_NAMES)
)
_EXCLUDED_PARTS = frozenset(("attempts", "checkpoints", ".locks", "wandb"))


@dataclass(frozen=True, slots=True)
class BundleDownload:
    """Public download metadata for a packed bundle archive."""

    download_link: str | None = None
    expected_sha256: str | None = None
    expected_size: int | None = None


@dataclass(frozen=True, slots=True)
class BundleDefinition:
    """Static policy for one reproducibility bundle."""

    name: BundleName
    archive_name: str
    download: BundleDownload = field(default_factory=BundleDownload)


@dataclass(frozen=True, slots=True)
class BundleResult:
    """Summary of a staged, packed, or downloaded bundle."""

    bundle_dir: Path
    staged_dir: Path
    archive_path: Path
    manifest_path: Path
    sha256_path: Path
    file_count: int = 0
    size_bytes: int = 0


_DEFINITIONS: dict[BundleName, BundleDefinition] = {
    PUBLIC_FULL_RUNS: BundleDefinition(
        name=PUBLIC_FULL_RUNS,
        archive_name="public-full-runs.tar.gz",
        download=BundleDownload(
            download_link=_PUBLIC_FULL_RUNS_DOWNLOAD_LINK,
            expected_sha256=(
                "46ef0ffee6cc578cd60a85809ef4b3de8a79979a1961ab41c6d34fd3ac614de7"
            ),
            expected_size=26_352_419_354,
        ),
    ),
    YEARBOOK_SALIENCY: BundleDefinition(
        name=YEARBOOK_SALIENCY,
        archive_name="yearbook-saliency.tar.gz",
        download=BundleDownload(
            download_link=_YEARBOOK_SALIENCY_DOWNLOAD_LINK,
            expected_sha256=(
                "72db66a698aa842307a2d13e01591645b57c4ef838b0a7f2b54ce36b5082e073"
            ),
            expected_size=32_100_167,
        ),
    ),
}


def bundle_definition(name: str) -> BundleDefinition:
    """Return a bundle definition or raise for unknown names."""
    if name not in _DEFINITIONS:
        valid = ", ".join(BUNDLE_NAMES)
        raise ValueError(f"unknown bundle {name!r}; choose one of: {valid}")
    return _DEFINITIONS[cast(BundleName, name)]


def stage_bundle(
    name: str,
    *,
    runs_root: Path = RUNS_DIR,
    bundle_root: Path = BUNDLES_DIR,
    overwrite: bool = False,
) -> BundleResult:
    """Create the unpacked staging directory for a reproducibility bundle."""
    definition = bundle_definition(name)
    bundle_dir = bundle_root / definition.name
    staged_dir = bundle_dir / "staged"
    _prepare_empty_dir(staged_dir, overwrite=overwrite)
    if definition.name == PUBLIC_FULL_RUNS:
        files = _stage_public_full_runs(runs_root=runs_root, staged_dir=staged_dir)
    elif definition.name == YEARBOOK_SALIENCY:
        files = _stage_yearbook_saliency(runs_root=runs_root, staged_dir=staged_dir)
    else:
        raise AssertionError(f"unhandled bundle: {definition.name}")
    manifest = _manifest(definition.name, files)
    _write_manifest(staged_dir / "manifest.json", manifest)
    _write_manifest(bundle_dir / "manifest.json", manifest)
    return _result(definition, bundle_root, files)


def pack_bundle(
    name: str,
    *,
    bundle_root: Path = BUNDLES_DIR,
    overwrite: bool = False,
) -> BundleResult:
    """Pack a staged bundle into a tar archive and write its sha256 file."""
    definition = bundle_definition(name)
    bundle_dir = bundle_root / definition.name
    staged_dir = bundle_dir / "staged"
    if not staged_dir.is_dir():
        raise FileNotFoundError(f"missing staged bundle directory: {staged_dir}")
    archive_path = bundle_dir / definition.archive_name
    sha256_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    if archive_path.exists() and not overwrite:
        raise FileExistsError(f"bundle archive already exists: {archive_path}")
    if sha256_path.exists() and not overwrite:
        raise FileExistsError(f"bundle sha256 already exists: {sha256_path}")
    archive_path.unlink(missing_ok=True)
    sha256_path.unlink(missing_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_reproducible_archive(archive_path, staged_dir)
    digest = sha256_file(archive_path)
    sha256_path.write_text(f"{digest}  {archive_path.name}\n")
    return _result(definition, bundle_root, _listed_files(staged_dir))


def build_bundle(
    name: str,
    *,
    runs_root: Path = RUNS_DIR,
    bundle_root: Path = BUNDLES_DIR,
    overwrite: bool = False,
) -> BundleResult:
    """Stage and pack a reproducibility bundle."""
    stage_bundle(
        name,
        runs_root=runs_root,
        bundle_root=bundle_root,
        overwrite=overwrite,
    )
    return pack_bundle(name, bundle_root=bundle_root, overwrite=overwrite)


def download_bundle(
    name: str,
    *,
    bundle_root: Path = BUNDLES_DIR,
    download_link: str | None = None,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    skip_integrity_check: bool = False,
    overwrite: bool = False,
) -> BundleResult:
    """Download and safely extract a public reproducibility bundle."""
    definition = bundle_definition(name)
    link = download_link or definition.download.download_link
    if link is None:
        raise ValueError(
            f"no public download link configured for {definition.name}; "
            "pass --download-link or update the bundle definition"
        )
    bundle_dir = bundle_root / definition.name
    staged_dir = bundle_dir / "staged"
    archive_path = bundle_dir / definition.archive_name
    if staged_dir.exists() and not overwrite:
        raise FileExistsError(f"staged bundle already exists: {staged_dir}")
    if archive_path.exists() and not overwrite:
        raise FileExistsError(f"bundle archive already exists: {archive_path}")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if skip_integrity_check:
        archive_sha256 = None
        archive_size = None
    else:
        archive_sha256 = expected_sha256 or definition.download.expected_sha256
        archive_size = (
            expected_size
            if expected_size is not None
            else definition.download.expected_size
        )
    download_pcloud_file(
        archive_path,
        download_link=link,
        expected_sha256=archive_sha256,
        expected_size=archive_size,
    )
    staging = Path(tempfile.mkdtemp(dir=bundle_dir, prefix=".extract-"))
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            safe_extract_tar(archive, staging)
        if staged_dir.exists():
            shutil.rmtree(staged_dir)
        staging.replace(staged_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    manifest = staged_dir / "manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, bundle_dir / "manifest.json")
    sha256_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    sha256_path.write_text(f"{sha256_file(archive_path)}  {archive_path.name}\n")
    return _result(definition, bundle_root, _listed_files(staged_dir))


def sha256_file(path: Path) -> str:
    """Return the sha256 digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _reproducible_member(member: tarfile.TarInfo) -> tarfile.TarInfo:
    """Strip machine- and time-specific metadata from a tar member."""
    member.mtime = 0
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mode = 0o644
    return member


def _write_reproducible_archive(archive_path: Path, staged_dir: Path) -> None:
    """Write a deterministic tar.gz so identical inputs yield identical bytes."""
    files = sorted(file for file in staged_dir.rglob("*") if file.is_file())
    with archive_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=9) as gz:
            with tarfile.open(
                fileobj=gz, mode="w", format=tarfile.PAX_FORMAT
            ) as archive:
                for path in files:
                    archive.add(
                        path,
                        arcname=path.relative_to(staged_dir).as_posix(),
                        filter=_reproducible_member,
                    )


def _stage_public_full_runs(
    *, runs_root: Path, staged_dir: Path
) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    if not runs_root.is_dir():
        raise FileNotFoundError(f"missing runs root: {runs_root}")
    for source in _iter_run_files(runs_root):
        if not _public_full_runs_includes(source, runs_root):
            continue
        relative = source.relative_to(runs_root)
        destination = staged_dir / "runs" / relative
        _copy_file(source, destination)
        files.append(_file_record(source, destination.relative_to(staged_dir)))
    if not files:
        raise FileNotFoundError(f"no public full-run files found below {runs_root}")
    return files


def _stage_yearbook_saliency(
    *, runs_root: Path, staged_dir: Path
) -> list[dict[str, object]]:
    trainers = ("cnn_l", "resnet_s", "mlp_l")
    cutoffs = (1950, 1970)
    files: list[dict[str, object]] = []
    for trainer in trainers:
        for cutoff in cutoffs:
            source = _find_saliency_checkpoint(runs_root, trainer, cutoff)
            relative = Path(trainer) / f"train_slice_{cutoff}" / "trained_model.pt"
            destination = staged_dir / relative
            _copy_file(source, destination)
            files.append(
                _file_record(
                    source,
                    relative,
                    sha256=sha256_file(destination),
                    trainer=trainer,
                    cutoff=cutoff,
                )
            )
    return files


def _find_saliency_checkpoint(runs_root: Path, trainer: str, cutoff: int) -> Path:
    pattern = (
        f"yearbook/{trainer}/yearbook-conference-{trainer}/seed=0/*/"
        f"stages/train/{trainer}/train_slice_{cutoff}/trained_model.pt"
    )
    matches = sorted(runs_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"missing saliency checkpoint for {trainer} cutoff {cutoff}"
        )
    if len(matches) > 1:
        joined = ", ".join(str(relative_to_project(path)) for path in matches)
        raise RuntimeError(
            f"multiple saliency checkpoints for {trainer} cutoff {cutoff}: {joined}"
        )
    return matches[0]


def _public_full_runs_includes(path: Path, runs_root: Path) -> bool:
    relative = path.relative_to(runs_root)
    parts = relative.parts
    if len(parts) < 5:
        return False
    dataset, _, experiment = parts[:3]
    if dataset not in _PUBLIC_DATASETS:
        return False
    if "conference" not in experiment or any("smoke" in part for part in parts[:3]):
        return False
    if any(part in _EXCLUDED_PARTS for part in parts):
        return False
    if _is_transient_path(parts):
        return False
    name = path.name
    if name in _ROOT_METADATA_NAMES:
        return True
    if any(part in {"logs", "metrics", "results"} for part in parts):
        return True
    if "stages" in parts and name == "completion.json":
        return True
    if "stages" in parts and "train" in parts and name == "training_history.json":
        return True
    return "stages" in parts and "train" in parts and name in _TRAINED_MODEL_NAMES


def _is_transient_path(parts: tuple[str, ...]) -> bool:
    for part in parts:
        if part.endswith(".lock") or part.endswith(".tmp"):
            return True
        if part.startswith(".tmp") or part.startswith(".nfs"):
            return True
    return False


def _iter_run_files(root: Path) -> Iterator[Path]:
    """Yield run files in deterministic order while pruning excluded subtrees."""
    for child in sorted(root.iterdir()):
        relative = child.relative_to(root)
        parts = relative.parts
        if any(part in _EXCLUDED_PARTS for part in parts):
            continue
        if _is_transient_path(parts):
            continue
        if child.is_symlink():
            if child.is_file():
                yield child
            continue
        if child.is_dir():
            yield from _iter_run_files(child)
        elif child.is_file():
            yield child


def _copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise ValueError(f"refusing to bundle symlink: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _file_record(
    source: Path,
    relative: Path,
    *,
    sha256: str | None = None,
    trainer: str | None = None,
    cutoff: int | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "path": relative.as_posix(),
        "size_bytes": source.stat().st_size,
        "source": str(relative_to_project(source)),
    }
    if sha256 is not None:
        record["sha256"] = sha256
    if trainer is not None:
        record["trainer"] = trainer
    if cutoff is not None:
        record["cutoff"] = cutoff
    return record


def _record_size_bytes(record: dict[str, object]) -> int:
    size_bytes = record["size_bytes"]
    if not isinstance(size_bytes, int):
        raise TypeError(f"file record size_bytes must be int: {size_bytes!r}")
    return size_bytes


def _manifest(name: BundleName, files: list[dict[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {
        "bundle": name,
        "file_count": len(files),
        "size_bytes": sum(_record_size_bytes(file) for file in files),
        "files": files,
    }
    if name == YEARBOOK_SALIENCY:
        payload.update(
            {
                "dataset": "yearbook",
                "seed": 0,
                "trainers": ["cnn_l", "resnet_s", "mlp_l"],
                "cutoffs": [1950, 1970],
                "eval_years": [1960, 1980, 2000],
                "saliency_command": "pixi run analysis-saliency",
            }
        )
    return payload


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _prepare_empty_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"staged bundle already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _listed_files(staged_dir: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(file for file in staged_dir.rglob("*") if file.is_file()):
        relative = path.relative_to(staged_dir)
        if relative == Path("manifest.json"):
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
            }
        )
    return files


def _result(
    definition: BundleDefinition,
    bundle_root: Path,
    files: list[dict[str, object]],
) -> BundleResult:
    bundle_dir = bundle_root / definition.name
    archive_path = bundle_dir / definition.archive_name
    return BundleResult(
        bundle_dir=bundle_dir,
        staged_dir=bundle_dir / "staged",
        archive_path=archive_path,
        manifest_path=bundle_dir / "manifest.json",
        sha256_path=archive_path.with_suffix(archive_path.suffix + ".sha256"),
        file_count=len(files),
        size_bytes=sum(_record_size_bytes(file) for file in files),
    )
