"""
Registry and cleanup helpers for on-disk embedding caches.

Every dataset that precomputes embeddings writes them through a
:class:`~drift_happens.dataset.dataset.TensorDatasetCache` under its own
``cache_tensor_dataset`` directory. This module collects those caches in one
place so operators can inspect and clear them.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

from drift_happens.dataset.dataset import TensorDatasetCache
from drift_happens.dataset.imdb_faces.const import IMDB_TENSOR_DATASET_CACHE
from drift_happens.dataset.yearbook.const import YB_TENSOR_DATASET_CACHE
from drift_happens.utils.log import get_logger
from drift_happens.utils.paths import ARTIFACTS_DIR

logger = get_logger()

EMBEDDING_CACHES: dict[str, TensorDatasetCache] = {
    "imdb_faces": TensorDatasetCache(
        cache_dir=IMDB_TENSOR_DATASET_CACHE, dataset_id="imdb_faces"
    ),
    "yearbook": TensorDatasetCache(
        cache_dir=YB_TENSOR_DATASET_CACHE, dataset_id="yearbook"
    ),
}


def select_caches(dataset: str | None) -> dict[str, TensorDatasetCache]:
    """
    Caches for ``dataset``, or every cache when ``dataset`` is ``None``.

    Raises ``KeyError`` for an unknown dataset name so the caller can surface it rather
    than silently clearing nothing.
    """
    if dataset is None:
        return dict(EMBEDDING_CACHES)
    if dataset not in EMBEDDING_CACHES:
        raise KeyError(dataset)
    return {dataset: EMBEDDING_CACHES[dataset]}


def delete_cache_files(files: tuple[Path, ...]) -> tuple[Path, ...]:
    """
    Delete embedding cache files after verifying each belongs to a cache.

    A path is deleted only when it sits inside one of the registered cache directories
    and its name starts with ``{dataset_id}_embed_``, so a stray path can never remove
    anything outside the embedding caches.
    """
    roots = {
        cache.cache_dir.resolve(): cache.dataset_id
        for cache in EMBEDDING_CACHES.values()
    }
    deleted: list[Path] = []
    for file in files:
        resolved = file.resolve()
        parent = resolved.parent
        dataset_id = roots.get(parent)
        if dataset_id is None or not resolved.name.startswith(f"{dataset_id}_embed_"):
            raise ValueError(f"refusing to delete non-embedding-cache path: {file}")
        if resolved.exists():
            resolved.unlink()
            deleted.append(resolved)
    return tuple(deleted)


# The conference text pipelines write manifest-backed feature caches under a
# single tree, one leaf directory per (dataset, variant, producer, kind).
FEATURE_CACHE_ROOT = ARTIFACTS_DIR / "cache"

# A feature cache leaf is exactly {dataset}/{variant}/{producer}/{kind} below the
# root, so a manifest at any other depth is not one of ours and is left alone.
_FEATURE_CACHE_DEPTH = 4


@dataclass(frozen=True)
class FeatureCacheEntry:
    """One text feature cache directory (a manifest.json plus its chunk files)."""

    path: Path
    dataset: str
    producer: str
    kind: str

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(sorted(p for p in self.path.iterdir() if p.is_file()))

    @property
    def size_bytes(self) -> int:
        return sum(file.stat().st_size for file in self.files)


def discover_feature_caches(root: Path | None = None) -> tuple[FeatureCacheEntry, ...]:
    """
    Every text feature cache (a directory holding a manifest.json) under ``root``.

    Only directories at the canonical ``{dataset}/{variant}/{producer}/{kind}`` depth
    count as caches; a manifest at any other depth is logged and skipped.
    """
    base = root or FEATURE_CACHE_ROOT
    if not base.exists():
        return ()
    entries: list[FeatureCacheEntry] = []
    for manifest_path in sorted(base.rglob("manifest.json")):
        cache_dir = manifest_path.parent
        parts = cache_dir.relative_to(base).parts
        if len(parts) != _FEATURE_CACHE_DEPTH:
            logger.warning(f"Ignoring manifest at an unexpected depth: {manifest_path}")
            continue
        entries.append(
            FeatureCacheEntry(
                path=cache_dir,
                dataset=parts[0],
                producer=parts[-2],
                kind=parts[-1],
            )
        )
    return tuple(entries)


def delete_feature_cache_dirs(dirs: tuple[Path, ...]) -> tuple[Path, ...]:
    """
    Delete text feature cache directories after verifying each is a real cache.

    A directory is removed only when it sits inside the feature cache root and holds a
    manifest.json, so the deletion can never escape the cache tree.
    """
    root = FEATURE_CACHE_ROOT.resolve()
    deleted: list[Path] = []
    for directory in dirs:
        resolved = directory.resolve()
        if (
            not resolved.is_relative_to(root)
            or not (resolved / "manifest.json").exists()
        ):
            raise ValueError(f"refusing to delete non-feature-cache path: {directory}")
        shutil.rmtree(resolved)
        deleted.append(resolved)
    return tuple(deleted)
