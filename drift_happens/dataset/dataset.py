"""TensorDataset cache utilities."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from drift_happens.configs.experiment import CacheReusePolicy
from drift_happens.utils.log import get_logger

logger = get_logger()


@dataclass
class TensorDatasetCache:
    """On-disk cache of embedded TensorDatasets; computes via a callback and reuses per
    cache policy."""

    cache_dir: Path
    dataset_id: str  # e.g. "yearbook"

    def _embed_cache_path(self, embed_id: str) -> Path:
        return self.cache_dir / f"{self.dataset_id}_embed_{embed_id}.pt"

    def cached_files(self) -> tuple[Path, ...]:
        """Existing on-disk embedding cache files for this dataset."""
        if not self.cache_dir.exists():
            return ()
        return tuple(sorted(self.cache_dir.glob(f"{self.dataset_id}_embed_*.pt")))

    def get_create_cache_embedding(
        self,
        embed_id: str,
        embedding_fn: Callable[[], TensorDataset],
        *,
        reuse_policy: CacheReusePolicy = "reuse",
    ) -> TensorDataset:
        """
        Forward-only embedding with caching.

        ``reuse_policy`` controls how the on-disk cache is used: ``reuse`` loads an
        existing file and otherwise computes and writes one; ``refresh`` recomputes and
        overwrites, ignoring any existing file; ``disabled`` computes without reading or
        writing the cache.
        """
        path = self._embed_cache_path(embed_id)

        if reuse_policy == "reuse" and path.exists():
            logger.info(f"Loading cached embedding from {path}")
            return _load_tensor_dataset(path)

        if reuse_policy == "disabled":
            logger.info(
                f"Computing embedding without cache (reuse_policy={reuse_policy})"
            )
            return embedding_fn()

        logger.info(f"Creating embedding at {path} (reuse_policy={reuse_policy})")
        embedded = embedding_fn()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so an interrupted save never leaves a truncated
        # file at a path a later run would happily load.
        tmp_path = path.with_name(path.name + ".tmp")
        try:
            torch.save(embedded, tmp_path)
            tmp_path.replace(path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return embedded


def _load_tensor_dataset(path: Path) -> TensorDataset:
    """Load a cached TensorDataset through PyTorch's safe unpickler."""
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        cached = torch.load(path, map_location="cpu")
    else:
        with safe_globals([TensorDataset]):
            cached = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(cached, TensorDataset):
        raise TypeError(f"cached embedding is not a TensorDataset: {path}")
    return cached
