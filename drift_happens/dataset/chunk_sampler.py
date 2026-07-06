"""
Chunk-blocked batch sampling for out-of-core sequence-embedding caches.

A globally shuffled loader over a lazy
:class:`~drift_happens.dataset.cache.ChunkedTensorDataset` re-deserializes a whole
chunk for nearly every sample, because consecutive shuffled rows land in different
chunks. At sequence-cache scale that is unrunnable.

This sampler shuffles within a window of ``shuffle_window`` chunks: it shuffles the
chunk visiting order, then pools the rows of each window of chunks and shuffles them
together before cutting batches. Only a window's chunks need to be resident at once,
so a wider window mixes rows across more chunks, approaching a global shuffle, at the
cost of holding more chunks in memory. A window of one keeps every batch inside a
single chunk; the pooled/materialized paths keep exact global shuffling.

The shuffle is fully deterministic in the supplied seed, so a resumed slice
reproduces an uninterrupted run.

``drop_last`` drops a short trailing batch within each window. The pipeline wires
``drop_last=False``; pass ``True`` only when that per-window semantics is intended.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from drift_happens.dataset.cache import ChunkedTensorDataset


class ChunkBlockedBatchSampler:
    """
    Yield batches of subset positions pooled from a window of cache chunks.

    The sampler indexes *positions* into a ``Subset`` view (``0..len-1``), the same
    values a ``DataLoader`` passes to ``Subset.__getitem__``. ``row_indices`` maps each
    position to its absolute row in the underlying cache, so the sampler can group
    positions by chunk and shuffle each window of chunks together.
    """

    def __init__(
        self,
        dataset: ChunkedTensorDataset,
        row_indices: Sequence[int],
        *,
        batch_size: int,
        drop_last: bool,
        seed: int,
        shuffle_window: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if shuffle_window <= 0:
            raise ValueError("shuffle_window must be positive")
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._seed = seed
        self._shuffle_window = shuffle_window

        # Group subset positions by the chunk their absolute row falls in.
        positions_by_chunk: dict[int, list[int]] = {}
        for position, row in enumerate(row_indices):
            chunk_index = dataset._chunk_index_for_row(row)
            positions_by_chunk.setdefault(chunk_index, []).append(position)
        self._positions_by_chunk = positions_by_chunk

    def _shuffled_chunks(self, generator: torch.Generator) -> list[int]:
        """Chunk indices in this epoch's visiting order (the first draw off
        ``generator``)."""
        chunk_order = sorted(self._positions_by_chunk)
        return [
            chunk_order[i]
            for i in torch.randperm(len(chunk_order), generator=generator).tolist()
        ]

    def _batches(self) -> list[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self._seed)
        shuffled_chunks = self._shuffled_chunks(generator)

        batches: list[list[int]] = []
        for start in range(0, len(shuffled_chunks), self._shuffle_window):
            window = shuffled_chunks[start : start + self._shuffle_window]
            positions = [
                position
                for chunk_index in window
                for position in self._positions_by_chunk[chunk_index]
            ]
            order = torch.randperm(len(positions), generator=generator).tolist()
            shuffled = [positions[i] for i in order]
            for batch_start in range(0, len(shuffled), self._batch_size):
                batch = shuffled[batch_start : batch_start + self._batch_size]
                if self._drop_last and len(batch) < self._batch_size:
                    continue
                batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        # Count batches without materializing them. Only the chunk visiting order (the
        # first draw off the seed) decides how chunks group into windows, and each
        # window's row count fixes its batch count; the within-window row shuffle changes
        # batch *contents*, never their number, so it need not be replayed here. Sharing
        # ``_shuffled_chunks`` keeps this exactly consistent with ``_batches``.
        generator = torch.Generator()
        generator.manual_seed(self._seed)
        shuffled_chunks = self._shuffled_chunks(generator)

        total = 0
        for start in range(0, len(shuffled_chunks), self._shuffle_window):
            window = shuffled_chunks[start : start + self._shuffle_window]
            window_rows = sum(len(self._positions_by_chunk[c]) for c in window)
            if self._drop_last:
                total += window_rows // self._batch_size
            else:
                total += (window_rows + self._batch_size - 1) // self._batch_size
        return total
