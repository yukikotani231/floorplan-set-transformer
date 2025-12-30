"""Custom samplers for dynamic batching based on element count."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from torch.utils.data import Sampler


class _DatasetProtocol(Protocol):
    """Protocol for datasets that can be used with compute_element_counts."""

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict[str, Any]: ...


class DynamicBatchSampler(Sampler[list[int]]):
    """Batch sampler that groups samples by element count for efficient GPU usage.

    Samples with similar element counts are batched together, with batch size
    determined by the element count to maintain consistent memory usage.
    """

    # Batch size thresholds based on element count
    DEFAULT_THRESHOLDS = [
        (500, 32),
        (1000, 16),
        (2000, 8),
        (4000, 4),
        (8000, 2),
        (float("inf"), 1),
    ]

    def __init__(
        self,
        element_counts: list[int],
        thresholds: list[tuple[int, int]] | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """Initialize the sampler.

        Args:
            element_counts: List of element counts for each sample in the dataset.
            thresholds: List of (max_elements, batch_size) tuples.
            shuffle: Whether to shuffle batches each epoch.
            seed: Random seed for shuffling.
        """
        self.element_counts = element_counts
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        # Pre-compute batches
        self._create_batches()

    def _get_batch_size(self, num_elements: int) -> int:
        """Get batch size for a given element count."""
        for max_elem, batch_size in self.thresholds:
            if num_elements <= max_elem:
                return batch_size
        return 1

    def _create_batches(self) -> None:
        """Create batches grouped by element count ranges."""
        # Group samples by their batch size category
        groups: dict[int, list[int]] = defaultdict(list)

        for idx, count in enumerate(self.element_counts):
            batch_size = self._get_batch_size(count)
            groups[batch_size].append(idx)

        # Create batches within each group
        self.batches: list[list[int]] = []

        for batch_size, indices in groups.items():
            # Shuffle within group
            if self.shuffle:
                indices = list(indices)
                self.rng.shuffle(indices)

            # Create batches of the appropriate size
            for i in range(0, len(indices), batch_size):
                batch = indices[i : i + batch_size]
                self.batches.append(batch)

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over batches."""
        if self.shuffle:
            # Shuffle batch order each epoch
            batch_order = list(range(len(self.batches)))
            self.rng.shuffle(batch_order)
            for idx in batch_order:
                yield self.batches[idx]
        else:
            yield from self.batches

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.batches)


def compute_element_counts(
    dataset: _DatasetProtocol,
    cache_dir: Path | str | None = None,
) -> list[int]:
    """Compute element counts for all samples in a dataset.

    Args:
        dataset: Dataset with svg_files attribute or __getitem__ returning num_elements.
        cache_dir: Directory to cache element counts. If None, uses dataset's data_dir.

    Returns:
        List of element counts.
    """
    from src.data.preprocess import parse_svg_file

    # Determine cache path
    cache_path: Path | None = None
    if cache_dir is None and hasattr(dataset, "data_dir"):
        cache_dir = dataset.data_dir
    if cache_dir is not None:
        cache_dir = Path(cache_dir)

    # Generate cache key from file list
    if hasattr(dataset, "svg_files"):
        svg_files: list[Path] = dataset.svg_files
        file_list_str = ",".join(str(f) for f in sorted(svg_files))
        cache_key = hashlib.md5(file_list_str.encode()).hexdigest()[:12]
        cache_path = cache_dir / f"element_counts_{cache_key}.json" if cache_dir else None

    # Try to load from cache
    if cache_path and cache_path.exists():
        print(f"Loading element counts from cache: {cache_path}")
        with open(cache_path) as f:
            counts: list[int] = json.load(f)
        if len(counts) == len(dataset):
            return counts
        print("Cache size mismatch, recomputing...")

    counts = []

    # If dataset has svg_files, use that for faster counting
    if hasattr(dataset, "svg_files"):
        from tqdm import tqdm

        svg_files = dataset.svg_files
        print("Computing element counts...")
        for svg_path in tqdm(svg_files, desc="Counting elements"):
            primitives = parse_svg_file(svg_path)
            counts.append(len(primitives))
    else:
        # Fall back to loading samples
        for i in range(len(dataset)):
            sample = dataset[i]
            counts.append(int(sample["num_elements"].item()))

    # Save to cache
    if cache_path:
        print(f"Saving element counts to cache: {cache_path}")
        with open(cache_path, "w") as f:
            json.dump(counts, f)

    return counts
