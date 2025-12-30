"""PyTorch Dataset for FloorPlanCAD."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import CompositeFeatureExtractor, FeatureNormalizer
from .preprocess import normalize_coordinates, parse_svg_file


class FloorPlanCADDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for FloorPlanCAD floor plans.

    Each sample contains:
        - features: (N, feature_dim) tensor of element features
        - semantic_labels: (N,) tensor of semantic class labels
        - instance_ids: (N,) tensor of instance IDs
        - num_elements: number of elements in this sample
    """

    # FloorPlanCAD semantic classes (30 categories)
    SEMANTIC_CLASSES = [
        "wall",
        "door",
        "window",
        "blind_window",
        "railing",
        "stairs",
        "elevator",
        "column",
        "curtain_wall",
        "parking",
        "room",
        "balcony",
        "corridor",
        "bathroom",
        "kitchen",
        "bedroom",
        "living_room",
        "storage",
        "equipment",
        "furniture",
        "sink",
        "toilet",
        "bathtub",
        "table",
        "chair",
        "bed",
        "sofa",
        "cabinet",
        "appliance",
        "other",
    ]

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        feature_config: dict[str, Any] | None = None,
        normalize_coords: bool = True,
        normalize_features: bool = True,
        max_elements: int | None = None,
    ):
        """Initialize dataset.

        Args:
            data_dir: Root directory containing the dataset.
            split: Dataset split ('train', 'val', 'test').
            feature_config: Configuration for feature extraction.
            normalize_coords: Whether to normalize coordinates to [0, 1].
            normalize_features: Whether to normalize features.
            max_elements: Maximum number of elements per sample (for memory).
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.normalize_coords = normalize_coords
        self.normalize_features = normalize_features
        self.max_elements = max_elements

        # Initialize feature extractor
        if feature_config is None:
            feature_config = {}
        self.feature_extractor = CompositeFeatureExtractor.from_config(feature_config)

        # Initialize normalizer (will be fitted on training data)
        self.normalizer = FeatureNormalizer() if normalize_features else None

        # Load file list
        self.svg_files = self._load_file_list()

        # Cache for parsed data
        self._cache: dict[int, dict[str, Any]] = {}

    def _load_file_list(self) -> list[Path]:
        """Load list of SVG files for this split."""
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            # Try flat structure
            svg_files = list(self.data_dir.glob("*.svg"))
        else:
            svg_files = list(split_dir.glob("**/*.svg"))

        return sorted(svg_files)

    def __len__(self) -> int:
        return len(self.svg_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx in self._cache:
            return self._cache[idx]

        svg_path = self.svg_files[idx]
        primitives = parse_svg_file(svg_path)

        # Normalize coordinates
        if self.normalize_coords:
            primitives = normalize_coordinates(primitives)

        # Limit elements if specified
        if self.max_elements and len(primitives) > self.max_elements:
            primitives = primitives[: self.max_elements]

        # Extract features
        features = self.feature_extractor.extract_batch(primitives)

        # Normalize features
        if self.normalizer and self.normalizer.fitted:
            features = self.normalizer.transform(features)

        # Extract labels
        semantic_labels = np.array(
            [p.semantic_label if p.semantic_label is not None else -1 for p in primitives],
            dtype=np.int64,
        )
        instance_ids = np.array(
            [p.instance_id if p.instance_id is not None else -1 for p in primitives],
            dtype=np.int64,
        )

        sample = {
            "features": torch.from_numpy(features),
            "semantic_labels": torch.from_numpy(semantic_labels),
            "instance_ids": torch.from_numpy(instance_ids),
            "num_elements": torch.tensor(len(primitives)),
        }

        return sample

    def fit_normalizer(self, num_samples: int = 1000) -> None:
        """Fit the feature normalizer on training data.

        Args:
            num_samples: Number of samples to use for fitting.
        """
        if self.normalizer is None:
            return

        all_features = []
        indices = np.random.choice(len(self), min(num_samples, len(self)), replace=False)

        for idx in indices:
            svg_path = self.svg_files[idx]
            primitives = parse_svg_file(svg_path)
            if self.normalize_coords:
                primitives = normalize_coordinates(primitives)
            if primitives:
                features = self.feature_extractor.extract_batch(primitives)
                all_features.append(features)

        if all_features:
            all_features_array = np.concatenate(all_features, axis=0)
            self.normalizer.fit(all_features_array)

    @property
    def feature_dim(self) -> int:
        """Dimension of element features."""
        return self.feature_extractor.feature_dim

    @property
    def num_classes(self) -> int:
        """Number of semantic classes."""
        return len(self.SEMANTIC_CLASSES)


def collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Custom collate function for variable-length sets.

    Pads sequences to the maximum length in the batch.

    Args:
        batch: List of samples from the dataset.

    Returns:
        Batched and padded tensors with a mask.
    """
    max_len = int(max(sample["num_elements"].item() for sample in batch))
    batch_size = len(batch)
    feature_dim = batch[0]["features"].shape[-1]

    # Initialize padded tensors
    features = torch.zeros(batch_size, max_len, feature_dim)
    semantic_labels = torch.full((batch_size, max_len), -1, dtype=torch.long)
    instance_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, sample in enumerate(batch):
        n = int(sample["num_elements"].item())
        features[i, :n] = sample["features"]
        semantic_labels[i, :n] = sample["semantic_labels"]
        instance_ids[i, :n] = sample["instance_ids"]
        mask[i, :n] = True

    return {
        "features": features,
        "semantic_labels": semantic_labels,
        "instance_ids": instance_ids,
        "mask": mask,
        "num_elements": torch.tensor([s["num_elements"].item() for s in batch]),
    }


class SyntheticFloorPlanDataset(Dataset[dict[str, torch.Tensor]]):
    """Synthetic dataset for testing without real data.

    Generates random floor plan-like data for debugging and testing.
    """

    def __init__(
        self,
        num_samples: int = 100,
        min_elements: int = 50,
        max_elements: int = 200,
        num_classes: int = 30,
        feature_dim: int = 12,
        seed: int = 42,
    ):
        """Initialize synthetic dataset.

        Args:
            num_samples: Number of samples to generate.
            min_elements: Minimum elements per sample.
            max_elements: Maximum elements per sample.
            num_classes: Number of semantic classes.
            feature_dim: Dimension of features.
            seed: Random seed for reproducibility.
        """
        self.num_samples = num_samples
        self.min_elements = min_elements
        self.max_elements = max_elements
        self.num_classes = num_classes
        self._feature_dim = feature_dim
        self.rng = np.random.default_rng(seed)

        # Pre-generate data
        self._data = self._generate_data()

    def _generate_data(self) -> list[dict[str, torch.Tensor]]:
        """Generate synthetic samples."""
        data = []
        for _ in range(self.num_samples):
            n_elements = self.rng.integers(self.min_elements, self.max_elements + 1)

            # Generate features (coordinates + type encoding)
            features = self.rng.random((n_elements, self._feature_dim)).astype(np.float32)

            # Generate labels (clustered to simulate instances)
            n_instances = self.rng.integers(5, 20)
            instance_ids = self.rng.integers(0, n_instances, size=n_elements)

            # Semantic labels based on instances (elements in same instance have same class)
            instance_to_class = self.rng.integers(0, self.num_classes, size=n_instances)
            semantic_labels = instance_to_class[instance_ids]

            data.append(
                {
                    "features": torch.from_numpy(features),
                    "semantic_labels": torch.from_numpy(semantic_labels),
                    "instance_ids": torch.from_numpy(instance_ids),
                    "num_elements": torch.tensor(n_elements),
                }
            )

        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._data[idx]

    @property
    def feature_dim(self) -> int:
        return self._feature_dim
