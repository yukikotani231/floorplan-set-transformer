"""Feature Extractor for CAD elements.

This module provides a flexible and extensible feature extraction system
for CAD primitives (lines, arcs, curves). Features can be easily added,
removed, or modified through configuration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class CADPrimitive:
    """Represents a single CAD primitive element."""

    primitive_type: str  # "line", "arc", "curve"
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    control_points: list[tuple[float, float]] | None = None  # For curves
    center: tuple[float, float] | None = None  # For arcs
    radius: float | None = None  # For arcs
    semantic_label: int | None = None
    instance_id: int | None = None


class FeatureExtractorBase(ABC):
    """Base class for feature extractors."""

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the dimension of the extracted features."""
        ...

    @abstractmethod
    def extract(self, primitive: CADPrimitive) -> NDArray[np.float32]:
        """Extract features from a single primitive."""
        ...


class CoordinateFeatureExtractor(FeatureExtractorBase):
    """Extract start and end point coordinates."""

    @property
    def feature_dim(self) -> int:
        return 4  # start_x, start_y, end_x, end_y

    def extract(self, primitive: CADPrimitive) -> NDArray[np.float32]:
        return np.array(
            [
                primitive.start_point[0],
                primitive.start_point[1],
                primitive.end_point[0],
                primitive.end_point[1],
            ],
            dtype=np.float32,
        )


class CenterFeatureExtractor(FeatureExtractorBase):
    """Extract center position of the primitive."""

    @property
    def feature_dim(self) -> int:
        return 2  # center_x, center_y

    def extract(self, primitive: CADPrimitive) -> NDArray[np.float32]:
        cx = (primitive.start_point[0] + primitive.end_point[0]) / 2
        cy = (primitive.start_point[1] + primitive.end_point[1]) / 2
        return np.array([cx, cy], dtype=np.float32)


class LengthFeatureExtractor(FeatureExtractorBase):
    """Extract length of the primitive."""

    @property
    def feature_dim(self) -> int:
        return 1

    def extract(self, primitive: CADPrimitive) -> NDArray[np.float32]:
        dx = primitive.end_point[0] - primitive.start_point[0]
        dy = primitive.end_point[1] - primitive.start_point[1]
        length = np.sqrt(dx**2 + dy**2)
        return np.array([length], dtype=np.float32)


class AngleFeatureExtractor(FeatureExtractorBase):
    """Extract angle of the primitive (for lines)."""

    @property
    def feature_dim(self) -> int:
        return 2  # sin(angle), cos(angle)

    def extract(self, primitive: CADPrimitive) -> NDArray[np.float32]:
        dx = primitive.end_point[0] - primitive.start_point[0]
        dy = primitive.end_point[1] - primitive.start_point[1]
        length = np.sqrt(dx**2 + dy**2)
        if length < 1e-8:
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.array([dy / length, dx / length], dtype=np.float32)  # sin, cos


class TypeFeatureExtractor(FeatureExtractorBase):
    """Extract primitive type as one-hot encoding."""

    TYPES = ["line", "arc", "curve"]

    @property
    def feature_dim(self) -> int:
        return len(self.TYPES)

    def extract(self, primitive: CADPrimitive) -> NDArray[np.float32]:
        one_hot = np.zeros(len(self.TYPES), dtype=np.float32)
        if primitive.primitive_type in self.TYPES:
            idx = self.TYPES.index(primitive.primitive_type)
            one_hot[idx] = 1.0
        return one_hot


class CompositeFeatureExtractor:
    """Combines multiple feature extractors.

    This class allows flexible composition of different features.
    Features can be enabled/disabled via configuration.
    """

    AVAILABLE_EXTRACTORS: dict[str, type[FeatureExtractorBase]] = {
        "coordinates": CoordinateFeatureExtractor,
        "center": CenterFeatureExtractor,
        "length": LengthFeatureExtractor,
        "angle": AngleFeatureExtractor,
        "type": TypeFeatureExtractor,
    }

    def __init__(self, enabled_features: list[str] | None = None):
        """Initialize with specified features.

        Args:
            enabled_features: List of feature names to enable.
                If None, all features are enabled.
        """
        if enabled_features is None:
            enabled_features = list(self.AVAILABLE_EXTRACTORS.keys())

        self.extractors: list[FeatureExtractorBase] = []
        for name in enabled_features:
            if name not in self.AVAILABLE_EXTRACTORS:
                raise ValueError(f"Unknown feature: {name}")
            self.extractors.append(self.AVAILABLE_EXTRACTORS[name]())

        self._feature_dim = sum(e.feature_dim for e in self.extractors)

    @property
    def feature_dim(self) -> int:
        """Total dimension of all enabled features."""
        return self._feature_dim

    def extract(self, primitive: CADPrimitive) -> NDArray[np.float32]:
        """Extract all enabled features from a primitive."""
        features = [e.extract(primitive) for e in self.extractors]
        return np.concatenate(features)

    def extract_batch(self, primitives: list[CADPrimitive]) -> NDArray[np.float32]:
        """Extract features from multiple primitives.

        Args:
            primitives: List of CAD primitives.

        Returns:
            Array of shape (N, feature_dim) where N is the number of primitives.
        """
        return np.stack([self.extract(p) for p in primitives])

    def add_extractor(self, name: str, extractor: FeatureExtractorBase) -> None:
        """Add a custom feature extractor.

        Args:
            name: Name for the extractor (for reference).
            extractor: The feature extractor instance.
        """
        self.extractors.append(extractor)
        self._feature_dim += extractor.feature_dim

    @classmethod
    def register_extractor(cls, name: str, extractor_class: type[FeatureExtractorBase]) -> None:
        """Register a new feature extractor class.

        Args:
            name: Name to register the extractor under.
            extractor_class: The feature extractor class.
        """
        cls.AVAILABLE_EXTRACTORS[name] = extractor_class

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CompositeFeatureExtractor":
        """Create a feature extractor from configuration.

        Args:
            config: Configuration dict with 'enabled_features' key.

        Returns:
            CompositeFeatureExtractor instance.
        """
        enabled = config.get("enabled_features")
        return cls(enabled_features=enabled)


class FeatureNormalizer:
    """Normalize features to a standard range."""

    def __init__(self) -> None:
        self.mean: NDArray[np.float32] | None = None
        self.std: NDArray[np.float32] | None = None
        self.fitted = False

    def fit(self, features: NDArray[np.float32]) -> None:
        """Compute mean and std from training data.

        Args:
            features: Array of shape (N, feature_dim).
        """
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        self.std[self.std < 1e-8] = 1.0  # Avoid division by zero
        self.fitted = True

    def transform(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalize features.

        Args:
            features: Array of shape (N, feature_dim) or (feature_dim,).

        Returns:
            Normalized features.
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        assert self.mean is not None and self.std is not None
        return (features - self.mean) / self.std

    def fit_transform(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features)
