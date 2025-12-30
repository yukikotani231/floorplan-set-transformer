"""Visualization utilities for FloorPlanCAD predictions."""

import colorsys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# FloorPlanCAD semantic class names (based on dataset documentation)
# Labels are 1-indexed in the dataset
CLASS_NAMES: dict[int, str] = {
    1: "Wall",
    2: "Door",
    3: "Window",
    4: "Blind Window",
    5: "Railing",
    9: "Parking",
    10: "Room",
    11: "Balcony",
    12: "Corridor",
    13: "Bathroom",
    14: "Kitchen",
    15: "Bedroom",
    16: "Living Room",
    17: "Storage",
    18: "Equipment",
    19: "Furniture",
    20: "Sink",
    21: "Toilet",
    22: "Bathtub",
    23: "Table",
    24: "Chair",
    25: "Bed",
    26: "Sofa",
    27: "Cabinet",
    28: "Appliance",
    29: "Stairs",
    30: "Elevator",
    31: "Column",
    33: "Curtain Wall",
    34: "Text/Dimension",
    35: "Other",
}


def get_class_name(class_id: int) -> str:
    """Get human-readable class name for a class ID."""
    return CLASS_NAMES.get(class_id, f"Class {class_id}")


def generate_class_colors(num_classes: int = 36) -> dict[int, tuple[float, float, float]]:
    """Generate distinct colors for each class.

    Args:
        num_classes: Number of classes.

    Returns:
        Dictionary mapping class ID to RGB tuple (0-1 range).
    """
    colors = {}
    for i in range(num_classes):
        hue = i / num_classes
        saturation = 0.7 + 0.3 * (i % 2)  # Alternate saturation
        value = 0.8 + 0.2 * ((i // 2) % 2)  # Alternate value
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[i] = rgb
    return colors


CLASS_COLORS = generate_class_colors(36)


def plot_floor_plan(
    primitives: list[Any],
    predictions: np.ndarray | None = None,
    ground_truth: np.ndarray | None = None,
    instance_ids: np.ndarray | None = None,
    figsize: tuple[int, int] = (12, 12),
    title: str = "Floor Plan",
    show_legend: bool = True,
) -> plt.Figure:
    """Plot floor plan with optional predictions/ground truth overlay.

    Args:
        primitives: List of CADPrimitive objects.
        predictions: Predicted semantic labels (N,).
        ground_truth: Ground truth semantic labels (N,).
        instance_ids: Instance IDs for coloring (N,).
        figsize: Figure size.
        title: Plot title.
        show_legend: Whether to show class legend.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2 if ground_truth is not None else 1, figsize=figsize)

    if ground_truth is not None:
        ax_pred, ax_gt = axes
    else:
        ax_pred = axes if not isinstance(axes, np.ndarray) else axes[0]
        ax_gt = None

    # Plot predictions
    _plot_primitives(ax_pred, primitives, predictions, "Predictions")

    # Plot ground truth if available
    if ax_gt is not None and ground_truth is not None:
        _plot_primitives(ax_gt, primitives, ground_truth, "Ground Truth")

    # Add legend
    if show_legend:
        used_classes: set[int] = set()
        if predictions is not None:
            used_classes.update(int(c) for c in predictions[predictions >= 0])
        if ground_truth is not None:
            used_classes.update(int(c) for c in ground_truth[ground_truth >= 0])

        legend_patches = [
            Patch(color=CLASS_COLORS.get(c, (0.5, 0.5, 0.5)), label=get_class_name(c))
            for c in sorted(used_classes)
        ]
        fig.legend(
            handles=legend_patches,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            fontsize=8,
        )

    fig.suptitle(title)
    plt.tight_layout()

    return fig


def _plot_primitives(
    ax: plt.Axes,
    primitives: list[Any],
    labels: np.ndarray | None,
    title: str,
) -> None:
    """Plot primitives on an axis with color coding."""
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.invert_yaxis()  # SVG coordinates have Y increasing downward

    for i, prim in enumerate(primitives):
        label = labels[i] if labels is not None else -1
        color = CLASS_COLORS.get(int(label), (0.5, 0.5, 0.5)) if label >= 0 else (0.8, 0.8, 0.8)
        alpha = 0.8 if label >= 0 else 0.3

        if prim.primitive_type == "line":
            ax.plot(
                [prim.start_point[0], prim.end_point[0]],
                [prim.start_point[1], prim.end_point[1]],
                color=color,
                alpha=alpha,
                linewidth=1.5,
            )
        elif prim.primitive_type == "arc":
            # Simplified arc drawing as line for now
            ax.plot(
                [prim.start_point[0], prim.end_point[0]],
                [prim.start_point[1], prim.end_point[1]],
                color=color,
                alpha=alpha,
                linewidth=1.5,
            )
        elif prim.primitive_type == "curve":
            # Draw curve as line between endpoints
            ax.plot(
                [prim.start_point[0], prim.end_point[0]],
                [prim.start_point[1], prim.end_point[1]],
                color=color,
                alpha=alpha,
                linewidth=1.5,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_ids: list[int] | None = None,
    figsize: tuple[int, int] = (12, 10),
    normalize: bool = True,
) -> plt.Figure:
    """Plot confusion matrix.

    Args:
        confusion_matrix: Confusion matrix (num_classes x num_classes).
        class_ids: List of class IDs for labels.
        figsize: Figure size.
        normalize: Whether to normalize rows.

    Returns:
        Matplotlib figure.
    """
    if normalize:
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        confusion_matrix = confusion_matrix / row_sums

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(confusion_matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)

    if class_ids is not None:
        labels = [get_class_name(c) for c in class_ids]
        ax.set_xticks(range(len(class_ids)))
        ax.set_yticks(range(len(class_ids)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

    plt.tight_layout()
    return fig


def plot_training_history(
    history: dict[str, list[float]],
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Plot training history curves.

    Args:
        history: Dictionary with metric names and values per epoch.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    n_plots = len(history)
    fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0], figsize[1]))

    if n_plots == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, history.items(), strict=True):
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Path | str, dpi: int = 150) -> None:
    """Save figure to file.

    Args:
        fig: Matplotlib figure.
        path: Output path.
        dpi: Resolution.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
