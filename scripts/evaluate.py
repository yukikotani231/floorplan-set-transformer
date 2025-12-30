#!/usr/bin/env python
"""Evaluation script for FloorPlan Set Transformer."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import FloorPlanCADDataset, collate_fn
from src.data.preprocess import normalize_coordinates, parse_svg_file
from src.models import create_panoptic_model
from src.training.metrics import cluster_instances
from src.utils.visualization import (
    get_class_name,
    plot_confusion_matrix,
    plot_floor_plan,
    save_figure,
)


def load_model(
    checkpoint_path: Path,
    config: dict,
    device: torch.device,
) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    model = create_panoptic_model(
        feature_dim=config["feature_dim"],
        num_classes=config.get("num_classes", 36),
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        instance_embed_dim=config["model"]["instance_embed_dim"],
        dropout=config["model"]["dropout"],
        use_isab=config["model"]["use_isab"],
        num_inducing_points=config["model"]["num_inducing_points"],
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def evaluate_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 36,
) -> dict:
    """Evaluate model on a dataset.

    Returns:
        Dictionary with evaluation metrics.
    """
    all_pred_semantic = []
    all_true_semantic = []
    all_pred_instances = []
    all_true_instances = []
    all_masks = []

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            features = batch["features"].to(device)
            semantic_labels = batch["semantic_labels"]
            instance_ids = batch["instance_ids"]
            mask = batch["mask"]

            # Forward pass
            outputs = model(features, mask.to(device))

            # Get predictions
            pred_semantic = outputs["semantic_logits"].argmax(dim=-1).cpu().numpy()
            instance_embeds = outputs["instance_embeds"].cpu().numpy()

            # Cluster instances
            batch_size = features.size(0)
            pred_instances = np.zeros_like(pred_semantic)

            for b in range(batch_size):
                valid = mask[b].numpy()
                embeds = instance_embeds[b][valid]
                if len(embeds) > 0:
                    pred_instances[b][valid] = cluster_instances(embeds)

            # Accumulate
            for b in range(batch_size):
                valid = mask[b].numpy()
                pred_sem = pred_semantic[b][valid]
                true_sem = semantic_labels[b].numpy()[valid]
                pred_inst = pred_instances[b][valid]
                true_inst = instance_ids[b].numpy()[valid]

                all_pred_semantic.append(pred_sem)
                all_true_semantic.append(true_sem)
                all_pred_instances.append(pred_inst)
                all_true_instances.append(true_inst)
                all_masks.append(valid)

                # Update confusion matrix (only for labeled elements)
                labeled = true_sem >= 0
                for pred, true in zip(pred_sem[labeled], true_sem[labeled], strict=True):
                    if 0 <= pred < num_classes and 0 <= true < num_classes:
                        confusion_matrix[true, pred] += 1

    # Compute overall metrics
    all_pred_sem = np.concatenate(all_pred_semantic)
    all_true_sem = np.concatenate(all_true_semantic)

    # Only evaluate on labeled elements
    labeled = all_true_sem >= 0
    accuracy = (all_pred_sem[labeled] == all_true_sem[labeled]).mean()

    # Per-class metrics
    per_class_metrics = {}
    for class_id in range(num_classes):
        true_mask = all_true_sem == class_id
        pred_mask = all_pred_sem == class_id

        tp = (true_mask & pred_mask).sum()
        fp = (~true_mask & pred_mask).sum()
        fn = (true_mask & ~pred_mask).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if true_mask.sum() > 0:  # Only include classes that appear in ground truth
            per_class_metrics[class_id] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(true_mask.sum()),
            }

    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix,
        "per_class_metrics": per_class_metrics,
        "num_samples": len(all_pred_semantic),
    }


def visualize_predictions(
    model: torch.nn.Module,
    dataset: FloorPlanCADDataset,
    device: torch.device,
    output_dir: Path,
    num_samples: int = 10,
) -> None:
    """Generate visualization for sample predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in tqdm(indices, desc="Visualizing"):
        # Get sample
        sample = dataset[int(idx)]
        features = sample["features"].unsqueeze(0).to(device)
        mask = torch.ones(1, features.size(1), dtype=torch.bool)

        # Get predictions
        with torch.no_grad():
            outputs = model(features, mask.to(device))

        pred_semantic = outputs["semantic_logits"].argmax(dim=-1).cpu().numpy()[0]
        true_semantic = sample["semantic_labels"].numpy()

        # Load original primitives for visualization
        svg_path = dataset.svg_files[int(idx)]
        primitives = parse_svg_file(svg_path)
        if dataset.normalize_coords:
            primitives = normalize_coordinates(primitives)

        # Limit to max_elements if needed
        if dataset.max_elements and len(primitives) > dataset.max_elements:
            primitives = primitives[: dataset.max_elements]

        # Create visualization
        fig = plot_floor_plan(
            primitives,
            predictions=pred_semantic,
            ground_truth=true_semantic,
            title=f"Sample {idx}: {svg_path.stem}",
        )

        save_figure(fig, output_dir / f"pred_{idx:04d}.png")


def print_metrics(metrics: dict) -> None:
    """Print evaluation metrics in a formatted way."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Accuracy: {metrics['accuracy']:.2%}")
    print(f"Number of Samples: {metrics['num_samples']}")

    print("\n" + "-" * 60)
    print("Per-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)

    for class_id, class_metrics in sorted(metrics["per_class_metrics"].items()):
        class_name = get_class_name(class_id)[:20]
        print(
            f"{class_name:<20} "
            f"{class_metrics['precision']:>10.2%} "
            f"{class_metrics['recall']:>10.2%} "
            f"{class_metrics['f1']:>10.2%} "
            f"{class_metrics['support']:>10}"
        )

    # Macro averages
    if metrics["per_class_metrics"]:
        avg_precision = np.mean([m["precision"] for m in metrics["per_class_metrics"].values()])
        avg_recall = np.mean([m["recall"] for m in metrics["per_class_metrics"].values()])
        avg_f1 = np.mean([m["f1"] for m in metrics["per_class_metrics"].values()])
        print("-" * 60)
        print(f"{'Macro Average':<20} {avg_precision:>10.2%} {avg_recall:>10.2%} {avg_f1:>10.2%}")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FloorPlan Set Transformer")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/floorplancad"),
        help="Data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=10,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Skip visualization",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create dataset
    dataset = FloorPlanCADDataset(
        data_dir=args.data_dir,
        split="train",  # Uses flat structure
        feature_config=config.get("features", {}),
        normalize_coords=config["data"]["normalize_coords"],
        normalize_features=False,  # Don't normalize for evaluation
        max_elements=config["data"].get("max_elements", 500),
    )

    # Update config with feature_dim
    config["feature_dim"] = dataset.feature_dim

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    print(f"Dataset: {len(dataset)} samples")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Evaluate
    print("\nRunning evaluation...")
    metrics = evaluate_dataset(
        model,
        dataloader,
        device,
        num_classes=config["model"].get("num_classes", 36),
    )

    # Print results
    print_metrics(metrics)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix
    used_classes = list(metrics["per_class_metrics"].keys())
    if used_classes:
        cm_subset = metrics["confusion_matrix"][np.ix_(used_classes, used_classes)]
        fig = plot_confusion_matrix(cm_subset, class_ids=used_classes)
        save_figure(fig, args.output_dir / "confusion_matrix.png")
        print(f"\nConfusion matrix saved to {args.output_dir / 'confusion_matrix.png'}")

    # Visualize predictions
    if not args.no_vis:
        print(f"\nGenerating {args.num_vis} visualizations...")
        visualize_predictions(
            model,
            dataset,
            device,
            args.output_dir / "visualizations",
            num_samples=args.num_vis,
        )
        print(f"Visualizations saved to {args.output_dir / 'visualizations'}")


if __name__ == "__main__":
    main()
