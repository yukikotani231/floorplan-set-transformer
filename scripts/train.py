#!/usr/bin/env python
"""Training script for FloorPlan Set Transformer."""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data import (
    DynamicBatchSampler,
    FloorPlanCADDataset,
    SyntheticFloorPlanDataset,
    collate_fn,
    compute_element_counts,
)
from src.models import create_panoptic_model
from src.training import Trainer


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FloorPlan Set Transformer")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for testing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command line arguments
    if args.data_dir:
        config["data"]["data_dir"] = str(args.data_dir)
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs

    print(f"Using device: {args.device}")

    # Create datasets
    # Type hints for datasets
    train_dataset: torch.utils.data.Dataset[dict[str, torch.Tensor]]
    val_dataset: torch.utils.data.Dataset[dict[str, torch.Tensor]]

    if args.synthetic:
        print("Using synthetic dataset for testing")
        feature_dim = 12  # Default synthetic feature dim
        train_dataset = SyntheticFloorPlanDataset(
            num_samples=500,
            min_elements=50,
            max_elements=200,
            num_classes=30,
            feature_dim=feature_dim,
            seed=42,
        )
        val_dataset = SyntheticFloorPlanDataset(
            num_samples=100,
            min_elements=50,
            max_elements=200,
            num_classes=30,
            feature_dim=feature_dim,
            seed=123,
        )
    else:
        from torch.utils.data import Subset

        data_dir = Path(config["data"]["data_dir"])
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Data directory {data_dir} not found. "
                "Use --synthetic flag to test with synthetic data."
            )

        feature_config = config.get("features", {})

        # Load full dataset WITHOUT max_elements limit
        full_dataset = FloorPlanCADDataset(
            data_dir=data_dir,
            split="train",  # Will use flat structure if no train subdir
            feature_config=feature_config,
            normalize_coords=config["data"]["normalize_coords"],
            normalize_features=config["data"]["normalize_features"],
            max_elements=None,  # No truncation - use dynamic batching
        )

        # Compute element counts for dynamic batching
        element_counts = compute_element_counts(full_dataset)
        print(
            f"Element counts: min={min(element_counts)}, max={max(element_counts)}, "
            f"mean={sum(element_counts) / len(element_counts):.0f}"
        )

        # Fit normalizer on full data
        print("Fitting feature normalizer...")
        full_dataset.fit_normalizer()

        # Split into train/val
        train_size = int(0.9 * len(full_dataset))

        # Create index splits
        indices = list(range(len(full_dataset)))
        rng = torch.Generator().manual_seed(42)
        perm = torch.randperm(len(indices), generator=rng).tolist()
        train_indices = perm[:train_size]
        val_indices = perm[train_size:]

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        # Get element counts for train/val splits
        train_counts = [element_counts[i] for i in train_indices]
        val_counts = [element_counts[i] for i in val_indices]

        feature_dim = full_dataset.feature_dim

    # Create data loaders
    num_workers = config["training"]["num_workers"]
    # Use persistent workers for faster epoch transitions
    persistent = num_workers > 0

    if args.synthetic:
        # Fixed batch size for synthetic data
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=persistent,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=persistent,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    else:
        # Dynamic batching for real data
        train_sampler = DynamicBatchSampler(
            train_counts,
            shuffle=True,
            seed=42,
        )
        val_sampler = DynamicBatchSampler(
            val_counts,
            shuffle=False,
            seed=42,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=persistent,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=persistent,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Feature dimension: {feature_dim}")

    # Create model
    model_config = config["model"]
    num_classes = config["model"].get("num_classes", 36)
    model = create_panoptic_model(
        feature_dim=feature_dim,
        num_classes=num_classes,
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        instance_embed_dim=model_config["instance_embed_dim"],
        dropout=model_config["dropout"],
        use_isab=model_config["use_isab"],
        num_inducing_points=model_config["num_inducing_points"],
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        num_epochs=config["training"]["num_epochs"],
        device=args.device,
        checkpoint_dir=checkpoint_dir,
        log_interval=config["training"]["log_interval"],
        use_amp=not args.no_amp,
        compile_model=not args.no_compile,
    )

    # Train
    print("Starting training...")
    trainer.train()

    print("Training complete!")
    print(f"Best validation PQ: {trainer.best_val_pq:.4f}")


if __name__ == "__main__":
    main()
