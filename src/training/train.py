"""Training loop for panoptic symbol spotting."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import PanopticLoss
from .metrics import MetricsAccumulator, cluster_instances


class Trainer:
    """Trainer for panoptic symbol spotting model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]] | None = None,
        num_classes: int = 30,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        num_epochs: int = 100,
        device: str | torch.device = "cuda",
        checkpoint_dir: Path | str | None = None,
        log_interval: int = 10,
        use_amp: bool = True,
        compile_model: bool = True,
    ):
        """Initialize trainer.

        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_classes: Number of semantic classes.
            learning_rate: Learning rate.
            weight_decay: Weight decay for AdamW.
            num_epochs: Number of training epochs.
            device: Device to train on.
            checkpoint_dir: Directory to save checkpoints.
            log_interval: Interval for logging.
            use_amp: Whether to use automatic mixed precision.
            compile_model: Whether to compile the model with torch.compile.
        """
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()

        # Optionally compile model for faster execution (PyTorch 2.0+)
        if compile_model and hasattr(torch, "compile"):
            print("Compiling model with torch.compile...")
            model = torch.compile(model)  # type: ignore[assignment]

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.log_interval = log_interval

        self.checkpoint_dir: Path | None = None
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.criterion = PanopticLoss(num_classes=num_classes)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
        )

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Best validation metric for checkpointing
        self.best_val_pq = 0.0

        if self.use_amp:
            print("Using automatic mixed precision (AMP)")

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        total_loss = 0.0
        total_semantic_loss = 0.0
        total_instance_loss = 0.0
        num_batches = 0

        total_correct = 0
        total_labeled = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.num_epochs}",
            dynamic_ncols=True,
        )

        for batch in pbar:
            # Move to device
            features = batch["features"].to(self.device)
            semantic_labels = batch["semantic_labels"].to(self.device)
            instance_ids = batch["instance_ids"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Forward pass with AMP
            self.optimizer.zero_grad()

            with autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self.model(features, mask)

                # Compute loss
                targets = {
                    "semantic_labels": semantic_labels,
                    "instance_ids": instance_ids,
                    "mask": mask,
                }
                losses = self.criterion(outputs, targets)

            # Backward pass with scaled gradients
            self.scaler.scale(losses["loss"]).backward()

            # Gradient clipping (unscale first for proper clipping)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate losses
            total_loss += losses["loss"].item()
            total_semantic_loss += losses["semantic_loss"].item()
            total_instance_loss += losses["instance_loss"].item()
            num_batches += 1

            # Compute accuracy
            pred = outputs["semantic_logits"].argmax(dim=-1)
            valid = semantic_labels >= 0
            total_correct += (pred[valid] == semantic_labels[valid]).sum().item()
            total_labeled += valid.sum().item()

            # Update progress bar with running averages
            avg_loss = total_loss / num_batches
            acc = total_correct / total_labeled if total_labeled > 0 else 0
            pbar.set_postfix(
                loss=f"{avg_loss:.3f}",
                acc=f"{acc:.1%}",
                sem=f"{losses['semantic_loss'].item():.3f}",
            )

        return {
            "train_loss": total_loss / num_batches,
            "train_semantic_loss": total_semantic_loss / num_batches,
            "train_instance_loss": total_instance_loss / num_batches,
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Validate the model.

        Returns:
            Dictionary with validation metrics.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        metrics_acc = MetricsAccumulator(self.num_classes)

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move to device
            features = batch["features"].to(self.device)
            semantic_labels = batch["semantic_labels"].to(self.device)
            instance_ids = batch["instance_ids"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Forward pass with AMP
            with autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self.model(features, mask)

                # Compute loss
                targets = {
                    "semantic_labels": semantic_labels,
                    "instance_ids": instance_ids,
                    "mask": mask,
                }
                losses = self.criterion(outputs, targets)
            total_loss += losses["loss"].item()
            num_batches += 1

            # Get predictions
            pred_semantic = outputs["semantic_logits"].argmax(dim=-1).cpu().numpy()
            instance_embeds = outputs["instance_embeds"].cpu().numpy()

            # Cluster instances for each sample
            batch_size = features.size(0)
            pred_instances = np.zeros_like(pred_semantic)

            for b in range(batch_size):
                valid = mask[b].cpu().numpy()
                embeds = instance_embeds[b][valid]
                if len(embeds) > 0:
                    inst_ids = cluster_instances(embeds)
                    pred_instances[b][valid] = inst_ids

            # Update metrics
            metrics_acc.update(
                pred_semantic,
                pred_instances,
                semantic_labels.cpu().numpy(),
                instance_ids.cpu().numpy(),
                mask.cpu().numpy(),
            )

        # Compute metrics
        metrics = metrics_acc.compute()
        metrics["val_loss"] = total_loss / num_batches

        return metrics

    def train(self) -> dict[str, list[float]]:
        """Run full training.

        Returns:
            Dictionary with training history.
        """
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_pq": [],
            "val_accuracy": [],
        }

        for epoch in range(self.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            history["train_loss"].append(train_metrics["train_loss"])

            # Validate
            val_metrics = self.validate()
            if val_metrics:
                history["val_loss"].append(val_metrics["val_loss"])
                history["val_pq"].append(val_metrics["pq"])
                history["val_accuracy"].append(val_metrics["accuracy"])

                # Log
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs} - "
                    f"Train Loss: {train_metrics['train_loss']:.4f} - "
                    f"Val Loss: {val_metrics['val_loss']:.4f} - "
                    f"Val PQ: {val_metrics['pq']:.4f} - "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

                # Save best model
                if val_metrics["pq"] > self.best_val_pq:
                    self.best_val_pq = val_metrics["pq"]
                    if self.checkpoint_dir:
                        self.save_checkpoint(
                            self.checkpoint_dir / "best_model.pt",
                            epoch,
                            val_metrics,
                        )
            else:
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs} - "
                    f"Train Loss: {train_metrics['train_loss']:.4f}"
                )

            # Update learning rate
            self.scheduler.step()

            # Save periodic checkpoint
            if self.checkpoint_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                    epoch,
                    val_metrics if val_metrics else {},
                )

        return history

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Save a checkpoint.

        Args:
            path: Path to save checkpoint.
            epoch: Current epoch.
            metrics: Current metrics.
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metrics": metrics,
                "best_val_pq": self.best_val_pq,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        """Load a checkpoint.

        Args:
            path: Path to checkpoint.

        Returns:
            Checkpoint dictionary.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_pq = checkpoint.get("best_val_pq", 0.0)
        return checkpoint
