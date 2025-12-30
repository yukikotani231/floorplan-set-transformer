"""Loss functions for panoptic symbol spotting."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: float | Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -1,
    ):
        """Initialize focal loss.

        Args:
            alpha: Class weights. Can be a scalar or tensor of shape (num_classes,).
            gamma: Focusing parameter. Higher values focus more on hard examples.
            reduction: 'none', 'mean', or 'sum'.
            ignore_index: Label to ignore in loss computation.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            inputs: Logits of shape (batch, N, num_classes) or (batch*N, num_classes).
            targets: Labels of shape (batch, N) or (batch*N,).

        Returns:
            Focal loss.
        """
        # Flatten if needed
        if inputs.dim() == 3:
            batch_size, N, num_classes = inputs.shape
            inputs = inputs.view(-1, num_classes)
            targets = targets.view(-1)

        # Create mask for valid labels
        valid_mask = targets != self.ignore_index

        # Filter valid samples
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        if inputs.numel() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # Compute softmax probabilities
        p = F.softmax(inputs, dim=-1)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Get probability of true class
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, Tensor):
                alpha_t: Tensor | float = self.alpha.to(inputs.device)[targets]
            else:
                alpha_t = self.alpha
            focal_weight = alpha_t * focal_weight

        # Compute focal loss
        focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DiscriminativeLoss(nn.Module):
    """Discriminative Loss for instance embedding.

    Encourages embeddings of same instance to be close (pull)
    and embeddings of different instances to be far (push).

    L = L_var + L_dist + L_reg
    """

    def __init__(
        self,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.001,
    ):
        """Initialize discriminative loss.

        Args:
            delta_v: Margin for variance (pull) loss.
            delta_d: Margin for distance (push) loss.
            alpha: Weight for variance loss.
            beta: Weight for distance loss.
            gamma: Weight for regularization loss.
        """
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        embeddings: Tensor,
        instance_ids: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute discriminative loss.

        Args:
            embeddings: Instance embeddings of shape (batch, N, embed_dim).
            instance_ids: Instance labels of shape (batch, N).
            mask: Valid element mask of shape (batch, N).

        Returns:
            Dictionary with loss components.
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        total_var_loss = torch.tensor(0.0, device=device)
        total_dist_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)

        num_valid_samples = 0

        for b in range(batch_size):
            embed = embeddings[b]  # (N, embed_dim)
            inst_ids = instance_ids[b]  # (N,)

            if mask is not None:
                valid = mask[b]
                embed = embed[valid]
                inst_ids = inst_ids[valid]

            # Filter out invalid instance IDs
            valid_inst = inst_ids >= 0
            embed = embed[valid_inst]
            inst_ids = inst_ids[valid_inst]

            if embed.numel() == 0:
                continue

            unique_ids = torch.unique(inst_ids)
            num_instances = len(unique_ids)

            if num_instances < 2:
                continue

            num_valid_samples += 1

            # Compute instance means
            means_list: list[Tensor] = []
            for inst_id in unique_ids:
                inst_mask = inst_ids == inst_id
                inst_embed = embed[inst_mask]
                means_list.append(inst_embed.mean(dim=0))

            means = torch.stack(means_list)  # (num_instances, embed_dim)

            # Variance loss (pull)
            var_loss = torch.tensor(0.0, device=device)
            for i, inst_id in enumerate(unique_ids):
                inst_mask = inst_ids == inst_id
                inst_embed = embed[inst_mask]
                distances = torch.norm(inst_embed - means[i], dim=1)
                var_loss += torch.clamp(distances - self.delta_v, min=0).pow(2).mean()

            var_loss /= num_instances
            total_var_loss += var_loss

            # Distance loss (push)
            dist_loss = torch.tensor(0.0, device=device)
            for i in range(num_instances):
                for j in range(i + 1, num_instances):
                    dist = torch.norm(means[i] - means[j])
                    dist_loss += torch.clamp(2 * self.delta_d - dist, min=0).pow(2)

            num_pairs = num_instances * (num_instances - 1) / 2
            if num_pairs > 0:
                dist_loss /= num_pairs
            total_dist_loss += dist_loss

            # Regularization loss
            reg_loss = means.norm(dim=1).mean()
            total_reg_loss += reg_loss

        # Average over batch
        if num_valid_samples > 0:
            total_var_loss /= num_valid_samples
            total_dist_loss /= num_valid_samples
            total_reg_loss /= num_valid_samples

        total_loss = (
            self.alpha * total_var_loss + self.beta * total_dist_loss + self.gamma * total_reg_loss
        )

        return {
            "loss": total_loss,
            "var_loss": total_var_loss,
            "dist_loss": total_dist_loss,
            "reg_loss": total_reg_loss,
        }


class PanopticLoss(nn.Module):
    """Combined loss for panoptic symbol spotting.

    Combines semantic classification loss and instance embedding loss.
    """

    def __init__(
        self,
        num_classes: int,
        semantic_weight: float = 1.0,
        instance_weight: float = 1.0,
        focal_gamma: float = 2.0,
        class_weights: Tensor | None = None,
    ):
        """Initialize panoptic loss.

        Args:
            num_classes: Number of semantic classes.
            semantic_weight: Weight for semantic loss.
            instance_weight: Weight for instance loss.
            focal_gamma: Gamma for focal loss.
            class_weights: Optional class weights for handling imbalance.
        """
        super().__init__()

        self.semantic_weight = semantic_weight
        self.instance_weight = instance_weight

        self.semantic_loss = FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            ignore_index=-1,
        )

        self.instance_loss = DiscriminativeLoss()

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute panoptic loss.

        Args:
            outputs: Model outputs with 'semantic_logits' and 'instance_embeds'.
            targets: Ground truth with 'semantic_labels', 'instance_ids', and 'mask'.

        Returns:
            Dictionary with loss components.
        """
        # Semantic loss
        semantic_loss = self.semantic_loss(
            outputs["semantic_logits"],
            targets["semantic_labels"],
        )

        # Instance loss
        instance_losses = self.instance_loss(
            outputs["instance_embeds"],
            targets["instance_ids"],
            targets.get("mask"),
        )

        # Total loss
        total_loss = (
            self.semantic_weight * semantic_loss + self.instance_weight * instance_losses["loss"]
        )

        return {
            "loss": total_loss,
            "semantic_loss": semantic_loss,
            "instance_loss": instance_losses["loss"],
            "var_loss": instance_losses["var_loss"],
            "dist_loss": instance_losses["dist_loss"],
        }
