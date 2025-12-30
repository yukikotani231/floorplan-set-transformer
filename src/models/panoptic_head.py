"""Panoptic prediction head for CAD element classification and instance segmentation."""

import torch.nn as nn
from torch import Tensor

from .set_transformer import SetTransformerEncoder


class PanopticHead(nn.Module):
    """Prediction head for panoptic symbol spotting.

    Outputs:
        - Semantic class logits for each element
        - Instance embeddings for clustering
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        instance_embed_dim: int = 32,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        """Initialize panoptic head.

        Args:
            d_model: Input dimension from encoder.
            num_classes: Number of semantic classes.
            instance_embed_dim: Dimension of instance embeddings.
            hidden_dim: Hidden layer dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = d_model

        # Semantic classification head
        self.semantic_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Instance embedding head
        self.instance_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, instance_embed_dim),
        )

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            X: Encoded features of shape (batch, N, d_model).

        Returns:
            Tuple of:
                - semantic_logits: (batch, N, num_classes)
                - instance_embeds: (batch, N, instance_embed_dim)
        """
        semantic_logits = self.semantic_head(X)
        instance_embeds = self.instance_head(X)

        # Normalize instance embeddings
        instance_embeds = nn.functional.normalize(instance_embeds, p=2, dim=-1)

        return semantic_logits, instance_embeds


class PanopticSetTransformer(nn.Module):
    """Complete model for panoptic symbol spotting using Set Transformer.

    Architecture:
        Input features -> Set Transformer Encoder -> Panoptic Head
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        instance_embed_dim: int = 32,
        d_ff: int | None = None,
        dropout: float = 0.0,
        use_isab: bool = True,
        num_inducing_points: int = 64,
    ):
        """Initialize model.

        Args:
            d_input: Input feature dimension.
            d_model: Model dimension.
            num_heads: Number of attention heads.
            num_layers: Number of encoder layers.
            num_classes: Number of semantic classes.
            instance_embed_dim: Dimension of instance embeddings.
            d_ff: Feed-forward dimension.
            dropout: Dropout probability.
            use_isab: Whether to use ISAB.
            num_inducing_points: Number of inducing points for ISAB.
        """
        super().__init__()

        self.encoder = SetTransformerEncoder(
            d_input=d_input,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_isab=use_isab,
            num_inducing_points=num_inducing_points,
        )

        self.panoptic_head = PanopticHead(
            d_model=d_model,
            num_classes=num_classes,
            instance_embed_dim=instance_embed_dim,
            dropout=dropout,
        )

        self.num_classes = num_classes
        self.instance_embed_dim = instance_embed_dim

    def forward(
        self,
        X: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            X: Input features of shape (batch, N, d_input).
            mask: Optional mask of shape (batch, N) where True = valid.

        Returns:
            Dictionary with:
                - semantic_logits: (batch, N, num_classes)
                - instance_embeds: (batch, N, instance_embed_dim)
                - encoded: (batch, N, d_model)
        """
        encoded = self.encoder(X, mask)
        semantic_logits, instance_embeds = self.panoptic_head(encoded)

        return {
            "semantic_logits": semantic_logits,
            "instance_embeds": instance_embeds,
            "encoded": encoded,
        }

    def predict(
        self,
        X: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Predict semantic classes and instance IDs.

        Args:
            X: Input features of shape (batch, N, d_input).
            mask: Optional mask of shape (batch, N).

        Returns:
            Dictionary with:
                - semantic_preds: (batch, N) predicted class indices
                - instance_embeds: (batch, N, instance_embed_dim)
        """
        outputs = self.forward(X, mask)

        semantic_preds = outputs["semantic_logits"].argmax(dim=-1)

        return {
            "semantic_preds": semantic_preds,
            "instance_embeds": outputs["instance_embeds"],
        }


def create_panoptic_model(
    feature_dim: int,
    num_classes: int = 30,
    d_model: int = 256,
    num_heads: int = 8,
    num_layers: int = 4,
    instance_embed_dim: int = 32,
    dropout: float = 0.1,
    use_isab: bool = True,
    num_inducing_points: int = 64,
) -> PanopticSetTransformer:
    """Factory function to create a panoptic model with default settings.

    Args:
        feature_dim: Input feature dimension.
        num_classes: Number of semantic classes.
        d_model: Model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of encoder layers.
        instance_embed_dim: Instance embedding dimension.
        dropout: Dropout probability.
        use_isab: Whether to use ISAB.
        num_inducing_points: Number of inducing points for ISAB.

    Returns:
        PanopticSetTransformer model.
    """
    return PanopticSetTransformer(
        d_input=feature_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        instance_embed_dim=instance_embed_dim,
        d_ff=d_model * 4,
        dropout=dropout,
        use_isab=use_isab,
        num_inducing_points=num_inducing_points,
    )
