"""Set Transformer model for CAD element processing."""

import torch.nn as nn
from torch import Tensor

from .modules import ISAB, PMA, SAB


class SetTransformerEncoder(nn.Module):
    """Set Transformer Encoder.

    Processes a set of elements using self-attention blocks.
    Can use either SAB (O(N^2)) or ISAB (O(NM)) for efficiency.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        use_isab: bool = True,
        num_inducing_points: int = 64,
    ):
        """Initialize encoder.

        Args:
            d_input: Input feature dimension.
            d_model: Model dimension.
            num_heads: Number of attention heads.
            num_layers: Number of attention layers.
            d_ff: Feed-forward dimension.
            dropout: Dropout probability.
            use_isab: Whether to use ISAB instead of SAB.
            num_inducing_points: Number of inducing points for ISAB.
        """
        super().__init__()

        self.input_proj = nn.Linear(d_input, d_model)

        layers: list[ISAB | SAB] = []
        for _ in range(num_layers):
            if use_isab:
                layers.append(ISAB(d_model, num_heads, num_inducing_points, d_ff, dropout))
            else:
                layers.append(SAB(d_model, num_heads, d_ff, dropout))

        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X: Tensor, mask: Tensor | None = None) -> Tensor:
        """Encode a set of elements.

        Args:
            X: Input features of shape (batch, N, d_input).
            mask: Optional mask of shape (batch, N) where True = valid.

        Returns:
            Encoded features of shape (batch, N, d_model).
        """
        # Project input to model dimension
        X = self.input_proj(X)

        # Apply attention layers
        for layer in self.layers:
            X = layer(X, mask)

        # Final normalization
        X = self.norm(X)

        return X


class SetTransformerPooling(nn.Module):
    """Set Transformer with pooling for set-level predictions.

    Uses PMA to pool the set into a fixed-size representation.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_pool_heads: int = 1,
        d_ff: int | None = None,
        dropout: float = 0.0,
        use_isab: bool = True,
        num_inducing_points: int = 64,
    ):
        """Initialize model.

        Args:
            d_input: Input feature dimension.
            d_model: Model dimension.
            num_heads: Number of attention heads in encoder.
            num_encoder_layers: Number of encoder layers.
            num_pool_heads: Number of output vectors from PMA.
            d_ff: Feed-forward dimension.
            dropout: Dropout probability.
            use_isab: Whether to use ISAB in encoder.
            num_inducing_points: Number of inducing points for ISAB.
        """
        super().__init__()

        self.encoder = SetTransformerEncoder(
            d_input=d_input,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_isab=use_isab,
            num_inducing_points=num_inducing_points,
        )

        self.pooling = PMA(d_model, num_heads, num_pool_heads, d_ff, dropout)

        # Optional: additional SAB after pooling for multi-head outputs
        self.post_pool: SAB | None
        if num_pool_heads > 1:
            self.post_pool = SAB(d_model, num_heads, d_ff, dropout)
        else:
            self.post_pool = None

    def forward(self, X: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            X: Input features of shape (batch, N, d_input).
            mask: Optional mask of shape (batch, N).

        Returns:
            Pooled representation of shape (batch, num_pool_heads, d_model).
        """
        # Encode
        encoded = self.encoder(X, mask)

        # Pool
        pooled = self.pooling(encoded, mask)

        # Optional post-pooling
        if self.post_pool is not None:
            pooled = self.post_pool(pooled)

        return pooled


class SetTransformerElementWise(nn.Module):
    """Set Transformer for element-wise predictions.

    Returns encoded representations for each element in the set.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_output: int,
        num_heads: int,
        num_layers: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        use_isab: bool = True,
        num_inducing_points: int = 64,
    ):
        """Initialize model.

        Args:
            d_input: Input feature dimension.
            d_model: Model dimension.
            d_output: Output dimension per element.
            num_heads: Number of attention heads.
            num_layers: Number of encoder layers.
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

        self.output_proj = nn.Linear(d_model, d_output)

    def forward(self, X: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            X: Input features of shape (batch, N, d_input).
            mask: Optional mask of shape (batch, N).

        Returns:
            Element-wise outputs of shape (batch, N, d_output).
        """
        encoded = self.encoder(X, mask)
        output = self.output_proj(encoded)
        return output
