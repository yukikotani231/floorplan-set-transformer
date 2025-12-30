"""Set Transformer building blocks.

Implementation of MAB, SAB, ISAB, and PMA modules from:
"Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"
(Lee et al., 2019)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiheadAttention(nn.Module):
    """Multi-head attention with optional mask support."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """Initialize multi-head attention.

        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute multi-head attention.

        Args:
            Q: Query tensor of shape (batch, seq_q, d_model).
            K: Key tensor of shape (batch, seq_k, d_model).
            V: Value tensor of shape (batch, seq_k, d_model).
            mask: Optional mask of shape (batch, seq_k) where True = valid.

        Returns:
            Output tensor of shape (batch, seq_q, d_model).
        """
        batch_size = Q.size(0)

        # Linear projections
        Q = self.W_q(Q)  # (batch, seq_q, d_model)
        K = self.W_k(K)  # (batch, seq_k, d_model)
        V = self.W_v(V)  # (batch, seq_k, d_model)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: (batch, num_heads, seq, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Shape: (batch, num_heads, seq_q, seq_k)

        # Apply mask if provided
        if mask is not None:
            # mask: (batch, seq_k) -> (batch, 1, 1, seq_k)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        # Shape: (batch, num_heads, seq_q, d_k)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(output)

        return output


class MAB(nn.Module):
    """Multihead Attention Block.

    MAB(X, Y) = LayerNorm(H + rFF(H))
    where H = LayerNorm(X + Multihead(X, Y, Y))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
    ):
        """Initialize MAB.

        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            d_ff: Feed-forward dimension. Defaults to 4 * d_model.
            dropout: Dropout probability.
        """
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.attention = MultiheadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        X: Tensor,
        Y: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            X: Query input of shape (batch, seq_x, d_model).
            Y: Key/Value input of shape (batch, seq_y, d_model).
            mask: Optional mask for Y of shape (batch, seq_y).

        Returns:
            Output tensor of shape (batch, seq_x, d_model).
        """
        # Multi-head attention with residual
        H = self.norm1(X + self.attention(X, Y, Y, mask))

        # Feed-forward with residual
        output = self.norm2(H + self.ff(H))

        return output


class SAB(nn.Module):
    """Set Attention Block.

    SAB(X) = MAB(X, X)

    Self-attention over the set.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
    ):
        """Initialize SAB.

        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            d_ff: Feed-forward dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.mab = MAB(d_model, num_heads, d_ff, dropout)

    def forward(self, X: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            X: Input set of shape (batch, N, d_model).
            mask: Optional mask of shape (batch, N).

        Returns:
            Output tensor of shape (batch, N, d_model).
        """
        return self.mab(X, X, mask)


class ISAB(nn.Module):
    """Induced Set Attention Block.

    ISAB_m(X) = MAB(X, H) where H = MAB(I, X)

    Uses m learnable inducing points to reduce complexity from O(N^2) to O(NM).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_inducing_points: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
    ):
        """Initialize ISAB.

        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            num_inducing_points: Number of inducing points (m).
            d_ff: Feed-forward dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing_points, d_model))
        nn.init.xavier_uniform_(self.inducing_points)

        self.mab1 = MAB(d_model, num_heads, d_ff, dropout)
        self.mab2 = MAB(d_model, num_heads, d_ff, dropout)

    def forward(self, X: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            X: Input set of shape (batch, N, d_model).
            mask: Optional mask of shape (batch, N).

        Returns:
            Output tensor of shape (batch, N, d_model).
        """
        batch_size = X.size(0)

        # Expand inducing points for batch
        inducing = self.inducing_points.expand(batch_size, -1, -1)

        # First MAB: inducing points attend to input
        H = self.mab1(inducing, X, mask)  # (batch, m, d_model)

        # Second MAB: input attends to compressed representation
        output = self.mab2(X, H)  # (batch, N, d_model)

        return output


class PMA(nn.Module):
    """Pooling by Multihead Attention.

    PMA_k(X) = MAB(S, rFF(X))

    Uses k learnable seed vectors to pool the set into k output vectors.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_seeds: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
    ):
        """Initialize PMA.

        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            num_seeds: Number of output vectors (k).
            d_ff: Feed-forward dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        self.seed_vectors = nn.Parameter(torch.randn(1, num_seeds, d_model))
        nn.init.xavier_uniform_(self.seed_vectors)

        if d_ff is None:
            d_ff = d_model

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
        )

        self.mab = MAB(d_model, num_heads, d_ff, dropout)

    def forward(self, X: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            X: Input set of shape (batch, N, d_model).
            mask: Optional mask of shape (batch, N).

        Returns:
            Output tensor of shape (batch, k, d_model).
        """
        batch_size = X.size(0)

        # Apply row-wise feed-forward
        X_ff = self.ff(X)

        # Expand seed vectors for batch
        S = self.seed_vectors.expand(batch_size, -1, -1)

        # Pool using attention
        output = self.mab(S, X_ff, mask)

        return output
