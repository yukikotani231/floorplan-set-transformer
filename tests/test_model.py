"""Tests for Set Transformer models."""

import torch

from src.models import (
    ISAB,
    MAB,
    PMA,
    SAB,
    PanopticSetTransformer,
    SetTransformerEncoder,
    create_panoptic_model,
)


class TestMAB:
    """Tests for Multihead Attention Block."""

    def test_forward_shape(self) -> None:
        """Test output shape of MAB."""
        batch_size = 4
        seq_x = 10
        seq_y = 20
        d_model = 64
        num_heads = 4

        mab = MAB(d_model, num_heads)
        X = torch.randn(batch_size, seq_x, d_model)
        Y = torch.randn(batch_size, seq_y, d_model)

        output = mab(X, Y)

        assert output.shape == (batch_size, seq_x, d_model)

    def test_forward_with_mask(self) -> None:
        """Test MAB with attention mask."""
        batch_size = 4
        seq_x = 10
        seq_y = 20
        d_model = 64
        num_heads = 4

        mab = MAB(d_model, num_heads)
        X = torch.randn(batch_size, seq_x, d_model)
        Y = torch.randn(batch_size, seq_y, d_model)
        mask = torch.ones(batch_size, seq_y, dtype=torch.bool)
        mask[:, 15:] = False  # Mask out last 5 elements

        output = mab(X, Y, mask)

        assert output.shape == (batch_size, seq_x, d_model)


class TestSAB:
    """Tests for Set Attention Block."""

    def test_forward_shape(self) -> None:
        """Test output shape of SAB."""
        batch_size = 4
        seq_len = 100
        d_model = 64
        num_heads = 4

        sab = SAB(d_model, num_heads)
        X = torch.randn(batch_size, seq_len, d_model)

        output = sab(X)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_permutation_equivariance(self) -> None:
        """Test that SAB is permutation equivariant."""
        batch_size = 2
        seq_len = 10
        d_model = 32
        num_heads = 4

        sab = SAB(d_model, num_heads)
        sab.eval()

        X = torch.randn(batch_size, seq_len, d_model)

        # Get output for original input
        with torch.no_grad():
            output1 = sab(X)

        # Permute input
        perm = torch.randperm(seq_len)
        X_perm = X[:, perm, :]

        # Get output for permuted input
        with torch.no_grad():
            output2 = sab(X_perm)

        # Reverse permutation on output
        inv_perm = torch.argsort(perm)
        output2_unperm = output2[:, inv_perm, :]

        # Should be approximately equal
        assert torch.allclose(output1, output2_unperm, atol=1e-5)


class TestISAB:
    """Tests for Induced Set Attention Block."""

    def test_forward_shape(self) -> None:
        """Test output shape of ISAB."""
        batch_size = 4
        seq_len = 100
        d_model = 64
        num_heads = 4
        num_inducing = 16

        isab = ISAB(d_model, num_heads, num_inducing)
        X = torch.randn(batch_size, seq_len, d_model)

        output = isab(X)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_with_mask(self) -> None:
        """Test ISAB with mask."""
        batch_size = 4
        seq_len = 100
        d_model = 64
        num_heads = 4
        num_inducing = 16

        isab = ISAB(d_model, num_heads, num_inducing)
        X = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[:, 80:] = False

        output = isab(X, mask)

        assert output.shape == (batch_size, seq_len, d_model)


class TestPMA:
    """Tests for Pooling by Multihead Attention."""

    def test_forward_shape(self) -> None:
        """Test output shape of PMA."""
        batch_size = 4
        seq_len = 100
        d_model = 64
        num_heads = 4
        num_seeds = 4

        pma = PMA(d_model, num_heads, num_seeds)
        X = torch.randn(batch_size, seq_len, d_model)

        output = pma(X)

        assert output.shape == (batch_size, num_seeds, d_model)


class TestSetTransformerEncoder:
    """Tests for Set Transformer Encoder."""

    def test_forward_shape_sab(self) -> None:
        """Test encoder with SAB."""
        batch_size = 4
        seq_len = 50
        d_input = 10
        d_model = 64

        encoder = SetTransformerEncoder(
            d_input=d_input,
            d_model=d_model,
            num_heads=4,
            num_layers=2,
            use_isab=False,
        )
        X = torch.randn(batch_size, seq_len, d_input)

        output = encoder(X)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_shape_isab(self) -> None:
        """Test encoder with ISAB."""
        batch_size = 4
        seq_len = 50
        d_input = 10
        d_model = 64

        encoder = SetTransformerEncoder(
            d_input=d_input,
            d_model=d_model,
            num_heads=4,
            num_layers=2,
            use_isab=True,
            num_inducing_points=16,
        )
        X = torch.randn(batch_size, seq_len, d_input)

        output = encoder(X)

        assert output.shape == (batch_size, seq_len, d_model)


class TestPanopticSetTransformer:
    """Tests for complete Panoptic Set Transformer model."""

    def test_forward_shape(self) -> None:
        """Test output shapes."""
        batch_size = 4
        seq_len = 50
        d_input = 12
        num_classes = 30
        instance_embed_dim = 32

        model = PanopticSetTransformer(
            d_input=d_input,
            d_model=64,
            num_heads=4,
            num_layers=2,
            num_classes=num_classes,
            instance_embed_dim=instance_embed_dim,
        )
        X = torch.randn(batch_size, seq_len, d_input)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        outputs = model(X, mask)

        assert outputs["semantic_logits"].shape == (batch_size, seq_len, num_classes)
        assert outputs["instance_embeds"].shape == (batch_size, seq_len, instance_embed_dim)
        assert outputs["encoded"].shape == (batch_size, seq_len, 64)

    def test_predict(self) -> None:
        """Test prediction method."""
        batch_size = 4
        seq_len = 50
        d_input = 12
        num_classes = 30

        model = PanopticSetTransformer(
            d_input=d_input,
            d_model=64,
            num_heads=4,
            num_layers=2,
            num_classes=num_classes,
        )
        X = torch.randn(batch_size, seq_len, d_input)

        preds = model.predict(X)

        assert preds["semantic_preds"].shape == (batch_size, seq_len)
        assert preds["semantic_preds"].dtype == torch.int64

    def test_create_panoptic_model(self) -> None:
        """Test factory function."""
        model = create_panoptic_model(
            feature_dim=12,
            num_classes=30,
            d_model=128,
            num_heads=4,
            num_layers=2,
        )

        assert isinstance(model, PanopticSetTransformer)
        assert model.num_classes == 30


class TestGradientFlow:
    """Tests for gradient flow through the model."""

    def test_gradients_flow(self) -> None:
        """Test that gradients flow through all parameters."""
        batch_size = 2
        seq_len = 20
        d_input = 12

        model = create_panoptic_model(
            feature_dim=d_input,
            num_classes=30,
            d_model=64,
            num_heads=4,
            num_layers=2,
        )

        X = torch.randn(batch_size, seq_len, d_input, requires_grad=True)
        outputs = model(X)

        # Compute a simple loss
        loss = outputs["semantic_logits"].sum() + outputs["instance_embeds"].sum()
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
