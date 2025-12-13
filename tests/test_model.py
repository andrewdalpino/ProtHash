import unittest

import torch

from src.prothash import model


class TestLoRA(unittest.TestCase):
    """Test cases for the LoRA class."""

    def setUp(self):
        torch.manual_seed(42)

    def test_from_linear_factory_method(self):
        """Test LoRA.from_linear factory method."""
        linear = torch.nn.Linear(8, 16, bias=False)
        lora = model.LoRA.from_linear(linear, rank=4, alpha=0.5)

        self.assertIsInstance(lora, model.LoRA)
        self.assertEqual(lora.lora_a.shape, (4, 8))
        self.assertEqual(lora.lora_b.shape, (16, 4))
        self.assertEqual(lora.alpha, 0.5)

    def test_forward_with_zero_lora_b(self):
        """Test that forward returns original weight when lora_b is zero."""
        linear = torch.nn.Linear(4, 6, bias=False)
        weight = linear.weight.detach().clone()

        lora = model.LoRA.from_linear(linear, rank=2, alpha=1.0)
        out = lora.forward(weight)

        # lora_b is initialized to zeros, so output should equal input
        self.assertTrue(torch.allclose(out, weight))

    def test_forward_with_nonzero_lora_b(self):
        """Test that forward modifies weight when lora_b is non-zero."""
        linear = torch.nn.Linear(4, 6, bias=False)
        weight = linear.weight.detach().clone()

        lora = model.LoRA.from_linear(linear, rank=2, alpha=2.0)

        # Set lora_b to non-zero values
        with torch.no_grad():
            lora.lora_b.data.fill_(0.1)

        out = lora.forward(weight)

        self.assertFalse(torch.allclose(out, weight))
        self.assertEqual(out.shape, weight.shape)

    def test_invalid_rank(self):
        """Test that invalid rank raises assertion error."""
        with self.assertRaises(AssertionError):
            model.LoRA(in_features=4, out_features=6, rank=0, alpha=1.0)

    def test_invalid_alpha(self):
        """Test that invalid alpha raises assertion error."""
        with self.assertRaises(AssertionError):
            model.LoRA(in_features=4, out_features=6, rank=2, alpha=0.0)


class TestAdapterHead(unittest.TestCase):
    """Test cases for the AdapterHead class."""

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shape(self):
        """Test AdapterHead forward pass preserves batch and sequence dimensions."""
        head = model.AdapterHead(in_dimensions=8, out_dimensions=16)
        x = torch.randn(2, 5, 8)
        out = head.forward(x)

        self.assertEqual(out.shape, (2, 5, 16))

    def test_different_dimensions(self):
        """Test AdapterHead with various dimension combinations."""
        test_cases = [(4, 8), (16, 8), (8, 8)]

        for in_dim, out_dim in test_cases:
            head = model.AdapterHead(in_dimensions=in_dim, out_dimensions=out_dim)
            x = torch.randn(3, 7, in_dim)
            out = head.forward(x)

            self.assertEqual(out.shape, (3, 7, out_dim))


class TestInvertedBottleneck(unittest.TestCase):
    """Test cases for the InvertedBottleneck class."""

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shape_preservation(self):
        """Test that InvertedBottleneck preserves input shape."""
        fb = model.InvertedBottleneck(
            embedding_dimensions=8, hidden_ratio=2, dropout=0.0
        )
        x = torch.randn(2, 6, 8)
        out = fb.forward(x)

        self.assertEqual(out.shape, x.shape)

    def test_valid_hidden_ratios(self):
        """Test that valid hidden ratios (1, 2, 4) work correctly."""
        for ratio in [1, 2, 4]:
            fb = model.InvertedBottleneck(
                embedding_dimensions=8, hidden_ratio=ratio, dropout=0.0
            )
            x = torch.randn(2, 4, 8)
            out = fb.forward(x)

            self.assertEqual(out.shape, x.shape)

    def test_invalid_hidden_ratio(self):
        """Test that invalid hidden ratio raises assertion error."""
        with self.assertRaises(AssertionError):
            model.InvertedBottleneck(
                embedding_dimensions=8, hidden_ratio=3, dropout=0.0
            )

    def test_add_lora_adapters(self):
        """Test adding LoRA adapters to InvertedBottleneck."""
        fb = model.InvertedBottleneck(
            embedding_dimensions=8, hidden_ratio=2, dropout=0.0
        )
        fb.add_lora_adapters(rank=2, alpha=1.0)

        # Check that parametrizations were added
        self.assertTrue(hasattr(fb.linear1, "parametrizations"))
        self.assertTrue(hasattr(fb.linear2, "parametrizations"))


class TestSelfAttention(unittest.TestCase):
    """Test cases for the SelfAttention class."""

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shape_preservation(self):
        """Test that SelfAttention preserves input shape."""
        sa = model.SelfAttention(
            embedding_dimensions=8, q_heads=2, kv_heads=1, dropout=0.0
        )
        x = torch.randn(3, 5, 8)
        out = sa.forward(x)

        self.assertEqual(out.shape, x.shape)

    def test_multi_head_attention(self):
        """Test multi-head attention with equal q and kv heads."""
        sa = model.SelfAttention(
            embedding_dimensions=16, q_heads=4, kv_heads=4, dropout=0.0
        )
        x = torch.randn(2, 10, 16)
        out = sa.forward(x)

        self.assertEqual(out.shape, x.shape)

    def test_grouped_query_attention(self):
        """Test grouped query attention (q_heads > kv_heads)."""
        sa = model.SelfAttention(
            embedding_dimensions=8, q_heads=4, kv_heads=2, dropout=0.0
        )
        x = torch.randn(2, 6, 8)
        out = sa.forward(x)

        self.assertEqual(out.shape, x.shape)
        self.assertTrue(sa.is_gqa)

    def test_invalid_embedding_dimensions(self):
        """Test that non-divisible embedding dimensions raise assertion error."""
        with self.assertRaises(AssertionError):
            model.SelfAttention(
                embedding_dimensions=7, q_heads=3, kv_heads=1, dropout=0.0
            )

    def test_invalid_head_relationship(self):
        """Test that q_heads < kv_heads raises assertion error."""
        with self.assertRaises(AssertionError):
            model.SelfAttention(
                embedding_dimensions=8, q_heads=1, kv_heads=2, dropout=0.0
            )

    def test_add_lora_adapters(self):
        """Test adding LoRA adapters to SelfAttention."""
        sa = model.SelfAttention(
            embedding_dimensions=8, q_heads=2, kv_heads=1, dropout=0.0
        )
        sa.add_lora_adapters(rank=2, alpha=1.0)

        # Check that parametrizations were added to all projection layers
        self.assertTrue(hasattr(sa.q_proj, "parametrizations"))
        self.assertTrue(hasattr(sa.k_proj, "parametrizations"))
        self.assertTrue(hasattr(sa.v_proj, "parametrizations"))
        self.assertTrue(hasattr(sa.out_proj, "parametrizations"))


class TestEncoderBlock(unittest.TestCase):
    """Test cases for the EncoderBlock class."""

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shape_preservation(self):
        """Test that EncoderBlock preserves input shape."""
        block = model.EncoderBlock(
            embedding_dimensions=8, q_heads=2, kv_heads=1, hidden_ratio=2, dropout=0.0
        )
        x = torch.randn(2, 4, 8)
        out = block.forward(x)

        self.assertEqual(out.shape, x.shape)

    def test_residual_connections(self):
        """Test that residual connections are working."""
        block = model.EncoderBlock(
            embedding_dimensions=8, q_heads=2, kv_heads=1, hidden_ratio=1, dropout=0.0
        )
        x = torch.randn(2, 4, 8)
        out = block.forward(x)

        # Output should differ from input due to transformations
        self.assertFalse(torch.allclose(out, x))

    def test_add_lora_adapters(self):
        """Test adding LoRA adapters to EncoderBlock."""
        block = model.EncoderBlock(
            embedding_dimensions=8, q_heads=2, kv_heads=1, hidden_ratio=2, dropout=0.0
        )
        block.add_lora_adapters(rank=2, alpha=1.0)

        # Check that both stages have parametrizations
        self.assertTrue(hasattr(block.stage1.q_proj, "parametrizations"))
        self.assertTrue(hasattr(block.stage2.linear1, "parametrizations"))


class TestEncoder(unittest.TestCase):
    """Test cases for the Encoder class."""

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shape_preservation(self):
        """Test that Encoder preserves input shape."""
        enc = model.Encoder(
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            num_layers=2,
            hidden_ratio=1,
            dropout=0.0,
        )
        x = torch.randn(2, 4, 8)
        out = enc.forward(x)

        self.assertEqual(out.shape, x.shape)

    def test_single_layer(self):
        """Test Encoder with single layer."""
        enc = model.Encoder(
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            num_layers=1,
            hidden_ratio=1,
            dropout=0.0,
        )
        x = torch.randn(2, 4, 8)
        out = enc.forward(x)

        self.assertEqual(out.shape, x.shape)
        self.assertEqual(len(enc.layers), 1)

    def test_invalid_num_layers(self):
        """Test that zero layers raises assertion error."""
        with self.assertRaises(AssertionError):
            model.Encoder(
                embedding_dimensions=8,
                q_heads=2,
                kv_heads=1,
                num_layers=0,
                hidden_ratio=1,
                dropout=0.0,
            )

    def test_enable_activation_checkpointing(self):
        """Test enabling activation checkpointing."""
        enc = model.Encoder(
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            num_layers=2,
            hidden_ratio=1,
            dropout=0.0,
        )
        enc.enable_activation_checkpointing()

        # Checkpoint should be callable
        self.assertTrue(callable(enc.checkpoint))

    def test_add_lora_adapters(self):
        """Test adding LoRA adapters to all encoder layers."""
        enc = model.Encoder(
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            num_layers=2,
            hidden_ratio=1,
            dropout=0.0,
        )
        enc.add_lora_adapters(rank=2, alpha=1.0)

        # Check that all layers have parametrizations
        for layer in enc.layers:
            self.assertTrue(hasattr(layer.stage1.q_proj, "parametrizations"))


class TestONNXModel(unittest.TestCase):
    """Test cases for the ONNXModel wrapper class."""

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_wraps_embed(self):
        """Test that ONNXModel.forward wraps ProtHash.embed."""
        prothash = model.ProtHash(
            vocabulary_size=10,
            padding_index=0,
            context_length=16,
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            hidden_ratio=1,
            num_encoder_layers=1,
            dropout=0.0,
        )

        onnx_model = model.ONNXModel(prothash)
        x = torch.randint(0, 9, (2, 5), dtype=torch.int64)

        out_onnx = onnx_model.forward(x)
        out_embed = prothash.embed(x)

        self.assertTrue(torch.allclose(out_onnx, out_embed))


class TestProtHash(unittest.TestCase):
    """Test cases for the ProtHash main model class."""

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_basic(self):
        """Test basic forward pass."""
        m = model.ProtHash(
            vocabulary_size=10,
            padding_index=0,
            context_length=16,
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            hidden_ratio=1,
            num_encoder_layers=1,
            dropout=0.0,
        )

        x = torch.randint(0, 9, (2, 5), dtype=torch.int64)
        out = m.forward(x)

        self.assertEqual(out.shape, (2, 5, 8))

    def test_embed_returns_cls_token(self):
        """Test that embed returns CLS token embeddings."""
        m = model.ProtHash(
            vocabulary_size=10,
            padding_index=0,
            context_length=16,
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            hidden_ratio=1,
            num_encoder_layers=1,
            dropout=0.0,
        )

        x = torch.randint(0, 9, (2, 5), dtype=torch.int64)
        emb = m.embed(x)

        # Should return only the first token (CLS) for each sequence
        self.assertEqual(emb.shape, (2, 8))

    def test_context_length_assertion(self):
        """Test that exceeding context length raises assertion error."""
        m = model.ProtHash(
            vocabulary_size=10,
            padding_index=0,
            context_length=8,
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            hidden_ratio=1,
            num_encoder_layers=1,
            dropout=0.0,
        )

        # Create input longer than context_length
        x = torch.randint(0, 9, (2, 10), dtype=torch.int64)

        with self.assertRaises(AssertionError):
            m.forward(x)

    def test_add_adapter_head(self):
        """Test adding adapter head."""
        m = model.ProtHash(
            vocabulary_size=10,
            padding_index=0,
            context_length=16,
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            hidden_ratio=1,
            num_encoder_layers=1,
            dropout=0.0,
        )

        self.assertFalse(hasattr(m, "head"))

        m.add_adapter_head(out_dimensions=16)
        self.assertTrue(hasattr(m, "head"))

        x = torch.randint(0, 9, (2, 5), dtype=torch.int64)
        out = m.forward(x)

        # Output should have adapter head dimensions
        self.assertEqual(out.shape, (2, 5, 16))

    def test_remove_adapter_head(self):
        """Test removing adapter head."""
        m = model.ProtHash(
            vocabulary_size=10,
            padding_index=0,
            context_length=16,
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            hidden_ratio=1,
            num_encoder_layers=1,
            dropout=0.0,
        )

        m.add_adapter_head(out_dimensions=16)
        self.assertTrue(hasattr(m, "head"))

        m.remove_adapter_head()
        self.assertFalse(hasattr(m, "head"))

    def test_num_params_property(self):
        """Test num_params property."""
        m = model.ProtHash(
            vocabulary_size=10,
            padding_index=0,
            context_length=16,
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            hidden_ratio=1,
            num_encoder_layers=1,
            dropout=0.0,
        )

        num_params = m.num_params
        self.assertGreater(num_params, 0)
        self.assertIsInstance(num_params, int)

    def test_num_trainable_parameters_property(self):
        """Test num_trainable_parameters property."""
        m = model.ProtHash(
            vocabulary_size=10,
            padding_index=0,
            context_length=16,
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            hidden_ratio=1,
            num_encoder_layers=1,
            dropout=0.0,
        )

        num_trainable = m.num_trainable_parameters
        self.assertGreater(num_trainable, 0)
        self.assertEqual(num_trainable, m.num_params)

    def test_add_lora_adapters(self):
        """Test adding LoRA adapters to the model."""
        m = model.ProtHash(
            vocabulary_size=10,
            padding_index=0,
            context_length=16,
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            hidden_ratio=1,
            num_encoder_layers=1,
            dropout=0.0,
        )

        # Should not raise any errors
        m.add_lora_adapters(rank=2, alpha=1.0)

        # Forward pass should still work
        x = torch.randint(0, 9, (2, 5), dtype=torch.int64)
        out = m.forward(x)
        self.assertEqual(out.shape, (2, 5, 8))

    def test_merge_lora_adapters(self):
        """Test merging LoRA adapters."""
        m = model.ProtHash(
            vocabulary_size=10,
            padding_index=0,
            context_length=16,
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            hidden_ratio=1,
            num_encoder_layers=1,
            dropout=0.0,
        )

        m.add_lora_adapters(rank=2, alpha=1.0)

        # Should not raise any errors
        m.merge_lora_adapters()

        # Forward pass should still work
        x = torch.randint(0, 9, (2, 5), dtype=torch.int64)
        out = m.forward(x)
        self.assertEqual(out.shape, (2, 5, 8))


if __name__ == "__main__":
    unittest.main()
