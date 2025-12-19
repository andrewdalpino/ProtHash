from math import sqrt
from functools import partial
from typing import Self

import torch

from torch import Tensor

from torch.nn import (
    Module,
    ModuleList,
    Embedding,
    Linear,
    SiLU,
    RMSNorm,
    Dropout1d,
    Identity,
    Parameter,
)

from torch.nn.functional import scaled_dot_product_attention
from torch.nn.utils.parametrize import register_parametrization, remove_parametrizations
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from torchao.quantization import Int8WeightOnlyConfig, quantize_

from torchao.quantization.qat import (
    IntxFakeQuantizeConfig,
    QATConfig,
    FromIntXQuantizationAwareTrainingConfig,
)

from huggingface_hub import PyTorchModelHubMixin


class ProtHash(Module, PyTorchModelHubMixin):
    """
    An encoder-only transformer model for protein sequence embedding.
    """

    def __init__(
        self,
        vocabulary_size: int,
        padding_index: int,
        context_length: int,
        teacher_dimensions: int,
        embedding_dimensions: int,
        q_heads: int,
        kv_heads: int,
        hidden_ratio: int,
        num_encoder_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.token_embeddings = Embedding(
            vocabulary_size, embedding_dimensions, padding_idx=padding_index
        )

        self.position_embeddings = Embedding(context_length, embedding_dimensions)

        self.encoder = Encoder(
            embedding_dimensions,
            q_heads,
            kv_heads,
            num_encoder_layers,
            hidden_ratio=hidden_ratio,
            dropout=dropout,
        )

        if embedding_dimensions != teacher_dimensions:
            head = AdapterHead(embedding_dimensions, teacher_dimensions)
        else:
            head = Identity()

        self.head = head

        self.vocabulary_size = vocabulary_size
        self.padding_index = padding_index
        self.context_length = context_length
        self.teacher_dimensions = teacher_dimensions
        self.embedding_dimensions = embedding_dimensions

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_weights(self) -> None:
        """Freeze all model parameters."""

        for module in self.modules():
            for param in module.parameters():
                param.requires_grad = False

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the model using LoRA adapters."""

        self.encoder.add_lora_adapters(rank, alpha)

        if isinstance(self.head, AdapterHead):
            self.head.add_lora_adapters(rank, alpha)

    def merge_lora_adapters(self) -> None:
        """Merge the LoRA adapters with the original parameters."""

        for module in self.modules():
            if not hasattr(module, "parametrizations"):
                continue

            lora_params = []

            for name, parameterizations in module.parametrizations.items():
                for parametrization in parameterizations:
                    if isinstance(parametrization, LoRA):
                        lora_params.append(name)

            for name in lora_params:
                remove_parametrizations(module, name)

    def add_fake_quantized_tensors(self) -> None:
        """Prepare the model for quantization-aware training."""

        self.encoder.add_fake_quantized_tensors()

    def remove_fake_quantized_tensors(self) -> None:
        """Convert fake quantized tensors back to regular tensors."""

        self.encoder.remove_fake_quantized_tensors()

    def quantize_weights(self) -> None:
        """Quantize the weights of the model."""

        self.encoder.quantize_weights()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): The token index sequence of shape (batch_size, sequence_length).
        """

        b, t = x.size()

        assert (
            t <= self.context_length
        ), f"Input sequence length {t} exceeds the maximum context length {self.context_length}."

        z_tok = self.token_embeddings(x)

        x_pos = torch.arange(t, dtype=torch.int64, device=x.device)
        x_pos = x_pos.unsqueeze(0).expand(b, t)

        z_pos = self.position_embeddings(x_pos)

        z = z_tok + z_pos

        z = self.encoder.forward(z)

        return z

    def forward_with_adapter(self, x: Tensor) -> Tensor:
        z = self.forward(x)

        z = self.head.forward(z)

        return z

    @torch.inference_mode()
    def embed_native(self, x: Tensor) -> Tensor:
        """
        Output the contextual embeddings of the input sequence in native dimensionality.

        Args:
            x (Tensor): The token index sequence of shape (batch_size, sequence_length).

        Returns:
            Tensor: The contextual embeddings of shape (batch_size, embedding_dimensions).
        """

        z = self.forward(x)

        # Grab the classification token <CLS> vectors.
        z = z[:, 0, :]

        return z

    @torch.inference_mode()
    def embed_teacher(self, x: Tensor) -> Tensor:
        """
        Output the contextual embeddings of the input sequence in the teacher's dimensionality.

        Args:
            x (Tensor): The token index sequence of shape (batch_size, sequence_length).

        Returns:
            Tensor: The contextual embeddings of shape (batch_size, teacher_dimensions).
        """

        z = self.forward_with_adapter(x)

        # Grab the classification token <CLS> vectors.
        z = z[:, 0, :]

        return z


class ONNXModel(Module):
    """A wrapper class for exporting the ProtHash model to ONNX format."""

    def __init__(self, model: ProtHash):
        super().__init__()

        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model.embed_native(x)


class Encoder(Module):
    """A deep stack of encoder blocks consisting of self-attention and feed-forward layers."""

    def __init__(
        self,
        embedding_dimensions: int,
        q_heads: int,
        kv_heads: int,
        num_layers: int,
        hidden_ratio: int,
        dropout: float,
    ):
        super().__init__()

        assert num_layers > 0, "Number of layers must be greater than 0."

        self.layers = ModuleList(
            [
                EncoderBlock(
                    embedding_dimensions,
                    q_heads,
                    kv_heads,
                    hidden_ratio,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.checkpoint = lambda layer, x: layer.forward(x)

    def enable_activation_checkpointing(self) -> None:
        """Instead of memorizing the activations of the forward pass, recompute them at various checkpoints."""

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the decoder using LoRA adapters."""

        for layer in self.layers:
            layer.add_lora_adapters(rank, alpha)

    def add_fake_quantized_tensors(self) -> None:
        """Prepare the model for quantization-aware training."""

        for layer in self.layers:
            layer.add_fake_quantized_tensors()

    def remove_fake_quantized_tensors(self) -> None:
        """Convert fake quantized tensors back to regular tensors."""

        for layer in self.layers:
            layer.remove_fake_quantized_tensors()

    def quantize_weights(self) -> None:
        """Quantize the weights of the model."""

        for layer in self.layers:
            layer.quantize_weights()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = self.checkpoint(layer, x)

        return x


class EncoderBlock(Module):
    """Encoder block with multi-head attention, wide activation layer, and residual connections."""

    def __init__(
        self,
        embedding_dimensions: int,
        q_heads: int,
        kv_heads: int,
        hidden_ratio: int,
        dropout: float,
    ):
        super().__init__()

        self.stage1 = SelfAttention(embedding_dimensions, q_heads, kv_heads, dropout)
        self.stage2 = InvertedBottleneck(embedding_dimensions, hidden_ratio, dropout)

        self.norm1 = RMSNorm(embedding_dimensions)
        self.norm2 = RMSNorm(embedding_dimensions)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the encoder using LoRA adapters."""

        self.stage1.add_lora_adapters(rank, alpha)
        self.stage2.add_lora_adapters(rank, alpha)

    def add_fake_quantized_tensors(self) -> None:
        """Prepare the model for quantization-aware training."""

        self.stage1.add_fake_quantized_tensors()
        self.stage2.add_fake_quantized_tensors(self.stage1.head_dimensions)

    def remove_fake_quantized_tensors(self) -> None:
        """Convert fake quantized tensors back to regular tensors."""

        self.stage1.remove_fake_quantized_tensors()
        self.stage2.remove_fake_quantized_tensors()

    def quantize_weights(self) -> None:
        """Quantize the weights of the model."""

        self.stage1.quantize_weights()
        self.stage2.quantize_weights(self.stage1.head_dimensions)

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm1.forward(x)
        z = self.stage1.forward(z)

        z1 = x + z  # Local residual connection

        z = self.norm2.forward(z1)
        z = self.stage2.forward(z)

        z2 = z1 + z  # Local residual connection

        return z2


class SelfAttention(Module):
    """Group query self-attention using fused scaled dot product attention kernel."""

    def __init__(
        self,
        embedding_dimensions: int,
        q_heads: int,
        kv_heads: int,
        dropout: float,
    ):
        super().__init__()

        assert embedding_dimensions > 0, "Embedding dimensions must be greater than 0."
        assert q_heads > 0, "Number of query heads must be greater than 0."
        assert kv_heads > 0, "Number of key-value heads must be greater than 0."

        assert (
            q_heads >= kv_heads
        ), "Number of query heads must be greater than or equal to the number of key-value heads."

        assert (
            embedding_dimensions % q_heads == 0
        ), "Embedding dimensions must be divisible by the number of query heads."

        head_dimensions = embedding_dimensions // q_heads

        kv_dimensions = kv_heads * head_dimensions

        self.q_proj = Linear(embedding_dimensions, embedding_dimensions, bias=False)
        self.k_proj = Linear(embedding_dimensions, kv_dimensions, bias=False)
        self.v_proj = Linear(embedding_dimensions, kv_dimensions, bias=False)

        self.out_proj = Linear(embedding_dimensions, embedding_dimensions, bias=False)

        scale = 1.0 / sqrt(head_dimensions)

        is_gqa = q_heads > kv_heads

        self.embedding_dimensions = embedding_dimensions
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dimensions = head_dimensions
        self.scale = scale
        self.is_gqa = is_gqa
        self.dropout = dropout

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the attention module using LoRA adapters."""

        register_parametrization(
            self.q_proj, "weight", LoRA.from_linear(self.q_proj, rank, alpha)
        )

        register_parametrization(
            self.k_proj, "weight", LoRA.from_linear(self.k_proj, rank, alpha)
        )

        register_parametrization(
            self.v_proj, "weight", LoRA.from_linear(self.v_proj, rank, alpha)
        )

        register_parametrization(
            self.out_proj, "weight", LoRA.from_linear(self.out_proj, rank, alpha)
        )

    def add_fake_quantized_tensors(self) -> None:
        """Prepare the model for quantization-aware training."""

        weight_config = IntxFakeQuantizeConfig(
            torch.int8, group_size=self.head_dimensions
        )

        config = QATConfig(weight_config=weight_config)

        quantize_(self, config)

    def remove_fake_quantized_tensors(self) -> None:
        """Convert fake quantized tensors back to regular tensors."""

        config = FromIntXQuantizationAwareTrainingConfig()

        quantize_(self, config)

    def quantize_weights(self) -> None:
        """Quantize the weights of the model."""

        config = Int8WeightOnlyConfig(group_size=self.head_dimensions)

        quantize_(self, config)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.size()

        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)

        q = q.view(b, t, self.q_heads, self.head_dimensions).transpose(1, 2)
        k = k.view(b, t, self.kv_heads, self.head_dimensions).transpose(1, 2)
        v = v.view(b, t, self.kv_heads, self.head_dimensions).transpose(1, 2)

        z = scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
            enable_gqa=self.is_gqa,
        )

        z = z.transpose(1, 2).contiguous().view(b, t, d)

        z = self.out_proj.forward(z)

        return z


class InvertedBottleneck(Module):
    """A two layer fully-connected network with a wide non-linear activation."""

    def __init__(self, embedding_dimensions: int, hidden_ratio: int, dropout: float):
        super().__init__()

        assert hidden_ratio in {1, 2, 4}, "Hidden ratio must be either 1, 2, or 4."

        hidden_dimensions = hidden_ratio * embedding_dimensions

        self.linear1 = Linear(embedding_dimensions, hidden_dimensions, bias=False)
        self.linear2 = Linear(hidden_dimensions, embedding_dimensions, bias=False)

        self.silu = SiLU()

        self.dropout = Dropout1d(p=dropout)

        self.hidden_dimensions = hidden_dimensions

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the feedforward module using LoRA adapters."""

        register_parametrization(
            self.linear1, "weight", LoRA.from_linear(self.linear1, rank, alpha)
        )

        register_parametrization(
            self.linear2, "weight", LoRA.from_linear(self.linear2, rank, alpha)
        )

    def add_fake_quantized_tensors(self, group_size: int) -> None:
        """Prepare the model for quantization-aware training."""

        weight_config = IntxFakeQuantizeConfig(torch.int8, group_size=group_size)

        config = QATConfig(weight_config=weight_config)

        quantize_(self, config)

    def remove_fake_quantized_tensors(self) -> None:
        """Convert fake quantized tensors back to regular tensors."""

        config = FromIntXQuantizationAwareTrainingConfig()

        quantize_(self, config)

    def quantize_weights(self, group_size: int) -> None:
        """Quantize the weights of the model."""

        config = Int8WeightOnlyConfig(group_size=group_size)

        quantize_(self, config)

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear1.forward(x)
        z = self.silu.forward(z)
        z = self.dropout.forward(z)
        z = self.linear2.forward(z)

        return z


class AdapterHead(Module):
    """A head for adapting to the teacher's embedding dimensionality."""

    def __init__(self, in_dimensions: int, out_dimensions: int):
        super().__init__()

        self.linear = Linear(in_dimensions, out_dimensions, bias=False)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the feedforward module using LoRA adapters."""

        register_parametrization(
            self.linear, "weight", LoRA.from_linear(self.linear, rank, alpha)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class LoRA(Module):
    """Low rank weight decomposition transformation."""

    @classmethod
    def from_linear(cls, linear: Linear, rank: int, alpha: float) -> Self:
        out_features, in_features = linear.weight.shape

        return cls(in_features, out_features, rank, alpha)

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()

        assert rank > 0, "Rank must be greater than 0."
        assert alpha > 0.0, "Alpha must be greater than 0."

        lora_a = torch.randn(rank, in_features) / sqrt(rank)
        lora_b = torch.zeros(out_features, rank)

        self.lora_a = Parameter(lora_a)
        self.lora_b = Parameter(lora_b)

        self.alpha = alpha

    def forward(self, weight: Tensor) -> Tensor:
        z = self.lora_b @ self.lora_a

        z *= self.alpha

        z = weight + z

        return z
