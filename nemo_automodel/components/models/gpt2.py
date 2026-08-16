# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT-2 / Modern nanoGPT model implementation for NeMo Automodel.

Provides a high-performance, pure-PyTorch causal language model incorporating
nanoGPT and modded-nanogpt best practices:
1. Bias-free linear & normalization layers (bias=False)
2. Hardware-aligned vocabulary sizing (multiple of 64/128, e.g. 50,304)
3. RMSNorm for low-latency memory-efficient normalization
4. Scaled residual projection initialization (0.02 / sqrt(2 * n_layer))
5. Rotary Position Embeddings (RoPE) + SwiGLU / Fast Tanh GELU
6. PyTorch SDPA attention with Zero-Dropout for pretraining
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["build_gpt2_model", "GPT2LMHeadModel", "RMSNorm", "SwiGLU"]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        norm = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).type_as(x)


def precompute_rope_freqs(head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0) -> torch.Tensor:
    """Precompute rotary position embedding cosine and sine frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(freqs_cis)  # (max_seq_len, head_dim // 2, 2)


def apply_rotary_emb(x: torch.Tensor, freqs_real: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to query or key tensor."""
    bsz, num_heads, seq_len, head_dim = x.shape
    x_reshaped = x.float().reshape(bsz, num_heads, seq_len, head_dim // 2, 2)
    cos = freqs_real[:seq_len, :, 0].unsqueeze(0).unsqueeze(0).to(x.device)
    sin = freqs_real[:seq_len, :, 1].unsqueeze(0).unsqueeze(0).to(x.device)
    x_out = torch.stack(
        [
            x_reshaped[..., 0] * cos - x_reshaped[..., 1] * sin,
            x_reshaped[..., 0] * sin + x_reshaped[..., 1] * cos,
        ],
        dim=-1,
    ).flatten(3)
    return x_out.type_as(x)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with SDPA and optional RoPE."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        attn_dropout: float = 0.0,
        use_rope: bool = True,
        max_seq_len: int = 4096,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = attn_dropout

        if use_rope:
            self.register_buffer("freqs_real", precompute_rope_freqs(self.head_dim, max_seq_len), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        bsz, seq_len, _ = x.shape

        # Project to QKV and reshape: (B, T, 3*C) → (B, n_head, T, head_dim)
        qkv = self.qkv_proj(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE if enabled
        if self.use_rope and hasattr(self, "freqs_real"):
            q = apply_rotary_emb(q, self.freqs_real)
            k = apply_rotary_emb(k, self.freqs_real)

        # Use PyTorch's optimized SDPA implementation.
        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0, is_causal=True
        )  # (B, n_head, T, head_dim)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (SiLU gating with hardware-aligned hidden dimension)."""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None, bias: bool = False):
        super().__init__()
        hidden_dim = hidden_dim or int(2 * (4 * embed_dim) / 3)
        # Align hidden_dim to multiple of 64 for optimal Tensor Core warp occupancy
        hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.gate_up_proj = nn.Linear(embed_dim, 2 * hidden_dim, bias=bias)
        self.fc_down = nn.Linear(hidden_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        """Apply the fused SwiGLU projection.

        Args:
            x: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden]. The returned tensor does
            not alias ``x``.
        """
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.fc_down(F.silu(gate) * up)


class MLP(nn.Module):
    """GPT-2 feed-forward network with fast Tanh-approximate GELU."""

    def __init__(self, embed_dim: int, expansion_factor: int = 4, bias: bool = False):
        super().__init__()
        hidden_dim = expansion_factor * embed_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """A single transformer block with configurable norm and FFN."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        dropout: float = 0.0,
        use_rope: bool = True,
        norm_type: str = "rmsnorm",
        mlp_type: str = "swiglu",
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.ln_1 = RMSNorm(embed_dim) if norm_type == "rmsnorm" else nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(
            embed_dim, num_heads, bias=bias, attn_dropout=dropout, use_rope=use_rope, max_seq_len=max_seq_len
        )
        self.ln_2 = RMSNorm(embed_dim) if norm_type == "rmsnorm" else nn.LayerNorm(embed_dim)

        if mlp_type == "swiglu":
            self.mlp = SwiGLU(embed_dim, bias=bias)
        else:
            self.mlp = MLP(embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2LMHeadModel(nn.Module):
    """Modern nanoGPT Causal-LM with weight tying, RoPE, RMSNorm, and scaled init."""

    def __init__(
        self,
        *,
        vocab_size: int = 50304,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        bias: bool = False,
        dropout: float = 0.0,
        use_rope: bool = True,
        norm_type: str = "rmsnorm",
        mlp_type: str = "swiglu",
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.use_rope = use_rope
        self.wte = nn.Embedding(vocab_size, n_embd)

        # If RoPE is disabled, fall back to learned absolute positional embeddings
        if not use_rope:
            self.wpe = nn.Embedding(n_positions, n_embd)
            self.drop = nn.Dropout(dropout)
        else:
            self.wpe = None
            self.drop = None

        self.h = nn.ModuleList(
            [
                TransformerBlock(
                    n_embd,
                    n_head,
                    bias=bias,
                    dropout=dropout,
                    use_rope=use_rope,
                    norm_type=norm_type,
                    mlp_type=mlp_type,
                    max_seq_len=n_positions,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = RMSNorm(n_embd) if norm_type == "rmsnorm" else nn.LayerNorm(n_embd)

        # Language model head (weights tied to token embedding matrix)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying

        # Scaled residual parameter initialization
        self._init_weights(n_layer)

    def initialize_weights(self):
        self._init_weights(len(self.h))

    def _init_weights(self, n_layer: int):
        """Parameter initialization with scaled residual standard deviation."""
        std = 0.02
        res_std = std / math.sqrt(2 * n_layer)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        # Scale residual projections
        for block in self.h:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=res_std)
            if hasattr(block.mlp, "fc_down"):
                nn.init.normal_(block.mlp.fc_down.weight, mean=0.0, std=res_std)
            elif hasattr(block.mlp, "fc2"):
                nn.init.normal_(block.mlp.fc2.weight, mean=0.0, std=res_std)

    def forward(
        self, input_ids: torch.LongTensor, logits_to_keep: int | None = None, **kwargs
    ) -> torch.Tensor | dict[str, torch.Tensor]:  # (B, T) → (B, T, V)
        """Compute causal-LM logits or final hidden states for fused loss.

        Args:
            input_ids: Tensor of shape [batch, sequence] containing vocabulary
                indices. ``sequence`` must not exceed ``n_positions``.
            logits_to_keep: When set, skip the vocabulary projection and return
                final hidden states for a fused linear cross-entropy loss.
            **kwargs: Compatibility arguments ignored by this model.

        Returns:
            When ``logits_to_keep`` is None, a tensor of shape [batch,
            sequence, vocab] in the model's computation dtype. Otherwise, a
            mapping containing ``hidden_states`` with shape [batch, sequence,
            hidden]. Returned tensors do not alias ``input_ids``.
        """
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}")
        if input_ids.size(1) > self.n_positions:
            raise ValueError(
                f"Sequence length {input_ids.size(1)} exceeds the configured context length {self.n_positions}."
            )
        x = self.wte(input_ids)

        if self.wpe is not None and self.drop is not None:
            batch_size, seq_len = input_ids.shape
            pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
            x = self.drop(x + self.wpe(pos_ids))

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        if logits_to_keep is not None:
            # cut-cross-entropy requires low-precision activations. torch.compile
            # may otherwise retain the final RMSNorm result in float32.
            fused_dtype = (
                torch.get_autocast_dtype("cuda")
                if x.is_cuda and torch.is_autocast_enabled("cuda")
                else self.lm_head.weight.dtype
            )
            if fused_dtype not in (torch.float16, torch.bfloat16):
                fused_dtype = torch.bfloat16
            return {"hidden_states": x.to(fused_dtype)}
        logits = self.lm_head(x)
        # Soft-cap logits while retaining the model's low-precision compute dtype.
        logits = 30.0 * torch.tanh(logits / 30.0)
        return logits


def build_gpt2_model(
    *,
    vocab_size: int = 50304,
    n_positions: int = 2048,
    n_ctx: int | None = None,
    n_embd: int = 768,
    n_layer: int = 12,
    n_head: int = 12,
    bias: bool = False,
    dropout: float = 0.0,
    use_rope: bool = True,
    norm_type: str = "rmsnorm",
    mlp_type: str = "swiglu",
    bos_token_id: int = 50256,  # kept for API backward-compat
    eos_token_id: int = 50256,  # kept for API backward-compat
    attn_implementation: str = "sdpa",
    torch_dtype: torch.dtype | str | None = None,
) -> nn.Module:
    """Instantiate and return a high-performance modern nanoGPT language model.

    Uses PyTorch SDPA for causal attention. The separately installed
    ``flash-attn`` package is not used by this model implementation.

    Supports all nanoGPT enhancements:
    - vocab_size=50304 (Hardware Tensor-Core aligned)
    - bias=False (Zero-bias pure GEMM)
    - use_rope=True (Rotary Position Embeddings)
    - norm_type="rmsnorm" (RMSNorm for lower latency)
    - mlp_type="swiglu" (SwiGLU gated FFN)
    - dropout=0.0 (Pretraining efficiency)
    - Scaled residual initialization (0.02 / sqrt(2 * n_layer))
    """
    if n_ctx is not None and n_ctx != n_positions:
        n_positions = n_ctx
    if attn_implementation != "sdpa":
        raise ValueError(f"GPT2LMHeadModel supports only attn_implementation='sdpa', got {attn_implementation!r}.")

    model = GPT2LMHeadModel(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        bias=bias,
        dropout=dropout,
        use_rope=use_rope,
        norm_type=norm_type,
        mlp_type=mlp_type,
    )
    if torch_dtype is None:
        return model
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype, None)
    if torch_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype!r}")
    return model.to(dtype=torch_dtype)
