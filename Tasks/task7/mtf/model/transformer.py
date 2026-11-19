"""Decoder-only transformer: Qwen2 / Qwen3 / Mistral, selected by flags in ModelArgs.

Module names match Hugging Face (minus the leading "model."), so HF checkpoints load with no key mapping.
Contract the parallelisms rely on -- keep it if you rewrite this file:
  * forward(tokens, positions) -> logits. RoPE is computed from `positions` on the fly (no buffers, so
    meta-device init just works, and context parallel can hand a rank any chunks of the sequence).
  * layers is a ModuleDict {"0": Block, ...}.
  * Attention calls self.attn_fn(q, k, v, positions); swap it to change attention (context parallel, kernels).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    vocab_size: int = 256
    dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2  # < n_heads: grouped-query attention
    head_dim: int = 0  # 0: dim // n_heads (Qwen3 sets it explicitly)
    ffn_dim: int = 256
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    qkv_bias: bool = False  # Qwen2
    qk_norm: bool = False  # Qwen3: RMSNorm on q and k per head, before RoPE
    tie_embeddings: bool = False  # output projection reuses the embedding matrix


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):  # computed in fp32 like HF, so bf16 runs match too
        x32 = x.float()
        return self.weight * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)


def rope_cos_sin(positions, head_dim: int, theta: float):
    """positions [B, S] -> cos, sin [B, S, head_dim] in HF's "rotate_half" layout (no weight permutation)."""
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim)
    freqs = positions[..., None].float() * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):  # x [B, H, S, D]
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos[:, None] + rotated * sin[:, None]).to(x.dtype)


def sdpa_causal(q, k, v, positions):
    # positions unused here: causal + contiguous sequence. Context parallel replaces this function.
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=q.shape[1] != k.shape[1])


class Attention(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.head_dim = a.head_dim or a.dim // a.n_heads
        self.q_proj = nn.Linear(a.dim, a.n_heads * self.head_dim, bias=a.qkv_bias)
        self.k_proj = nn.Linear(a.dim, a.n_kv_heads * self.head_dim, bias=a.qkv_bias)
        self.v_proj = nn.Linear(a.dim, a.n_kv_heads * self.head_dim, bias=a.qkv_bias)
        self.o_proj = nn.Linear(a.n_heads * self.head_dim, a.dim, bias=False)
        if a.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, a.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, a.norm_eps)
        self.qk_norm = a.qk_norm
        self.attn_fn = sdpa_causal

    def forward(self, x, cos, sin, positions):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, -1, self.head_dim)
        k = self.k_proj(x).view(b, s, -1, self.head_dim)
        v = self.v_proj(x).view(b, s, -1, self.head_dim)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = self.attn_fn(q, k, v, positions)
        return self.o_proj(out.transpose(1, 2).reshape(b, s, -1))


class MLP(nn.Module):  # SwiGLU
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(a.dim, a.ffn_dim, bias=False)
        self.up_proj = nn.Linear(a.dim, a.ffn_dim, bias=False)
        self.down_proj = nn.Linear(a.ffn_dim, a.dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.input_layernorm = RMSNorm(a.dim, a.norm_eps)
        self.self_attn = Attention(a)
        self.post_attention_layernorm = RMSNorm(a.dim, a.norm_eps)
        self.mlp = MLP(a)

    def forward(self, x, cos, sin, positions):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, positions)
        return x + self.mlp(self.post_attention_layernorm(x))


class Transformer(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.args = a
        self.embed_tokens = nn.Embedding(a.vocab_size, a.dim)
        self.layers = nn.ModuleDict({str(i): Block(a) for i in range(a.n_layers)})
        self.norm = RMSNorm(a.dim, a.norm_eps)
        self.lm_head = None if a.tie_embeddings else nn.Linear(a.dim, a.vocab_size, bias=False)

    def forward(self, tokens, positions=None):
        if positions is None:
            positions = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        h = self.embed_tokens(tokens)
        cos, sin = rope_cos_sin(positions, self.args.head_dim or self.args.dim // self.args.n_heads, self.args.rope_theta)
        for layer in self.layers.values():
            h = layer(h, cos, sin, positions)
        h = self.norm(h)
        return self.lm_head(h) if self.lm_head is not None else F.linear(h, self.embed_tokens.weight)

    @torch.no_grad()
    def init_weights(self, std: float = 0.02):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
            if isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)

    def flops_per_token(self, seq_len: int) -> float:
        """Training FLOPs per token: 6 * matmul params + causal attention scores (for MFU)."""
        a = self.args
        n = sum(p.numel() for p in self.parameters())
        if self.lm_head is not None:  # the embedding lookup is not a matmul (a tied one doubles as lm_head)
            n -= self.embed_tokens.weight.numel()
        return 6 * n + 6 * a.n_layers * a.n_heads * (a.head_dim or a.dim // a.n_heads) * seq_len
