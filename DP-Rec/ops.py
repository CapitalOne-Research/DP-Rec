"""
ops.py — shared primitives for all DP-Rec modules.

Provides:
  _build_rope_matrix       — standard positional RoPE
  _build_time_rope_matrix  — timestamp-based temporal RoPE
  _apply_rope              — apply rotation matrices to Q, K
  _make_causal_mask        — sliding-window causal boolean mask
  _RopeMixin               — cached mask + rope helpers for transformer modules
  _FFN                     — two-layer MLP
  _SelfAttention           — multi-head self-attention with RoPE
  _TransformerBlock        — pre-norm residual block
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotary position embedding
# ---------------------------------------------------------------------------

def _build_rope_matrix(dim: int, length: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute (length, dim//2, 2, 2) rotation matrices for standard positional RoPE.

    Each position gets a set of 2×2 rotation matrices — one per head-dimension pair.
    The matrices are [[cos, -sin], [sin, cos]] at frequency f_k for pair k.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(length, dtype=torch.float32)
    outer = torch.outer(t, freqs)                      # (length, dim//2)
    cos, sin = outer.cos(), outer.sin()
    return torch.stack([cos, -sin, sin, cos], dim=-1).view(length, dim // 2, 2, 2)


def _build_time_rope_matrix(
    dim: int,
    timestamps: torch.Tensor,   # (B, S) float — raw Unix-style timestamps
    theta: float = 100.0,
) -> torch.Tensor:
    """
    Compute (B, S, dim//2, 2, 2) rotation matrices from raw timestamps.

    log1p is applied to compress large epoch values before computing frequencies.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=timestamps.device).float() / dim))
    t = torch.log1p(timestamps.float())                # compress large epoch values
    outer = torch.einsum("bs,d->bsd", t, freqs)        # (B, S, dim//2)
    cos, sin = outer.cos(), outer.sin()
    return torch.stack([cos, -sin, sin, cos], dim=-1).view(*outer.shape, 2, 2)


def _apply_rope(
    xq: torch.Tensor,           # (B, S, H, head_dim)
    xk: torch.Tensor,           # (B, S, H, head_dim)
    rope_matrix: torch.Tensor,  # (S, D/2, 2, 2)  or  (B, S, D/2, 2, 2)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotation matrices to query and key tensors.

    Handles both standard positional RoPE (4-D matrix) and batched temporal
    RoPE (5-D matrix) via ndim dispatch.
    """
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)   # (B, S, H, D/2, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    if rope_matrix.ndim == 5:
        # Batched temporal rope (B, S, D/2, 2, 2) — insert head dim
        rm = rope_matrix.unsqueeze(2).float()      # (B, S, 1, D/2, 2, 2)
    else:
        # Standard positional rope (S, D/2, 2, 2) — insert batch + head dims
        rm = rope_matrix.unsqueeze(0).unsqueeze(2).float()  # (1, S, 1, D/2, 2, 2)
    xq_out = (xq_ * rm).sum(-1).flatten(3)
    xk_out = (xk_ * rm).sum(-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)



# ---------------------------------------------------------------------------
# Causal attention mask
# ---------------------------------------------------------------------------

def _make_causal_mask(
    seqlen: int,
    attn_window: Optional[int],
    device,
) -> torch.Tensor:
    """
    Build a (seqlen, seqlen) boolean causal mask.
    True = attend, False = masked out.
    When attn_window is None, this is a standard full lower-triangular mask.
    """
    i = torch.arange(seqlen, device=device).unsqueeze(1)  # (S, 1)
    j = torch.arange(seqlen, device=device).unsqueeze(0)  # (1, S)
    mask = i >= j
    if attn_window is not None:
        mask = mask & (i - j < attn_window)
    return mask


# ---------------------------------------------------------------------------
# Mixin: causal-mask + RoPE caching shared by all transformer modules
# ---------------------------------------------------------------------------

class _RopeMixin(nn.Module):
    """
    Mixin for transformer modules that need a cached causal mask and RoPE.

    Subclasses must set before calling these helpers:
        self.attn_window      : Optional[int]
        self.use_time_rope    : bool
        self.time_rope_theta  : float
        self.head_dim         : int
        self.rope_matrix      : Optional[Tensor]  (registered buffer or None)
    """

    def _get_causal_mask(self, seqlen: int, device) -> Optional[torch.Tensor]:
        if getattr(self, "_causal_mask_len", -1) != seqlen or (
            getattr(self, "_causal_mask", None) is not None
            and self._causal_mask.device != device
        ):
            self._causal_mask = _make_causal_mask(seqlen, self.attn_window, device)
            self._causal_mask_len = seqlen
        return self._causal_mask

    def _get_rope(
        self, seqlen: int, time_seq: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        if self.rope_matrix is None:
            return None
        pos_rm = self.rope_matrix[:seqlen]
        if self.use_time_rope and time_seq is not None:
            time_rm = _build_time_rope_matrix(self.head_dim, time_seq, self.time_rope_theta)
            pos_b = pos_rm.unsqueeze(0).expand(time_rm.shape[0], -1, -1, -1, -1)
            return torch.einsum("...ij,...jk->...ik", pos_b, time_rm)
        return pos_rm


# ---------------------------------------------------------------------------
# Feed-forward network
# ---------------------------------------------------------------------------

class _FFN(nn.Module):
    """Two-layer MLP (no activation — matches the simplified FFN in the base transformer)."""

    def __init__(self, dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, dim, bias=True)
        self.w2 = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)))


# ---------------------------------------------------------------------------
# Self-attention (multi-head, with RoPE support)
# ---------------------------------------------------------------------------

class _SelfAttention(nn.Module):
    """Multi-head self-attention using F.scaled_dot_product_attention (no xformers)."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=True)
        self.wk = nn.Linear(dim, dim, bias=True)
        self.wv = nn.Linear(dim, dim, bias=True)
        self.wo = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,                           # (B, S, D)
        rope_matrix: Optional[torch.Tensor],
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, D = x.shape
        xq = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, S, self.n_heads, self.head_dim)
        xv = self.wv(x).view(B, S, self.n_heads, self.head_dim)

        if rope_matrix is not None:
            xq, xk = _apply_rope(xq, xk, rope_matrix)

        # SDPA expects (B, H, S, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=causal_mask, is_causal=False)
        return self.wo(out.transpose(1, 2).contiguous().view(B, S, D))


# ---------------------------------------------------------------------------
# Transformer block (pre-norm residual)
# ---------------------------------------------------------------------------

class _TransformerBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → SelfAttention + FFN."""

    def __init__(self, dim: int, n_heads: int, norm_eps: float = 1e-5):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = _SelfAttention(dim, n_heads)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn = _FFN(dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_matrix: Optional[torch.Tensor],
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_matrix, causal_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
