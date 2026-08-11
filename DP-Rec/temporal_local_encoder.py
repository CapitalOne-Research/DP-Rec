"""
TemporalLocalEncoder — local token-level transformer for DP-Rec.

Key design decisions:
  - No xformers: sliding-window causal attention via F.scaled_dot_product_attention
    with a precomputed boolean mask.
  - Cross-attention path is excluded when use_cross_attention=False (the default
    in DP-Rec). With use_cross_attention=True, a lightweight cross-attention block
    is appended after the final transformer layer.
  - Downsampling uses scatter_reduce with 'amax' (max-pooling over segment tokens).
  - Time-RoPE composes positional and temporal rotation matrices via einsum.
  - No apex FusedRMSNorm dependency — uses nn.LayerNorm throughout.
  - All shared primitives imported from ops.py.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import (
    _build_rope_matrix,
    _RopeMixin,
    _TransformerBlock,
)


# ---------------------------------------------------------------------------
# Optional cross-attention block (used when use_cross_attention=True)
# ---------------------------------------------------------------------------

class _CrossAttention(nn.Module):
    """
    Cross-attention for segment pooling: queries from segment embeddings,
    keys/values from token-level encoder hidden states.
    """

    def __init__(self, dim: int, n_heads: int = 1, norm_eps: float = 1e-5):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm_q = nn.RMSNorm(dim, eps=norm_eps)
        self.norm_kv = nn.RMSNorm(dim, eps=norm_eps)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        queries: torch.Tensor,    # (B, P, D) — segment embeddings
        keys_vals: torch.Tensor,  # (B, S, D) — token hidden states
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, P, D = queries.shape
        _, S, _ = keys_vals.shape

        q = self.wq(self.norm_q(queries)).view(B, P, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(self.norm_kv(keys_vals)).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(keys_vals).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, P, D)
        return queries + self.wo(out)


# ---------------------------------------------------------------------------
# Max-pooling downsampler
# ---------------------------------------------------------------------------

def _max_pool_to_segments(
    h: torch.Tensor,           # (B, S, D)
    segment_ids: torch.Tensor, # (B, S) long — which segment each token belongs to
    num_segments: int,
) -> torch.Tensor:
    """
    Aggregate token hidden states into segment hidden states via max-pooling.

    Uses scatter_reduce 'amax' — semantically equivalent to selecting the
    most activated token representation per segment.

    Returns: (B, num_segments, D)
    """
    B, S, D = h.shape
    idx = segment_ids.unsqueeze(-1).expand(-1, -1, D)          # (B, S, D)
    out = torch.zeros(B, num_segments, D, dtype=h.dtype, device=h.device)
    out.scatter_reduce_(1, idx, h, reduce="amax", include_self=False)
    return out


# ---------------------------------------------------------------------------
# TemporalLocalEncoder
# ---------------------------------------------------------------------------

class TemporalLocalEncoder(_RopeMixin):
    """
    Encodes a raw item token sequence into per-token hidden states and then
    downsamples them into per-segment embeddings.

    Architecture:
        item_embeddings  →  positional/temporal RoPE
        → N × _TransformerBlock (sliding-window causal)
        → max-pool to segments  (when use_cross_attention=False)
          OR
          cross-attention after final layer  (when use_cross_attention=True)

    Args:
        vocab_size:           Size of item vocabulary (item_num + 1).
        dim:                  Hidden dimension.
        n_layers:             Number of local transformer blocks.
        n_heads:              Number of attention heads.
        max_seqlen:           Maximum sequence length (for RoPE precomputation).
        attn_window:          Local attention window size (None = full causal).
        use_rope:             Whether to use rotary position embeddings.
        use_time_rope:        Whether to compose positional with temporal RoPE.
        time_rope_theta:      Theta for temporal RoPE (default 100.0).
        rope_theta:           Theta for standard positional RoPE (default 10000.0).
        dropout:              Dropout rate (applied after embedding lookup).
        norm_eps:             LayerNorm epsilon.
        use_cross_attention:  If True, use cross-attention pooling instead of max-pool.
        cross_attn_nheads:    Number of heads for cross-attention.
        cross_attn_init_by_pooling: If True (default), initialise segment queries from
                              max-pooled token hidden states before cross-attention.
                              If False, segment queries start as zeros.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        max_seqlen: int,
        attn_window: Optional[int],
        use_rope: bool = True,
        use_time_rope: bool = False,
        time_rope_theta: float = 100.0,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        use_cross_attention: bool = False,
        cross_attn_nheads: int = 1,
        cross_attn_init_by_pooling: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.attn_window = attn_window
        self.use_rope = use_rope
        self.use_time_rope = use_time_rope
        self.time_rope_theta = time_rope_theta
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.use_cross_attention = use_cross_attention
        self.cross_attn_init_by_pooling = cross_attn_init_by_pooling

        self.item_embeddings = nn.Embedding(vocab_size, dim, padding_idx=0)

        if use_rope:
            self.register_buffer(
                "rope_matrix",
                _build_rope_matrix(self.head_dim, max_seqlen, rope_theta),
                persistent=False,
            )
            self.pos_embeddings = None
        else:
            self.rope_matrix = None
            self.pos_embeddings = nn.Embedding(max_seqlen, dim)

        self.layers = nn.ModuleList(
            [_TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_layers)]
        )

        if use_cross_attention:
            self.cross_attn = _CrossAttention(dim, n_heads=cross_attn_nheads, norm_eps=norm_eps)
        else:
            self.cross_attn = None

    # ------------------------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor,                     # (B, S) long
        segment_ids: torch.Tensor,                 # (B, S) long — segment index per token
        num_segments: int,                         # total segments in batch
        time_seq: Optional[torch.Tensor] = None,  # (B, S) float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            token_hidden:   (B, S, D)  — per-token hidden states (used by decoder).
            segment_hidden: (B, P, D)  — per-segment pooled hidden states (input to global).
        """
        B, S = tokens.shape

        h = self.item_embeddings(tokens)

        if self.pos_embeddings is not None:
            positions = torch.arange(S, device=tokens.device).unsqueeze(0).expand(B, -1)
            h = h + self.pos_embeddings(positions)

        h = F.dropout(h, p=self.dropout, training=self.training)

        causal_mask = self._get_causal_mask(S, tokens.device)
        rope_matrix = self._get_rope(S, time_seq)

        for layer in self.layers:
            h = layer(h, rope_matrix, causal_mask)

        token_hidden = h  # (B, S, D) — preserved for local decoder residual

        if self.use_cross_attention:
            if self.cross_attn_init_by_pooling:
                segment_init = _max_pool_to_segments(h, segment_ids, num_segments)
            else:
                segment_init = torch.zeros(
                    h.shape[0], num_segments, h.shape[-1], dtype=h.dtype, device=h.device
                )
            segment_hidden = self.cross_attn(segment_init, h)
        else:
            segment_hidden = _max_pool_to_segments(h, segment_ids, num_segments)

        return token_hidden, segment_hidden
