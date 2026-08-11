"""
LocalDecoder — local token-level decoder for DP-Rec.

Key design decisions:
  - sliding-window causal attention via F.scaled_dot_product_attention
    with a precomputed boolean mask.
  - Cross-attention path is excluded when use_cross_attention=False (default).
    When disabled, global segment embeddings are gathered and added directly to
    token embeddings before the transformer blocks.
  - When use_cross_attention=True, per-layer cross-attention injects global
    context into the token-level representations.
  - Output head: Linear → vocab for BCE loss, identity (dim passthrough) for BPR.
  - No apex FusedRMSNorm — uses nn.LayerNorm.
  - No positional embeddings in the decoder (tokens receive their positions
    implicitly through the encoder residual + segment context).
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
# Optional per-layer cross-attention (used when use_cross_attention=True)
# ---------------------------------------------------------------------------

class _ContextAttention(nn.Module):
    """
    Per-decoder-layer cross-attention: token hidden states (queries) attend to
    global segment embeddings (keys + values).

    When cross_attn_k > 1, each segment's global vector is projected and split
    into k latent vectors before being used as keys/values, giving the decoder
    k times more capacity to read from each segment.
    """

    def __init__(self, dim: int, dim_global: int, n_heads: int = 1, norm_eps: float = 1e-5, cross_attn_k: int = 1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.cross_attn_k = cross_attn_k
        self.norm_q = nn.RMSNorm(dim, eps=norm_eps)
        self.norm_kv = nn.RMSNorm(dim_global, eps=norm_eps)
        self.wq = nn.Linear(dim, dim, bias=False)
        # Project each segment to k * dim so it can be reshaped to (P*k, dim)
        self.wk = nn.Linear(dim_global, dim * cross_attn_k, bias=False)
        self.wv = nn.Linear(dim_global, dim * cross_attn_k, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        token_hidden: torch.Tensor,   # (B, S, D_dec)   — token hidden states (queries)
        global_hidden: torch.Tensor,  # (B, P, D_global) — global outputs (keys/values)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, D = token_hidden.shape
        _, P, _ = global_hidden.shape
        k = self.cross_attn_k

        q = self.wq(self.norm_q(token_hidden)).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Project then expand: (B, P, D*k) → (B, P*k, D)
        kv_norm = self.norm_kv(global_hidden)
        kv_k = self.wk(kv_norm).view(B, P * k, D)
        kv_v = self.wv(kv_norm).view(B, P * k, D)
        kv_k = kv_k.view(B, P * k, self.n_heads, self.head_dim).transpose(1, 2)
        kv_v = kv_v.view(B, P * k, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, kv_k, kv_v, attn_mask=mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return token_hidden + self.wo(out)


class LocalDecoder(_RopeMixin):
    """
    Decodes global segment context back to token-level predictions.

    Architecture (when use_cross_attention=False — default):
        token_hidden (encoder token hidden states)
        + gathered global segment embedding for each token
        → N × _TransformerBlock (sliding-window causal)
        → LayerNorm → output head

    Architecture (when use_cross_attention=True):
        token_hidden
        → at each block i: _ContextAttention(h, global_hidden)
        → self-attention
        → LayerNorm → output head

    Output:
        BCE loss:  (B, S, vocab_size) logits
        BPR loss:  (B, S, dim) embeddings (dot-producted with item embeddings by caller)

    Args:
        dim:                  Hidden dimension of the decoder.
        dim_global:           Hidden dimension of the global transformer output.
        n_layers:             Number of local transformer blocks.
        n_heads:              Number of attention heads.
        vocab_size:           Item vocabulary size (for BCE output head).
        max_seqlen:           Maximum sequence length (for RoPE precomputation).
        attn_window:          Local attention window size.
        loss:                 'bce' or 'bpr'.
        use_rope:             Whether to use RoPE.
        rope_theta:           Theta for positional RoPE.
        dropout:              Dropout rate.
        norm_eps:             LayerNorm epsilon.
        use_cross_attention:  If True, use per-layer cross-attention from global.
        cross_attn_nheads:    Number of heads for cross-attention.
        cross_attn_k:         Number of latent vectors per segment in cross-attention.
                              When > 1, each segment's global vector is projected to
                              k vectors (B, P*k, D) before being used as keys/values.
    """

    def __init__(
        self,
        dim: int,
        dim_global: int,
        n_layers: int,
        n_heads: int,
        vocab_size: int,
        max_seqlen: int,
        attn_window: Optional[int],
        loss: str = "bpr",
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        use_cross_attention: bool = False,
        cross_attn_nheads: int = 1,
        cross_attn_k: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.dim_global = dim_global
        self.attn_window = attn_window
        self.use_rope = use_rope
        self.dropout = dropout
        self.loss = loss
        self.use_cross_attention = use_cross_attention
        self.head_dim = dim // n_heads

        # Project global dim → decoder dim when they differ (for gather-add path)
        if not use_cross_attention and dim_global != dim:
            self.global_proj = nn.Linear(dim_global, dim, bias=False)
        else:
            self.global_proj = None

        self.layers = nn.ModuleList(
            [_TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_layers)]
        )

        if use_cross_attention:
            self.context_attn_layers = nn.ModuleList(
                [_ContextAttention(dim, dim_global, cross_attn_nheads, norm_eps, cross_attn_k)
                 for _ in range(n_layers)]
            )
        else:
            self.context_attn_layers = None

        self.norm = nn.LayerNorm(dim, eps=norm_eps)

        if loss == "bce":
            self.output = nn.Linear(dim, vocab_size, bias=True)
        else:
            self.output = None  # BPR: caller uses item_embeddings dot product

        if use_rope:
            self.register_buffer(
                "rope_matrix",
                _build_rope_matrix(self.head_dim, max_seqlen, rope_theta),
                persistent=False,
            )
        else:
            self.rope_matrix = None
        self.use_time_rope = False   # decoder uses positional RoPE only

    # ------------------------------------------------------------------

    def forward(
        self,
        token_hidden: torch.Tensor,       # (B, S, D_dec) — encoder token hidden states
        global_hidden: torch.Tensor,      # (B, P, D_global) — global transformer output
        ctx_segment_ids: torch.Tensor,    # (B, S) long — which segment each token reads from
    ) -> torch.Tensor:
        """
        Args:
            token_hidden:     Per-token encoder hidden states (residual stream).
            global_hidden:    Per-segment global transformer outputs.
            ctx_segment_ids:  For each decoder token position, the index into
                              global_hidden. Shape (B, S).
                              Typically ctx_segment_ids = (segment_ids - 1).clamp(min=0),
                              so each token reads the preceding segment's global context.

        Returns:
            (B, S, vocab_size)  for BCE loss.
            (B, S, dim)         for BPR loss (item embedding dot-product in caller).
        """
        B, S, _ = token_hidden.shape

        h = token_hidden

        if not self.use_cross_attention:
            # Gather the global embedding for each token's segment and add to h
            gathered = torch.gather(
                global_hidden,
                dim=1,
                index=ctx_segment_ids.unsqueeze(-1).expand(-1, -1, global_hidden.shape[-1]),
            )
            if self.global_proj is not None:
                gathered = self.global_proj(gathered)
            h = h + gathered

        h = F.dropout(h, p=self.dropout, training=self.training)

        causal_mask = self._get_causal_mask(S, h.device)
        rope_matrix = self._get_rope(S)

        for i, layer in enumerate(self.layers):
            if self.use_cross_attention:
                h = self.context_attn_layers[i](h, global_hidden)
            h = layer(h, rope_matrix, causal_mask)

        h = self.norm(h)

        if self.loss == "bce":
            h = F.dropout(h, p=self.dropout, training=self.training)
            return self.output(h).float()   # (B, S, vocab_size)

        return h.float()                    # (B, S, dim) for BPR
