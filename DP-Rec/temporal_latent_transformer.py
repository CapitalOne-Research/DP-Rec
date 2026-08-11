"""
TemporalLatentTransformer — segment-level global transformer for DP-Rec.

Key design decisions:
  - No xformers: full causal attention via F.scaled_dot_product_attention with
    a standard lower-triangular boolean mask (no sliding window at global level).
  - Time-RoPE at the segment level: uses the last timestamp of each segment,
    propagated from the caller (DPREC model).
  - Input projection from encoder dim → global dim is handled here when dims differ.
  - No apex FusedRMSNorm — uses nn.LayerNorm.
  - Global transformer operates over segment sequences (no sliding window).
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
# TemporalLatentTransformer — public module
# ---------------------------------------------------------------------------

class TemporalLatentTransformer(_RopeMixin):
    """
    Global (segment-level) transformer that models behavioural patterns across
    an entire session by attending over the compressed segment sequence.

    Architecture:
        [optional input projection:  D_enc → D_global]
        → N × _TransformerBlock  (full causal, no sliding window)
        → returns (B, P, D_global) segment hidden states

    RoPE is applied using segment-level positions. If use_time_rope is True,
    temporal RoPE is additionally composed using the last timestamp of each
    segment (passed in by the caller).

    Args:
        dim_in:          Dimension of incoming segment embeddings (encoder dim).
        dim:             Hidden dimension of the global transformer.
        n_layers:        Number of global transformer blocks.
        n_heads:         Number of attention heads.
        max_segments:    Maximum number of segments (for RoPE precomputation).
                         Typically set to max_seqlen (upper bound).
        use_rope:        Whether to use RoPE for segment positions.
        use_time_rope:   Whether to compose positional with temporal RoPE.
        time_rope_theta: Theta for temporal RoPE (default 100.0).
        rope_theta:      Theta for standard positional RoPE (default 10000.0).
        dropout:         Dropout applied to segment embeddings before layers.
        norm_eps:        LayerNorm epsilon.
    """

    def __init__(
        self,
        dim_in: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        max_segments: int,
        use_rope: bool = True,
        use_time_rope: bool = False,
        time_rope_theta: float = 100.0,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.dim_in = dim_in
        self.use_rope = use_rope
        self.use_time_rope = use_time_rope
        self.time_rope_theta = time_rope_theta
        self.dropout = dropout
        self.head_dim = dim // n_heads

        if dim_in != dim:
            self.input_proj = nn.Linear(dim_in, dim, bias=False)
        else:
            self.input_proj = None

        self.layers = nn.ModuleList(
            [_TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_layers)]
        )

        if use_rope:
            self.register_buffer(
                "rope_matrix",
                _build_rope_matrix(self.head_dim, max_segments, rope_theta),
                persistent=False,
            )
        else:
            self.rope_matrix = None
        self.attn_window = None   # global transformer uses full causal attention

    # ------------------------------------------------------------------

    def forward(
        self,
        segment_embeds: torch.Tensor,                       # (B, P, D_enc)
        time_seq_segments: Optional[torch.Tensor] = None,  # (B, P) float timestamps
    ) -> torch.Tensor:
        """
        Args:
            segment_embeds:      Per-segment embeddings from TemporalLocalEncoder.
            time_seq_segments:   Timestamp of the last token in each segment.
                                 Only used when use_time_rope=True.

        Returns:
            (B, P, D_global) — contextualised segment representations.
        """
        B, P, _ = segment_embeds.shape

        h = segment_embeds
        if self.input_proj is not None:
            h = self.input_proj(h)

        h = F.dropout(h, p=self.dropout, training=self.training)

        causal_mask = self._get_causal_mask(P, h.device)
        rope_matrix = self._get_rope(P, time_seq_segments)

        for layer in self.layers:
            h = layer(h, rope_matrix, causal_mask)

        return h
