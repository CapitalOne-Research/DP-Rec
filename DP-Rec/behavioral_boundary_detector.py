"""
BehavioralBoundaryDetector — entropy-based dynamic patching for DP-Rec.

Key design decisions:
  - Only entropy-based patching is supported (no BPE/space/static modes).
  - Entropy is computed via BPR CE-scoring (pos/neg dot-product), matching
    the recommendation training objective.
  - Threshold optimisation is fused into patch() — entropies are computed once.
  - Global threshold uses torch.kthvalue (O(N) on GPU) instead of a full sort.
  - Sliding-window causal attention via F.scaled_dot_product_attention with a
    precomputed boolean mask — no xformers dependency.
  - All padding tokens (item_id == 0) are zeroed out in entropy before
    patching so they always collapse into a single leading segment (segment 0).
"""

from __future__ import annotations

from contextlib import nullcontext
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
# BehavioralBoundaryModel — small LM that scores item transitions
# ---------------------------------------------------------------------------

class BehavioralBoundaryModel(_RopeMixin):
    """
    Small transformer that predicts whether a token position is a behavioral
    boundary (high-entropy transition in the user's item sequence).

    Used only as a frozen inference engine inside BehavioralBoundaryDetector;
    it is never trained inside this module — training happens separately via
    the same architecture loaded from a checkpoint.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        max_seqlen: int,
        attn_window: Optional[int],
        loss: str = "bpr",
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        use_time_rope: bool = False,
        time_rope_theta: float = 100.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_seqlen = max_seqlen
        self.attn_window = attn_window
        self.loss = loss
        self.use_time_rope = use_time_rope
        self.time_rope_theta = time_rope_theta

        self.item_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [_TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim, eps=norm_eps)

        output_dim = dim if loss == "bpr" else vocab_size
        self.output = nn.Linear(dim, output_dim, bias=True)

        self.register_buffer(
            "rope_matrix",
            _build_rope_matrix(self.head_dim, max_seqlen, rope_theta),
            persistent=False,
        )

    # ------------------------------------------------------------------

    def predict(
        self,
        user_ids,
        log_seqs,                   # (B, S) array/tensor — item ids
        item_indices,               # (C,) or (B, C) array/tensor — candidate items
        time_seq=None,
    ) -> torch.Tensor:
        """
        Inference: score candidate items against the last hidden state..py.
        Returns (B, C) logit scores.
        """
        tokens = torch.tensor(log_seqs, device=self.item_embeddings.weight.device).long()
        item_idx = torch.tensor(item_indices, device=tokens.device).long()
        ts = None
        if self.use_time_rope and time_seq is not None:
            ts = (torch.tensor(time_seq, device=tokens.device).float()
                  if not isinstance(time_seq, torch.Tensor)
                  else time_seq.to(tokens.device).float())

        B, S = tokens.shape
        h = self.item_embeddings(tokens)
        causal_mask = self._get_causal_mask(S, tokens.device)
        rope_matrix = self._get_rope(S, ts)
        for layer in self.layers:
            h = layer(h, rope_matrix, causal_mask)
        final_feat = self.output(self.norm(h))[:, -1, :]  # (B, D)

        item_embs = self.item_embeddings(item_idx)         # (C, D) or (B, C, D)
        if item_embs.ndim == 2:
            return final_feat.matmul(item_embs.t())        # (B, C)
        return (item_embs * final_feat.unsqueeze(1)).sum(-1)  # (B, C)

    # ------------------------------------------------------------------

    def forward(
        self,
        item_ids: torch.Tensor,                      # (B, S) long
        pos_seqs: Optional[torch.Tensor] = None,     # (B, S) long
        neg_seqs: Optional[torch.Tensor] = None,     # (B, S) long
        time_seq: Optional[torch.Tensor] = None,     # (B, S) float
    ):
        """
        BCE mode → returns logits (B, S, vocab_size)
        BPR mode → returns (pos_logits (B, S), neg_logits (B, S))
        """
        B, S = item_ids.shape
        h = self.item_embeddings(item_ids)

        causal_mask = self._get_causal_mask(S, item_ids.device)
        rope_matrix = self._get_rope(S, time_seq)

        for layer in self.layers:
            h = layer(h, rope_matrix, causal_mask)

        logits = self.output(self.norm(h))

        if self.loss == "bpr":
            assert pos_seqs is not None and neg_seqs is not None
            pos_embs = self.item_embeddings(pos_seqs)   # (B, S, D)
            neg_embs = self.item_embeddings(neg_seqs)
            return (logits * pos_embs).sum(-1), (logits * neg_embs).sum(-1)

        return logits


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) from raw logits. Returns (B, S) or (B*S,)."""
    log_p = F.log_softmax(logits.float(), dim=-1)
    p = log_p.exp()
    return -(p * log_p).sum(dim=-1)


def _compute_ce_entropies(
    item_ids: torch.Tensor,
    model: BehavioralBoundaryModel,
    pos_seqs: torch.Tensor,
    neg_seqs: torch.Tensor,
    time_seq: Optional[torch.Tensor] = None,
    enable_grad: bool = False,
) -> torch.Tensor:
    """
    Compute per-token entropy scores using BPR CE-scoring.

    The model outputs (pos_logit, neg_logit) per token. We treat these as a
    2-class distribution and compute entropy over it.

    Returns: (B, S) entropy tensor.
    """
    ctx = nullcontext() if enable_grad else torch.no_grad()
    with ctx:
        max_len = getattr(model, "max_length", model.max_seqlen)
        B, S = item_ids.shape

        if S <= max_len:
            pos_logits, neg_logits = model(
                item_ids=item_ids,
                pos_seqs=pos_seqs,
                neg_seqs=neg_seqs,
                time_seq=time_seq,
            )
            combined = torch.stack([pos_logits, neg_logits], dim=-1)  # (B, S, 2)
            return _entropy_from_logits(combined)

        # Chunked path: split along batch×seq flattened dimension
        batch_numel = max_len
        all_entropies = []

        flat_ids = item_ids.flatten()
        flat_pos = pos_seqs.flatten()
        flat_neg = neg_seqs.flatten()
        flat_ts = time_seq.flatten() if time_seq is not None else None

        for start in range(0, flat_ids.numel(), batch_numel):
            end = min(start + batch_numel, flat_ids.numel())
            chunk_len = end - start
            pad = (max_len - chunk_len % max_len) % max_len

            def _pad(t, s=start, e=end, p=pad):
                return torch.cat([t[s:e], t.new_zeros(p)])

            ids_c = _pad(flat_ids).view(-1, max_len)
            pos_c = _pad(flat_pos).view(-1, max_len)
            neg_c = _pad(flat_neg).view(-1, max_len)
            ts_c = _pad(flat_ts).view(-1, max_len) if flat_ts is not None else None

            pos_l, neg_l = model(item_ids=ids_c, pos_seqs=pos_c, neg_seqs=neg_c, time_seq=ts_c)
            combined = torch.stack([pos_l, neg_l], dim=-1)
            ent = _entropy_from_logits(combined).flatten()[:chunk_len]
            all_entropies.append(ent)

        return torch.cat(all_entropies).view(B, S)


# ---------------------------------------------------------------------------
# Global threshold optimisation (vectorised, GPU-friendly)
# ---------------------------------------------------------------------------

def _global_threshold(
    entropies: torch.Tensor,     # (B, S) — padding already zeroed
    real_mask: torch.Tensor,     # (B, S) bool — True for non-padding tokens
    target_segments: float,
) -> float:
    """
    Find scalar threshold T such that the average number of tokens with
    entropy > T (per non-padded sequence) ≈ target_segments - 1.

    Uses torch.kthvalue for O(N) selection on GPU instead of a full sort.
    """
    valid_ent = entropies[real_mask.bool()]
    num_valid = valid_ent.numel()
    batch_size = entropies.shape[0]

    if num_valid == 0:
        return 1.1

    avg_valid_len = num_valid / batch_size
    target = min(float(target_segments), avg_valid_len)
    total_target = int(round(target * batch_size))

    k = total_target - batch_size
    if k <= 0:
        return 1.1
    if k >= num_valid:
        return -0.1

    # kthvalue returns the k-th smallest; we want the k-th largest boundary
    rank = max(1, min(num_valid - k, num_valid))
    noisy = valid_ent + torch.rand_like(valid_ent) * 1e-9   # break ties
    return float(max(0.0, min(1.1, torch.kthvalue(noisy, rank).values.item())))


# ---------------------------------------------------------------------------
# Segment-id construction (shared by all patching strategies)
# ---------------------------------------------------------------------------

def build_segment_ids(
    tokens: torch.Tensor,      # (B, S) long — item ids (0 = padding)
    split_mask: torch.Tensor,  # (B, S) bool — proposed boundaries
) -> torch.Tensor:
    """
    Turn a boolean boundary mask into segment ids under the DP-Rec contract.

    All padding tokens (item id 0) collapse into segment 0; the first real token
    opens segment 1; each remaining True in split_mask (among real tokens) opens
    a new segment. Boundaries proposed inside padding are ignored. Returns (B, S).

    Shared by the entropy detector and the parameter-free fixed/random patchers
    so the padding/first-token invariants live in exactly one place.
    """
    B, S = tokens.shape
    real_mask = (tokens != 0)
    first_real = real_mask.int().argmax(dim=1).clamp(min=0)          # (B,)
    positions = torch.arange(S, device=tokens.device).unsqueeze(0)   # (1, S)
    in_padding = positions < first_real.unsqueeze(1)                 # (B, S)

    split = split_mask.clone()
    split[in_padding] = False                                        # no splits in padding
    split[:, 0] = True                                               # segment 0 starts at pos 0
    split.scatter_(1, first_real.unsqueeze(1), True)                 # first real token → seg 1
    return split.int().cumsum(dim=1) - 1                             # 0-indexed segment ids


# ---------------------------------------------------------------------------
# BehavioralBoundaryDetector
# ---------------------------------------------------------------------------

class BehavioralBoundaryDetector:
    """
    Detects behavioral boundaries in item interaction sequences and segments
    them into variable-length patches.

    Args:
        model: A BehavioralBoundaryModel (frozen, eval mode).
    """

    def __init__(self, model: BehavioralBoundaryModel):
        self.model = model

    # ------------------------------------------------------------------

    def patch(
        self,
        tokens: torch.Tensor,                   # (B, S) long  — item ids
        pos_seqs: torch.Tensor,                  # (B, S) long
        neg_seqs: torch.Tensor,                  # (B, S) long
        time_seq: Optional[torch.Tensor] = None, # (B, S) float
        expected_patches: Optional[int] = None,  # target segments per sequence
        threshold: Optional[float] = None,       # explicit threshold (overrides expected_patches)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute segment ids for a batch of sequences.

        Returns:
            segment_ids: (B, S) long — 0-indexed segment index per token.
                         Segment 0 is reserved for all padding tokens.
                         Real tokens start at segment 1.
            entropies:   (B, S) float — per-token entropy scores.
        """
        B, S = tokens.shape

        # 1. Compute entropies via a forward pass through BehavioralBoundaryModel
        entropies = _compute_ce_entropies(tokens, self.model, pos_seqs, neg_seqs, time_seq)

        # 2. Zero padding positions so they never trigger boundaries
        scores = entropies.clone()
        real_mask = (tokens != 0)                                  # (B, S) bool
        scores[~real_mask] = 0.0

        # 3. Determine threshold
        if threshold is None:
            if expected_patches is not None:
                threshold = _global_threshold(scores, real_mask, expected_patches)
            else:
                raise ValueError(
                    "BehavioralBoundaryDetector.patch() requires either "
                    "'expected_patches' or an explicit 'threshold'."
                )

        # 4. Build split mask (threshold-driven), then derive segment ids.
        #    build_segment_ids handles padding → seg 0 and first real → seg 1.
        split_mask = scores > threshold
        segment_ids = build_segment_ids(tokens, split_mask)

        return segment_ids, entropies

    # ------------------------------------------------------------------
    # Factory method: load from an existing checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        device: str = "cpu",
    ) -> "BehavioralBoundaryDetector":
        """
        Build a BehavioralBoundaryDetector by loading an existing
        checkpoint (params.json + state_dict.pth).
        """
        import json, os
        with open(os.path.join(checkpoint_dir, "params.json")) as f:
            params = json.load(f)["entropy_model"]

        model = BehavioralBoundaryModel(
            vocab_size=params["vocab_size"],
            dim=params["dim"],
            n_layers=params["n_layers"],
            n_heads=params["n_heads"],
            max_seqlen=params["max_seqlen"],
            attn_window=params.get("attn_window"),
            loss=params.get("loss", "bpr"),
            use_time_rope=params.get("use_time_rope", False),
        )

        state = torch.load(
            os.path.join(checkpoint_dir, "state_dict.pth"),
            weights_only=False,
            map_location=device,
        )
        model.load_state_dict(state, strict=False)
        model.max_length = params["max_seqlen"]
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        return cls(model)
