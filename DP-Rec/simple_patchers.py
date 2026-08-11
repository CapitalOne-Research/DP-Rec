"""
Parameter-free patchers for DP-Rec: fixed-count and random boundaries.

These skip the entropy BehavioralBoundaryModel entirely — no checkpoint, no
patcher_model_path. Both honor `expected_patches` as a target patch *count* per
sequence, so they are directly comparable to the entropy patcher and share its
FLOPs accounting (only the entropy term drops).

Each exposes the same patch() signature as BehavioralBoundaryDetector, returns
(segment_ids, None), and reuses build_segment_ids so all padding semantics are
identical (padding → segment 0, first real token → segment 1). Boundaries are
computed in real-token rank space, so left-padding is handled automatically.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .behavioral_boundary_detector import build_segment_ids


class FixedCountPatcher:
    """Slice each sequence's real tokens into ~expected_patches equal patches.

    For real length L and target P, token at real-rank r joins patch
    floor(r * min(P, L) / L); a boundary opens wherever that index increments.
    This yields exactly min(P, L) near-equal patches per sequence.
    """

    def patch(
        self,
        tokens: torch.Tensor,                         # (B, S) long
        pos_seqs: Optional[torch.Tensor] = None,
        neg_seqs: Optional[torch.Tensor] = None,
        time_seq: Optional[torch.Tensor] = None,
        expected_patches: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, None]:
        if expected_patches is None:
            raise ValueError("FixedCountPatcher requires expected_patches.")

        real_mask = (tokens != 0)
        L = real_mask.sum(dim=1, keepdim=True).clamp(min=1)           # (B,1) real length
        P = torch.full_like(L, int(expected_patches)).clamp(min=1)
        P = torch.minimum(P, L)                                       # cap at real length
        rank = real_mask.int().cumsum(dim=1) - 1                      # (B,S) real-token rank
        group = torch.div(rank.long() * P, L, rounding_mode="floor")  # (B,S) patch index
        prev = F.pad(group[:, :-1], (1, 0), value=-1)
        split_mask = real_mask & (group != prev)
        return build_segment_ids(tokens, split_mask), None


class RandomPatcher:
    """Place expected_patches-1 random boundaries among each sequence's real tokens.

    Deterministic per sequence: the RNG is seeded from the sequence content, so a
    given user history always segments the same way (stable eval) while different
    histories segment differently.
    """

    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    def patch(
        self,
        tokens: torch.Tensor,                         # (B, S) long
        pos_seqs: Optional[torch.Tensor] = None,
        neg_seqs: Optional[torch.Tensor] = None,
        time_seq: Optional[torch.Tensor] = None,
        expected_patches: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, None]:
        if expected_patches is None:
            raise ValueError("RandomPatcher requires expected_patches.")

        B, S = tokens.shape
        real_mask = (tokens != 0)
        first_real = real_mask.int().argmax(dim=1)                   # (B,)
        split_mask = torch.zeros_like(real_mask)

        # Pull the few per-row scalars to CPU once to avoid per-element syncs.
        weights = torch.arange(1, S + 1, device=tokens.device)
        L_list = real_mask.sum(dim=1).tolist()
        first_real_list = first_real.tolist()
        keys = (tokens.long() * weights).sum(dim=1).tolist()          # order-sensitive content key

        for b in range(B):
            Lb = L_list[b]
            if Lb <= 1:
                continue
            n_cuts = min(int(expected_patches), Lb) - 1
            if n_cuts <= 0:
                continue
            gen = torch.Generator()
            gen.manual_seed((self.seed * 1000003 + keys[b]) & 0x7FFFFFFF)
            # Candidate ranks 1..Lb-1 (rank 0 = first real token = auto boundary).
            ranks = torch.randperm(Lb - 1, generator=gen)[:n_cuts] + 1
            positions = (ranks + first_real_list[b]).to(tokens.device)
            split_mask[b, positions] = True

        return build_segment_ids(tokens, split_mask), None
