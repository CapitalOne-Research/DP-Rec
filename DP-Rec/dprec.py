"""
DPREC — top-level model that wires together the four DP-Rec components.

This module contains:
  - segment_time_seq()  — extract last timestamp per segment
  - DPREC               — the full recommendation model

Usage:
    model = DPREC(user_num, item_num, args)
    pos_logits, neg_logits = model(user_ids, log_seqs, pos_seqs, neg_seqs, time_seq=ts)
    predictions = model.predict(user_ids, log_seqs, item_indices, pos_seqs, neg_seqs, time_seq=ts)
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .behavioral_boundary_detector import BehavioralBoundaryDetector, BehavioralBoundaryModel
from .simple_patchers import FixedCountPatcher, RandomPatcher
from .temporal_local_encoder import TemporalLocalEncoder
from .temporal_latent_transformer import TemporalLatentTransformer
from .local_decoder import LocalDecoder


# ---------------------------------------------------------------------------
# segment_time_seq
# ---------------------------------------------------------------------------

def segment_time_seq(
    time_seq: torch.Tensor,      # (B, S) float
    segment_ids: torch.Tensor,   # (B, S) long
    num_segments: int,
) -> torch.Tensor:
    """
    For each segment, extract the timestamp of its last token.

    Uses scatter_reduce 'amax' — timestamps are non-decreasing within a
    sequence, so the max timestamp in a segment is its last token's timestamp.

    Returns (B, P) float.
    """
    B, S = time_seq.shape
    out = torch.zeros(B, num_segments, dtype=time_seq.dtype, device=time_seq.device)
    out.scatter_reduce_(1, segment_ids, time_seq, reduce="amax", include_self=False)
    return out


# ---------------------------------------------------------------------------
# DPREC — full model
# ---------------------------------------------------------------------------

class DPREC(nn.Module):
    """
    Dynamic-Patch Recommendation model (DP-Rec).

    Integrates BehavioralBoundaryDetector, TemporalLocalEncoder,
    TemporalLatentTransformer, and LocalDecoder into a single
    trainable module that accepts item interaction sequences and produces
    next-item recommendations.
    """

    def __init__(self, user_num: int, item_num: int, args):
        super().__init__()

        self.dev = args.training_args.device
        self.loss = args.training_args.loss
        self.use_time_rope = getattr(args.model_args, "use_time_rope", False)
        self.expected_patches = args.training_args.get("expected_patches", None)

        vocab_size = item_num + 1
        maxlen = args.model_args.maxlen
        enc_cfg = args.model_args.local_encoder
        dec_cfg = args.model_args.local_decoder
        attn_window = args.model_args.sliding_window
        use_rope = args.model_args.use_rope
        use_cross_attn = getattr(args.model_args, "use_cross_attention", False)
        cross_attn_k = getattr(args.model_args, "cross_attention_k", 1)
        cross_attn_init_by_pooling = getattr(args.model_args, "cross_attn_init_by_pooling", True)
        dropout = args.model_args.dropout_rate

        dim_enc = enc_cfg.hidden_units
        dim_global = args.model_args.hidden_units
        dim_dec = dec_cfg.hidden_units

        # ── 1. Patching strategy ─────────────────────────────────────────
        # entropy → learned BehavioralBoundaryModel (needs patcher_model_path).
        # fixed/random → parameter-free, no checkpoint; both use expected_patches
        # as a target patch count.
        strategy = getattr(args.model_args, "patching_strategy", "entropy")
        if strategy in ("fixed", "random") and self.expected_patches is None:
            raise ValueError(
                f"patching_strategy='{strategy}' requires training_args.expected_patches."
            )
        if strategy == "entropy":
            print("\nLoading BehavioralBoundaryDetector (entropy)....")
            boundary_model = self._load_boundary_model(vocab_size, args)
            self.detector = BehavioralBoundaryDetector(boundary_model)
        elif strategy == "fixed":
            print("\nUsing FixedCountPatcher (no patcher model)....")
            self.detector = FixedCountPatcher()
        elif strategy == "random":
            seed = int(getattr(args.model_args, "patch_seed", 42))
            print(f"\nUsing RandomPatcher (seed={seed}, no patcher model)....")
            self.detector = RandomPatcher(seed=seed)
        else:
            raise ValueError(f"Unknown patching_strategy: {strategy}")

        # ── 2. Temporal Local Encoder ─────────────────────────────────────
        print("\nCreating TemporalLocalEncoder....")
        self.encoder = TemporalLocalEncoder(
            vocab_size=vocab_size,
            dim=dim_enc,
            n_layers=enc_cfg.num_blocks,
            n_heads=enc_cfg.num_heads,
            max_seqlen=enc_cfg.maxlen,
            attn_window=attn_window,
            use_rope=use_rope,
            use_time_rope=self.use_time_rope,
            dropout=dropout,
            use_cross_attention=use_cross_attn,
            cross_attn_nheads=1,
            cross_attn_init_by_pooling=cross_attn_init_by_pooling,
        ).to(self.dev)

        # ── 3. Temporal Latent Transformer (Global) ────────────────────────
        print("\nCreating TemporalLatentTransformer....")
        self.global_transformer = TemporalLatentTransformer(
            dim_in=dim_enc,
            dim=dim_global,
            n_layers=args.model_args.num_blocks,
            n_heads=args.model_args.num_heads,
            max_segments=maxlen,
            use_rope=use_rope,
            use_time_rope=self.use_time_rope,
            dropout=dropout,
        ).to(self.dev)

        # ── 4. Temporal Local Decoder ─────────────────────────────────────
        print("\nCreating LocalDecoder....")
        self.decoder = LocalDecoder(
            dim=dim_dec,
            dim_global=dim_global,
            n_layers=dec_cfg.num_blocks,
            n_heads=dec_cfg.num_heads,
            vocab_size=vocab_size,
            max_seqlen=dec_cfg.maxlen,
            attn_window=attn_window,
            loss=self.loss,
            use_rope=use_rope,
            dropout=dropout,
            use_cross_attention=use_cross_attn,
            cross_attn_nheads=1,
            cross_attn_k=cross_attn_k,
        ).to(self.dev)

        self._print_param_stats()

    # ------------------------------------------------------------------
    # Weight loading helper
    # ------------------------------------------------------------------

    def _load_boundary_model(
        self, vocab_size: int, args
    ) -> BehavioralBoundaryModel:
        """
        Build and load a BehavioralBoundaryModel from the patcher checkpoint.

        Architecture is read from params.json inside the checkpoint directory
        so it is independent of the encoder/decoder/global-transformer config.
        """
        import json

        ckpt_path = args.patcher_model_path
        params_path = os.path.join(ckpt_path, "params.json")
        state_path = os.path.join(ckpt_path, "state_dict.pth")

        # Read boundary model architecture from checkpoint metadata
        if not os.path.exists(params_path):
            raise FileNotFoundError(
                f"[DPREC] params.json not found at {params_path}. "
                "The patcher checkpoint must contain params.json with the "
                "boundary model architecture."
            )

        with open(params_path) as f:
            raw = json.load(f)
        p = raw.get("entropy_model", raw)  # support both wrapped and flat formats
        model = BehavioralBoundaryModel(
            vocab_size=vocab_size,
            dim=p["dim"],
            n_layers=p["n_layers"],
            n_heads=p["n_heads"],
            max_seqlen=p["max_seqlen"],
            attn_window=p.get("attn_window", p.get("sliding_window")),
            loss=p.get("loss", "bpr"),
            use_time_rope=p.get("use_time_rope", False),
            time_rope_theta=p.get("time_rope_theta", 100.0),
        )

        if not os.path.exists(state_path):
            raise FileNotFoundError(
                f"[DPREC] state_dict.pth not found at {state_path}."
            )

        state = torch.load(state_path, weights_only=False, map_location=self.dev)
        # Older checkpoints use 'tok_embeddings'; remap to 'item_embeddings'.
        state = {k.replace("tok_embeddings", "item_embeddings"): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.max_length = model.max_seqlen
        model.to(self.dev).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"[DPREC] BehavioralBoundaryModel loaded: {n_params:,} params (frozen)")
        return model

    # ------------------------------------------------------------------
    # Parameter statistics
    # ------------------------------------------------------------------

    def _print_param_stats(self) -> None:
        def count_all(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        def count_no_emb(m):
            emb_params = {p for mod in m.modules() if isinstance(mod, nn.Embedding)
                          for p in mod.parameters()}
            return sum(p.numel() for p in m.parameters()
                       if p.requires_grad and p not in emb_params)
        def count_emb(m):
            return sum(p.numel() for mod in m.modules() if isinstance(mod, nn.Embedding)
                       for p in mod.parameters() if p.requires_grad)

        enc_emb   = count_emb(self.encoder)
        enc       = count_no_emb(self.encoder)
        glo       = count_no_emb(self.global_transformer)
        dec       = count_no_emb(self.decoder)
        total_no_emb = enc + glo + dec
        total_all    = count_all(self.encoder) + count_all(self.global_transformer) + count_all(self.decoder)

        print("\n" + "=" * 60)
        print("DPREC Parameter Statistics")
        print("=" * 60)
        print(f"  TemporalLocalEncoder:       {count_all(self.encoder):>12,}")
        print(f"    of which item embeddings: {enc_emb:>12,}")
        print(f"    of which transformer:     {enc:>12,}")
        print(f"  TemporalLatentTransformer:  {glo:>12,}")
        print(f"  LocalDecoder:               {dec:>12,}")
        print(f"  ─────────────────────────────────────────")
        print(f"  Total (trainable, w/ emb):  {total_all:>12,}")
        print(f"  Total (trainable, no emb):  {total_no_emb:>12,}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Core forward logic (shared between forward() and predict())
    # ------------------------------------------------------------------

    def _encode(
        self,
        tokens: torch.Tensor,          # (B, S) long
        pos_seqs: torch.Tensor,        # (B, S) long
        neg_seqs: torch.Tensor,        # (B, S) long
        time_seq: Optional[torch.Tensor] = None,  # (B, S) float
    ) -> torch.Tensor:
        """
        Run the full encoder→global→decoder pipeline.
        Returns (B, S, dim_dec) or (B, S, vocab_size) depending on loss.
        """
        # 1. Detect behavioral boundaries → segment_ids (B, S)
        segment_ids, _ = self.detector.patch(
            tokens=tokens,
            pos_seqs=pos_seqs,
            neg_seqs=neg_seqs,
            time_seq=time_seq,
            expected_patches=self.expected_patches,
        )
        num_segments = int(segment_ids.max().item()) + 1
        # Each token reads the preceding segment's global context
        ctx_segment_ids = (segment_ids - 1).clamp(min=0)

        # 2. TemporalLocalEncoder
        token_hidden, segment_hidden = self.encoder(
            tokens=tokens,
            segment_ids=segment_ids,
            num_segments=num_segments,
            time_seq=time_seq,
        )
        # token_hidden: (B, S, D_enc), segment_hidden: (B, P, D_enc)

        # 3. Segment-level timestamps for global time-RoPE
        time_segments = None
        if self.use_time_rope and time_seq is not None:
            time_segments = segment_time_seq(time_seq, segment_ids, num_segments)

        # 4. TemporalLatentTransformer
        global_hidden = self.global_transformer(segment_hidden, time_seq_segments=time_segments)
        # global_hidden: (B, P, D_global)

        # 5. LocalDecoder
        output = self.decoder(
            token_hidden=token_hidden,
            global_hidden=global_hidden,
            ctx_segment_ids=ctx_segment_ids,
        )
        return output  # (B, S, vocab_size) or (B, S, D_dec)

    def forward(
        self,
        user_ids,
        log_seqs,
        pos_seqs,
        neg_seqs,
        include_next_token: bool = False,
        time_seq=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.tensor(log_seqs, device=self.dev)
        pos = torch.tensor(pos_seqs, device=self.dev).long()
        neg = torch.tensor(neg_seqs, device=self.dev).long()
        ts = None
        if self.use_time_rope and time_seq is not None:
            ts = (torch.tensor(time_seq, device=self.dev).float()
                  if not isinstance(time_seq, torch.Tensor)
                  else time_seq.to(self.dev).float())

        output = self._encode(tokens, pos, neg, ts)   # (B, S, D or vocab)

        if self.loss == "bpr":
            pos_embs = self.encoder.item_embeddings(pos)   # (B, S, D)
            neg_embs = self.encoder.item_embeddings(neg)
            pos_logits = (output * pos_embs).sum(-1)        # (B, S)
            neg_logits = (output * neg_embs).sum(-1)
            return pos_logits, neg_logits

        # BCE: output is (B, S, vocab_size) — return dummy second tensor
        return output, torch.tensor([], device=self.dev)

    def predict(
        self,
        user_ids,
        log_seqs,
        item_indices,
        pos_seqs,
        neg_seqs=None,
        include_next_token: bool = False,
        time_seq=None,
    ) -> torch.Tensor:
        """
        Inference: score a list of candidate items against the last hidden state.
        Returns (B, num_candidates) logit scores.
        """
        tokens = torch.tensor(log_seqs, device=self.dev)
        pos = torch.tensor(pos_seqs, device=self.dev).long()
        neg_t = (torch.tensor(neg_seqs, device=self.dev).long()
                 if neg_seqs is not None
                 else torch.zeros_like(pos))
        item_idx = torch.tensor(item_indices, device=self.dev).long()
        ts = None
        if self.use_time_rope and time_seq is not None:
            ts = (torch.tensor(time_seq, device=self.dev).float()
                  if not isinstance(time_seq, torch.Tensor)
                  else time_seq.to(self.dev).float())

        output = self._encode(tokens, pos, neg_t, ts)  # (B, S, D or vocab)

        if self.loss == "bpr":
            final_feat = output[:, -1, :]                               # (B, D)
            item_embs = self.encoder.item_embeddings(item_idx)          # (C, D) or (B, C, D)
            if item_embs.ndim == 2:
                logits = final_feat.matmul(item_embs.t())               # (B, C)
            else:
                logits = (item_embs * final_feat.unsqueeze(1)).sum(-1)  # (B, C)
            return logits

        # BCE: use final-position logits over the candidate items
        final_logits = output[:, -1, :]   # (B, vocab_size)
        return final_logits[:, item_idx]  # (B, C)
