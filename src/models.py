import numpy as np
import torch
import torch.nn.functional as F

import copy
import math

def clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class RelativeAttentionBias(torch.nn.Module):
    """
    Computes relative time and positional attention biases.

    Based on HSTU's implementation from RecTools.

    Parameters
    ----------
    session_max_len : int
        Maximum sequence length
    relative_time_attention : bool
        Whether to compute relative time attention from timestamps
    relative_pos_attention : bool
        Whether to compute relative positional attention
    num_buckets : int
        Number of buckets for quantizing timestamp differences
    """

    def __init__(
        self,
        session_max_len: int,
        relative_time_attention: bool,
        relative_pos_attention: bool,
        num_buckets: int = 128,
    ) -> None:
        super().__init__()
        self.session_max_len = session_max_len
        self.num_buckets = num_buckets
        self.relative_time_attention = relative_time_attention
        self.relative_pos_attention = relative_pos_attention
        if relative_time_attention:
            self.time_weights = torch.nn.Parameter(
                torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
            )
        if relative_pos_attention:
            self.pos_weights = torch.nn.Parameter(
                torch.empty(2 * session_max_len - 1).normal_(mean=0, std=0.02),
            )

    def _quantization_func(self, diff_timestamps: torch.Tensor) -> torch.Tensor:
        """Quantizes the differences between timestamps into discrete buckets."""
        return (torch.log(torch.abs(diff_timestamps).clamp(min=1)) / 0.301).long()

    def forward_time_attention(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ---------
        all_timestamps: torch.Tensor (batch_size, session_max_len+1)
            User interaction timestamps including the target item timestamp
        Returns
        ---------
        torch.Tensor (batch_size, session_max_len, session_max_len)
            relative time attention
        """
        len_expanded = self.session_max_len + 1
        batch_size = all_timestamps.size(0)
        extended_timestamps = torch.cat([all_timestamps, all_timestamps[:, len_expanded - 1 : len_expanded]], dim=1)
        early_time_binding = extended_timestamps[:, 1:].unsqueeze(2) - extended_timestamps[:, :-1].unsqueeze(1)
        bucketed_timestamps = torch.clamp(
            self._quantization_func(early_time_binding),
            min=0,
            max=self.num_buckets,
        ).detach()
        rel_time_attention = torch.index_select(self.time_weights, dim=0, index=bucketed_timestamps.view(-1)).view(
            batch_size, len_expanded, len_expanded
        )
        # reducted target time
        rel_time_attention = rel_time_attention[:, :-1, :-1]
        return rel_time_attention

    def forward_pos_attention(self) -> torch.Tensor:
        """
        Compute and return the relative positional attention bias matrix.

        Returns
        -------
        torch.Tensor (1, session_max_len, session_max_len)
        """
        n = self.session_max_len
        t = torch.nn.functional.pad(self.pos_weights[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        rel_pos_attention = t[:, :, r:-r]
        return rel_pos_attention

    def forward(self, batch: dict = None) -> torch.Tensor:
        """
        Compute combined relative attention bias.

        Parameters
        ----------
        batch : dict, optional
            Dictionary containing 'unix_ts' key with timestamps

        Returns
        -------
        torch.Tensor
            Combined attention bias
        """
        bias = 0.0

        if self.relative_pos_attention:
            bias = bias + self.forward_pos_attention()

        if self.relative_time_attention and batch is not None and 'unix_ts' in batch:
            bias = bias + self.forward_time_attention(batch['unix_ts'])

        return bias

    
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.training_args.device
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.model_args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.model_args.maxlen, args.model_args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.model_args.dropout_rate)
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.model_args.hidden_units, eps=1e-8)
        for _ in range(args.model_args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.model_args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer =  torch.nn.MultiheadAttention(
                args.model_args.hidden_units,
                args.model_args.num_heads,
                args.model_args.dropout_rate
            )
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = torch.nn.LayerNorm(args.model_args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(args.model_args.hidden_units, args.model_args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        log_seqs = torch.tensor(log_seqs).to(self.dev).long()
        seqs = self.item_emb(log_seqs.to(self.dev).long())
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
        # timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        timeline_mask = (log_seqs == 0).to(self.dev).bool()
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim
        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, 
                attn_mask=attention_mask
            ) # key_padding_mask=timeline_mask # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)
        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(torch.tensor(pos_seqs).to(self.dev).long())

        neg_seqs_tensor = torch.tensor(neg_seqs).to(self.dev).long()

        # Handle multiple negatives: neg_seqs can be (batch, maxlen) or (batch, maxlen, num_negatives)
        if neg_seqs_tensor.dim() == 3:
            # Multiple negatives: (batch, maxlen, num_negatives)
            neg_embs = self.item_emb(neg_seqs_tensor)  # (batch, maxlen, num_neg, hidden)
            log_feats_expanded = log_feats.unsqueeze(2)  # (batch, maxlen, 1, hidden)
            pos_logits = (log_feats * pos_embs).sum(dim=-1)  # (batch, maxlen)
            neg_logits = (log_feats_expanded * neg_embs).sum(dim=-1)  # (batch, maxlen, num_neg)
        else:
            # Single negative: (batch, maxlen)
            neg_embs = self.item_emb(neg_seqs_tensor)
            pos_logits = (log_feats * pos_embs).sum(dim=-1)
            neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        item_embs = self.item_emb(torch.tensor(item_indices).to(self.dev).long()) # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        # preds = self.pos_sigmoid(logits) # rank same item list for different users
        return logits # preds # (U, I)


class LONGER(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(LONGER, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.training_args.device
        self.loss = args.training_args.loss

        hidden_units = args.model_args.hidden_units
        dropout_rate = args.model_args.dropout_rate
        num_heads = args.model_args.num_heads
        num_blocks = args.model_args.num_blocks
        maxlen = args.model_args.maxlen

        self.maxlen = maxlen
        self.hidden_units = hidden_units

        # Support both group_size and expected_patches for semantic understanding
        # If expected_patches is provided, derive group_size from it
        # Formula: group_size = maxlen / expected_patches
        # expected_patches lives in training_args (parity with DP-Rec); group_size
        # remains a model_args knob.
        if hasattr(args.training_args, 'expected_patches') and args.training_args.expected_patches is not None:
            self.group_size = maxlen // args.training_args.expected_patches
            print(f"LoNGER: Using expected_patches={args.training_args.expected_patches} → derived group_size={self.group_size}")
        elif hasattr(args.model_args, 'group_size') and args.model_args.group_size is not None:
            self.group_size = args.model_args.group_size
            print(f"LoNGER: Using explicit group_size={self.group_size}")
        else:
            raise ValueError("LoNGER requires either 'expected_patches' in training_args or 'group_size' in model_args")

        # num_recent_tokens: If not specified, default to 5% of maxlen
        if hasattr(args.model_args, 'num_recent_tokens') and args.model_args.num_recent_tokens is not None:
            self.num_recent_tokens = args.model_args.num_recent_tokens
            print(f"LoNGER: Using explicit num_recent_tokens={self.num_recent_tokens}")
        else:
            self.num_recent_tokens = max(1, int(maxlen * 0.05))  # Default: 5% of maxlen
            print(f"LoNGER: Using default num_recent_tokens={self.num_recent_tokens} (5% of maxlen={maxlen})")

        inner_dim = args.model_args.inner_trans.dim
        inner_num_blocks = args.model_args.inner_trans.num_blocks
        inner_num_heads = args.model_args.inner_trans.num_heads

        # --- Time-difference feature ---
        self.use_abs_time_diff = getattr(args.model_args, 'use_abs_time_diff', False)
        if self.use_abs_time_diff:
            time_emb_dim = args.model_args.time_emb_dim
            self.max_weeks = getattr(args.model_args, 'max_weeks', 52)
            self.hour_emb = torch.nn.Embedding(24, time_emb_dim)        # 0-23
            self.day_emb = torch.nn.Embedding(7, time_emb_dim)          # 0-6
            self.week_emb = torch.nn.Embedding(self.max_weeks + 1, time_emb_dim)  # 0-max_weeks
            self.time_proj = torch.nn.Linear(hidden_units + time_emb_dim, hidden_units)

        # --- Embeddings ---
        self.item_emb = torch.nn.Embedding(self.item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)

        # --- InnerTrans: lightweight transformer within each group ---
        # Dimension projections if inner_dim != hidden_units
        self.inner_dim = inner_dim
        if inner_dim != hidden_units:
            self.inner_proj_in = torch.nn.Linear(hidden_units, inner_dim)
            self.inner_proj_out = torch.nn.Linear(inner_dim, hidden_units)
        else:
            self.inner_proj_in = None
            self.inner_proj_out = None

        self.inner_attn_layernorms = torch.nn.ModuleList()
        self.inner_attn_layers = torch.nn.ModuleList()
        self.inner_ffn_layernorms = torch.nn.ModuleList()
        self.inner_ffn_layers = torch.nn.ModuleList()
        for _ in range(inner_num_blocks):
            self.inner_attn_layernorms.append(torch.nn.LayerNorm(inner_dim, eps=1e-8))
            self.inner_attn_layers.append(
                torch.nn.MultiheadAttention(inner_dim, inner_num_heads, dropout_rate, batch_first=True)
            )
            self.inner_ffn_layernorms.append(torch.nn.LayerNorm(inner_dim, eps=1e-8))
            self.inner_ffn_layers.append(PointWiseFeedForward(inner_dim, dropout_rate))

        # --- Cross-Causal Attention (Layer 1): Q=O, K/V=R ---
        self.cross_attn_layernorm_q = torch.nn.LayerNorm(hidden_units, eps=1e-8)
        self.cross_attn_layernorm_kv = torch.nn.LayerNorm(hidden_units, eps=1e-8)
        self.cross_attn = torch.nn.MultiheadAttention(hidden_units, num_heads, dropout_rate, batch_first=True)
        self.cross_ffn_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
        self.cross_ffn = PointWiseFeedForward(hidden_units, dropout_rate)

        # --- Self-Causal Attention Layers (Layers 2..N) ---
        self.self_attn_layernorms = torch.nn.ModuleList()
        self.self_attn_layers = torch.nn.ModuleList()
        self.self_ffn_layernorms = torch.nn.ModuleList()
        self.self_ffn_layers = torch.nn.ModuleList()
        for _ in range(num_blocks):
            self.self_attn_layernorms.append(torch.nn.LayerNorm(hidden_units, eps=1e-8))
            self.self_attn_layers.append(
                torch.nn.MultiheadAttention(hidden_units, num_heads, dropout_rate, batch_first=True)
            )
            self.self_ffn_layernorms.append(torch.nn.LayerNorm(hidden_units, eps=1e-8))
            self.self_ffn_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        self.last_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)

        # --- Prediction head for BCE loss ---
        if self.loss == 'bce':
            self.output_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_units, hidden_units),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_units, item_num + 1),
            )

    def _compress_sequence(self, seqs):
        """Apply InnerTrans within groups then max pool to produce compressed representation R.

        Args:
            seqs: (batch, maxlen, hidden_units) - embedded input sequence
        Returns:
            R: (batch, num_groups, hidden_units) - compressed group representations
        """
        batch_size, seq_len, hidden_dim = seqs.shape
        K = self.group_size

        # Pad sequence length to be divisible by K
        if seq_len % K != 0:
            pad_len = K - (seq_len % K)
            seqs = F.pad(seqs, (0, 0, pad_len, 0))  # left-pad with zeros to match data convention
            seq_len = seqs.shape[1]

        num_groups = seq_len // K

        # Reshape: (batch, num_groups, K, hidden_dim)
        grouped = seqs.view(batch_size, num_groups, K, hidden_dim)

        # Flatten for parallel InnerTrans: (batch * num_groups, K, hidden_dim)
        grouped = grouped.view(batch_size * num_groups, K, hidden_dim)

        # Project to inner_dim
        if self.inner_proj_in is not None:
            grouped = self.inner_proj_in(grouped)  # (batch * num_groups, K, inner_dim)

        # InnerTrans
        for i in range(len(self.inner_attn_layers)):
            normed = self.inner_attn_layernorms[i](grouped)
            attn_out, _ = self.inner_attn_layers[i](normed, normed, normed)
            grouped = grouped + attn_out
            normed = self.inner_ffn_layernorms[i](grouped)
            grouped = normed + self.inner_ffn_layers[i](normed)

        # Project back to hidden_units
        if self.inner_proj_out is not None:
            grouped = self.inner_proj_out(grouped)  # (batch * num_groups, K, hidden_units)

        # Reshape back: (batch, num_groups, K, hidden_units)
        grouped = grouped.view(batch_size, num_groups, K, -1)

        # Max pool across tokens in each group: (batch, num_groups, hidden_units)
        # Pooling not mentioned in LONGER paper; using max pooling for LocalEncoder
        R = grouped.max(dim=2)[0]
        return R

    def _create_cross_causal_mask(self, M, num_groups):
        """Create causal mask for cross-attention (Q=recent tokens, K/V=groups).

        Token at absolute position (maxlen - M + i) can attend to group j
        only if j * K <= (maxlen - M + i), meaning the group started at or
        before the token's position.

        Args:
            M: number of recent tokens
            num_groups: number of compressed groups
        Returns:
            mask: (M, num_groups) bool tensor, True = masked (cannot attend)
        """
        K = self.group_size
        # Absolute positions of the M recent tokens
        abs_positions = torch.arange(self.maxlen - M, self.maxlen, device=self.dev)  # (M,)
        # Start position of each group
        group_starts = torch.arange(0, num_groups, device=self.dev) * K  # (num_groups,)
        # Token i can attend to group j if group_starts[j] <= abs_positions[i]
        # Mask is True where we should NOT attend
        mask = group_starts.unsqueeze(0) > abs_positions.unsqueeze(1)  # (M, num_groups)
        return mask

    def log2feats(self, log_seqs, time_seqs=None):
        """Convert input sequences to feature representations.

        Args:
            log_seqs: numpy array (batch, maxlen) of item IDs
            time_seqs: numpy array (batch, maxlen) of timestamps in seconds (optional)
        Returns:
            log_feats: (batch, maxlen, hidden_units) - zero-padded on left
        """
        log_seqs = torch.tensor(log_seqs).to(self.dev).long()
        batch_size = log_seqs.shape[0]

        # Item + positional embeddings
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5

        # Time-difference feature: decompose into hours, days, weeks
        if self.use_abs_time_diff and time_seqs is not None:
            time_seqs = torch.tensor(time_seqs).to(self.dev).long()
            # Target timestamp is the last non-zero timestamp per sequence
            # Use the last position (rightmost) as target
            target_time = time_seqs[:, -1:]  # (batch, 1)
            abs_diff = (target_time - time_seqs).abs()  # (batch, maxlen) in seconds
            total_hours = abs_diff // 3600
            weeks = (total_hours // 168).clamp(max=self.max_weeks)
            days = (total_hours % 168) // 24   # 0-6
            hours = total_hours % 24            # 0-23
            time_feat = self.hour_emb(hours) + self.day_emb(days) + self.week_emb(weeks)  # (batch, maxlen, time_emb_dim)
            seqs = self.time_proj(torch.cat([seqs, time_feat], dim=-1))

        positions = np.tile(np.array(range(self.maxlen)), [batch_size, 1])
        seqs = seqs + self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        # Zero out padding positions
        timeline_mask = (log_seqs == 0).unsqueeze(-1)  # (batch, maxlen, 1)
        seqs = seqs * ~timeline_mask

        # Extract matrix O - most recent M token embeddings (pre-compression)
        M = self.num_recent_tokens
        O = seqs[:, -M:, :]  # (batch, M, hidden_units)

        # Compress sequence -> matrix R
        R = self._compress_sequence(seqs)  # (batch, num_groups, hidden_units)
        num_groups = R.shape[1]

        # Cross-Causal Attention (Q=O, K/V=R)
        cross_mask = self._create_cross_causal_mask(M, num_groups)  # (M, num_groups)
        Q_norm = self.cross_attn_layernorm_q(O)
        KV_norm = self.cross_attn_layernorm_kv(R)
        cross_out, _ = self.cross_attn(Q_norm, KV_norm, KV_norm, attn_mask=cross_mask)
        h = O + cross_out
        h_norm = self.cross_ffn_layernorm(h)
        h = h_norm + self.cross_ffn(h_norm)

        # Self-Causal Attention layers
        causal_mask = ~torch.tril(torch.ones((M, M), dtype=torch.bool, device=self.dev))
        for i in range(len(self.self_attn_layers)):
            normed = self.self_attn_layernorms[i](h)
            attn_out, _ = self.self_attn_layers[i](normed, normed, normed, attn_mask=causal_mask)
            h = h + attn_out
            normed = self.self_ffn_layernorms[i](h)
            h = normed + self.self_ffn_layers[i](normed)

        h = self.last_layernorm(h)  # (batch, M, hidden_units)

        # Pad to (batch, maxlen, hidden_units) with zeros on left
        if M < self.maxlen:
            padding = torch.zeros(batch_size, self.maxlen - M, self.hidden_units, device=self.dev)
            log_feats = torch.cat([padding, h], dim=1)
        else:
            log_feats = h

        return log_feats

    def forward(self, user_ids, log_seqs, time_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs, time_seqs)

        if self.loss == 'bce':
            logits = self.output_mlp(log_feats)  # (batch, maxlen, vocab_size)
            return (logits,) + (torch.tensor([]),)

        # BPR loss: dot product scoring
        pos_embs = self.item_emb(torch.tensor(pos_seqs).to(self.dev).long())
        neg_embs = self.item_emb(torch.tensor(neg_seqs).to(self.dev).long())
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices, time_seqs=None):
        log_feats = self.log2feats(log_seqs, time_seqs)

        if self.loss == 'bce':
            logits = self.output_mlp(log_feats)
            return logits

        final_feat = log_feats[:, -1, :]  # (batch, hidden_units)
        item_embs = self.item_emb(torch.tensor(item_indices).to(self.dev).long())  # (batch, num_items, hidden_units)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


class HSTU(torch.nn.Module):
    """HSTU wrapper using rectools STULayers with the same forward/predict interface as other models."""

    def __init__(self, user_num, item_num, args):
        super(HSTU, self).__init__()
        from rectools.models.nn.transformers.hstu import STULayers
        from rectools.models.nn.transformers.net_blocks import LearnableInversePositionalEncoding

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.training_args.device
        self.loss = args.training_args.loss

        hidden_units = args.model_args.hidden_units
        num_heads = args.model_args.num_heads
        num_blocks = args.model_args.num_blocks
        dropout_rate = args.model_args.dropout_rate
        maxlen = args.model_args.maxlen
        self.maxlen = maxlen

        self.relative_time_attention = args.model_args.get('relative_time_attention', True)
        self.relative_pos_attention = args.model_args.get('relative_pos_attention', True)
        use_pos_emb = args.model_args.get('use_pos_emb', True)
        use_scale_factor = args.model_args.get('use_scale_factor', True)
        attn_dropout_rate = args.model_args.get('attn_dropout_rate', 0.0)
        self.similarity = args.model_args.get('similarity', 'dot')  # dot or cosine

        # Item embedding
        self.item_emb = torch.nn.Embedding(self.item_num + 1, hidden_units, padding_idx=0)

        # Use RecTools' positional encoding with scale factor (matching HSTU)
        self.pos_encoding = LearnableInversePositionalEncoding(
            use_pos_emb=use_pos_emb,
            session_max_len=maxlen,
            n_factors=hidden_units,
            use_scale_factor=use_scale_factor
        )
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)

        head_dim = hidden_units // num_heads

        # STULayers from rectools
        self.stu_layers = STULayers(
            n_blocks=num_blocks,
            n_factors=hidden_units,
            n_heads=num_heads,
            linear_hidden_dim=head_dim,
            attention_dim=head_dim,
            session_max_len=maxlen,
            relative_time_attention=self.relative_time_attention,
            relative_pos_attention=self.relative_pos_attention,
            attn_dropout_rate=attn_dropout_rate,
            dropout_rate=dropout_rate,
        )

        # Final layer norm (matching RecTools transformer models)
        self.final_norm = torch.nn.LayerNorm(hidden_units)

        # Prediction head for BCE loss
        if self.loss == 'bce':
            self.output_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_units, hidden_units),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_units, item_num + 1),
            )

        # Apply Xavier normal initialization (matching RecTools)
        self._xavier_normal_init()

    def _xavier_normal_init(self):
        """Initialize all parameters with Xavier normal distribution (matching RecTools)."""
        for name, param in self.named_parameters():
            if param.data.dim() > 1:
                torch.nn.init.xavier_normal_(param.data)

    def log2feats(self, log_seqs, time_seqs=None):
        log_seqs = torch.tensor(log_seqs).to(self.dev).long()
        seqs = self.item_emb(log_seqs)

        # Apply RecTools positional encoding with scale factor
        # This handles: item_emb * sqrt(dim), add pos_emb, scale by learnable factor
        seqs = self.pos_encoding(seqs)
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs != 0).unsqueeze(-1)  # (batch, maxlen, 1)

        # Causal attention mask
        attn_mask = ~torch.tril(
            torch.ones((self.maxlen, self.maxlen), dtype=torch.bool, device=self.dev)
        )

        # Build batch dict for STULayers (needed by RelativeAttentionBias)
        batch = {"x": log_seqs}
        if self.relative_time_attention:
            if time_seqs is not None:
                time_seqs_t = torch.tensor(time_seqs).to(self.dev).long()
            else:
                time_seqs_t = torch.zeros(log_seqs.shape[0], self.maxlen, device=self.dev).long()

            # For relative time attention, we need session_max_len+1 timestamps
            # The last timestamp should be the target item's timestamp
            # Since we don't have a separate target timestamp, we use the last position
            # but this should ideally be provided separately in training
            # For now, append the last timestamp to create the required shape
            target_time = time_seqs_t[:, -1:]  # (batch, 1)
            all_timestamps = torch.cat([time_seqs_t, target_time], dim=1)  # (batch, maxlen+1)
            batch["unix_ts"] = all_timestamps

        seqs = self.stu_layers(seqs, timeline_mask, attn_mask, key_padding_mask=None, batch=batch)

        # Apply final layer norm
        seqs = self.final_norm(seqs) * timeline_mask

        return seqs

    def forward(self, user_ids, log_seqs, time_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs, time_seqs)

        if self.loss == 'bce':
            logits = self.output_mlp(log_feats)
            return (logits,) + (torch.tensor([]),)

        pos_embs = self.item_emb(torch.tensor(pos_seqs).to(self.dev).long())
        neg_seqs_tensor = torch.tensor(neg_seqs).to(self.dev).long()

        # Handle multiple negatives: neg_seqs can be (batch, maxlen) or (batch, maxlen, num_negatives)
        if neg_seqs_tensor.dim() == 3:
            # Multiple negatives: (batch, maxlen, num_negatives)
            neg_embs = self.item_emb(neg_seqs_tensor)  # (batch, maxlen, num_neg, hidden)
        else:
            # Single negative: (batch, maxlen)
            neg_embs = self.item_emb(neg_seqs_tensor)  # (batch, maxlen, hidden)

        if self.similarity == 'cosine':
            # Cosine similarity (matching RecTools HSTU default)
            log_feats_norm = F.normalize(log_feats, p=2, dim=-1)
            pos_embs_norm = F.normalize(pos_embs, p=2, dim=-1)
            neg_embs_norm = F.normalize(neg_embs, p=2, dim=-1)
            pos_logits = (log_feats_norm * pos_embs_norm).sum(dim=-1)

            if neg_embs_norm.dim() == 4:
                # Multiple negatives
                log_feats_expanded = log_feats_norm.unsqueeze(2)  # (batch, maxlen, 1, hidden)
                neg_logits = (log_feats_expanded * neg_embs_norm).sum(dim=-1)  # (batch, maxlen, num_neg)
            else:
                neg_logits = (log_feats_norm * neg_embs_norm).sum(dim=-1)
        else:
            # Dot product (original implementation)
            pos_logits = (log_feats * pos_embs).sum(dim=-1)  # (batch, maxlen)

            if neg_embs.dim() == 4:
                # Multiple negatives: (batch, maxlen, num_neg, hidden)
                log_feats_expanded = log_feats.unsqueeze(2)  # (batch, maxlen, 1, hidden)
                neg_logits = (log_feats_expanded * neg_embs).sum(dim=-1)  # (batch, maxlen, num_neg)
            else:
                # Single negative
                neg_logits = (log_feats * neg_embs).sum(dim=-1)  # (batch, maxlen)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices, time_seqs=None):
        log_feats = self.log2feats(log_seqs, time_seqs)

        if self.loss == 'bce':
            logits = self.output_mlp(log_feats)
            return logits

        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.tensor(item_indices).to(self.dev).long())

        if self.similarity == 'cosine':
            # Cosine similarity (matching RecTools HSTU default)
            final_feat_norm = F.normalize(final_feat, p=2, dim=-1)
            item_embs_norm = F.normalize(item_embs, p=2, dim=-1)
            logits = item_embs_norm.matmul(final_feat_norm.unsqueeze(-1)).squeeze(-1)
        else:
            # Dot product (original implementation)
            logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
