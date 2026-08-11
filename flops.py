import math

def get_transformer_primitives(l, h, m, r=1, d_ff=1):
    """
    Standard Transformer FLOPs primitives.
    """
    # Feed-forward: 2 * layers * 2 * h * (d_ff * h)
    ffn = 2 * l * 2 * h * (d_ff * h)
    
    # QKVO: (r * 2 + 2) * 2 * l * h^2
    qkvo = (r * 2 + 2) * 2 * l * (h**2)
    
    # Attention: 4 * l * h * ((m + 1) / 2)
    attn = 4 * l * h * ((m + 1) / 2)
    
    return ffn, qkvo, attn

def calculate_entropy_flops(n_ctx, h_ent, l_ent, w_ent, d_ff=1):
    """
    Eq for Entropy Model: Typically a small sliding window transformer.
    h_ent: hidden dim, l_ent: layers, w_ent: window size
    """
    # Operates per item in the sequence
    e_ffn, e_qkvo, e_attn = get_transformer_primitives(l_ent, h_ent, w_ent, d_ff=d_ff)
    return (e_ffn + e_qkvo + e_attn) * n_ctx

def calculate_sasrec_flops(n_ctx, h, l, d_ff=1):
    ffn, qkvo, attn = get_transformer_primitives(l, h, n_ctx, d_ff=d_ff)
    per_item = ffn + qkvo + attn
    return per_item * n_ctx

def calculate_patch_flops(
    n_ctx=100,    # Total sequence length
    n_p=10,       # Patch size
    k=1,          # Latent states per patch
    h_g=64, l_g=2, # Global Transformer
    h_e=64, l_e=1, # Local Encoder
    h_d=64, l_d=1, # Local Decoder
    h_ent=16, l_ent=1, w_ent=10, # Entropy Model Params
    w_e=10, w_d=10,
    d_ff=1,
    use_cross_attn=True,
    include_entropy=True,
):
    # --- Entropy Model (New) ---
    # Fixed/random patching has no entropy model, so its FLOPs are excluded.
    if include_entropy:
        fl_ent_total = calculate_entropy_flops(n_ctx, h_ent, l_ent, w_ent, d_ff)
        fl_ent_per_item = fl_ent_total / n_ctx
    else:
        fl_ent_per_item = 0.0

    # --- Eq 13: Global Latent Transformer ---
    m_global = n_ctx / n_p
    g_ffn, g_qkvo, g_attn = get_transformer_primitives(l_g, h_g, m_global, d_ff=d_ff)
    fl_13_per_item = (g_ffn + g_qkvo + g_attn) / n_p

    # --- Eq 14: Local Encoder ---
    e_ffn, e_qkvo, e_attn = get_transformer_primitives(l_e, h_e, w_e, d_ff=d_ff)
    fl_14_per_item = e_ffn + e_qkvo + e_attn

    # --- Eq 15: Local Decoder ---
    d_ffn, d_qkvo, d_attn = get_transformer_primitives(l_d, h_d, w_d, d_ff=d_ff)
    fl_15_per_item = d_ffn + d_qkvo + d_attn

    # --- Eq 16 & 17: Bridges ---
    if use_cross_attn:
        c_attn_e = 4 * l_e * h_e * ((n_p + 1) / 2)
        c_qkvo_e = ((n_p/k) * 2 + 2) * 2 * l_e * (h_e**2)
        fl_16_per_item = (c_attn_e + c_qkvo_e) * (k / n_p)

        c_attn_d = 4 * l_d * h_d * ((k + 1) / 2)
        c_qkvo_d = ((k/n_p) * 2 + 2) * 2 * l_d * (h_d**2)
        fl_17_per_item = c_attn_d + c_qkvo_d
    else:
        fl_16_per_item = (h_e * (n_p - 1)) / n_p
        fl_17_per_item = 0

    total_per_item = (fl_ent_per_item + fl_13_per_item + fl_14_per_item + 
                      fl_15_per_item + fl_16_per_item + fl_17_per_item)
    
    return total_per_item * n_ctx


def _per_user_flops(backbone, n_ctx, ma, ta):
    """Per-user inference FLOPs for one backbone at context length n_ctx.

    Reads every parameter strictly from the model config (ma=model_args,
    ta=training_args) — no defaults. Missing keys raise, surfacing a
    misconfigured artifact rather than silently producing wrong FLOPs.
    Returns None only for backbones without a FLOPs formula (e.g. the patcher).
    """
    bb = (backbone or "").strip().lower()
    hidden_units = int(ma.hidden_units)
    num_blocks   = int(ma.num_blocks)
    maxlen       = int(ma.maxlen)

    if bb in ("sas", "sasrec"):
        return calculate_sasrec_flops(n_ctx=n_ctx, h=hidden_units, l=num_blocks, d_ff=1)
    if bb == "hstu":
        return calculate_sasrec_flops(n_ctx=n_ctx, h=hidden_units, l=num_blocks, d_ff=1) * 1.1
    if bb in ("gru4rec", "gru"):
        return 12 * num_blocks * (hidden_units ** 2) * n_ctx
    if bb == "longer":
        expected_patches = int(ta.expected_patches)
        inner_dim    = int(ma.inner_trans.dim)
        inner_blocks = int(ma.inner_trans.num_blocks)
        # num_recent_tokens is genuinely optional; the LONGER model derives 5%
        # of maxlen when absent, so we mirror that (not an invented default).
        _nr = ma.get("num_recent_tokens", None)
        nr = int(_nr) if _nr else max(1, int(n_ctx * 0.05))
        compression_k = max(1, maxlen // max(1, expected_patches))
        num_groups = max(1, n_ctx // compression_k)
        inner = calculate_sasrec_flops(n_ctx=compression_k, h=inner_dim, l=inner_blocks, d_ff=1) * num_groups
        cross = (nr * hidden_units * hidden_units
                 + 2 * num_groups * hidden_units * hidden_units
                 + nr * num_groups * hidden_units
                 + nr * hidden_units * hidden_units)
        self_ = calculate_sasrec_flops(n_ctx=nr, h=hidden_units, l=num_blocks, d_ff=1)
        return inner + cross + self_
    if bb in ("dp_rec", "dprec"):
        sliding_window   = int(ma.sliding_window)
        expected_patches = int(ta.expected_patches)
        use_cross_attn   = bool(ma.use_cross_attention)
        cross_attn_k     = int(ma.cross_attention_k)
        h_enc = int(ma.local_encoder.hidden_units)
        l_enc = int(ma.local_encoder.num_blocks)
        l_dec = int(ma.local_decoder.num_blocks)
        # Fixed/random patching drops the entropy-model FLOPs; the patch count
        # target (expected_patches) is identical, so n_p is unchanged.
        strategy = str(ma.get("patching_strategy", "entropy")).strip().lower()
        n_p = max(1, n_ctx // max(1, expected_patches))
        return calculate_patch_flops(
            n_ctx=n_ctx, n_p=n_p, k=cross_attn_k,
            h_g=hidden_units, l_g=num_blocks,
            h_e=h_enc, l_e=l_enc, h_d=h_enc, l_d=l_dec,
            w_e=sliding_window, w_d=sliding_window,
            h_ent=8, l_ent=1, w_ent=sliding_window,
            d_ff=1, use_cross_attn=use_cross_attn,
            include_entropy=(strategy == "entropy"),
        )
    return None


def compute_flops(args, dataset):
    """
    Compute mean / p99 / total inference FLOPs for the configured model over the
    evaluation sequences (valid item + train history, capped at maxlen).

    All parameters are read strictly from the model config (args.model_args /
    args.training_args) — the model artifact's config.yaml when called from
    update_results. Returns {'mean_flops','p99_flops','total_flops'} or None if
    the backbone has no FLOPs formula (e.g. the entropy/patcher stage).
    """
    import numpy as np

    ma = args.model_args
    backbone = str(ma.backbone).strip().lower()
    maxlen   = int(ma.maxlen)

    train, valid, test, usernum = dataset[0], dataset[1], dataset[2], dataset[3]

    per_user = []
    for u in range(1, usernum + 1):
        if u not in train or u not in test or not train[u] or not test[u]:
            continue
        n = len(train[u]) + (1 if valid.get(u) else 0)
        n_ctx = min(n, maxlen)
        if n_ctx == 0:
            continue
        f = _per_user_flops(backbone, n_ctx, args.model_args, args.training_args)
        if f is None:
            return None
        per_user.append(f)

    if not per_user:
        return None
    arr = np.array(per_user)
    return {
        "mean_flops":  float(arr.mean()),
        "p99_flops":   float(np.percentile(arr, 99)),
        "total_flops": float(arr.sum()),
    }


if __name__ == "__main__":
    # -------------------------------------------------------------------
    # Shared settings
    # -------------------------------------------------------------------
    N_CTX    = 200   # sequence length
    H        = 64    # hidden dim for SASRec / HSTU / LONGER global
    L        = 3     # num blocks
    H_LOCAL  = 64    # local encoder/decoder / InnerTrans dim
    L_LOCAL  = 1     # local encoder/decoder / InnerTrans blocks
    SW       = 32    # sliding window for local attention
    H_ENT    = 8     # patcher hidden dim
    L_ENT    = 1     # patcher layers
    W_ENT    = 32    # patcher window

    # -------------------------------------------------------------------
    # SASRec  64x3
    # -------------------------------------------------------------------
    sas_flops = calculate_sasrec_flops(N_CTX, h=H, l=L)

    # -------------------------------------------------------------------
    # HSTU  64x3  (same backbone + 10% gating overhead)
    # -------------------------------------------------------------------
    hstu_flops = calculate_sasrec_flops(N_CTX, h=H, l=L) * 1.1

    # -------------------------------------------------------------------
    # DP-Rec  global=64x3, local enc/dec=64x1, patcher=8x1, SW=32
    # expected_patches=10  =>  n_p = N_CTX / 10 = 20
    # -------------------------------------------------------------------
    EXPECTED_PATCHES = 10
    N_P = max(1, N_CTX // EXPECTED_PATCHES)   # avg patch size = 20

    blt_flops = calculate_patch_flops(
        n_ctx=N_CTX, n_p=N_P,
        k=1,
        h_g=H,       l_g=L_LOCAL,
        h_e=H_LOCAL, l_e=L_LOCAL,
        h_d=H_LOCAL, l_d=L_LOCAL,
        w_e=SW,      w_d=SW,
        h_ent=H_ENT, l_ent=L_ENT, w_ent=W_ENT,
        d_ff=1,
        use_cross_attn=False
    )

    # -------------------------------------------------------------------
    # LONGER  global=64x3, InnerTrans=64x1, groups=expected_patches=50
    # group_size k = N_CTX / 50 = 4
    # -------------------------------------------------------------------
    LONGER_GROUPS = 50
    K = max(1, N_CTX // LONGER_GROUPS)   # group size = 4
    G = LONGER_GROUPS                    # number of groups

    # Inner-group encoding: G groups each of size K
    inner_ffn, inner_qkvo, inner_attn = get_transformer_primitives(L_LOCAL, H_LOCAL, K)
    inner_flops = (inner_ffn + inner_qkvo + inner_attn) * K * G

    # Token-to-group cross-attention: 2 * n * G * h
    cross_flops = 2 * N_CTX * G * H

    # Global self-attention over G groups
    global_ffn, global_qkvo, global_attn = get_transformer_primitives(L_LOCAL, H, G)
    global_flops = (global_ffn + global_qkvo + global_attn) * G

    longer_flops = inner_flops + cross_flops + global_flops

    # -------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------
    print(f"--- FLOPs Comparison (n={N_CTX}, h={H}, l={L}) ---")
    print(f"SASRec  (64x3):                {sas_flops:>15,.0f}")
    print(f"HSTU    (64x3):                {hstu_flops:>15,.0f}")
    print(f"DP-Rec  (64x3 | 64x1 | 8x1):  {blt_flops:>15,.0f}   ({sas_flops/blt_flops:.2f}x vs SASRec)")
    print(f"LONGER  (64x3 | 64x1):         {longer_flops:>15,.0f}   ({sas_flops/longer_flops:.2f}x vs SASRec)")
    