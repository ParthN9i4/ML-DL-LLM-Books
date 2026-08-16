"""
ex14 — Scaled dot-product and multi-head attention.  (Book: Chapter 18)

Attention is content-addressable soft retrieval:

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Three things to actually understand, each of which the checks below force:

  1. WHY sqrt(d_k). For q, k with i.i.d. unit-variance entries, the dot product
     q.k has variance d_k. Un-scaled logits at d_k = 64 have std 8, softmax
     saturates to near one-hot, and the gradient through it collapses. The
     scaling holds logit variance at 1 regardless of head size. This is checked
     numerically, not asserted rhetorically.

  2. THE SOFTMAX JACOBIAN. d softmax_i / d z_j = p_i (delta_ij - p_j), i.e.
     J = diag(p) - p p^T. Every gradient bug in every from-scratch attention
     implementation lives here or in the mask.

  3. THE MASK GOES BEFORE THE SOFTMAX. Masking probabilities after the softmax
     leaves them normalized over the wrong support — the future tokens still
     stole probability mass. The "break it" section shows the numerical
     difference, which is not small.

Multi-head is then bookkeeping: split d_model into h heads of d_k = d_model/h,
attend independently, concatenate, and mix with an output projection.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


def softmax(z, axis=-1):
    """Stable softmax along `axis` (shift by the max; see tier0 ex02)."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention(Q, K, V, mask=None):
    """Scaled dot-product attention.

    Q: (..., n_q, d_k), K: (..., n_k, d_k), V: (..., n_k, d_v).
    mask: boolean (..., n_q, n_k), True = MAY attend. Applied BEFORE softmax
    by setting disallowed logits to -inf (in practice a large negative number).

    Returns (output, weights) with output (..., n_q, d_v).
    """
    d_k = Q.shape[-1]
    # === YOUR CODE HERE ===
    logits = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        logits = np.where(mask, logits, -1e30)
    w = softmax(logits, axis=-1)
    return w @ V, w


def causal_mask(n):
    """(n, n) boolean lower-triangular mask: position i may attend to j <= i."""
    # === YOUR CODE HERE ===
    return np.tril(np.ones((n, n), dtype=bool))


class MultiHeadAttention:
    """Multi-head attention with combined QKV projection, matching torch's layout.

    torch.nn.MultiheadAttention stores in_proj_weight as the (3d, d) stack
    [W_q; W_k; W_v] acting on the RIGHT of row-vector inputs (x @ W.T), and an
    out_proj of shape (d, d). Matching its layout exactly is what lets the
    cross-check below transplant weights and demand agreement to 1e-10.
    """

    def __init__(self, d_model, n_heads, rng):
        assert d_model % n_heads == 0
        self.d, self.h, self.dk = d_model, n_heads, d_model // n_heads
        s = 1.0 / np.sqrt(d_model)
        self.w_in = rng.uniform(-s, s, (3 * d_model, d_model))   # [Wq; Wk; Wv]
        self.b_in = np.zeros(3 * d_model)
        self.w_out = rng.uniform(-s, s, (d_model, d_model))
        self.b_out = np.zeros(d_model)

    def __call__(self, x, mask=None):
        """x: (n, d) -> (n, d).  Split heads, attend, concat, project."""
        n, d = x.shape
        # === YOUR CODE HERE ===
        qkv = x @ self.w_in.T + self.b_in                 # (n, 3d)
        q, k, v = np.split(qkv, 3, axis=-1)               # each (n, d)

        def heads(t):                                     # (n, d) -> (h, n, dk)
            return t.reshape(n, self.h, self.dk).transpose(1, 0, 2)

        hm = None if mask is None else np.broadcast_to(mask, (self.h, n, n))
        out, w = attention(heads(q), heads(k), heads(v), hm)   # (h, n, dk)
        out = out.transpose(1, 0, 2).reshape(n, d)             # concat heads
        return out @ self.w_out.T + self.b_out, w


def attention_with_kv_cache(mha, x, mask):
    """Decode x one position at a time, reusing cached K and V.

    The assertion that cached decoding equals full recomputation is THE test
    that catches real cache bugs (wrong slot, stale cache, off-by-one mask).
    """
    n, d = x.shape
    # === YOUR CODE HERE ===
    k_cache = np.zeros((mha.h, 0, mha.dk))
    v_cache = np.zeros((mha.h, 0, mha.dk))
    outs = []
    for t in range(n):
        qkv = x[t:t + 1] @ mha.w_in.T + mha.b_in
        q, k, v = np.split(qkv, 3, axis=-1)

        def heads(z):
            return z.reshape(1, mha.h, mha.dk).transpose(1, 0, 2)

        k_cache = np.concatenate([k_cache, heads(k)], axis=1)
        v_cache = np.concatenate([v_cache, heads(v)], axis=1)
        # A causal query at position t may see everything cached so far.
        out, _ = attention(heads(q), k_cache, v_cache)
        outs.append(out.transpose(1, 0, 2).reshape(1, d))
    return np.vstack(outs) @ mha.w_out.T + mha.b_out


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("\n--- why sqrt(d_k) ---")
    # Logit variance without scaling grows linearly in d_k; with it, stays ~1.
    for dk in (16, 64, 256):
        q = rng.standard_normal((20000, dk))
        k = rng.standard_normal((20000, dk))
        raw = np.einsum("nd,nd->n", q, k)
        check(f"d_k={dk}: unscaled logit variance ~ d_k",
              float(raw.var()), dk, tol=0.06 * dk)
        check(f"d_k={dk}: scaled logit variance ~ 1",
              float((raw / np.sqrt(dk)).var()), 1.0, tol=0.06)
    # And what saturation does to the gradient: max_j p_j -> 1, so the softmax
    # Jacobian diag(p) - pp^T -> 0 in every entry.
    z_sat = rng.standard_normal(64) * np.sqrt(256)     # unscaled, d_k=256 regime
    z_ok = z_sat / np.sqrt(256)
    for name, z in (("unscaled", z_sat), ("scaled", z_ok)):
        p = softmax(z)
        J = np.diag(p) - np.outer(p, p)
        print(f"      {name:9s}: max prob {p.max():.4f}, ||Jacobian||_F = {np.linalg.norm(J):.3e}")
    p_sat, p_ok = softmax(z_sat), softmax(z_ok)
    J_sat = np.diag(p_sat) - np.outer(p_sat, p_sat)
    J_ok = np.diag(p_ok) - np.outer(p_ok, p_ok)
    check("saturation crushes the softmax Jacobian by >10x",
          np.linalg.norm(J_ok) > 10 * np.linalg.norm(J_sat))

    print("\n--- softmax Jacobian, analytically vs numerically ---")
    z = rng.standard_normal(7)
    p = softmax(z)
    J_analytic = np.diag(p) - np.outer(p, p)
    eps = 1e-6
    J_numeric = np.column_stack([
        (softmax(z + eps * np.eye(7)[j]) - softmax(z - eps * np.eye(7)[j])) / (2 * eps)
        for j in range(7)
    ])
    check("J = diag(p) - p p^T matches finite differences", J_analytic, J_numeric, tol=1e-9)
    check("Jacobian rows sum to zero (probabilities stay normalized)",
          J_analytic.sum(axis=1), np.zeros(7), tol=1e-15)

    print("\n--- attention basics ---")
    Q = rng.standard_normal((5, 8))
    K = rng.standard_normal((6, 8))
    V = rng.standard_normal((6, 4))
    out, w = attention(Q, K, V)
    check("output shape (n_q, d_v)", out.shape == (5, 4))
    check("attention weights are a distribution per query",
          w.sum(axis=-1), np.ones(5), tol=1e-12)
    # Retrieval sanity: a query aligned with one key attends there.
    K_id = np.eye(4) * 10
    Q_id = np.eye(4)[2:3] * 10
    _, w_id = attention(Q_id, K_id, np.eye(4))
    check("aligned query retrieves its key", float(w_id[0, 2]) > 0.99)

    print("\n--- causal masking ---")
    n = 6
    x_pos = rng.standard_normal((n, 8))
    m = causal_mask(n)
    _, w_causal = attention(x_pos, x_pos, rng.standard_normal((n, 4)), m)
    check("no attention to the future (upper triangle exactly 0)",
          float(np.abs(np.triu(w_causal, k=1)).max()), 0.0, tol=1e-12)
    check("rows still normalize over the allowed prefix",
          w_causal.sum(axis=-1), np.ones(n), tol=1e-12)
    # Changing a FUTURE token must not change the current output — causality
    # as an end-to-end property, not just a mask shape.
    v_seq = rng.standard_normal((n, 4))
    out_a, _ = attention(x_pos, x_pos, v_seq, m)
    x_mod = x_pos.copy()
    x_mod[-1] += 5.0
    out_b, _ = attention(x_mod[:-1], x_mod, v_seq, m[:-1])  # queries 0..n-2 see all keys
    check("perturbing the last token leaves earlier outputs unchanged",
          out_a[:-1], out_b, tol=1e-12)

    print("\n--- multi-head vs torch ---")
    d_model, n_heads, n_seq = 32, 4, 10
    mha = MultiHeadAttention(d_model, n_heads, rng)
    x = rng.standard_normal((n_seq, d_model))
    ours, _ = mha(x, causal_mask(n_seq))

    try:
        import torch
        t = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=True)
        with torch.no_grad():
            t.in_proj_weight.copy_(torch.tensor(mha.w_in))
            t.in_proj_bias.copy_(torch.tensor(mha.b_in))
            t.out_proj.weight.copy_(torch.tensor(mha.w_out))
            t.out_proj.bias.copy_(torch.tensor(mha.b_out))
        am = torch.triu(torch.ones(n_seq, n_seq, dtype=torch.bool), diagonal=1)
        ref, _ = t(torch.tensor(x[None]).float(), torch.tensor(x[None]).float(),
                   torch.tensor(x[None]).float(), attn_mask=am, need_weights=False)
        check("multi-head output matches torch.nn.MultiheadAttention",
              ours, ref[0].detach().double().numpy(), tol=2e-5)
    except ImportError:
        print("  ....  [skipped: torch not installed] torch MHA cross-check")

    print("\n--- KV cache ---")
    full, _ = mha(x, causal_mask(n_seq))
    cached = attention_with_kv_cache(mha, x, causal_mask(n_seq))
    check("incremental KV-cached decode == full recompute", cached, full, tol=1e-10)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # THE classic: masking AFTER the softmax. The surviving weights are
    # normalized over all positions including the masked ones, so each row
    # under-weights its allowed prefix — badly, for early positions.
    logits = x_pos @ x_pos.T / np.sqrt(8)
    w_wrong = softmax(logits) * m                     # mask applied too late
    row_mass = w_wrong.sum(axis=-1)
    print(f"      row mass after post-softmax masking: {np.round(row_mass, 3)}")
    print(f"      row 0 keeps only {row_mass[0]*100:.0f}% of its probability mass")
    check("post-softmax masking breaks normalization", float(row_mass[0]) < 0.7)
    v_cmp = rng.standard_normal((n, 4))
    out_right, _ = attention(x_pos, x_pos, v_cmp, m)
    out_wrong = w_wrong @ v_cmp
    rel = np.linalg.norm(out_right - out_wrong) / np.linalg.norm(out_right)
    print(f"      relative output error vs correct masking: {rel:.2f}")
    check("the resulting outputs differ materially", rel > 0.1)

    # And the subtle variant: masking with 0 instead of -inf. exp(0) = 1, so a
    # "masked" logit still receives probability as if its logit were 0.
    w_zeroed = softmax(np.where(m, logits, 0.0))
    leak = float(np.abs(np.triu(w_zeroed, k=1)).max())
    print(f"      masking by zeroing logits leaks {leak:.3f} probability to the future")
    check("zeroing logits does not mask (exp(0)=1)", leak > 0.01)

    summary()
