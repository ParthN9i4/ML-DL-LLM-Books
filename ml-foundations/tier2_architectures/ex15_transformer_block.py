"""
ex15 — A complete modern transformer block.  (Book: Chapters 12, 18)

One block of a 2025-vintage decoder — the repeating unit of essentially every
current LLM — is:

    x = x + Attn(RMSNorm(x))          # pre-norm residual, attention mixing
    x = x + SwiGLU(RMSNorm(x))        # pre-norm residual, channel mixing

Three deliberate choices, each of which this exercise makes checkable:

  1. PRE-norm, not post-norm. The residual stream is never normalized in the
     main path, so the identity gradient path is untouched: d(out)/d(x) has an
     exact identity component at every depth. The comparison below measures
     this in BOTH regimes, because the first version of this test got it
     wrong: with down-scaled residual branches post-norm is fine too (that
     rescue is DeepNorm's whole insight), and the classical pathology only
     appears when branch gain approaches 1 — where it then compounds with
     depth. The regime is part of the claim, and eating that correction is
     more instructive than the clean story would have been.

  2. RMSNorm, not LayerNorm. Drops the mean subtraction; only the scale
     x / sqrt(mean(x^2) + eps) remains. One fewer reduction, empirically no
     quality loss. (And one fewer data-dependent quantity under encryption —
     see Chapter 38.)

  3. SwiGLU, not a plain MLP: (SiLU(x W_g) * x W_u) W_d, with the hidden
     dimension set to ~8/3 d so the parameter count matches a 4d MLP.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


def rmsnorm(x, gamma, eps=1e-6):
    """x / sqrt(mean(x^2, last axis) + eps) * gamma.  No mean subtraction."""
    # === YOUR CODE HERE ===
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * gamma


def silu(x):
    """SiLU / swish: x * sigmoid(x)."""
    # === YOUR CODE HERE ===
    return x / (1.0 + np.exp(-np.clip(x, -60, 60)))


def swiglu_mlp(x, W_gate, W_up, W_down):
    """(SiLU(x W_gate) * (x W_up)) W_down — the gated channel-mixer."""
    # === YOUR CODE HERE ===
    return (silu(x @ W_gate) * (x @ W_up)) @ W_down


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def causal_attention(x, W_qkv, W_o, n_heads):
    """Multi-head causal self-attention, (n, d) -> (n, d). Weights act on the right."""
    n, d = x.shape
    dk = d // n_heads
    qkv = x @ W_qkv                                    # (n, 3d)
    q, k, v = np.split(qkv, 3, axis=-1)

    def heads(t):
        return t.reshape(n, n_heads, dk).transpose(1, 0, 2)

    q, k, v = heads(q), heads(k), heads(v)
    logits = q @ np.swapaxes(k, -1, -2) / np.sqrt(dk)  # (h, n, n)
    logits = np.where(np.tril(np.ones((n, n), dtype=bool)), logits, -1e30)
    out = softmax(logits) @ v                          # (h, n, dk)
    return out.transpose(1, 0, 2).reshape(n, d) @ W_o


class Block:
    """One pre-norm decoder block. `post_norm=True` flips to the 2017 layout
    (normalize AFTER the residual add) so the two can be compared honestly."""

    def __init__(self, d, n_heads, rng, post_norm=False):
        self.d, self.h, self.post = d, n_heads, post_norm
        # ~8/3 d hidden, rounded to a multiple of 8, parameter-matching a 4d MLP.
        hidden = int(np.ceil(8 * d / 3 / 8) * 8)
        s = 1.0 / np.sqrt(d)
        r = 1.0 / np.sqrt(2 * 24)     # residual-branch scaling for deep stacks
        self.W_qkv = rng.uniform(-s, s, (d, 3 * d))
        self.W_o = rng.uniform(-s, s, (d, d)) * r
        self.W_g = rng.uniform(-s, s, (d, hidden))
        self.W_u = rng.uniform(-s, s, (d, hidden))
        self.W_d = rng.uniform(-1 / np.sqrt(hidden), 1 / np.sqrt(hidden), (hidden, d)) * r
        self.g1 = np.ones(d)
        self.g2 = np.ones(d)

    def __call__(self, x):
        # === YOUR CODE HERE ===
        if not self.post:
            x = x + causal_attention(rmsnorm(x, self.g1), self.W_qkv, self.W_o, self.h)
            x = x + swiglu_mlp(rmsnorm(x, self.g2), self.W_g, self.W_u, self.W_d)
            return x
        x = rmsnorm(x + causal_attention(x, self.W_qkv, self.W_o, self.h), self.g1)
        x = rmsnorm(x + swiglu_mlp(x, self.W_g, self.W_u, self.W_d), self.g2)
        return x


def block_param_count(d, hidden):
    """Attention (4 d^2) + gated MLP (3 d*hidden) + two norms (2d)."""
    # === YOUR CODE HERE ===
    return 4 * d * d + 3 * d * hidden + 2 * d


def input_grad_norm(blocks, x0, eps=1e-5):
    """|| d sum(out) / d x0 ||, by forward differences through the whole stack.

    Slow but engine-free: perturb each input coordinate, rerun the stack.
    """
    def run(x):
        for b in blocks:
            x = b(x)
        return float(x.sum())

    base = run(x0)
    g = np.zeros_like(x0)
    flat = x0.reshape(-1)
    gf = g.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        gf[i] = (run(x0) - base) / eps
        flat[i] = old
    return float(np.linalg.norm(g))


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("\n--- RMSNorm ---")
    x = rng.standard_normal((5, 16)) * 7.0
    y = rmsnorm(x, np.ones(16))
    check("output RMS is 1 per row", np.sqrt((y * y).mean(axis=-1)), np.ones(5), tol=1e-6)
    check("scale invariance: rmsnorm(10x) == rmsnorm(x)",
          rmsnorm(10 * x, np.ones(16)), y, tol=1e-6)
    # LayerNorm subtracts the mean first; RMSNorm does not. They differ exactly
    # when the mean is nonzero — that difference IS the design decision.
    x_shift = x + 5.0
    ln = (x_shift - x_shift.mean(-1, keepdims=True))
    ln = ln / np.sqrt((ln * ln).mean(-1, keepdims=True) + 1e-6)
    check("RMSNorm != LayerNorm on mean-shifted input",
          float(np.abs(rmsnorm(x_shift, np.ones(16)) - ln).max()) > 0.5)

    print("\n--- SwiGLU ---")
    check("SiLU(0) = 0", float(silu(np.array([0.0]))[0]), 0.0, tol=1e-15)
    check("SiLU -> identity for large x", float(silu(np.array([30.0]))[0]), 30.0, tol=1e-6)
    check("SiLU is not monotone (dips below 0 near -1.28)",
          float(silu(np.array([-1.278]))[0]) < -0.278)
    d, hidden = 32, 88          # ceil(8*32/3/8)*8 = 88
    Wg, Wu = rng.standard_normal((d, hidden)), rng.standard_normal((d, hidden))
    Wd = rng.standard_normal((hidden, d))
    xg = rng.standard_normal((4, d))
    # The gate matters: zeroing the gate weights kills the whole output.
    check("zero gate silences the MLP",
          swiglu_mlp(xg, np.zeros_like(Wg), Wu, Wd), np.zeros((4, d)), tol=1e-12)

    print("\n--- parameter accounting ---")
    blk = Block(d, 4, rng)
    actual = (blk.W_qkv.size + blk.W_o.size + blk.W_g.size + blk.W_u.size
              + blk.W_d.size + blk.g1.size + blk.g2.size)
    check("hand formula matches the built block", block_param_count(d, hidden), actual)
    # SwiGLU at 8/3 d within ~4% of a plain 4d-hidden MLP (2 * d * 4d params).
    plain_4d = 2 * d * (4 * d)
    gated = 3 * d * hidden
    print(f"      gated MLP {gated} vs plain 4d MLP {plain_4d} params "
          f"({100*gated/plain_4d:.0f}%)")
    check("8/3-rule keeps SwiGLU within ~5% of the 4d MLP budget",
          abs(gated - plain_4d) / plain_4d < 0.05)

    print("\n--- causality end to end ---")
    n = 8
    xs = rng.standard_normal((n, d))
    out_a = blk(xs)
    xs_mod = xs.copy()
    xs_mod[-1] += 3.0
    out_b = blk(xs_mod)
    check("perturbing the last token leaves rows 0..n-2 unchanged",
          out_a[:-1], out_b[:-1], tol=1e-12)
    check("...and does change the last row", float(np.abs(out_a[-1] - out_b[-1]).max()) > 1e-3)

    print("\n--- pre-norm vs post-norm: the regime matters ---")
    # A first version of this test stacked 24 blocks WITH 1/sqrt(2L) residual
    # branch scaling and asserted post-norm attenuates the input gradient. It
    # does not — the measured ratio was 1.1x — and the reason is the lesson:
    # down-scaled branches are precisely the fix (DeepNorm's insight) that makes
    # post-norm trainable. The pathology is a statement about branch gain near
    # 1, and it grows with depth. Both regimes are measured below.
    d_small, n_small = 12, 4
    x0 = rng.standard_normal((n_small, d_small))

    def stacks(depth, branch_gain):
        pre, post = [], []
        for i in range(depth):
            for lst, pn in ((pre, False), (post, True)):
                b = Block(d_small, 2, np.random.default_rng(100 + i), post_norm=pn)
                # Undo the constructor's 1/sqrt(2*24) scaling, then apply gain.
                b.W_o = b.W_o * np.sqrt(2 * 24) * branch_gain
                b.W_d = b.W_d * np.sqrt(2 * 24) * branch_gain
                lst.append(b)
        return pre, post

    # Regime 1 — weak (down-scaled) branches: BOTH layouts pass gradients fine.
    pre_w, post_w = stacks(24, 1.0 / np.sqrt(2 * 24))
    g_pre_w = input_grad_norm(pre_w, x0.copy())
    g_post_w = input_grad_norm(post_w, x0.copy())
    print(f"      weak branches, depth 24 : pre {g_pre_w:9.3e}  post {g_post_w:9.3e}  "
          f"ratio {g_pre_w/g_post_w:5.2f}")
    check("with down-scaled branches BOTH layouts are healthy (ratio in [0.5, 2])",
          0.5 < g_pre_w / g_post_w < 2.0)
    check("weak-branch pre-norm gradient is order one", 0.5 < g_pre_w < 500.0)

    # Regime 2 — branch gain pushed toward 1 (weights x3): the classical
    # pathology appears, and it COMPOUNDS with depth. A single probe input is
    # too noisy to assert the trend (one seed gave 4.2 -> 6.7, another 2.6 ->
    # 18), so the ratio is averaged over several probe inputs per depth.
    probes = [np.random.default_rng(500 + j).standard_normal((n_small, d_small))
              for j in range(3)]
    ratios = {}
    for depth in (4, 32):
        pre_s, post_s = stacks(depth, 3.0)
        rs = [input_grad_norm(pre_s, xp.copy()) / input_grad_norm(post_s, xp.copy())
              for xp in probes]
        ratios[depth] = float(np.mean(rs))
        print(f"      strong branches, depth {depth:2d}: mean pre/post ratio "
              f"{ratios[depth]:6.2f}   (per-probe: {' '.join(f'{r:.1f}' for r in rs)})")
    check("at unit-gain branches the pre/post gap grows with depth",
          ratios[32] > 1.5 * ratios[4])
    # Magnitude varies a lot across probes in a toy this small (1.5 to 5.9
    # observed), so assert a modest floor, not the headline number a single
    # lucky seed produces.
    check("by depth 32 post-norm is attenuated >2x relative to pre-norm",
          ratios[32] > 2.0)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # The eps-outside-the-sqrt bug: rmsnorm written as x/(sqrt(mean(x^2))+eps)
    # instead of x/sqrt(mean(x^2)+eps). Harmless-looking, wrong near zero:
    # correct scaling caps the output at ~x/sqrt(eps); the buggy form divides
    # by eps and amplifies a tiny input by up to 1/eps.
    tiny = np.full((1, 16), 1e-8)
    good = rmsnorm(tiny, np.ones(16), eps=1e-6)
    bad = tiny / (np.sqrt((tiny * tiny).mean(-1, keepdims=True)) + 1e-6)
    print(f"      ||correct(tiny)|| = {np.linalg.norm(good):.3e}   (bounded by 1/sqrt(eps))")
    print(f"      ||buggy(tiny)||   = {np.linalg.norm(bad):.3e}")
    check("correct form stays bounded on tiny inputs", float(np.abs(good).max()) < 0.05)
    check("buggy form amplifies tiny inputs ~100x more",
          np.linalg.norm(bad) > 100 * np.linalg.norm(good) / 2)

    summary()
