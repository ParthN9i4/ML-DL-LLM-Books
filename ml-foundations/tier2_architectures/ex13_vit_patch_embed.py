"""
ex13 — The ViT stem: patches, CLS token, position embeddings.  (Book: Chapter 20)

A Vision Transformer's first layer is the only part of it that knows it is
looking at an image, and it is four lines long: cut the picture into
non-overlapping PxP blocks, flatten each block into a vector, project all of
them with ONE shared matrix, prepend a learned CLS token, add a learned
position embedding. Everything after that is a plain transformer chewing on a
sequence. So the whole of "vision" in a ViT lives in this file.

The identity worth internalizing is that step one is not a new operation:

    patchify -> flatten -> shared projection  ==  Conv2d(C, D, kernel=P, stride=P)

Kernel size equal to stride means the windows tile the image with no overlap,
so each window's dot products are exactly one patch's projections, and the
projection matrix reshaped from (D, C*P*P) to (D, C, P, P) IS the convolution
kernel. This file implements both paths independently — one a reshape and a
matmul, the other an honest strided cross-correlation — and finds them equal to
1.8e-15, and equal to torch's nn.Conv2d to 3.1e-15.

The price of the stem is quadratic and it is paid by every layer above it.
Tokens number (H/P)*(W/P) + 1, so halving P quadruples the patch count and
multiplies attention's T^2 by just under 16 — measured here as 15.52x going
from P=32 to P=16 and 15.88x from 16 to 8, the shortfall being the one CLS
token. At 224x224 with d_model=768 that is 7.7, 119.2 and 1893.0 MFLOP of
attention per layer for P = 32, 16, 8.

Position embeddings are not decoration, and this file measures why. Attention
over a SET is exactly permutation-equivariant: shuffle the patch tokens and the
outputs come back shuffled, to 1.3e-15, with the CLS output moving by 4.4e-16.
A positionless ViT cannot tell a photograph from its jigsaw. Add the position
table and the same shuffle moves CLS by 3.0x the output's own rms. Change the
input resolution and that table has to be resampled onto the new grid — a
low-pass operation whose 14->16->14 round trip loses 1.2% of a smooth table and
28.1% of a white-noise one.

To learn: replace each function body with `pass` and reimplement.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402

try:
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


# ---------------------------------------------------------------------------
# The stem
# ---------------------------------------------------------------------------

def patchify(x, P):
    """(N, C, H, W) -> (N, T, C*P*P) with T = (H/P)*(W/P) non-overlapping blocks.

    The whole exercise lives in the middle line. Splitting H into (gh, P) and W
    into (gw, P) is free; what matters is that the two GRID axes must be pulled
    to the front TOGETHER and the two WITHIN-PATCH axes left together behind
    them. Get that transpose wrong and every shape still checks out.
    """
    N, C, H, W = x.shape
    assert H % P == 0 and W % P == 0, f"{H}x{W} is not divisible by P={P}"
    gh, gw = H // P, W // P
    # === YOUR CODE HERE ===
    x = x.reshape(N, C, gh, P, gw, P)
    x = x.transpose(0, 2, 4, 1, 3, 5)            # (N, gh, gw, C, P, P)
    return x.reshape(N, gh * gw, C * P * P)


def unpatchify(tokens, P, gh, gw, C):
    """Exact inverse of `patchify`: (N, T, C*P*P) -> (N, C, gh*P, gw*P)."""
    N = tokens.shape[0]
    # === YOUR CODE HERE ===
    x = tokens.reshape(N, gh, gw, C, P, P)
    x = x.transpose(0, 3, 1, 4, 2, 5)            # (N, C, gh, P, gw, P)
    return x.reshape(N, C, gh * P, gw * P)


def patch_embed_reshape(x, W_lin, b_lin, P):
    """Patchify, then ONE shared linear projection. Returns (N, T, D)."""
    # === YOUR CODE HERE ===
    return patchify(x, P) @ W_lin.T + b_lin


def patch_embed_conv(x, W_lin, b_lin, P):
    """The same map written as a stride-P convolution with a PxP kernel.

    Nothing here calls `patchify`. The loop runs P*P times, each iteration
    grabbing the strided view of every window's (a, c) pixel and accumulating
    its contribution — an honest cross-correlation, exactly as in ex11. The
    kernel is the projection matrix reshaped: W_lin is (D, C*P*P) with feature
    order (channel, row-in-patch, col-in-patch), which is precisely the
    (D, C, P, P) layout `nn.Conv2d` wants. That reshape being a no-op is the
    whole equivalence.
    """
    N, C, H, W = x.shape
    D = W_lin.shape[0]
    gh, gw = H // P, W // P
    k = W_lin.reshape(D, C, P, P)
    # === YOUR CODE HERE ===
    out = np.zeros((N, D, gh, gw))
    for a in range(P):
        for c in range(P):
            win = x[:, :, a:a + P * gh:P, c:c + P * gw:P]      # (N, C, gh, gw)
            out += np.einsum("ncij,dc->ndij", win, k[:, :, a, c])
    tokens = out.reshape(N, D, gh * gw).transpose(0, 2, 1)     # (N, T, D)
    return tokens + b_lin


def add_cls_and_pos(tokens, cls, pos):
    """Prepend the CLS token to every sequence and add the learned position table.

    tokens (N, T, D), cls (D,), pos (1+T, D) -> (N, 1+T, D).
    The CLS row of `pos` is a real learned parameter too, but it encodes no
    location — which is exactly why it must be carved out again whenever the
    table is resized or the sequence is folded back into a spatial grid.
    """
    N, T, D = tokens.shape
    # === YOUR CODE HERE ===
    cls_row = np.broadcast_to(cls, (N, 1, D))
    return np.concatenate([cls_row, tokens], axis=1) + pos


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def self_attention(z, Wq, Wk, Wv):
    """Single-head self-attention, no mask. (..., T, D) -> (..., T, D).

    Deliberately positionless: nothing in this function can tell you where a
    token sits in the sequence. That is the property section 3 measures.
    """
    d = Wq.shape[1]
    # === YOUR CODE HERE ===
    q, k, v = z @ Wq, z @ Wk, z @ Wv
    w = softmax(q @ np.swapaxes(k, -1, -2) / np.sqrt(d))
    return w @ v


# ---------------------------------------------------------------------------
# Position-embedding interpolation
# ---------------------------------------------------------------------------

def _cubic_taps(t, A=-0.75):
    """Keys' cubic convolution kernel at the four tap distances t+1, t, 1-t, 2-t.

    A = -0.75 is the constant OpenCV and torch use for `bicubic`, and it is the
    one this file matches to 1e-12. Pillow is the notable exception: its
    BICUBIC filter uses A = -0.5, so a PIL-resized position table does NOT
    agree with a torch-resized one — measured offline against Pillow 12.3 on a
    32->64 upsample of data in [0,1), interior max|PIL - ours| is 6.7e-08 with
    A = -0.5 but 7.3e-02 with A = -0.75 (and max|PIL - torch| is the same
    7.3e-02). If a ported ViT's interpolated position table is subtly off, this
    constant is the first thing to check.
    """
    def wk(x):
        x = abs(x)
        if x <= 1.0:
            return (A + 2) * x ** 3 - (A + 3) * x ** 2 + 1
        if x < 2.0:
            return A * x ** 3 - 5 * A * x ** 2 + 8 * A * x - 4 * A
        return 0.0
    return [wk(t + 1.0), wk(t), wk(1.0 - t), wk(2.0 - t)]


def resample_matrix(n_in, n_out, mode="bicubic"):
    """(n_out, n_in) matrix implementing 1-D resampling.

    Interpolation is LINEAR in the samples, so the entire operation is a matrix
    — which is what makes "interpolate up, interpolate back" a composition of
    two matrices whose product can be inspected directly.

    Uses torch's align_corners=False convention: output pixel i has centre
    (i + 0.5)/scale - 0.5 in input coordinates, and taps that fall off the edge
    are clamped to the border sample.
    """
    Mx = np.zeros((n_out, n_in))
    scale = n_in / n_out
    # === YOUR CODE HERE ===
    for i in range(n_out):
        src = (i + 0.5) * scale - 0.5
        if mode == "bilinear":
            src = max(src, 0.0)                       # torch clamps for bilinear
            i0 = int(np.floor(src))
            t = src - i0
            taps, wts = (0, 1), (1.0 - t, t)
        else:
            i0 = int(np.floor(src))                   # ...but not for bicubic
            t = src - i0
            taps, wts = (-1, 0, 1, 2), _cubic_taps(t)
        for off, wgt in zip(taps, wts):
            Mx[i, min(max(i0 + off, 0), n_in - 1)] += wgt
    return Mx


def resample_grid(grid, new_h, new_w, mode="bicubic"):
    """(h, w, D) -> (new_h, new_w, D), separably: one matrix per spatial axis."""
    h, w, D = grid.shape
    # === YOUR CODE HERE ===
    Aw = resample_matrix(w, new_w, mode)
    Ah = resample_matrix(h, new_h, mode)
    g = np.einsum("qw,hwd->hqd", Aw, grid)
    return np.einsum("ph,hqd->pqd", Ah, g)


def interpolate_pos_embed(pos, gh, gw, new_gh, new_gw, mode="bicubic"):
    """Resize a (1 + gh*gw, D) position table to a (1 + new_gh*new_gw, D) one.

    Row 0 is CLS and is copied through untouched: it has no spatial location,
    so folding it into the grid would resample garbage into every corner.
    """
    D = pos.shape[1]
    assert pos.shape[0] == 1 + gh * gw
    # === YOUR CODE HERE ===
    cls_pos, grid = pos[:1], pos[1:].reshape(gh, gw, D)
    new_grid = resample_grid(grid, new_gh, new_gw, mode)
    return np.concatenate([cls_pos, new_grid.reshape(new_gh * new_gw, D)], axis=0)


# ---------------------------------------------------------------------------
# Cost bookkeeping
# ---------------------------------------------------------------------------

def n_tokens(H, W, P, cls=True):
    """(H/P)*(W/P) patches, plus one for CLS."""
    # === YOUR CODE HERE ===
    return (H // P) * (W // P) + (1 if cls else 0)


def attention_flops(T, D):
    """Q K^T and (weights) V are both (T,D)x(D,T) and (T,T)x(T,D) matmuls:
    2*T*T*D each, so 4*T^2*D. The projections are only O(T*D^2)."""
    # === YOUR CODE HERE ===
    return 4 * T * T * D


# ---------------------------------------------------------------------------
# The classic bug, hoisted to module level so both sections can use it
# ---------------------------------------------------------------------------

def patchify_wrong_axis_order(x, P):
    """THE bug: flatten H and W together, then chop the pixel stream into T.

    `patchify` splits H into (gh, P) and W into (gw, P) BEFORE grouping; this
    collapses H and W into one flat axis of H*W pixels per channel and cuts it
    into T contiguous chunks of P*P. The shapes come out identical to `patchify`'s, so
    nothing raises. What you get is not a grid of PxP blocks but a stack of
    horizontal STRIPS: each token is (P*P/W) whole image rows.
    """
    N, C, H, W = x.shape
    gh, gw = H // P, W // P
    return (x.reshape(N, C, gh * gw, P * P)
             .transpose(0, 2, 1, 3)
             .reshape(N, gh * gw, C * P * P))


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # -----------------------------------------------------------------------
    print("\n--- patch embedding IS a stride-P convolution ---")
    # -----------------------------------------------------------------------
    N, C, H, W, P, D = 2, 3, 32, 32, 8, 16
    gh, gw = H // P, W // P
    x = rng.standard_normal((N, C, H, W))
    W_lin = rng.standard_normal((D, C * P * P)) / np.sqrt(C * P * P)
    b_lin = rng.standard_normal(D)

    tok_reshape = patch_embed_reshape(x, W_lin, b_lin, P)
    tok_conv = patch_embed_conv(x, W_lin, b_lin, P)
    print(f"      image {N}x{C}x{H}x{W}, P={P} -> {gh}x{gw} grid, "
          f"{gh*gw} patches of {C*P*P} numbers -> D={D}")
    check("patch embed output shape is (N, T, D)", tok_reshape.shape == (N, gh * gw, D))
    check("reshape-and-project == stride-P convolution", tok_conv, tok_reshape, tol=1e-13)

    # patchify is a pure permutation of the pixels: nothing is lost.
    check("unpatchify(patchify(x)) == x exactly", unpatchify(patchify(x, P), P, gh, gw, C),
          x, tol=0)
    # And patch 0 really is the top-left PxP block of channel 0.
    check("token 0 is the top-left PxP block",
          patchify(x, P)[0, 0, :P * P].reshape(P, P), x[0, 0, :P, :P], tol=0)

    if HAVE_TORCH:
        conv = torch.nn.Conv2d(C, D, kernel_size=P, stride=P, bias=True).double()
        with torch.no_grad():
            conv.weight.copy_(torch.tensor(W_lin.reshape(D, C, P, P)))
            conv.bias.copy_(torch.tensor(b_lin))
        ref = conv(torch.tensor(x)).detach().numpy()             # (N, D, gh, gw)
        ref_tok = ref.reshape(N, D, gh * gw).transpose(0, 2, 1)
        check("torch nn.Conv2d(kernel=P, stride=P) gives the same tokens",
              tok_reshape, ref_tok, tol=1e-13)
        # ...and the projection matrix reshaped to (D,C,P,P) IS the conv kernel.
        check("the conv kernel is literally W_lin.reshape(D, C, P, P)",
              conv.weight.detach().numpy(), W_lin.reshape(D, C, P, P), tol=0)
    else:
        print("  ....  [skipped: torch not installed] nn.Conv2d cross-check")

    # -----------------------------------------------------------------------
    print("\n--- CLS token and position table ---")
    # -----------------------------------------------------------------------
    T_p = gh * gw
    # Scale CLS and the position table to the token rms. A freshly initialised
    # ViT uses std 0.02 for both, but in a TRAINED model the position signal is
    # comparable in size to the patch signal — which is the regime where the
    # permutation experiment below is honest.
    tok_rms = float(np.sqrt((tok_reshape ** 2).mean()))
    cls = rng.standard_normal(D) * tok_rms
    pos = rng.standard_normal((1 + T_p, D)) * tok_rms
    print(f"      patch-token rms {tok_rms:.3f}; position table drawn at the same scale")
    z = add_cls_and_pos(tok_reshape, cls, pos)
    check("sequence grows by exactly one token", z.shape == (N, 1 + T_p, D))
    check("row 0 is CLS + its own position row", z[0, 0], cls + pos[0], tol=1e-15)
    check("row 1 is patch 0 + position row 1", z[0, 1], tok_reshape[0, 0] + pos[1], tol=1e-15)

    # -----------------------------------------------------------------------
    print("\n--- what the patch size costs: T and T^2 ---")
    # -----------------------------------------------------------------------
    H_v = W_v = 224
    D_v = 768                                   # ViT-Base
    print(f"      {H_v}x{W_v} input, d_model={D_v}")
    print(f"      {'P':>4} {'grid':>8} {'patches':>9} {'T (+CLS)':>9} "
          f"{'T^2':>10} {'attn MFLOP':>11} {'vs P=32':>9}")
    rows = []
    for Pv in (32, 16, 8):
        Tv = n_tokens(H_v, W_v, Pv)
        fl = attention_flops(Tv, D_v)
        rows.append((Pv, Tv, fl))
        base = rows[0]
        print(f"      {Pv:4d} {f'{H_v//Pv}x{W_v//Pv}':>8} {Tv-1:9d} {Tv:9d} "
              f"{Tv*Tv:10d} {fl/1e6:11.1f} {fl/base[2]:8.1f}x")
    (_, T32, f32), (_, T16, f16), (_, T8, f8) = rows
    # The mechanism: halving P quadruples the PATCH count exactly, so the
    # quadratic factor grows by 16 — approached from below because CLS adds a
    # constant 1 that matters less and less.
    check("halving P exactly quadruples the patch count",
          (T16 - 1) == 4 * (T32 - 1) and (T8 - 1) == 4 * (T16 - 1))
    r_32_16 = (T16 * T16) / (T32 * T32)
    r_16_8 = (T8 * T8) / (T16 * T16)
    print(f"      quadratic factor grows {r_32_16:.4f}x (32->16) then "
          f"{r_16_8:.4f}x (16->8); the CLS-free limit is exactly 16")
    check("each halving of P costs just under 16x in attention", r_32_16 < 16.0 and r_16_8 < 16.0)
    check("and the deficit shrinks as CLS becomes negligible", r_16_8 > r_32_16)
    check("attention flops track T^2 exactly", f8 / f32, (T8 / T32) ** 2, tol=1e-9)

    # ...and the honest version of "quadratic is bad": at ViT-Base sizes the
    # T^2 term is NOT yet the bottleneck. Everything else in a block is linear
    # in T and quadratic in d_model — 8*T*d^2 for the q,k,v,out projections plus
    # 16*T*d^2 for a 4x MLP — so attention only takes over when T gets large.
    def linear_flops(T, D):
        """(q,k,v,out) projections + a 4x MLP, all O(T * D^2)."""
        return 8 * T * D * D + 16 * T * D * D

    print(f"      {'T':>6} {'score matrix':>14} {'attn MFLOP':>11} "
          f"{'linear MFLOP':>13} {'attn share':>11}")
    shares = []
    for Tv in (T32, T16, T8):
        a_fl, l_fl = attention_flops(Tv, D_v), linear_flops(Tv, D_v)
        shares.append(a_fl / (a_fl + l_fl))
        print(f"      {Tv:6d} {Tv * Tv * 8 / 2**20:11.2f} MB {a_fl/1e6:11.1f} "
              f"{l_fl/1e6:13.1f} {shares[-1]*100:10.1f}%")
    check("attention's share of the block grows with every halving of P",
          shares[0] < shares[1] < shares[2])
    # The crossover is a closed form: 4*T^2*d = 24*T*d^2  <=>  T = 6*d.
    T_cross = 6 * D_v
    print(f"      attention overtakes the linear work at T = 6*d_model = {T_cross}, "
          f"i.e. P = {H_v / np.sqrt(T_cross):.1f} on a {H_v}x{W_v} image")
    check("attention flops equal the linear flops exactly at T = 6*d_model",
          attention_flops(T_cross, D_v), linear_flops(T_cross, D_v))
    check("...so at ViT-Base P=16 the quadratic term is still a minority cost",
          shares[1] < 0.5)

    # -----------------------------------------------------------------------
    print("\n--- why position embeddings are not optional ---")
    # -----------------------------------------------------------------------
    # Attention over a SET is permutation-equivariant. Permute the patch tokens
    # (leaving CLS at slot 0) and the outputs come back permuted the same way.
    Wq = rng.standard_normal((D, D)) / np.sqrt(D)
    Wk = rng.standard_normal((D, D)) / np.sqrt(D)
    Wv = rng.standard_normal((D, D)) / np.sqrt(D)

    # Before measuring what self-attention does to a permutation, pin down that
    # it IS attention. Permutation equivariance holds for essentially any
    # token-symmetric map, so on its own it says nothing about the 1/sqrt(d)
    # scale, which axis the softmax normalises, or which projection feeds Q, K
    # and V. Hand-compute a 3-token, d=2 case in pure Python — no numpy, no
    # reuse of this file's `softmax` — and demand agreement. The three weight
    # matrices are deliberately all different (so swapping any two changes the
    # answer), d = 2 makes 1/sqrt(d) differ from 1/d, and the scores are
    # asymmetric so normalising over queries differs from normalising over keys.
    z_h = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
    Wq_h = [[1.0, 0.5], [0.0, 1.0]]
    Wk_h = [[0.0, 1.0], [1.0, -0.5]]
    Wv_h = [[2.0, 0.0], [0.0, -1.0]]

    def _mm(a, b):
        return [[sum(a[i][t] * b[t][j] for t in range(len(b)))
                 for j in range(len(b[0]))] for i in range(len(a))]

    q_h, k_h, v_h = _mm(z_h, Wq_h), _mm(z_h, Wk_h), _mm(z_h, Wv_h)
    hand = []
    for i_h in range(3):
        sc = [sum(q_h[i_h][c] * k_h[j][c] for c in range(2)) / math.sqrt(2.0)
              for j in range(3)]
        ex = [math.exp(t_) for t_ in sc]
        tot = sum(ex)
        hand.append([sum((ex[j] / tot) * v_h[j][m] for j in range(3))
                     for m in range(2)])
    got_h = self_attention(np.array(z_h), np.array(Wq_h), np.array(Wk_h),
                           np.array(Wv_h))
    print(f"      hand-computed 3-token attention row 0 = "
          f"({hand[0][0]:.6f}, {hand[0][1]:.6f}); self_attention gives "
          f"({got_h[0, 0]:.6f}, {got_h[0, 1]:.6f})")
    check("self_attention == softmax(Q K^T / sqrt(d)) V, hand-computed",
          got_h, np.array(hand), tol=1e-14)

    # And a property that pins the softmax axis on the real-sized problem: the
    # weights normalise over KEYS, so every output row is a CONVEX combination
    # of the value rows and cannot leave their per-coordinate range.
    z_dbg = add_cls_and_pos(tok_reshape, cls, pos)
    v_dbg = z_dbg @ Wv
    out_dbg = self_attention(z_dbg, Wq, Wk, Wv)
    lo_v = v_dbg.min(axis=1, keepdims=True)
    hi_v = v_dbg.max(axis=1, keepdims=True)
    margin = float(np.minimum(out_dbg - lo_v, hi_v - out_dbg).min())
    print(f"      every attention output sits inside the hull of the value rows, "
          f"tightest margin {margin:.3e}")
    check("attention output is a convex combination of the value rows",
          margin >= -1e-12)

    perm = rng.permutation(T_p)
    seq_ids = np.concatenate([[0], 1 + perm])          # CLS stays put

    def stem(tokens, use_pos):
        z_ = np.concatenate([np.broadcast_to(cls, (tokens.shape[0], 1, D)), tokens], axis=1)
        return z_ + pos if use_pos else z_

    for use_pos in (False, True):
        base = self_attention(stem(tok_reshape, use_pos), Wq, Wk, Wv)
        shuf = self_attention(stem(tok_reshape[:, perm], use_pos), Wq, Wk, Wv)
        equivar = float(np.abs(shuf - base[:, seq_ids]).max())
        cls_gap = float(np.abs(shuf[:, 0] - base[:, 0]).max())
        scale = float(np.sqrt((base ** 2).mean()))
        tag = "with pos" if use_pos else "no pos  "
        print(f"      {tag}: |attn(perm z) - perm attn(z)|max = {equivar:.3e}, "
              f"|CLS shift| = {cls_gap:.3e}   (output rms {scale:.3f})")
        if not use_pos:
            eq_nopos, cls_nopos = equivar, cls_gap
        else:
            eq_pos, cls_pos_gap, rms_pos = equivar, cls_gap, scale
    check("without position embeddings attention is exactly permutation-equivariant",
          eq_nopos, 0.0, tol=1e-14)
    check("...so the CLS output cannot tell a shuffled image from the original",
          cls_nopos, 0.0, tol=1e-14)
    check("adding position embeddings destroys equivariance by >1e10x",
          eq_pos / max(eq_nopos, np.finfo(float).tiny) > 1e10)
    print(f"      with pos, shuffling the patches moves the CLS output by "
          f"{cls_pos_gap/rms_pos:.2f}x the output's own rms")
    check("and the CLS shift is a sizeable fraction of the signal itself",
          cls_pos_gap > 0.1 * rms_pos)

    # -----------------------------------------------------------------------
    print("\n--- position-embedding interpolation for a new resolution ---")
    # -----------------------------------------------------------------------
    g0, g1, Dp = 14, 16, 32                       # 224/16 -> 256/16
    yy, xx = np.meshgrid(np.linspace(0, 1, g0), np.linspace(0, 1, g0), indexing="ij")
    # A learned ViT position table is empirically SMOOTH and low-frequency
    # (neighbouring patches get similar embeddings). Model that with a few
    # low-order 2-D cosines, and keep a white-noise table around as the control.
    smooth = np.zeros((g0, g0, Dp))
    for (a, b) in [(0, 1), (1, 0), (1, 1), (2, 1), (1, 2)]:
        ca = rng.standard_normal(Dp)
        smooth += (np.cos(np.pi * (a * yy + b * xx))[:, :, None] * ca)
    smooth /= np.sqrt((smooth ** 2).mean())
    noise = rng.standard_normal((g0, g0, Dp))

    # `resample_grid` takes new_h and new_w separately, and every grid this file
    # actually resizes is square (14->16, 14->7, 14->14), so a square-only
    # cross-check applies the SAME matrix to both axes and cannot tell them
    # apart — the one bug the two-argument signature exists to expose. Slice a
    # rectangular sub-grid out of `smooth` (no fresh rng draws, so the stream
    # below is untouched) and make the reference comparisons non-square.
    rect = np.ascontiguousarray(smooth[:10, :6])                # (10, 6, Dp)
    cases = ((smooth, g1, g1), (smooth, g1, 7), (rect, 13, 9), (rect, 5, 11))
    if HAVE_TORCH:
        for mode in ("bilinear", "bicubic"):
            for (src, sh, sw) in cases:
                mine = resample_grid(src, sh, sw, mode)
                t_in = torch.tensor(src.transpose(2, 0, 1)[None])
                ref = F.interpolate(t_in, size=(sh, sw), mode=mode,
                                    align_corners=False).numpy()[0].transpose(1, 2, 0)
                check(f"{mode} {src.shape[0]}x{src.shape[1]}->{sh}x{sw} matches "
                      f"torch F.interpolate", mine, ref, tol=1e-12)
    else:
        print("  ....  [skipped: torch not installed] F.interpolate cross-check")

    # Torch-free versions of the same discrimination, so the numpy-only run also
    # separates the axes. Resampling is separable, so (1) transposing the two
    # spatial axes of the input must transpose the output, and (2) since
    # resample_matrix(n, n) is exactly the identity, doing the axes in two calls
    # must equal doing both in one.
    for mode in ("bilinear", "bicubic"):
        check(f"{mode} resample commutes with transposing the spatial axes",
              resample_grid(rect, 13, 9, mode).transpose(1, 0, 2),
              resample_grid(np.ascontiguousarray(rect.transpose(1, 0, 2)), 9, 13, mode),
              tol=1e-13)
        check(f"{mode} resample is separable: one axis at a time == both at once",
              resample_grid(rect, 13, 9, mode),
              resample_grid(resample_grid(rect, 13, 6, mode), 13, 9, mode), tol=1e-13)

    # Partition of unity: both kernels sum to 1, so a constant table survives.
    for mode in ("bilinear", "bicubic"):
        Amat = resample_matrix(g0, g1, mode)
        check(f"{mode} rows sum to 1 (a constant grid stays constant)",
              Amat.sum(axis=1), np.ones(g1), tol=1e-14)

    print(f"      {'grid path':>16} {'mode':>9} {'smooth table':>14} {'white noise':>13}")
    for (ga, gb) in ((g0, g1), (g0, 7)):
        for mode in ("bilinear", "bicubic"):
            errs = []
            for tbl in (smooth, noise):
                back = resample_grid(resample_grid(tbl, gb, gb, mode), ga, ga, mode)
                errs.append(float(np.sqrt(((back - tbl) ** 2).mean() / (tbl ** 2).mean())))
            print(f"      {f'{ga}->{gb}->{ga}':>16} {mode:>9} {errs[0]:13.4f}  {errs[1]:12.4f}")
            if (ga, gb, mode) == (g0, g1, "bicubic"):
                up_smooth, up_noise = errs
            if (ga, gb, mode) == (g0, 7, "bicubic"):
                down_smooth, down_noise = errs
    print(f"      interpolate 14->16 and back with bicubic: the smooth table returns "
          f"with {up_smooth*100:.2f}% relative error, white noise with {up_noise*100:.1f}%")
    # Mechanism 1: resampling is a low-pass, so what it destroys is high spatial
    # frequency. The smooth table survives the round trip far better than noise.
    check("interpolation round-trip hurts a white-noise table far more than a smooth one",
          up_noise > 5 * up_smooth)
    # Mechanism 2: going through a COARSER grid throws away rank. The 14->7->14
    # round trip factors through a 7x7 grid, so at most 49/196 of the table can
    # survive, whatever the interpolation.
    R_up = resample_matrix(g1, g0) @ resample_matrix(g0, g1)      # 14 -> 16 -> 14
    R_down = resample_matrix(7, g0) @ resample_matrix(g0, 7)       # 14 ->  7 -> 14
    rank_up = int(np.linalg.matrix_rank(np.kron(R_up, R_up), tol=1e-8))
    rank_down = int(np.linalg.matrix_rank(np.kron(R_down, R_down), tol=1e-8))
    print(f"      round-trip operator rank: 14->16->14 is {rank_up}/{g0*g0}, "
          f"14->7->14 is {rank_down}/{g0*g0}")
    check("upsizing and coming back is full rank (nothing is structurally lost)",
          rank_up == g0 * g0)
    check("downsizing and coming back is rank-limited by the coarse grid",
          rank_down == 49)
    check("...which is why the coarse round trip is far more destructive",
          down_smooth > 3 * up_smooth)
    # And the residual that remains even at full rank is real: R is not I.
    print(f"      ||R(14->16->14) - I||_max = {np.abs(R_up - np.eye(g0)).max():.4f}")
    check("the full-rank round trip is still not the identity",
          float(np.abs(R_up - np.eye(g0)).max()) > 1e-3)

    # The whole-table API keeps CLS out of the resampling.
    pos_v = np.concatenate([rng.standard_normal((1, Dp)) * 5.0,
                            smooth.reshape(g0 * g0, Dp)], axis=0)
    pos_big = interpolate_pos_embed(pos_v, g0, g0, g1, g1)
    check("interpolated table has 1 + new_gh*new_gw rows",
          pos_big.shape == (1 + g1 * g1, Dp))
    check("the CLS position row is carried through untouched", pos_big[0], pos_v[0], tol=0)
    # The row count and row 0 say nothing about the grid, and the off-by-one the
    # docstring warns about — `pos[:-1]` instead of `pos[1:]`, folding CLS into
    # the grid and dropping the last patch — leaves both of them intact. Pin the
    # grid itself, three ways.
    # (i) The patch rows ARE `resample_grid` of the CLS-free table, exactly.
    check("the patch rows are exactly resample_grid of the CLS-free table",
          pos_big[1:],
          resample_grid(pos_v[1:].reshape(g0, g0, Dp), g1, g1).reshape(g1 * g1, Dp),
          tol=0)
    # (ii) Linearity, with no reference implementation at all: the CLS row is
    # not an input to any patch row, so rewriting it must leave every patch row
    # bit-identical. Folding CLS into the grid fails this immediately.
    pos_alt = pos_v.copy()
    pos_alt[0] = -7.0
    alt_big = interpolate_pos_embed(pos_alt, g0, g0, g1, g1)
    check("rewriting ONLY the CLS row cannot move a single patch row",
          alt_big[1:], pos_big[1:], tol=0)
    check("...and the rewritten CLS row is the one that comes back",
          alt_big[0], pos_alt[0], tol=0)
    # (iii) Partition of unity end to end: a constant grid must resize to the
    # same constant no matter what the CLS row holds. This is what catches a
    # resampler that quietly returns zeros, and it needs no reference either.
    const_pos = np.concatenate([np.full((1, Dp), 5.0),
                                np.full((g0 * g0, Dp), 0.25)], axis=0)
    check("a constant position grid resizes to the same constant",
          interpolate_pos_embed(const_pos, g0, g0, g1, g1)[1:],
          np.full((g1 * g1, Dp), 0.25), tol=1e-14)
    # The CLS row was drawn 5x the grid's scale precisely so a leak would be
    # loud. The ceiling is derived, not guessed: |out[p,q,d]| =
    # |sum_h sum_w Ah[p,h] Aw[q,w] grid[h,w,d]| <= max|grid| * (max row L1 of
    # Ah) * (max row L1 of Aw), and both axes use the same 14->16 matrix here.
    gain = float(np.abs(resample_matrix(g0, g1)).sum(axis=1).max()) ** 2
    grid_in_max = float(np.abs(pos_v[1:]).max())
    leak_bound = grid_in_max * gain
    grid_out_max = float(np.abs(pos_big[1:]).max())
    print(f"      max|CLS row| {float(np.abs(pos_v[0]).max()):.3f}, "
          f"max|grid in| {grid_in_max:.3f}, max|grid out| {grid_out_max:.3f}; "
          f"bicubic gain {gain:.4f}x bounds the grid by {leak_bound:.3f}")
    check("the 5x CLS row never leaks into the resampled grid",
          grid_out_max < leak_bound)

    # -----------------------------------------------------------------------
    print("\n--- break it ---")
    # -----------------------------------------------------------------------

    # (a) The flat-chunking bug: collapse H and W into one axis of H*W pixels
    # and cut it into T contiguous chunks of P*P, instead of splitting H into
    # (gh, P) and W into (gw, P) first. Every shape matches `patchify`'s
    # exactly, so nothing raises.
    print("\n  (a) patchify that chops the flat pixel stream instead of tiling it")
    good = patchify(x, P)
    bad = patchify_wrong_axis_order(x, P)
    print(f"      correct shape {good.shape}, buggy shape {bad.shape}  "
          f"-> {'identical, nothing raises' if good.shape == bad.shape else 'DIFFERENT'}")
    check("the bug leaves every shape intact", good.shape == bad.shape)

    # What the tokens actually are: correct = a PxP block, buggy = whole rows.
    rows_per_tok = P * P // W
    check("buggy token 0 is a strip of whole image rows, not a PxP block",
          bad[0, 0, :P * P].reshape(rows_per_tok, W), x[0, 0, :rows_per_tok, :], tol=0)
    # Measure the scrambling with a coordinate image: how many distinct image
    # columns does one token touch?
    col_img = np.broadcast_to(np.arange(W, dtype=float), (1, 1, H, W)).copy()
    row_img = np.broadcast_to(np.arange(H, dtype=float)[:, None], (1, 1, H, W)).copy()
    for nm, fn in (("correct", patchify), ("buggy", patchify_wrong_axis_order)):
        cols = fn(col_img, P)[0, 0]
        rws = fn(row_img, P)[0, 0]
        print(f"      {nm:>8} token 0 spans {len(np.unique(rws)):3d} image rows "
              f"x {len(np.unique(cols)):3d} image columns")
        if nm == "correct":
            n_r_ok, n_c_ok = len(np.unique(rws)), len(np.unique(cols))
        else:
            n_r_bad, n_c_bad = len(np.unique(rws)), len(np.unique(cols))
    check("a correct token is a PxP square window", (n_r_ok, n_c_ok) == (P, P))
    check("the buggy token spans the full image width", n_c_bad == W)
    check("...and only P*P/W rows", n_r_bad == rows_per_tok)

    # And now the part that makes this bug survive code review: NOTHING CRASHES
    # AND IT STILL TRAINS. Fit a ViT-shaped head — shared patch projection to D
    # channels, then a linear readout over all tokens — to a target that is a
    # linear functional of the image, built to be exactly rank-1 in the CORRECT
    # patch basis.
    print("\n      does it still train?")
    Ptr, Htr, Ftr = 8, 32, 8 * 8              # 1 channel, 4x4 grid -> T=16, F=64
    Ttr = (Htr // Ptr) ** 2
    idx_img = np.arange(Htr * Htr, dtype=float).reshape(1, 1, Htr, Htr)
    idx_ok = patchify(idx_img, Ptr)[0].astype(int)             # (T, F) pixel ids
    idx_bad = patchify_wrong_axis_order(idx_img, Ptr)[0].astype(int)

    a_true = rng.standard_normal(Ttr)
    w_true = rng.standard_normal(Ftr)
    G_pix = np.zeros(Htr * Htr)
    G_pix[idx_ok.ravel()] = np.outer(a_true, w_true).ravel()   # rank 1 in the right basis
    G_pix /= np.linalg.norm(G_pix)                             # so E[y^2] = 1
    M_ok = G_pix[idx_ok]
    M_bad = G_pix[idx_bad]
    rk_ok = int(np.linalg.matrix_rank(M_ok, tol=1e-9))
    rk_bad = int(np.linalg.matrix_rank(M_bad, tol=1e-9))
    print(f"      the very same pixel functional, viewed in the two patch bases: "
          f"rank {rk_ok} -> rank {rk_bad}")
    check("the target is exactly rank 1 in the correct patch basis", rk_ok == 1)
    check("...and full rank once the bug regroups the same pixels",
          rk_bad == min(Ttr, Ftr))

    n_tr, n_te, D_tr = 3000, 3000, 8
    X_tr = rng.standard_normal((n_tr, 1, Htr, Htr))
    X_te = rng.standard_normal((n_te, 1, Htr, Htr))
    y_tr = X_tr.reshape(n_tr, -1) @ G_pix
    y_te = X_te.reshape(n_te, -1) @ G_pix

    def train_stem(patch_fn, Xd, yd, steps=1200, lr=0.4):
        """Patch-embed (F->D, shared) + linear readout over tokens. The model
        realises M_hat = Head @ Emb^T, i.e. any rank-D matrix in whatever patch
        basis `patch_fn` induces. Plain full-batch gradient descent."""
        Pm = patch_fn(Xd, Ptr)                         # (n, T, F)
        n_d = len(yd)
        g = np.random.default_rng(7)
        Emb = g.standard_normal((Ftr, D_tr)) * 0.05
        Head = g.standard_normal((Ttr, D_tr)) * 0.05
        first = None
        for step in range(steps):
            M_hat = Head @ Emb.T                       # (T, F)
            r = np.einsum("ntf,tf->n", Pm, M_hat) - yd
            loss = float(np.mean(r * r))
            if step == 0:
                first = loss
            gM = 2.0 * np.einsum("n,ntf->tf", r, Pm) / n_d
            gH, gE = gM @ Emb, gM.T @ Head
            Head -= lr * gH
            Emb -= lr * gE
        M_hat = Head @ Emb.T
        Pm_te = patch_fn(X_te, Ptr)
        te = float(np.mean((np.einsum("ntf,tf->n", Pm_te, M_hat) - y_te) ** 2))
        return first, loss, te, M_hat

    var_y = float(np.mean(y_te ** 2))
    f_ok, l_ok, te_ok, Mh_ok = train_stem(patchify, X_tr, y_tr)
    f_bad, l_bad, te_bad, Mh_bad = train_stem(patchify_wrong_axis_order, X_tr, y_tr)
    print(f"      held-out target variance {var_y:.3f}")
    print(f"      correct patchify: train {f_ok:7.4f} -> {l_ok:9.2e}   "
          f"test {te_ok:9.2e}  ({te_ok/var_y*100:7.3f}% of signal)")
    print(f"      buggy   patchify: train {f_bad:7.4f} -> {l_bad:9.2e}   "
          f"test {te_bad:9.2e}  ({te_bad/var_y*100:7.3f}% of signal)")
    check("the buggy stem trains without raising anything and the loss falls a lot",
          l_bad < 0.5 * f_bad)
    check("but it never gets near the correct stem's loss", te_bad > 100 * te_ok)
    # The mechanism, exactly: with i.i.d. pixels the population loss is
    # ||M_true - M_hat||_F^2 in whichever basis the model uses, so a rank-D
    # model can at best keep the top D singular values. The correct basis needs
    # rank 1; the scrambled basis spreads the same numbers over the full
    # spectrum, and the tail is the irreducible error.
    def rank_floor(M, r):
        s = np.linalg.svd(M, compute_uv=False)
        return float(1.0 - (s[:r] ** 2).sum() / (s ** 2).sum())

    floor_ok = rank_floor(M_ok, D_tr)
    floor_bad = rank_floor(M_bad, D_tr)
    obs_ok = float(((M_ok - Mh_ok) ** 2).sum() / (M_ok ** 2).sum())
    obs_bad = float(((M_bad - Mh_bad) ** 2).sum() / (M_bad ** 2).sum())
    print(f"      spectral floor for a rank-{D_tr} stem: {floor_ok:.2e} (correct basis) "
          f"vs {floor_bad:.4f} (scrambled)")
    print(f"      population residual reached: {obs_ok:.2e} (correct) "
          f"vs {obs_bad:.4f} (buggy)")
    # The floor is a rank statement, not a magnitude to guess at: a rank-D stem
    # keeps the top D singular values of whatever matrix the basis hands it.
    # Rank 1 <= D leaves nothing behind; rank 16 > D always leaves a tail.
    check("a rank-D stem fits the rank-1 correct basis with nothing left over",
          floor_ok < 1e-25)
    check("...and can never clear the full-rank scrambled basis", floor_bad > 0.0)
    check("Eckart-Young: the buggy run cannot beat its own spectral floor",
          obs_bad > floor_bad - 1e-9)
    # The run is converged, though not to six digits by 500 steps. Measured on
    # this exact stream, the buggy run's population residual is 0.0929203953 at
    # 500 steps, 0.0929133790 at 1200, 0.0929133777 at 3000 and 0.0929133777 at
    # 10000: 500 steps is already within 7.1e-6 of the limit, and from 1200 on
    # the residual is settled to 1.3e-9 absolute (1.4e-8 relative), i.e. seven
    # significant digits. What sits above the Frobenius floor is a
    # finite-sample effect, not an optimization failure: the empirical Gram of
    # n images in a 1024-dimensional patch space is not the identity, so the
    # empirically best rank-8 subspace is not quite the Frobenius one. Double
    # the training set and the gap shrinks.
    _, _, _, Mh_bad2 = train_stem(patchify_wrong_axis_order,
                                  np.concatenate([X_tr, X_te]),
                                  np.concatenate([y_tr, y_te]))
    obs_bad2 = float(((M_bad - Mh_bad2) ** 2).sum() / (M_bad ** 2).sum())
    print(f"      excess over the Eckart-Young floor: {obs_bad/floor_bad:.3f}x at "
          f"n={n_tr}, {obs_bad2/floor_bad:.3f}x at n={n_tr + n_te}")
    check("the excess over the floor is finite-sample and shrinks with more data",
          floor_bad < obs_bad2 < obs_bad)
    check("held-out error is the population residual", te_bad / var_y, obs_bad, tol=0.05)

    # (b) Folding tokens back to a grid without dropping CLS.
    print("\n  (b) forgetting to strip CLS before reshaping to a grid")
    # A dense-prediction head takes the T = 1 + gh*gw sequence and wants the
    # gh x gw spatial map back. Use per-patch mean intensity of a blob image so
    # the map is visually meaningful and the damage is measurable in pixels.
    Hb, Pb = 32, 4
    gb = Hb // Pb
    yy2, xx2 = np.mgrid[0:Hb, 0:Hb]
    blob = np.exp(-(((yy2 - 10.0) ** 2 + (xx2 - 14.0) ** 2) / 18.0))[None, None]
    feat = patchify(blob, Pb)[0].mean(axis=1)                  # (gb*gb,)
    cls_feat = float(feat.mean()) * 3.0                        # CLS carries a summary
    seq = np.concatenate([[cls_feat], feat])                   # (1 + gb*gb,)

    map_ok = seq[1:].reshape(gb, gb)                           # correct
    map_bad = seq[:-1].reshape(gb, gb)                         # off-by-one
    check("the off-by-one keeps the grid shape", map_ok.shape == map_bad.shape)
    check("but every patch is shifted one slot in raster order",
          map_bad.ravel()[1:], map_ok.ravel()[:-1], tol=0)
    check("and slot (0,0) now holds the CLS summary, not a patch",
          float(map_bad[0, 0]), cls_feat, tol=0)

    def centroid(m):
        s = m.sum()
        return (float((m * np.arange(gb)[:, None]).sum() / s),
                float((m * np.arange(gb)[None, :]).sum() / s))

    r_ok, c_ok = centroid(map_ok)
    r_bad, c_bad = centroid(map_bad)
    pk_ok = tuple(int(v) for v in np.unravel_index(int(np.argmax(map_ok)), map_ok.shape))
    pk_bad = tuple(int(v) for v in np.unravel_index(int(np.argmax(map_bad)), map_bad.shape))
    print(f"      blob centroid in patch units: correct ({r_ok:.2f}, {c_ok:.2f}), "
          f"buggy ({r_bad:.2f}, {c_bad:.2f})")
    print(f"      brightest patch: correct {pk_ok} -> buggy {pk_bad}")
    check("every patch slides one slot along the raster, so the peak moves one column",
          pk_bad == (pk_ok[0], pk_ok[1] + 1))
    # Anchor "misaligned" against a computed floor: the shifted map is as wrong
    # as the map of a DIFFERENT blob one patch away.
    blob_shift = np.exp(-(((yy2 - 10.0) ** 2 + (xx2 - (14.0 + Pb)) ** 2) / 18.0))[None, None]
    map_shift = patchify(blob_shift, Pb)[0].mean(axis=1).reshape(gb, gb)
    e_bug = float(np.sqrt(((map_bad - map_ok) ** 2).mean()))
    e_ref = float(np.sqrt(((map_shift - map_ok) ** 2).mean()))
    print(f"      rms(buggy - correct) = {e_bug:.4f};  "
          f"rms(one-patch-translated image - correct) = {e_ref:.4f}")
    check("the CLS off-by-one is as damaging as translating the image by a patch",
          e_bug > 0.5 * e_ref)

    summary()
