"""
ex12 — The convolution backward pass.  (Book: Chapter 16)

ex11 turned the forward convolution into a matrix product. This file asks the
only question that remains: what are its three gradients, and why do they look
the way they do?

A conv layer is bilinear — Y is linear in X for fixed W, and linear in W for
fixed X — so both gradients are themselves convolutions, and you can read them
off the forward index expression without ever writing a chain rule. Start from
Y[n,o,i,j] = sum_{c,a,b} W[o,c,a,b] * Xpad[n,c,i*s+a, j*s+b] and differentiate
by inspection. Differentiating by W[o,c,a,b] leaves whatever multiplied it, so
dW is a CORRELATION of the padded input with the upstream gradient, summed over
the batch and over every output position. Differentiating by Xpad[n,c,y,x]
collects every (i,j,a,b) with i*s+a = y, which reverses the sign on a and b:
dX is a CONVOLUTION of the upstream gradient with the kernel — equivalently a
correlation with the kernel rotated 180 degrees, zero-dilated by the stride,
and padded to full width. Nobody remembers those two sentences. Everybody can
re-derive them from the index expression in a minute, which is the actual skill.

The insight this file is built to make undeniable is that dX is not "some
transpose"; it is a convolution, and it is the ADJOINT of the forward map, so
<conv(X), G> = <X, dX> to machine precision for every X and G. That identity
is checked here with no reference implementation involved at all.

Both classic bugs are then committed on purpose. Forgetting to sum dW over the
batch is wrong by exactly the batch size and utterly invisible at N=1, which is
the size every unit test uses. Forgetting the 180-degree rotation in dX is
catastrophic — and vanishes identically for a symmetric kernel, which is the
kind every sanity check reaches for.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402

try:
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _pair(v):
    """Broadcast an int into an (h, w) pair, the way every conv API does."""
    return (v, v) if isinstance(v, int) else tuple(v)


def conv_output_size(size, k, stride=1, padding=0):
    """floor((size + 2*padding - k) / stride) + 1 — how many starts fit."""
    # === YOUR CODE HERE ===
    return (size + 2 * padding - k) // stride + 1


def im2col(x, kh, kw, stride=1, padding=0):
    """Every sliding window as a column: (N, C*kh*kw, H_out*W_out).

    Row order is (channel, ki, kj) with kj fastest, matching
    `w.reshape(C_out, -1)`. Same routine as ex11, minus dilation.
    """
    N, C, H, W = x.shape
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    H_out = conv_output_size(H, kh, sh, ph)
    W_out = conv_output_size(W, kw, sw, pw)
    # === YOUR CODE HERE ===
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    cols = np.empty((N, C, kh, kw, H_out, W_out), dtype=np.float64)
    for a in range(kh):
        for b in range(kw):
            cols[:, :, a, b] = xp[:, :, a: a + sh * H_out: sh,
                                        b: b + sw * W_out: sw]
    return cols.reshape(N, C * kh * kw, H_out * W_out)


def col2im(cols, x_shape, kh, kw, stride=1, padding=0):
    """The exact adjoint of im2col: SCATTER-ADD every column back where it came
    from, then crop the padding away.

    im2col GATHERS (one input pixel is read into several columns, because
    windows overlap), so its transpose must ACCUMULATE. Overwriting instead of
    += is the third classic conv-backward bug and it silently keeps only the
    last window that touched each pixel.
    """
    N, C, H, W = x_shape
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    H_out = conv_output_size(H, kh, sh, ph)
    W_out = conv_output_size(W, kw, sw, pw)
    # === YOUR CODE HERE ===
    cols = cols.reshape(N, C, kh, kw, H_out, W_out)
    xp = np.zeros((N, C, H + 2 * ph, W + 2 * pw))
    for a in range(kh):
        for b in range(kw):
            xp[:, :, a: a + sh * H_out: sh,
                     b: b + sw * W_out: sw] += cols[:, :, a, b]
    return xp[:, :, ph: ph + H, pw: pw + W]


def conv2d_forward(x, w, b=None, stride=1, padding=0):
    """Cross-correlation as one matmul. Returns (y, cache) — the cache holds
    the im2col buffer, because the backward pass needs it and rebuilding it is
    the single most expensive thing a naive implementation does twice."""
    N, C_in, H, W = x.shape
    C_out, _, kh, kw = w.shape
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    H_out = conv_output_size(H, kh, sh, ph)
    W_out = conv_output_size(W, kw, sw, pw)
    # === YOUR CODE HERE ===
    cols = im2col(x, kh, kw, stride, padding)                 # (N, C*kh*kw, L)
    y = np.matmul(w.reshape(C_out, -1), cols)                 # (N, C_out, L)
    y = y.reshape(N, C_out, H_out, W_out)
    if b is not None:
        y = y + b.reshape(1, -1, 1, 1)
    cache = dict(x_shape=x.shape, w=w, cols=cols, k=(kh, kw),
                 stride=stride, padding=padding)
    return y, cache


def conv2d_backward(dy, cache):
    """All three gradients, by transposing the forward matmul.

    Forward, per sample n:   Yf[n] = Wf @ cols[n]     (Wf is (C_out, C*kh*kw))
    so by the matmul rule:   dWf   = sum_n dYf[n] @ cols[n]^T        <- SUM
                             dcols[n] = Wf^T @ dYf[n]
                             dX    = col2im(dcols)

    The sum over n is not decoration: W is SHARED by every sample, so its
    gradient is the sum of the per-sample gradients. Same rule as the shared
    bias in ex07 — replication forward means summation backward.
    """
    N, C_out, H_out, W_out = dy.shape
    w, cols = cache["w"], cache["cols"]
    kh, kw = cache["k"]
    # === YOUR CODE HERE ===
    dy_flat = dy.reshape(N, C_out, H_out * W_out)
    db = dy.sum(axis=(0, 2, 3))                       # broadcast fwd -> sum bwd
    dw = np.einsum("nol,nml->om", dy_flat, cols).reshape(w.shape)
    dcols = np.matmul(w.reshape(C_out, -1).T, dy_flat)
    dx = col2im(dcols, cache["x_shape"], kh, kw, cache["stride"], cache["padding"])
    return dx, dw, db


def dw_as_correlation(x, dy, kh, kw, stride=1, padding=0):
    """dW written out as what it is: a correlation of X with dY.

        dW[o,c,a,b] = sum_{n,i,j} dY[n,o,i,j] * Xpad[n,c, i*s+a, j*s+b]

    One strided slice per kernel offset (a,b) — kh*kw of them — and the (n,i,j)
    contraction is the same sum-over-batch-and-position that conv2d_backward
    hides inside its einsum.
    """
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    N, C_in, H, W = x.shape
    C_out, H_out, W_out = dy.shape[1], dy.shape[2], dy.shape[3]
    # === YOUR CODE HERE ===
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    dw = np.zeros((C_out, C_in, kh, kw))
    for a in range(kh):
        for b in range(kw):
            patch = xp[:, :, a: a + sh * H_out: sh,
                             b: b + sw * W_out: sw]      # (N, C_in, H_out, W_out)
            dw[:, :, a, b] = np.einsum("noij,ncij->oc", dy, patch)
    return dw


def dx_as_full_convolution(dy, w, x_shape, stride=1, padding=0, rotate=True):
    """dX written out as what it is: a CONVOLUTION of dY with the kernel.

    Three steps, each forced by the index algebra:
      1. DILATE dy by the stride (insert s-1 zeros between samples), because a
         strided forward pass only ever wrote to every s-th position.
      2. Rotate the kernel 180 degrees and swap its in/out channel axes, turning
         the convolution back into a correlation the forward code can run.
      3. Correlate with FULL padding (kh-1, kw-1), because an input pixel near
         the border still contributed to the few outputs that reached it.
    Then paste into the padded-input canvas and crop the padding off. Any input
    row the stride skipped past entirely keeps a gradient of exactly zero.

    `rotate=False` is the classic bug, kept as a parameter so the break-it
    section can run the same code path with the flip removed.
    """
    N, C_out, H_out, W_out = dy.shape
    _, C_in, kh, kw = w.shape
    N_x, C_in_x, H, W = x_shape
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    # === YOUR CODE HERE ===
    # 1. zero-dilate the upstream gradient back onto the strided grid
    dyd = np.zeros((N, C_out, (H_out - 1) * sh + 1, (W_out - 1) * sw + 1))
    dyd[:, :, ::sh, ::sw] = dy
    # 2. rot180 + channel swap: (C_out, C_in, kh, kw) -> (C_in, C_out, kh, kw)
    w_use = w[:, :, ::-1, ::-1] if rotate else w
    w_t = np.ascontiguousarray(w_use.transpose(1, 0, 2, 3))
    # 3. full-padded correlation, stride 1
    full, _ = conv2d_forward(dyd, w_t, None, stride=1, padding=(kh - 1, kw - 1))
    dxp = np.zeros((N, C_in, H + 2 * ph, W + 2 * pw))
    dxp[:, :, :full.shape[2], :full.shape[3]] = full
    return dxp[:, :, ph: ph + H, pw: pw + W]


def db_from_dy(dy):
    """The bias was broadcast over (N, H_out, W_out); sum those axes back."""
    # === YOUR CODE HERE ===
    return dy.sum(axis=(0, 2, 3))


def numeric_grad(f, arr, eps=1e-3):
    """Central differences over every entry of `arr`, in place.

    The loss here is exactly LINEAR in each of x, w, b, so central differences
    carry ZERO truncation error — the only error left is roundoff, and a large
    eps is therefore strictly better than a small one.
    """
    g = np.zeros_like(arr)
    flat, gflat = arr.reshape(-1), g.reshape(-1)
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + eps
        up = f()
        flat[i] = orig - eps
        dn = f()
        flat[i] = orig
        gflat[i] = (up - dn) / (2 * eps)
    return g


def dw_batch_reduced(dy, cache, mode="sum"):
    """dW with the batch reduction deliberately parameterized.

    'sum' is correct. 'mean' is the copy-paste-from-the-loss bug. 'first' is
    the debugging shortcut that never got removed.
    """
    N, C_out, H_out, W_out = dy.shape
    cols = cache["cols"]
    dy_flat = dy.reshape(N, C_out, H_out * W_out)
    per_sample = np.einsum("nol,nml->nom", dy_flat, cols)     # (N, C_out, C*kh*kw)
    if mode == "sum":
        r = per_sample.sum(axis=0)
    elif mode == "mean":
        r = per_sample.mean(axis=0)
    elif mode == "first":
        r = per_sample[0]
    else:
        raise ValueError(mode)
    return r.reshape(cache["w"].shape)


def best_scalar_residual(g, ref):
    """Fraction of ||g|| that no rescaling of `ref` can explain.

    A gradient that is merely off by a constant factor is a learning-rate bug:
    the direction is right and the residual after the best scalar fit is 0. A
    gradient pointing somewhere else entirely is a correctness bug. This is the
    number that separates the two, and it needs no invented threshold.
    """
    alpha = float(np.vdot(g, ref) / np.vdot(ref, ref))
    return float(np.linalg.norm(g - alpha * ref) / np.linalg.norm(g)), alpha


if __name__ == "__main__":
    t_start = time.perf_counter()
    rng = np.random.default_rng(0)

    # The upstream gradient is RANDOM, not all-ones. A loss of Y.sum() makes
    # dY constant, and a constant dY hides the kernel rotation (see break-it c).
    # Everything below differentiates L = sum(Y * G) so that dL/dY = G exactly.

    print("\n--- the three gradients, against torch.autograd ---")
    N, C_in, C_out, H, W = 3, 3, 4, 9, 11
    kh, kw = 3, 4
    x = rng.standard_normal((N, C_in, H, W))
    w = rng.standard_normal((C_out, C_in, kh, kw))
    b = rng.standard_normal(C_out)

    configs = [
        dict(stride=1, padding=0),        # the easy case that hides bugs
        dict(stride=2, padding=1),        # stride>1 AND padding>0 — the real test
        dict(stride=3, padding=2),
        dict(stride=(2, 3), padding=(1, 2)),   # anisotropic, catches axis swaps
    ]

    def torch_grads(x_, w_, b_, G_, stride, padding):
        xt = torch.tensor(x_, requires_grad=True)
        wt = torch.tensor(w_, requires_grad=True)
        bt = torch.tensor(b_, requires_grad=True)
        yt = F.conv2d(xt, wt, bt, stride=stride, padding=padding)
        (yt * torch.tensor(G_)).sum().backward()
        return (xt.grad.numpy(), wt.grad.numpy(), bt.grad.numpy(), yt.detach().numpy())

    if HAVE_TORCH:
        for cfg in configs:
            y, cache = conv2d_forward(x, w, b, **cfg)
            G = rng.standard_normal(y.shape)
            dx, dw, db = conv2d_backward(G, cache)
            gx, gw, gb, yt = torch_grads(x, w, b, G, cfg["stride"], cfg["padding"])
            tag = f"s={cfg['stride']} p={cfg['padding']}"
            check(f"forward matches torch          [{tag}]", y, yt, tol=1e-12)
            check(f"dL/dX matches torch.autograd   [{tag}]", dx, gx, tol=1e-10)
            check(f"dL/dW matches torch.autograd   [{tag}]", dw, gw, tol=1e-10)
            check(f"dL/db matches torch.autograd   [{tag}]", db, gb, tol=1e-10)
    else:
        print("  ....  [skipped: torch not installed] torch.autograd cross-check")

    print("\n--- dL/dX IS a convolution, dL/dW IS a correlation ---")
    for cfg in configs:
        y, cache = conv2d_forward(x, w, b, **cfg)
        G = rng.standard_normal(y.shape)
        dx, dw, db = conv2d_backward(G, cache)
        dx_conv = dx_as_full_convolution(G, w, x.shape, **cfg)
        dw_corr = dw_as_correlation(x, G, kh, kw, **cfg)
        tag = f"s={cfg['stride']} p={cfg['padding']}"
        check(f"dX == full-padded correlation with rot180(W)  [{tag}]",
              dx_conv, dx, tol=1e-12)
        check(f"dW == correlation of X with dY                [{tag}]",
              dw_corr, dw, tol=1e-12)
        check(f"db == sum of dY over (N, H_out, W_out)        [{tag}]",
              db_from_dy(G), db, tol=0)

    print("\n--- dX is the ADJOINT of the forward map ---")
    # Y = A X with A linear, so <A X, G> = <X, A^T G> = <X, dX> for ANY X and G.
    # No reference implementation appears in this test at all: it is the
    # defining property of the backward pass, checked directly.
    print(f"      {'config':>18} {'<Y,G>':>14} {'<X,dX>':>14} {'rel. gap':>10}")
    for cfg in configs:
        y0, cache0 = conv2d_forward(x, w, None, **cfg)      # no bias: pure linear map
        G = rng.standard_normal(y0.shape)
        dx0, dw0, _ = conv2d_backward(G, cache0)
        lhs = float(np.sum(y0 * G))
        rhs_x = float(np.sum(x * dx0))
        rhs_w = float(np.sum(w * dw0))
        rel = abs(lhs - rhs_x) / abs(lhs)
        tag = f"s={cfg['stride']} p={cfg['padding']}"
        print(f"      {tag:>18} {lhs:14.6f} {rhs_x:14.6f} {rel:10.2e}")
        # The observed relative gaps are 0 to 8.6e-16 (see the column above);
        # 1e-12 leaves ~3 orders of headroom over the worst of them.
        check(f"<conv(X),G> == <X, dL/dX>  [{tag}]", rhs_x, lhs, tol=1e-12 * abs(lhs))
        # Y is bilinear, so the same identity holds in W.
        check(f"<conv(X),G> == <W, dL/dW>  [{tag}]", rhs_w, lhs, tol=1e-12 * abs(lhs))

    print("\n--- pixels the stride skipped get exactly zero gradient ---")
    # H=12, k=3, s=3, p=1: padded width 14, and (H_out-1)*s + k = 12, so the
    # last two padded columns — one of which is a real input column — are never
    # inside any window.
    xs = rng.standard_normal((1, 1, 12, 12))
    ws = rng.standard_normal((1, 1, 3, 3))
    ys, cs = conv2d_forward(xs, ws, None, stride=3, padding=1)
    Gs = rng.standard_normal(ys.shape)
    dxs, _, _ = conv2d_backward(Gs, cs)
    covered = (ys.shape[-1] - 1) * 3 + 3
    print(f"      input 12x12, k=3 s=3 p=1 -> output {ys.shape[-2]}x{ys.shape[-1]}, "
          f"windows cover {covered} of {12 + 2} padded columns")
    print(f"      max|dX| on the last row/col: {np.abs(dxs[:, :, -1, :]).max():.1e} / "
          f"{np.abs(dxs[:, :, :, -1]).max():.1e}   (elsewhere {np.abs(dxs[:, :, :-1, :-1]).max():.3f})")
    check("the skipped last row has identically zero gradient",
          dxs[:, :, -1, :], np.zeros((1, 1, 12)), tol=0)
    check("the skipped last column too", dxs[:, :, :, -1], np.zeros((1, 1, 12)), tol=0)
    # And nothing ELSE is zero: the exactly-zero entries are precisely the last
    # row plus the last column, counted once at the shared corner.
    n_zero = int((dxs == 0).sum())
    print(f"      exactly-zero entries of dX: {n_zero} of {dxs.size} "
          f"(predicted 12 + 12 - 1 = {12 + 12 - 1})")
    check("exactly the skipped row and column are zero, nothing else",
          n_zero, 12 + 12 - 1)
    if HAVE_TORCH:
        gxs, _, _, _ = torch_grads(xs, ws, np.zeros(1), Gs, 3, 1)
        check("...and torch agrees, zeros included", dxs, gxs, tol=1e-12)

    print("\n--- finite differences on a small case ---")
    Nf, Cf, Of, Hf = 2, 2, 2, 5
    xf = rng.standard_normal((Nf, Cf, Hf, Hf))
    wf = rng.standard_normal((Of, Cf, 3, 3))
    bf = rng.standard_normal(Of)
    yf, _ = conv2d_forward(xf, wf, bf, stride=2, padding=1)
    Gf = rng.standard_normal(yf.shape)

    def loss():
        yy, _ = conv2d_forward(xf, wf, bf, stride=2, padding=1)
        return float(np.sum(yy * Gf))

    _, cache_f = conv2d_forward(xf, wf, bf, stride=2, padding=1)
    dxf, dwf, dbf = conv2d_backward(Gf, cache_f)
    t0 = time.perf_counter()
    fd = [numeric_grad(loss, arr) for arr in (xf, wf, bf)]
    t_fd = time.perf_counter() - t0
    n_evals = 2 * (xf.size + wf.size + bf.size)
    print(f"      {n_evals} forward passes in {t_fd*1e3:.0f} ms "
          f"(x:{xf.size} w:{wf.size} b:{bf.size} parameters, eps=1e-3)")
    for name, ana, num in zip(("dL/dX", "dL/dW", "dL/db"), (dxf, dwf, dbf), fd):
        err = float(np.abs(ana - num).max())
        scale = float(np.abs(ana).max())
        print(f"      {name}: max|analytic - numeric| = {err:.2e}   "
              f"(gradient scale {scale:.2f}, relative {err/scale:.2e})")
        check(f"{name} matches central differences", ana, num, tol=1e-9)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    print("\n  (a) forgetting to accumulate dL/dW over the batch")
    Nb = 8
    xb = rng.standard_normal((Nb, C_in, H, W))
    yb, cb = conv2d_forward(xb, w, b, stride=2, padding=1)
    Gb = rng.standard_normal(yb.shape)
    dw_sum = dw_batch_reduced(Gb, cb, "sum")
    dw_mean = dw_batch_reduced(Gb, cb, "mean")
    dw_first = dw_batch_reduced(Gb, cb, "first")

    if HAVE_TORCH:
        _, gw_t, _, _ = torch_grads(xb, w, b, Gb, 2, 1)
        check("the summing version is the one torch computes", dw_sum, gw_t, tol=1e-10)

    ratio = dw_sum / dw_mean
    print(f"      dW_sum / dW_mean over all {dw_sum.size} entries: "
          f"min {ratio.min():.6f}, max {ratio.max():.6f}   (batch size {Nb})")
    check("the mean version is wrong by exactly the batch size",
          ratio, np.full(dw_sum.shape, float(Nb)), tol=1e-9)
    rel_mean = float(np.linalg.norm(dw_mean - dw_sum) / np.linalg.norm(dw_sum))
    print(f"      relative error of the mean version: {rel_mean:.6f}  "
          f"(1 - 1/N = {1 - 1/Nb:.6f})")
    check("...i.e. a relative error of exactly 1 - 1/N", rel_mean, 1 - 1 / Nb, tol=1e-12)

    # And now the part that makes it ship: at N=1 all three variants agree BITWISE.
    x1 = xb[:1]
    y1, c1 = conv2d_forward(x1, w, b, stride=2, padding=1)
    G1 = Gb[:1]
    v1 = {m: dw_batch_reduced(G1, c1, m) for m in ("sum", "mean", "first")}
    d_mean_1 = float(np.abs(v1["mean"] - v1["sum"]).max())
    d_first_1 = float(np.abs(v1["first"] - v1["sum"]).max())
    print(f"      at N=1: max|mean - sum| = {d_mean_1:.1e}, "
          f"max|first - sum| = {d_first_1:.1e}  <- the bug is invisible")
    check("at batch size 1 the mean bug is bit-for-bit undetectable",
          v1["mean"], v1["sum"], tol=0)
    check("at batch size 1 the take-first bug is too", v1["first"], v1["sum"], tol=0)
    if HAVE_TORCH:
        _, gw1, _, _ = torch_grads(x1, w, b, G1, 2, 1)
        check("and a torch gradcheck at N=1 passes all three variants",
              max(float(np.abs(v1[m] - gw1).max()) for m in v1) < 1e-10)

    # The two bugs are not equally bad, and the distinction is measurable: the
    # mean version is a pure RESCALING (direction exact, fixable by the learning
    # rate), the take-first version points somewhere else entirely.
    res_mean, a_mean = best_scalar_residual(dw_mean, dw_sum)
    res_first, a_first = best_scalar_residual(dw_first, dw_sum)
    print(f"      after the best scalar rescale: mean leaves {res_mean:.2e} of its norm "
          f"(alpha={a_mean:.4f} = 1/N)")
    print(f"                                     first leaves {res_first:.4f} "
          f"(alpha={a_first:.4f})")
    check("the mean bug is a pure rescaling — nothing survives the best scalar fit",
          res_mean < 1e-14)
    check("its scale factor is exactly 1/N", a_mean, 1 / Nb, tol=1e-12)
    check("the take-first bug is NOT a rescaling: most of it is a wrong direction",
          res_first > 0.5)

    print("\n  (b) omitting the 180-degree kernel rotation in dL/dX")
    y2, c2 = conv2d_forward(x, w, b, stride=2, padding=1)
    G2 = rng.standard_normal(y2.shape)
    dx_ok = dx_as_full_convolution(G2, w, x.shape, stride=2, padding=1, rotate=True)
    dx_bad = dx_as_full_convolution(G2, w, x.shape, stride=2, padding=1, rotate=False)
    disc = float(np.abs(dx_ok - dx_bad).max())
    print(f"      max|dX| = {np.abs(dx_ok).max():.4f}, "
          f"max|correct - unrotated| = {disc:.4f}")
    # The right anchor for "wrong" is not a percentage: it is how wrong an
    # entirely UNRELATED kernel would be.
    w_other = rng.standard_normal(w.shape)
    dx_other = dx_as_full_convolution(G2, w_other, x.shape, stride=2, padding=1)
    disc_indep = float(np.abs(dx_ok - dx_other).max())
    print(f"      for reference, an unrelated random kernel differs by {disc_indep:.4f}")
    check("dropping the rotation is as damaging as using a random kernel",
          disc > 0.3 * disc_indep)
    if HAVE_TORCH:
        gx2, _, _, _ = torch_grads(x, w, b, G2, 2, 1)
        check("the rotated version is the one torch computes", dx_ok, gx2, tol=1e-10)
        check("the unrotated version is not, by the same random-kernel yardstick",
              float(np.abs(dx_bad - gx2).max()) > 0.3 * disc_indep)

    # ...and now the reason it survives review. Rotating a centro-symmetric
    # kernel is the identity, so the bug disappears. The floor to compare
    # against is not zero: the two paths sum the same terms in a different
    # memory order, so they can differ by float64 re-association.
    def rounding_floor(arr, n_terms):
        """eps * peak * (number of accumulations) — the most two orderings of
        the same sum can differ by in float64."""
        return np.finfo(float).eps * float(np.abs(arr).max()) * n_terms

    n_terms = C_out * kh * kw          # accumulations per input-gradient entry
    w_sym = 0.5 * (w + w[:, :, ::-1, ::-1])
    dx_s_ok = dx_as_full_convolution(G2, w_sym, x.shape, stride=2, padding=1, rotate=True)
    dx_s_bad = dx_as_full_convolution(G2, w_sym, x.shape, stride=2, padding=1, rotate=False)
    disc_sym = float(np.abs(dx_s_ok - dx_s_bad).max())
    floor_sym = rounding_floor(dx_s_ok, n_terms)
    print(f"      with a SYMMETRIC kernel the discrepancy falls to {disc_sym:.2e} "
          f"(float64 re-association floor {floor_sym:.2e})")
    check("a centro-symmetric kernel leaves nothing above float64 rounding",
          disc_sym <= floor_sym)
    # Comparing to disc_sym itself is useless when it is exactly 0. Compare the
    # asymmetric damage to the FLOOR that bounds the symmetric case instead —
    # measured at 17.4 vs 9.1e-14, i.e. 14 orders of magnitude.
    check("...over 12 orders of magnitude above what a symmetric kernel can reveal",
          disc > 1e12 * floor_sym)
    # The kernels everyone reaches for when writing a conv test are exactly the
    # centro-symmetric ones, and a 1x1 conv cannot see the bug at all.
    box = np.ones((1, 1, 3, 3)) / 9.0
    gauss = np.array([[[[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]]]) / 16.0
    lap = np.array([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]])
    onexone = rng.standard_normal((1, 1, 1, 1))
    x_t = rng.standard_normal((1, 1, 12, 12))
    for nm, kk in (("box blur", box), ("Gaussian", gauss),
                   ("Laplacian", lap), ("1x1 conv", onexone)):
        y_t, _ = conv2d_forward(x_t, kk, None, stride=2, padding=1)
        G_t = rng.standard_normal(y_t.shape)
        a_ = dx_as_full_convolution(G_t, kk, x_t.shape, stride=2, padding=1, rotate=True)
        b_ = dx_as_full_convolution(G_t, kk, x_t.shape, stride=2, padding=1, rotate=False)
        d_ = float(np.abs(a_ - b_).max())
        floor_ = rounding_floor(a_, 1 * kk.shape[2] * kk.shape[3])
        print(f"      {nm:>10}: discrepancy {d_:.2e}  (rounding floor {floor_:.2e})")
        check(f"the classic test kernel '{nm}' cannot detect the missing rotation",
              d_ <= floor_)

    print("\n  (c) the weak test vector: a loss of Y.sum()")
    # dY is then all ones, and in the interior every input pixel collects
    # sum(W) regardless of how W is indexed — so the rotation bug survives
    # anywhere the kernel window is fully inside. Only the border disagrees.
    xw = rng.standard_normal((1, 1, 12, 12))
    ww = rng.standard_normal((1, 1, 3, 3))
    yw, _ = conv2d_forward(xw, ww, None, stride=1, padding=0)
    G_ones = np.ones(yw.shape)
    a_ones = dx_as_full_convolution(G_ones, ww, xw.shape, stride=1, padding=0, rotate=True)
    b_ones = dx_as_full_convolution(G_ones, ww, xw.shape, stride=1, padding=0, rotate=False)
    d_all = np.abs(a_ones - b_ones)
    interior = d_all[:, :, 2:-2, 2:-2]
    G_rand = rng.standard_normal(yw.shape)
    a_rand = dx_as_full_convolution(G_rand, ww, xw.shape, stride=1, padding=0, rotate=True)
    b_rand = dx_as_full_convolution(G_rand, ww, xw.shape, stride=1, padding=0, rotate=False)
    d_rand_int = float(np.abs(a_rand - b_rand)[:, :, 2:-2, 2:-2].max())
    # Same yardstick as (b): how wrong would an unrelated kernel have been?
    w_alt = rng.standard_normal(ww.shape)
    alt = dx_as_full_convolution(G_rand, w_alt, xw.shape, stride=1, padding=0)
    d_alt_int = float(np.abs(a_rand - alt)[:, :, 2:-2, 2:-2].max())
    print(f"      dY = all ones : interior discrepancy {float(interior.max()):.2e}, "
          f"border {float(d_all.max()):.4f}")
    print(f"      dY = random   : interior discrepancy {d_rand_int:.4f} "
          f"(unrelated-kernel yardstick {d_alt_int:.4f})")
    check("with a constant upstream gradient the interior cannot see the bug",
          float(interior.max()) <= rounding_floor(a_ones, 1 * 3 * 3))
    check("...but a random upstream gradient exposes it throughout the interior",
          d_rand_int > 0.3 * d_alt_int)
    check("the constant-dY test only ever probes the border",
          float(d_all.max()) > 100 * max(float(interior.max()), np.finfo(float).tiny))

    print(f"\n      total runtime {time.perf_counter() - t_start:.2f} s")
    summary()
