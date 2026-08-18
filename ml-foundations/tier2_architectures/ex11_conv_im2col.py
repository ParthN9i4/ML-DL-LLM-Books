"""
ex11 — Convolution as a matmul: im2col.  (Book: Chapter 16)

A 2-D convolution layer looks nothing like a matrix multiply, and for twenty
years that was a problem: every hardware vendor had a hand-tuned GEMM and
nobody had a hand-tuned six-deep loop nest. im2col is the trick that made the
whole field portable. You take every sliding window the kernel will ever land
on, flatten each one into a column of length C_in*k*k, stack those columns
side by side into one big matrix, and then the entire layer — every output
channel, every spatial position — is a single matrix product:

    Y  =  W_flat  @  im2col(X)
    (C_out, k*k*C_in) @ (k*k*C_in, H_out*W_out)

The arithmetic is not reduced by one multiply. The same 2*N*C_out*H_out*W_out*
C_in*k*k floating-point operations happen either way. What changes is that they
now happen inside a BLAS call that has been optimized to within a few percent
of the machine's peak, instead of inside an interpreted loop. The price is
memory: the im2col buffer holds k*k*C_in*H_out*W_out numbers where the input
held C_in*H*W, and for a stride-1 3x3 layer that is a nine-fold blow-up of the
activation. This exercise measures that expansion factor and the wall-clock
payoff rather than asserting them from memory, because the trade is the whole
point and the numbers are the argument.

Two things hide in the details. The first is the output-size formula,
floor((H + 2p - d*(k-1) - 1)/s) + 1, which is just "how many starting
positions fit" and which this file pins against the shapes torch actually
produces. The second is that what every framework calls a convolution is a
CROSS-CORRELATION: the kernel is not flipped. That discrepancy is invisible
for a symmetric kernel, which is exactly why it survives code review.

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

try:
    from scipy.signal import convolve2d, correlate2d
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def _pair(v):
    """Broadcast an int into an (h, w) pair, the way every conv API does."""
    return (v, v) if isinstance(v, int) else tuple(v)


def conv_output_size(size, k, stride=1, padding=0, dilation=1):
    """floor((size + 2*padding - dilation*(k - 1) - 1) / stride) + 1.

    Read it as counting starting positions: the kernel's true footprint is
    dilation*(k-1) + 1 samples wide, the padded signal is size + 2*padding
    wide, so the last valid start is at index size + 2p - (d*(k-1) + 1), and
    you step to it in units of `stride`.
    """
    # === YOUR CODE HERE ===
    return (size + 2 * padding - dilation * (k - 1) - 1) // stride + 1


def conv2d_naive(x, w, b=None, stride=1, padding=0, dilation=1):
    """Cross-correlation the honest way: one Python iteration per output pixel.

    x: (N, C_in, H, W)   w: (C_out, C_in, kh, kw)   b: (C_out,) or None
    NOTE the sign convention on the kernel indices: x[..., i*s + a*d, ...],
    a PLUS. That makes this a correlation, not a convolution. Every deep
    learning framework does the same thing under the name "conv2d".
    """
    N, C_in, H, W = x.shape
    C_out, C_in_w, kh, kw = w.shape
    assert C_in == C_in_w, f"channel mismatch: x has {C_in}, w wants {C_in_w}"
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    H_out = conv_output_size(H, kh, sh, ph, dh)
    W_out = conv_output_size(W, kw, sw, pw, dw)
    # === YOUR CODE HERE ===
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    out = np.zeros((N, C_out, H_out, W_out))
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    patch = xp[n, :,
                               i * sh: i * sh + dh * (kh - 1) + 1: dh,
                               j * sw: j * sw + dw * (kw - 1) + 1: dw]
                    out[n, co, i, j] = np.sum(patch * w[co])
    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    return out


def im2col(x, kh, kw, stride=1, padding=0, dilation=1):
    """Materialize every sliding window as a column.

    Returns (N, C_in*kh*kw, H_out*W_out). The row order is (channel, ki, kj)
    with kj fastest — which is exactly the order `w.reshape(C_out, -1)` walks,
    so the two line up with no transposes.

    The loop below runs kh*kw times, not H_out*W_out times: each iteration
    grabs one strided VIEW of the padded input and copies it wholesale. That
    is what makes the buffer cheap to build and expensive to hold.
    """
    N, C, H, W = x.shape
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    H_out = conv_output_size(H, kh, sh, ph, dh)
    W_out = conv_output_size(W, kw, sw, pw, dw)
    # === YOUR CODE HERE ===
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    cols = np.empty((N, C, kh, kw, H_out, W_out), dtype=np.float64)
    for a in range(kh):
        i0 = a * dh
        for bkw in range(kw):
            j0 = bkw * dw
            cols[:, :, a, bkw] = xp[:, :,
                                    i0: i0 + sh * H_out: sh,
                                    j0: j0 + sw * W_out: sw]
    return cols.reshape(N, C * kh * kw, H_out * W_out)


def conv2d_im2col(x, w, b=None, stride=1, padding=0, dilation=1):
    """The same cross-correlation as one batched matrix product.

    Everything interesting already happened in im2col; this is a reshape, a
    matmul, and a reshape back.
    """
    N, C_in, H, W = x.shape
    C_out, _, kh, kw = w.shape
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    H_out = conv_output_size(H, kh, sh, ph, dh)
    W_out = conv_output_size(W, kw, sw, pw, dw)
    # === YOUR CODE HERE ===
    cols = im2col(x, kh, kw, stride, padding, dilation)      # (N, C*kh*kw, L)
    w_flat = w.reshape(C_out, -1)                            # (C_out, C*kh*kw)
    out = np.matmul(w_flat, cols)                            # (N, C_out, L)
    out = out.reshape(N, C_out, H_out, W_out)
    if b is not None:
        out = out + b.reshape(1, -1, 1, 1)
    return out


def conv2d_true_convolution(x, w, b=None, stride=1, padding=0, dilation=1):
    """A REAL convolution: flip the kernel in both spatial axes, then correlate.

    This is the textbook definition, the one that makes convolution commutative
    and turns into pointwise multiplication under the Fourier transform. It is
    not what `nn.Conv2d` computes.
    """
    # === YOUR CODE HERE ===
    return conv2d_im2col(x, w[:, :, ::-1, ::-1], b, stride, padding, dilation)


def im2col_expansion(H, W, C_in, kh, kw, stride=1, padding=0, dilation=1):
    """Analytic buffer blow-up: (kh*kw*H_out*W_out) / (H*W).

    The C_in factor cancels — it multiplies both the buffer and the input —
    so the expansion is purely a statement about how many output positions
    each input pixel gets copied into.
    """
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    # === YOUR CODE HERE ===
    H_out = conv_output_size(H, kh, sh, ph, dh)
    W_out = conv_output_size(W, kw, sw, pw, dw)
    return (kh * kw * H_out * W_out) / (H * W)


def conv_flops(N, C_in, C_out, H_out, W_out, kh, kw):
    """Multiply-accumulates x 2. Identical for the loop form and the matmul
    form — im2col buys layout, not arithmetic."""
    # === YOUR CODE HERE ===
    return 2 * N * C_out * H_out * W_out * C_in * kh * kw


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("\n--- the three implementations agree ---")
    N, C_in, C_out, H, W = 2, 3, 4, 11, 13
    kh = kw = 3
    x = rng.standard_normal((N, C_in, H, W))
    w = rng.standard_normal((C_out, C_in, kh, kw))
    bias = rng.standard_normal(C_out)

    combos = [
        dict(stride=1, padding=0, dilation=1),   # the default
        dict(stride=1, padding=1, dilation=1),   # "same" for 3x3
        dict(stride=2, padding=1, dilation=1),   # downsampling
        dict(stride=1, padding=2, dilation=2),   # dilated, "same"
        dict(stride=2, padding=2, dilation=2),   # all three at once, non-default
    ]
    for cfg in combos:
        y_naive = conv2d_naive(x, w, bias, **cfg)
        y_col = conv2d_im2col(x, w, bias, **cfg)
        tag = f"s={cfg['stride']} p={cfg['padding']} d={cfg['dilation']}"
        check(f"naive loops == im2col matmul  [{tag}]", y_col, y_naive, tol=1e-12)

    if HAVE_TORCH:
        for cfg in combos:
            y_col = conv2d_im2col(x, w, bias, **cfg)
            y_t = F.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(bias),
                           **cfg).numpy()
            tag = f"s={cfg['stride']} p={cfg['padding']} d={cfg['dilation']}"
            check(f"im2col matmul == torch conv2d  [{tag}]", y_col, y_t, tol=1e-12)
    else:
        print("  ....  [skipped: torch not installed] torch.nn.functional cross-check")

    print("\n--- against scipy.signal, single channel ---")
    if HAVE_SCIPY:
        x1 = rng.standard_normal((1, 1, 9, 10))
        w1 = rng.standard_normal((1, 1, 3, 4))
        mine = conv2d_im2col(x1, w1)[0, 0]
        sci = correlate2d(x1[0, 0], w1[0, 0], mode="valid")
        check("conv2d_im2col == scipy.signal.correlate2d (valid)", mine, sci, tol=1e-12)
        # And the flipped-kernel version is scipy's *convolve*2d.
        mine_conv = conv2d_true_convolution(x1, w1)[0, 0]
        sci_conv = convolve2d(x1[0, 0], w1[0, 0], mode="valid")
        check("conv2d_true_convolution == scipy.signal.convolve2d (valid)",
              mine_conv, sci_conv, tol=1e-12)
    else:
        print("  ....  [skipped: scipy not installed] scipy.signal cross-check")

    print("\n--- the output-size formula ---")
    print(f"      {'H':>4} {'k':>3} {'s':>3} {'p':>3} {'d':>3} {'formula':>8} {'torch':>7}")
    shape_cases = [
        (32, 3, 1, 0, 1), (32, 3, 1, 1, 1), (32, 3, 2, 1, 1), (32, 3, 2, 0, 1),
        (32, 5, 1, 2, 1), (32, 3, 1, 2, 2), (32, 3, 3, 4, 3), (32, 7, 4, 3, 1),
        (31, 3, 2, 1, 1), (31, 4, 3, 2, 2), (17, 2, 5, 1, 1), (17, 3, 1, 0, 5),
    ]
    all_match = True
    for (Hs, k, s, p, d) in shape_cases:
        want = conv_output_size(Hs, k, s, p, d)
        got_t = None
        if HAVE_TORCH:
            xt = torch.zeros(1, 1, Hs, Hs, dtype=torch.float64)
            wt = torch.zeros(1, 1, k, k, dtype=torch.float64)
            got_t = F.conv2d(xt, wt, stride=s, padding=p, dilation=d).shape[-1]
            all_match &= (want == got_t)
        print(f"      {Hs:4d} {k:3d} {s:3d} {p:3d} {d:3d} {want:8d} "
              f"{got_t if got_t is not None else '   n/a':>7}")
        # The formula must also agree with what our own im2col actually built.
        L = im2col(np.zeros((1, 1, Hs, Hs)), k, k, s, p, d).shape[-1]
        check(f"im2col column count == formula^2  [H={Hs} k={k} s={s} p={p} d={d}]",
              L == want * want)
    if HAVE_TORCH:
        check("formula matches torch's shape on all 12 (H,k,s,p,d) combos", all_match)

    print("\n--- what im2col costs: memory ---")
    mem_cases = [(64, 64, 3, 1, 1, 1), (64, 64, 3, 1, 1, 2), (64, 64, 7, 3, 1, 1),
                 (64, 64, 1, 0, 1, 1), (56, 64, 3, 1, 2, 1)]
    print(f"      {'H=W':>5} {'C_in':>5} {'k':>3} {'p':>3} {'s':>3} {'d':>3} "
          f"{'input MB':>9} {'im2col MB':>10} {'expansion':>10}")
    for (Hm, Cm, k, p, s, d) in mem_cases:
        xm = np.zeros((1, Cm, Hm, Hm))
        cm = im2col(xm, k, k, s, p, d)
        factor_measured = cm.nbytes / xm.nbytes
        factor_analytic = im2col_expansion(Hm, Hm, Cm, k, k, s, p, d)
        print(f"      {Hm:5d} {Cm:5d} {k:3d} {p:3d} {s:3d} {d:3d} "
              f"{xm.nbytes/2**20:9.2f} {cm.nbytes/2**20:10.2f} "
              f"{factor_measured:9.2f}x")
        check(f"measured buffer expansion == k*k*L/(H*W)  [k={k} s={s} p={p} d={d}]",
              factor_measured, factor_analytic, tol=1e-12)
    # The headline case: stride-1 "same" 3x3 replicates every pixel exactly 9 times.
    check("stride-1 same-padded 3x3 costs exactly 9x the activation memory",
          im2col_expansion(64, 64, 64, 3, 3, 1, 1, 1), 9.0, tol=1e-12)
    # A 1x1 conv is already a matmul, so im2col is free — the degenerate case
    # that shows the expansion is entirely about window overlap.
    check("a 1x1 conv has expansion exactly 1 (im2col is a no-op)",
          im2col_expansion(64, 64, 64, 1, 1, 1, 0, 1), 1.0, tol=1e-12)

    print("\n--- what im2col buys: one BLAS call ---")
    Nt, Cit, Cot, Ht = 4, 8, 16, 32
    xt_np = rng.standard_normal((Nt, Cit, Ht, Ht))
    wt_np = rng.standard_normal((Cot, Cit, 3, 3))

    t0 = time.perf_counter(); y_loop = conv2d_naive(xt_np, wt_np, padding=1); t_loop = time.perf_counter() - t0
    t0 = time.perf_counter(); y_gemm = conv2d_im2col(xt_np, wt_np, padding=1); t_gemm = time.perf_counter() - t0
    check("the fast path computes the same thing", y_gemm, y_loop, tol=1e-12)

    H_out = conv_output_size(Ht, 3, 1, 1, 1)
    flops = conv_flops(Nt, Cit, Cot, H_out, H_out, 3, 3)
    print(f"      layer: N={Nt} C_in={Cit} C_out={Cot} {Ht}x{Ht} k=3 p=1")
    print(f"      identical work either way: {flops/1e6:.1f} MFLOP")
    print(f"      naive nested loops : {t_loop*1e3:8.1f} ms   "
          f"({flops/t_loop/1e9:.3f} GFLOP/s)")
    print(f"      im2col + one matmul: {t_gemm*1e3:8.1f} ms   "
          f"({flops/t_gemm/1e9:.3f} GFLOP/s)")
    print(f"      speedup: {t_loop/t_gemm:.1f}x")
    # Assert the MECHANISM, not the wall clock: the two forms do exactly the
    # same number of flops, and the loop form pays one Python iteration per
    # output pixel while the matmul form pays kh*kw slice copies total.
    py_iters_loop = Nt * Cot * H_out * H_out
    py_iters_gemm = 3 * 3
    print(f"      Python-level iterations: {py_iters_loop} (loops) vs "
          f"{py_iters_gemm} (im2col) = {py_iters_loop // py_iters_gemm}x fewer")
    check("both forms perform identical arithmetic",
          conv_flops(Nt, Cit, Cot, H_out, H_out, 3, 3), flops)
    check("the loop form pays one interpreted step per output element",
          py_iters_loop == y_loop.size)
    check("im2col pays kh*kw interpreted steps regardless of output size",
          py_iters_gemm == 9)
    # Measured on this machine at 205-299x across runs (0.037 vs 7.5-11.3
    # GFLOP/s); the assertion keeps a 20x margin for a slower or busier box.
    check("and that turns into a real wall-clock win", t_loop / t_gemm > 10.0)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) "conv" is a cross-correlation. Flip the kernel and you get the OTHER
    #     operation. Nothing raises, both are valid linear maps, and the output
    #     shapes are identical — so the only symptom is that your ported filter
    #     does the wrong thing.
    print("\n  (a) convolution vs cross-correlation")
    xa = rng.standard_normal((1, 1, 16, 16))
    wa = rng.standard_normal((1, 1, 3, 3))
    y_corr = conv2d_im2col(xa, wa, padding=1)              # what frameworks do
    y_conv = conv2d_true_convolution(xa, wa, padding=1)    # what textbooks mean
    disc = float(np.abs(y_corr - y_conv).max())
    rel = disc / float(np.abs(y_corr).max())
    corr_coef = float(np.corrcoef(y_corr.ravel(), y_conv.ravel())[0, 1])
    print(f"      max|correlation - convolution| = {disc:.4f} "
          f"({rel*100:.0f}% of the signal's own peak)")
    print(f"      correlation between the two outputs: {corr_coef:+.4f}")
    check("the two operations disagree at the scale of the signal itself",
          disc > 0.5 * float(np.abs(y_corr).max()))
    # The right anchor for "different": the discrepancy is as large as swapping
    # in an INDEPENDENT random kernel would make it.
    w_other = rng.standard_normal((1, 1, 3, 3))
    y_other = conv2d_im2col(xa, w_other, padding=1)
    disc_indep = float(np.abs(y_corr - y_other).max())
    print(f"      for reference, an unrelated random kernel differs by {disc_indep:.4f}")
    check("flipping the kernel is about as damaging as using a random one",
          disc > 0.3 * disc_indep)

    # ...and now the reason the bug hides. Symmetrize the kernel and the two
    # operations become bit-identical.
    # The right floor to compare against is not zero — flipping the kernel
    # reorders a 9-term dot product, so the two answers differ by rounding.
    # Bound that: 9 accumulations, each carrying at most eps*|peak| of error.
    n_terms = 1 * 3 * 3          # C_in * kh * kw accumulations per output pixel

    def rounding_floor(y_):
        return np.finfo(float).eps * float(np.abs(y_).max()) * n_terms

    w_sym = 0.5 * (wa + wa[:, :, ::-1, ::-1])
    y_corr_s = conv2d_im2col(xa, w_sym, padding=1)
    y_conv_s = conv2d_true_convolution(xa, w_sym, padding=1)
    disc_sym = float(np.abs(y_corr_s - y_conv_s).max())
    print(f"      with a SYMMETRIC kernel the discrepancy falls to {disc_sym:.2e} "
          f"(float64 re-association floor {rounding_floor(y_corr_s):.2e})")
    check("a centro-symmetric kernel leaves nothing above float64 rounding",
          disc_sym <= rounding_floor(y_corr_s))
    check("...which is ~15 orders of magnitude below the asymmetric case",
          disc / max(disc_sym, np.finfo(float).tiny) > 1e12)
    # Box blur, the 3x3 Laplacian, and any Gaussian are all centro-symmetric —
    # which is precisely the set of kernels people reach for when testing.
    box = np.ones((1, 1, 3, 3)) / 9.0
    lap = np.array([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]])
    for nm, kk in (("box blur", box), ("Laplacian", lap)):
        y_c = conv2d_im2col(xa, kk, padding=1)
        d_ = float(np.abs(y_c - conv2d_true_convolution(xa, kk, padding=1)).max())
        print(f"      {nm:>10}: discrepancy {d_:.2e}  (rounding floor "
              f"{rounding_floor(y_c):.2e})")
        check(f"the classic test kernel '{nm}' cannot detect the bug",
              d_ <= rounding_floor(y_c))

    # (b) Off-by-one padding. Put pad+1 on the left and pad-1 on the right: the
    #     TOTAL padding is unchanged, so the output shape is unchanged, nothing
    #     raises, and every feature map is silently translated by one pixel.
    print("\n  (b) an off-by-one in padding")

    def conv2d_offbyone_pad(x_, w_, pad=1):
        """Same layer, same output shape, padding on the wrong side."""
        xp = np.pad(x_, ((0, 0), (0, 0), (pad, pad), (pad + 1, pad - 1)))
        return conv2d_im2col(xp, w_, stride=1, padding=0, dilation=1)

    xb = rng.standard_normal((1, 1, 12, 12))
    wb = rng.standard_normal((1, 1, 3, 3))
    y_ok = conv2d_im2col(xb, wb, padding=1)
    y_bad = conv2d_offbyone_pad(xb, wb, pad=1)
    print(f"      correct output shape {y_ok.shape}, buggy output shape {y_bad.shape}"
          f"  -> {'identical, nothing raises' if y_ok.shape == y_bad.shape else 'DIFFERENT'}")
    check("the off-by-one leaves the output shape untouched", y_ok.shape == y_bad.shape)
    # The damage is a pure translation: the buggy map, shifted back one column,
    # is bit-identical to the correct one.
    shift_residual = float(np.abs(y_bad[:, :, :, 1:] - y_ok[:, :, :, :-1]).max())
    raw_diff = float(np.abs(y_bad - y_ok).max())
    print(f"      max|buggy - correct| = {raw_diff:.4f}")
    print(f"      max|buggy shifted back by one column - correct| = {shift_residual:.2e}")
    check("the buggy feature map is the correct one translated by exactly one pixel",
          shift_residual, 0.0, tol=1e-15)
    check("but compared elementwise it is badly wrong", raw_diff > 1.0)

    # Does it survive training? Fit a 3x3 kernel to targets generated by the
    # CORRECT convolution, once with the correct implementation and once with
    # the buggy one. Both are linear least-squares problems, so gradient
    # descent converges to the global optimum and we can read off exactly what
    # the bug costs.
    N_tr = 64
    x_tr = rng.standard_normal((N_tr, 1, 12, 12))
    w_true = rng.standard_normal((1, 1, 3, 3))
    y_tr = conv2d_im2col(x_tr, w_true, padding=1)

    def design_matrix(build):
        """Stack the per-output-pixel feature rows: out = A @ w_flat."""
        cols = build(x_tr)                              # (N, 9, L)
        return cols.transpose(0, 2, 1).reshape(-1, 9)   # (N*L, 9)

    A_ok = design_matrix(lambda z: im2col(z, 3, 3, 1, 1, 1))
    A_bad = design_matrix(
        lambda z: im2col(np.pad(z, ((0, 0), (0, 0), (1, 1), (2, 0))), 3, 3, 1, 0, 1))
    target = y_tr.reshape(-1)

    def train(A, t, steps=4000):
        """Plain gradient descent on 0.5*mean((A w - t)^2), step size set from
        the Hessian's top eigenvalue so convergence needs no tuning."""
        Hess = A.T @ A / len(t)
        lr = 1.0 / float(np.linalg.eigvalsh(Hess)[-1])
        wv = np.zeros(A.shape[1])
        first = None
        for step in range(steps):
            r = A @ wv - t
            loss = float(np.mean(r * r))
            if step == 0:
                first = loss
            wv -= lr * (A.T @ r) / len(t)
        return wv, first, float(np.mean((A @ wv - t) ** 2))

    w_fit_ok, l0_ok, l1_ok = train(A_ok, target)
    w_fit_bad, l0_bad, l1_bad = train(A_bad, target)
    var_y = float(np.mean(target ** 2))
    print(f"      target variance = {var_y:.4f}")
    print(f"      correct impl: loss {l0_ok:.4f} -> {l1_ok:.3e}")
    print(f"      buggy   impl: loss {l0_bad:.4f} -> {l1_bad:.4f}  "
          f"({l1_bad/var_y*100:.1f}% of the signal, and it does not move)")
    check("with correct padding, training recovers the true kernel",
          w_fit_ok, w_true.reshape(-1), tol=1e-6)
    check("the loss still falls by more than half, so the run looks healthy",
          l1_bad < 0.5 * l0_bad)
    # The mechanism, and it is exactly predictable: shifting the window one
    # column left leaves the true kernel's RIGHTMOST column with nowhere to
    # live. With unit-variance inputs the irreducible residual is therefore the
    # energy of that column, as a fraction of the kernel's total energy.
    kt = w_true.reshape(3, 3)
    predicted_frac = float((kt[:, -1] ** 2).sum() / (kt ** 2).sum())
    observed_frac = l1_bad / var_y
    print(f"      unexplainable fraction of the signal: {observed_frac:.4f} observed, "
          f"{predicted_frac:.4f} predicted from ||w_true[:, -1]||^2 / ||w_true||^2")
    check("the plateau is exactly the kernel column the shift pushed off the edge",
          observed_frac, predicted_frac, tol=0.05)
    check("and it sits orders of magnitude above the clean fit's residual",
          l1_bad / l1_ok > 1e6)

    # And what training settled on is the TRUE kernel, shifted one column over,
    # with the column that fell off the edge dropped.
    w_shifted = np.zeros((3, 3))
    w_shifted[:, 1:] = w_true.reshape(3, 3)[:, :-1]
    d_to_shift = float(np.abs(w_fit_bad.reshape(3, 3) - w_shifted).max())
    d_to_true = float(np.abs(w_fit_bad.reshape(3, 3) - w_true.reshape(3, 3)).max())
    print(f"      learned kernel is {d_to_shift:.3f} away from the SHIFTED true kernel")
    print(f"      ...and {d_to_true:.3f} away from the true kernel itself")
    check("training compensated by shifting the weights, not by fixing the bug",
          d_to_shift < 0.25 * d_to_true)
    # Which is the deployment failure: export those weights into a correct
    # implementation and the layer is wrong, even though training looked fine.
    y_deploy = conv2d_im2col(x_tr, w_fit_bad.reshape(1, 1, 3, 3), padding=1)
    deploy_mse = float(np.mean((y_deploy - y_tr) ** 2))
    print(f"      exporting those weights to a CORRECT conv gives MSE "
          f"{deploy_mse:.3f} vs signal {var_y:.3f}")
    check("weights trained against the buggy padding are wrong in a correct kernel",
          deploy_mse > 0.5 * var_y)

    summary()
