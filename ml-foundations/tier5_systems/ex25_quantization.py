"""
ex25 — Quantization: RTN, groups, and the outlier problem.  (Book: Chapter 34)

Round-to-nearest quantization is three lines of arithmetic: pick a step size s,
divide, round, clamp. Everything interesting about it is a statement about
WHICH weights share a step size, and the whole engineering history of LLM
quantization is the story of shrinking that sharing set.

The arithmetic itself is exactly characterized, and this exercise measures the
characterization rather than asserting it. For a uniform grid of step s the
rounding error of any weight obeys |e| <= s/2 — an exact, tight bound — and if
the weights are spread over many bins the error looks uniform on [-s/2, s/2],
so its RMS is s/sqrt(12) = 0.2887 s. The sqrt(12) is the standard deviation of
a uniform distribution; almost every treatment asserts it and moves on. Here it
comes out within 3% on every grid whose bins are actually populated, and is
wrong by a factor of 9 on one that is not — which turns out to be the more
instructive half.

That failure is the one insight the rest of the file is built around: the error
of every weight in a group is set by the LARGEST weight in that group, because
that weight alone sets s. One outlier taxes everybody. Real transformer weight
matrices have systematic outlier channels — the weight-side companion of the
outlier activation features reported in LLM.int8() — and with a single
per-tensor step size the tax is fatal. At int4 here, half a step (1.022) is
wider than the largest non-outlier weight (0.919), so all 129,536 weights
outside the three outlier columns round to exactly zero and the layer keeps
nothing but its outliers. Per-output-channel scales, then group-wise scales of
128 and 64, buy that back at a metered 0.03 / 0.125 / 0.25 bits per weight of
scale storage — and int4 with 64-wide groups ends up both smaller and more
accurate than int8 per-tensor.

Two closing traps. The step size must be constant along the axis the matmul
contracts over, or it cannot be factored out of the integer accumulator — and
choosing the other axis costs 5x downstream accuracy while being invisible to
any weight-space metric. And clipping outliers instead of isolating them
IMPROVES the weight error 1.2x while making the layer's output error 3.4x
worse: the metric moves opposite to the model. The same disagreement, exploited
deliberately, is what makes GPTQ work.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402

try:
    import warnings

    import torch
    warnings.filterwarnings("ignore", message=".*quantize_per.*")
    HAVE_TORCH = True
except Exception:                                    # pragma: no cover
    HAVE_TORCH = False


# ---------------------------------------------------------------------------
# Grouping: which weights share a step size
# ---------------------------------------------------------------------------

def to_groups(W, group_size=None, axis=0):
    """Reshape (R, C) so that each group's members lie along the LAST axis.

    axis=0   : a group is `group_size` consecutive ROWS inside one column, i.e.
               it runs along the contraction dimension of y = x @ W.
               group_size=None means the whole column (per-output-channel).
    axis=1   : a group is `group_size` consecutive COLUMNS inside one row.
    axis=None: one group for the entire tensor.
    """
    R, C = W.shape
    # === YOUR CODE HERE ===
    if axis is None:
        return W.reshape(1, 1, R * C)
    if axis == 0:
        g = R if group_size is None else group_size
        return W.reshape(R // g, g, C).transpose(0, 2, 1)      # (n_groups, C, g)
    g = C if group_size is None else group_size
    return W.reshape(R, C // g, g)                             # (R, n_groups, g)


def from_groups(V, shape, axis=0):
    """Inverse of to_groups."""
    R, C = shape
    # === YOUR CODE HERE ===
    if axis is None:
        return V.reshape(R, C)
    if axis == 0:
        return V.transpose(0, 2, 1).reshape(R, C)
    return V.reshape(R, C)


# ---------------------------------------------------------------------------
# The quantizers themselves
# ---------------------------------------------------------------------------

def rtn_symmetric(V, bits):
    """Symmetric round-to-nearest over the last axis. Returns (codes, scale).

    Levels are -qmax..qmax with qmax = 2^(bits-1) - 1, so the grid is centred on
    zero and zero is exactly representable. One code (-2^(bits-1)) is thrown
    away to keep the symmetry; that lost level is the price, and it is visible
    in the comparison against the asymmetric quantizer below.
    """
    qmax = 2 ** (bits - 1) - 1
    # === YOUR CODE HERE ===
    amax = np.max(np.abs(V), axis=-1, keepdims=True)
    scale = np.where(amax > 0, amax / qmax, 1.0)
    codes = np.clip(np.rint(V / scale), -qmax, qmax)
    return codes, scale


def rtn_asymmetric(V, bits):
    """Asymmetric (affine) RTN over the last axis. Returns (codes, scale, zero).

    Codes run 0..2^bits-1 and the reconstruction is (code - zero) * scale, so
    the grid covers [min, max] exactly instead of [-max|w|, +max|w|].
    """
    n_levels = 2 ** bits - 1
    # === YOUR CODE HERE ===
    lo = np.min(V, axis=-1, keepdims=True)
    hi = np.max(V, axis=-1, keepdims=True)
    scale = np.where(hi > lo, (hi - lo) / n_levels, 1.0)
    zero = np.rint(-lo / scale)
    codes = np.clip(np.rint(V / scale) + zero, 0, n_levels)
    return codes, scale, zero


def quantize_rtn(W, bits, group_size=None, axis=0, mode="symmetric"):
    """Quantize-then-dequantize W. Returns (W_hat, codes, scales, zeros).

    `codes` has W's shape and holds integers; `scales` (and `zeros`) are the
    per-group values with the group axis squeezed out, so for axis=0 they have
    shape (n_groups, C) — one number per group per output channel.
    """
    V = to_groups(W, group_size, axis)
    # === YOUR CODE HERE ===
    if mode == "symmetric":
        codes, scale = rtn_symmetric(V, bits)
        deq = codes * scale
        zeros = None
    else:
        codes, scale, zero = rtn_asymmetric(V, bits)
        deq = (codes - zero) * scale
        zeros = np.squeeze(zero, axis=-1)
    return (from_groups(deq, W.shape, axis),
            from_groups(codes, W.shape, axis),
            np.squeeze(scale, axis=-1),
            zeros)


def error_model(V, codes, scale):
    """Measured vs predicted RMS rounding error, both in units of the step.

    The textbook value is 1/sqrt(12), and it assumes every weight sits at a
    generic position inside its bin. Weights that round to code 0 break that
    assumption: for them the "error" is the whole weight, and if the step is
    much wider than the weight distribution they are all crammed against zero
    and their error is much SMALLER than a uniform draw from [-s/2, s/2].

    So split the population and predict:

        rms(e)^2 = p0 * rms(w | code 0)^2 + (1 - p0) * s^2/12

    where only the second term is a claim — that the occupied bins really are
    filled uniformly. Returns (measured, predicted, dead-bin fraction), the
    first two divided by the step so 1/sqrt(12) is the reference value.
    """
    w_over_s = V / scale
    dead = codes == 0
    p0 = float(np.mean(dead))
    # === YOUR CODE HERE ===
    measured = float(np.sqrt(np.mean((w_over_s - codes) ** 2)))
    dead_ms = float(np.mean(w_over_s[dead] ** 2)) if p0 > 0 else 0.0
    predicted = float(np.sqrt(p0 * dead_ms + (1.0 - p0) / 12.0))
    return measured, predicted, p0


def matmul_int_accum(X, codes, scales, group_size):
    """y = X @ dequant(codes) computed as integer accumulation per group.

    For axis=0 grouping the step size is constant along the contraction index,
    so it factors straight out of the inner sum:

        y[:, j] = sum_g  s[g, j] * ( sum_{i in g} x[:, i] * code[i, j] )
                          \\_____/    \\_______________________________/
                        one float mul          pure integer dot product

    That factorization is the entire reason weight quantization is done down
    the contraction axis: the accumulator stays in int32.
    """
    n, d_in = X.shape
    d_out = codes.shape[1]
    y = np.zeros((n, d_out))
    # === YOUR CODE HERE ===
    for g in range(d_in // group_size):
        sl = slice(g * group_size, (g + 1) * group_size)
        y += (X[:, sl] @ codes[sl, :]) * scales[g]
    return y


# ---------------------------------------------------------------------------
# Error-compensating quantization — the IDEA of GPTQ, not GPTQ itself
# ---------------------------------------------------------------------------

def gptq_like(W, H, bits, percdamp=0.01):
    """Sequential quantization along the contraction axis with error feedback.

    This is the OBQ/GPTQ idea reduced to its core and nothing more: no lazy
    batching, no act-order permutation, no blocked Cholesky, no grouping. Do
    not mistake it for the published algorithm.

    The objective is the layer reconstruction error ||X W - X W_hat||_F^2, whose
    Hessian in each output column is H = X^T X. Quantizing row i of W leaves a
    residual (w_i - q_i); the optimal correction to the NOT-YET-QUANTIZED rows
    that minimizes the objective given that residual is

        W[i+1:, :] -= Hinv[i+1:, i] / Hinv[i, i] * (w_i - q_i)

    which, with Hinv = L L^T from a Cholesky factorization, is the update below.
    The step sizes come from the ORIGINAL weights, so the comparison against
    plain RTN at the same bit width is like-for-like: identical grid, identical
    storage, only the rounding decisions differ.
    """
    d_in = W.shape[0]
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(W).max(axis=0, keepdims=True) / qmax          # per output column
    H = H.copy()
    d = np.arange(d_in)
    H[d, d] += percdamp * np.mean(np.diag(H))                    # damping, as in GPTQ
    L = np.linalg.cholesky(np.linalg.inv(H))                     # Hinv = L L^T
    Wc = W.copy()
    W_hat = np.zeros_like(W)
    # === YOUR CODE HERE ===
    for i in range(d_in):
        w = Wc[i, :]
        q = np.clip(np.rint(w / scale[0]), -qmax, qmax) * scale[0]
        W_hat[i, :] = q
        err = (w - q) / L[i, i]
        if i + 1 < d_in:
            Wc[i + 1:, :] -= L[i + 1:, i][:, None] * err[None, :]
    return W_hat


def rtn_per_column(W, bits):
    """Plain RTN with the same per-output-column grid gptq_like uses."""
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(W).max(axis=0, keepdims=True) / qmax
    return np.clip(np.rint(W / scale), -qmax, qmax) * scale


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------

def make_layer(d_in=512, d_out=256, n=256, seed=0):
    """A weight matrix with the two structures real transformers have.

    Bulk weights are heavy-tailed (Student-t, df=4), not Gaussian. On top:
      * OUTLIER OUTPUT CHANNELS — a few columns whose weights are ~25x larger.
      * SALIENT INPUT CHANNELS  — a few rows whose weights are ~6x larger AND
        whose activations have ~12x the standard deviation. These are the
        outlier features of LLM.int8(); the weights that read them are the
        "salient" weights AWQ is named after.
    """
    rng = np.random.default_rng(seed)
    out_cols = np.array([17, 103, 200])
    in_rows = np.array([5, 61, 130, 288, 401, 470])
    # === YOUR CODE HERE ===
    W = 0.02 * rng.standard_t(4.0, size=(d_in, d_out))
    W[:, out_cols] *= 25.0
    W[in_rows, :] *= 6.0
    act_std = np.ones(d_in)
    act_std[in_rows] = 12.0
    X = rng.standard_normal((n, d_in)) * act_std
    return W, X, out_cols, in_rows


def rel_err(A, A_hat):
    """Relative Frobenius error."""
    return float(np.linalg.norm(A - A_hat) / np.linalg.norm(A))


def footprint_bytes(numel, bits, n_scales, asymmetric=False):
    """Bytes to store the codes plus fp16 scales (and zero points)."""
    # === YOUR CODE HERE ===
    payload = numel * bits / 8.0
    meta = n_scales * 2.0 * (2 if asymmetric else 1)
    return payload + meta


if __name__ == "__main__":
    W, X, out_cols, in_rows = make_layer()
    d_in, d_out = W.shape
    Y = X @ W
    bulk = np.ones(d_out, dtype=bool)
    bulk[out_cols] = False

    print("\n--- the exact bound: |e| <= s/2 and RMS = s/sqrt(12) ---")
    # The bound is a fact about rounding; the sqrt(12) is a fact about where
    # inside its bin a generic weight falls. Measure both, for every grid.
    print(f"      {'grid':>22} {'max|e|/s':>10} {'dead bin':>9} "
          f"{'rms x sqrt12':>13} {'predicted':>10}")
    for label, bits, gs, ax in (("int8 per-tensor", 8, None, None),
                                ("int8 per-column", 8, None, 0),
                                ("int8 group-128", 8, 128, 0),
                                ("int8 group-64", 8, 64, 0),
                                ("int4 per-tensor", 4, None, None),
                                ("int4 per-column", 4, None, 0),
                                ("int4 group-128", 4, 128, 0),
                                ("int4 group-64", 4, 64, 0)):
        V = to_groups(W, gs, ax)
        codes, s = rtn_symmetric(V, bits)
        worst = float(np.abs((V - codes * s) / s).max())
        rms, pred, p0 = error_model(V, codes, s)
        print(f"      {label:>22} {worst:10.6f} {p0*100:8.1f}% "
              f"{rms*np.sqrt(12):13.4f} {pred*np.sqrt(12):10.4f}")
        # The bound is a theorem about rounding and holds unconditionally.
        check(f"{label}: no error exceeds s/2", worst <= 0.5 + 1e-12)
        check(f"{label}: the s/2 bound is tight, not loose", worst > 0.499)
        # The two-population model holds unconditionally too, because it puts
        # the dead bin in a term of its own.
        check(f"{label}: two-population RMS model within 10%", rms, pred, tol=0.10 * rms)
        if p0 < 2.0 / 3.0:
            # Bins actually populated -> the textbook 1/sqrt(12) is right.
            check(f"{label}: RMS error is s/sqrt(12) within 5%",
                  rms * np.sqrt(12), 1.0, tol=0.05)
        else:
            # A single step size shared with the outlier columns puts nearly
            # everything in the dead bin, and the naive law is wildly wrong --
            # too OPTIMISTIC in error terms, which is why "RMS error" alone is
            # such a poor way to judge a quantizer.
            check(f"{label}: {p0*100:.1f}% of weights are in the dead bin", p0 > 0.9)
            check(f"{label}: so s/sqrt(12) overpredicts the RMS by >20%",
                  rms * np.sqrt(12) < 0.8)

    # Symmetric vs asymmetric is not a matter of taste: the step sizes stand in
    # an exact ratio. Symmetric spends max|w|/qmax; asymmetric spends
    # (hi-lo)/(2^b - 1). The ratio decomposes into the wasted level and the
    # asymmetry of the group's actual range.
    V = to_groups(W, 64, 0)
    _, s_sym = rtn_symmetric(V, 4)
    _, s_asym, _ = rtn_asymmetric(V, 4)
    span = V.max(-1, keepdims=True) - V.min(-1, keepdims=True)
    amax = np.abs(V).max(-1, keepdims=True)
    predicted = (15.0 / 14.0) * (2 * amax / span)
    print(f"      int4 group-64: symmetric step is {float(np.median(s_sym/s_asym)):.3f}x the "
          f"asymmetric step (median)")
    print(f"        = 15/14 (the discarded level) x {float(np.median(2*amax/span)):.3f} "
          f"(range asymmetry, median)")
    check("s_sym / s_asym = (15/14) * (2 max|w| / range), exactly",
          s_sym / s_asym, predicted, tol=1e-12)

    if HAVE_TORCH:
        A = W[:64, :64].astype(np.float32)
        lo, hi = float(A.min()), float(A.max())
        sc = (hi - lo) / 255.0
        zp = int(round(-lo / sc))
        ref = torch.quantize_per_tensor(torch.from_numpy(A), sc, zp,
                                        torch.quint8).dequantize().numpy()
        ours = ((np.clip(np.rint(A / sc) + zp, 0, 255) - zp) * sc).astype(np.float32)
        print(f"      torch asymmetric uint8 cross-check: max|diff| = {np.abs(ours-ref).max():.1e}")
        check("our asymmetric uint8 quantizer is bit-identical to torch",
              float(np.abs(ours - ref).max()), 0.0, tol=0.0)

        colmax = np.abs(A).max(axis=0) / 127.0
        ref2 = torch.quantize_per_channel(torch.from_numpy(A),
                                          torch.from_numpy(colmax.astype(np.float64)),
                                          torch.zeros(64, dtype=torch.long), 1,
                                          torch.qint8).dequantize().numpy()
        ours2 = (np.clip(np.rint(A / colmax), -127, 127) * colmax).astype(np.float32)
        print(f"      torch symmetric  int8 per-channel:  max|diff| = {np.abs(ours2-ref2).max():.1e}")
        check("our symmetric int8 per-channel quantizer is bit-identical to torch",
              float(np.abs(ours2 - ref2).max()), 0.0, tol=0.0)

    print("\n--- the outlier problem, made arithmetic ---")
    bulk_max = float(np.abs(W[:, bulk]).max())
    bulk_std = float(W[:, bulk].std())
    print(f"      3 outlier columns reach {np.abs(W[:, out_cols]).max():.2f}; "
          f"everything else stays under {bulk_max:.3f}")
    print(f"      global max / bulk max = {np.abs(W).max()/bulk_max:.1f}x  "
          f"-> per-tensor step is {np.abs(W).max()/bulk_max:.1f}x too big for the bulk")

    W8, _, _, _ = quantize_rtn(W, 8, axis=None)
    step8 = float(np.abs(W).max() / 127)
    zero_frac8 = float(np.mean(W8[:, bulk] == 0.0))
    print(f"      int8 per-tensor: step = {step8:.4f} = {step8/bulk_std:.1f} x the bulk "
          f"std -> {zero_frac8*100:.1f}% of non-outlier weights round to 0")
    check("int8 per-tensor already zeroes most of the matrix", zero_frac8 > 0.9)

    W4, _, _, _ = quantize_rtn(W, 4, axis=None)
    step4 = float(np.abs(W).max() / 7)
    print(f"      int4 per-tensor: step = {step4:.4f}, half-step = {step4/2:.4f} "
          f"> bulk max {bulk_max:.4f}")
    print(f"      -> nonzero weights outside the outlier columns: "
          f"{int(np.count_nonzero(W4[:, bulk]))} of {W[:, bulk].size}")
    # This is not a tolerance, it is a syllogism: every bulk weight is closer to
    # 0 than to +-step, so RTN sends all of them to code 0 and the layer's
    # entire non-outlier content is annihilated.
    check("half the int4 per-tensor step exceeds the largest bulk weight",
          step4 / 2 > bulk_max)
    check("so every weight outside the 3 outlier columns becomes exactly zero",
          int(np.count_nonzero(W4[:, bulk])), 0)
    check("and the bulk relative error is exactly 1 (nothing survives)",
          rel_err(W[:, bulk], W4[:, bulk]), 1.0, tol=1e-12)

    print("\n--- granularity buys it back ---")
    print(f"      {'scheme':>22} {'#scales':>8} {'||dW||/||W||':>13} {'||dY||/||Y||':>13} {'zeros':>7}")
    grid_rows = []
    for label, bits, gs, ax, mode in (
            ("int8 per-tensor", 8, None, None, "symmetric"),
            ("int8 per-column", 8, None, 0, "symmetric"),
            ("int8 group-128", 8, 128, 0, "symmetric"),
            ("int8 group-64", 8, 64, 0, "symmetric"),
            ("int4 per-tensor", 4, None, None, "symmetric"),
            ("int4 per-column", 4, None, 0, "symmetric"),
            ("int4 group-128", 4, 128, 0, "symmetric"),
            ("int4 group-64", 4, 64, 0, "symmetric"),
            ("int4 group-64 asym", 4, 64, 0, "asymmetric")):
        W_hat, codes, scales, zeros = quantize_rtn(W, bits, gs, ax, mode)
        ew, ey = rel_err(W, W_hat), rel_err(Y, X @ W_hat)
        nsc = int(np.asarray(scales).size)
        grid_rows.append((label, bits, nsc, ew, ey, mode == "asymmetric"))
        print(f"      {label:>22} {nsc:8d} {ew:13.5f} {ey:13.5f} "
              f"{np.mean(W_hat == 0)*100:6.1f}%")

    by_name = {r[0]: r for r in grid_rows}
    for bits in (8, 4):
        chain = [f"int{bits} per-tensor", f"int{bits} per-column",
                 f"int{bits} group-128", f"int{bits} group-64"]
        ew = [by_name[c][3] for c in chain]
        ey = [by_name[c][4] for c in chain]
        check(f"int{bits}: weight error falls monotonically as groups shrink",
              all(ew[i] > ew[i + 1] for i in range(3)))
        check(f"int{bits}: matmul error falls monotonically as groups shrink",
              all(ey[i] > ey[i + 1] for i in range(3)))
        print(f"      int{bits}: per-tensor -> group-64 cuts the matmul error "
              f"{ey[0]/ey[-1]:.1f}x using {by_name[chain[-1]][2]} scales instead of 1")

    print("\n--- the scale must factor out of the contraction ---")
    # With axis=0 grouping the step is constant along the summation index, so
    # the inner product can run entirely in integers and be scaled once per
    # (group, output channel). Check that identity exactly.
    W_hat, codes, scales, _ = quantize_rtn(W, 4, 128, 0, "symmetric")
    y_float = X @ W_hat
    y_int = matmul_int_accum(X, codes, scales, 128)
    check("integer accumulation reproduces the dequantized matmul",
          rel_err(y_float, y_int), 0.0, tol=1e-13)
    n_groups = d_in // 128
    print(f"      float multiplies by the scale: {n_groups*d_out} "
          f"(one per group per output channel) instead of {d_in*d_out} (one per weight)")
    check("codes really are integers", np.all(codes == np.rint(codes)))
    check("and they fit in 4 bits", float(np.abs(codes).max()) <= 7.0)

    print("\n--- the accuracy / size curve ---")
    numel = W.size
    fp32 = numel * 4.0
    print(f"      fp32 baseline: {fp32/1024:.1f} KB")
    print(f"      {'scheme':>22} {'eff. bits/w':>12} {'KB':>8} {'compress':>9} "
          f"{'||dW||/||W||':>13} {'||dY||/||Y||':>13}")
    curve = []
    for label, bits, nsc, ew, ey, asym in grid_rows:
        by = footprint_bytes(numel, bits, nsc, asym)
        eff = by * 8.0 / numel
        curve.append((label, eff, by, ew, ey))
        print(f"      {label:>22} {eff:12.4f} {by/1024:8.1f} {fp32/by:8.1f}x "
              f"{ew:13.5f} {ey:13.5f}")

    eff = {c[0]: c for c in curve}
    # The headline of the whole curve: fine-grained int4 is cheaper AND better
    # than per-tensor int8. Bits alone do not order the configurations.
    a, b = eff["int8 per-tensor"], eff["int4 group-64"]
    print(f"      int4 group-64 uses {a[1]/b[1]:.2f}x fewer bits per weight than "
          f"int8 per-tensor and is {a[4]/b[4]:.1f}x better downstream")
    check("int4 group-64 costs fewer effective bits than int8 per-tensor", b[1] < a[1])
    check("and has a lower weight error", b[3] < a[3])
    check("and a lower matmul error", b[4] < a[4])
    # Scale storage is a real, measurable overhead, not a rounding detail.
    check("group-64 pays 0.25 bits/weight for its scales",
          eff["int4 group-64"][1] - 4.0, 0.25, tol=1e-12)
    check("group-128 pays half that", eff["int4 group-128"][1] - 4.0, 0.125, tol=1e-12)
    check("asymmetric group-64 pays double (scale + zero point)",
          eff["int4 group-64 asym"][1] - 4.0, 0.5, tol=1e-12)

    print("\n--- error compensation: the idea behind GPTQ ---")
    # Its own small layer, so the Hessian is cheap and the sequential loop is fast.
    g_in, g_out, g_n = 128, 64, 1024
    grng = np.random.default_rng(3)
    Wg = 0.05 * grng.standard_t(4.0, size=(g_in, g_out))
    Wg[:, [3, 40]] *= 15.0

    def ar1_inputs(rho, seed=11):
        idx = np.arange(g_in)
        C = rho ** np.abs(idx[:, None] - idx[None, :])
        Lc = np.linalg.cholesky(C + 1e-10 * np.eye(g_in))
        return np.random.default_rng(seed).standard_normal((g_n, g_in)) @ Lc.T

    # First the mechanism, isolated. If the layer's inputs are orthogonal, H is
    # diagonal, Hinv is diagonal, and the compensation term is identically zero:
    # error feedback has nothing to feed into. GPTQ MUST reduce to RTN there.
    Q_orth, _ = np.linalg.qr(np.random.default_rng(5).standard_normal((g_n, g_in)))
    X_orth = Q_orth * 3.0
    H_orth = X_orth.T @ X_orth
    print(f"      orthogonal inputs: max off-diagonal of H = "
          f"{np.abs(H_orth - np.diag(np.diag(H_orth))).max():.1e}")
    for bits in (4, 3):
        diff = float(np.abs(gptq_like(Wg, H_orth, bits) - rtn_per_column(Wg, bits)).max())
        check(f"int{bits}: with a diagonal Hessian, compensation is exactly RTN",
              diff, 0.0, tol=0.0)

    print(f"      {'rho':>6} {'cond(H)':>10} {'RTN dY':>10} {'GPTQ dY':>10} {'gain':>7} "
          f"{'RTN dW':>9} {'GPTQ dW':>9}")
    gains = []
    for rho in (0.0, 0.7, 0.9, 0.95):
        Xg = ar1_inputs(rho)
        Hg = Xg.T @ Xg
        Yg = Xg @ Wg
        Wr, Wq = rtn_per_column(Wg, 4), gptq_like(Wg, Hg, 4)
        er, eq = rel_err(Yg, Xg @ Wr), rel_err(Yg, Xg @ Wq)
        wr, wq = rel_err(Wg, Wr), rel_err(Wg, Wq)
        gains.append((rho, float(np.linalg.cond(Hg)), er, eq, wr, wq))
        print(f"      {rho:6.2f} {np.linalg.cond(Hg):10.1f} {er:10.5f} {eq:10.5f} "
              f"{er/eq:6.2f}x {wr:9.5f} {wq:9.5f}")

    check("the gain from error feedback grows with input correlation",
          all(gains[i][2] / gains[i][3] < gains[i + 1][2] / gains[i + 1][3]
              for i in range(len(gains) - 1)))
    rho0, rho_hi = gains[0], gains[-1]
    check("with white-ish inputs there is essentially nothing to gain",
          rho0[2] / rho0[3], 1.0, tol=0.05)
    check("with correlated inputs error feedback beats RTN at the same bit width",
          rho_hi[2] / rho_hi[3] > 2.0)
    # And the reason it is not free lunch: it trades weight-space accuracy for
    # output accuracy, deliberately. Same trade the clipping trap below gets
    # backwards.
    print(f"      at rho=0.95 GPTQ is {rho_hi[2]/rho_hi[3]:.1f}x better on the OUTPUT "
          f"while being {rho_hi[5]/rho_hi[4]:.2f}x WORSE on the weights")
    check("compensation deliberately worsens the weight error", rho_hi[5] > rho_hi[4])

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) WRONG AXIS. Group along axis=1 (runs of columns inside a row) instead
    #     of axis=0. Two things break. Structurally, the step now varies with
    #     the summation index i, so it cannot be pulled out of sum_i x_i W_ij --
    #     there is no per-output-channel scale to multiply the accumulator by,
    #     and the integer kernel is gone. Numerically, a per-row step ties each
    #     weight's absolute error to its own row's magnitude, so the salient
    #     rows -- the ones whose activations are 12x larger -- receive the
    #     largest errors, precisely where the layer can least afford them.
    X_white = np.random.default_rng(0).standard_normal(X.shape)
    Y_white = X_white @ W
    print(f"      {'scheme':>22} {'#scales':>8} {'dW':>9} {'dY white':>10} {'dY real':>10}")
    for label, gs, ax in (("int8 per-column (ax0)", None, 0),
                          ("int8 per-row (ax1)", None, 1),
                          ("int8 group-128 (ax0)", 128, 0),
                          ("int8 group-128 (ax1)", 128, 1)):
        W_hat, _, sc, _ = quantize_rtn(W, 8, gs, ax, "symmetric")
        print(f"      {label:>22} {np.asarray(sc).size:8d} {rel_err(W, W_hat):9.5f} "
              f"{rel_err(Y_white, X_white @ W_hat):10.5f} {rel_err(Y, X @ W_hat):10.5f}")

    W_col, _, sc_col, _ = quantize_rtn(W, 8, None, 0, "symmetric")
    W_row, _, sc_row, _ = quantize_rtn(W, 8, None, 1, "symmetric")
    ey_col, ey_row = rel_err(Y, X @ W_col), rel_err(Y, X @ W_row)
    ew_col, ew_row = rel_err(W, W_col), rel_err(W, W_row)
    yw_col, yw_row = rel_err(Y_white, X_white @ W_col), rel_err(Y_white, X_white @ W_row)
    print(f"      per-row uses {np.asarray(sc_row).size} scales vs per-column's "
          f"{np.asarray(sc_col).size}, and is still {ey_row/ey_col:.1f}x worse downstream")
    # With white inputs E[x x^T] = I, so ||X dW||_F is proportional to ||dW||_F
    # and the two axes look nearly the same. The damage is invisible to any
    # weight-only metric; it only appears once the input has structure.
    check("with white inputs the wrong axis looks almost fine",
          yw_row / yw_col < 1.3)
    check("weight-space error also fails to reveal the problem", ew_row / ew_col < 1.3)
    check("but with outlier input features the wrong axis is far worse downstream",
          ey_row / ey_col > 3.0)
    check("even though it stores twice as many scales",
          np.asarray(sc_row).size > np.asarray(sc_col).size)

    # (b) CLIPPING THE OUTLIERS. The obvious fix: shrink the range so the step
    #     shrinks. Sweep a per-column clip fraction and pick the one that
    #     MINIMIZES the weight-space error -- exactly what a range-search
    #     calibration does. That choice makes the layer worse.
    alphas = np.arange(0.20, 1.01, 0.05)
    ew_a, ey_a = [], []
    for a in alphas:
        cm = a * np.abs(W).max(axis=0, keepdims=True)
        W_hat, _, _, _ = quantize_rtn(np.clip(W, -cm, cm), 4, None, 0, "symmetric")
        ew_a.append(rel_err(W, W_hat))
        ey_a.append(rel_err(Y, X @ W_hat))
    ew_a, ey_a = np.array(ew_a), np.array(ey_a)
    best = int(np.argmin(ew_a))
    print(f"      {'clip':>6} {'||dW||/||W||':>13} {'||dY||/||Y||':>13}")
    for k in range(0, len(alphas), 2):
        mark = "  <- min weight error" if k == best else ""
        print(f"      {alphas[k]:6.2f} {ew_a[k]:13.5f} {ey_a[k]:13.5f}{mark}")
    print(f"      clipping at {alphas[best]:.2f}x improves the weight error "
          f"{ew_a[-1]/ew_a[best]:.2f}x and makes the output {ey_a[best]/ey_a[-1]:.1f}x worse")
    check("the weight-error-optimal clip really does clip (alpha < 1)", alphas[best] < 1.0)
    check("clipping improves the metric everybody reports", ew_a[best] < ew_a[-1])
    check("while making the layer's actual output error several times worse",
          ey_a[best] / ey_a[-1] > 2.0)
    # The two curves are not merely offset, they point in opposite directions
    # over this range: the output error is monotonically increasing in clipping.
    check("output error only ever gets worse as you clip harder",
          all(ey_a[i] > ey_a[i + 1] for i in range(len(ey_a) - 1)))

    summary()
