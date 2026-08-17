"""Artifact 34.1 -- GPTQ from scratch, with the error-propagation ablation,
per-group vs per-tensor scaling, and an AWQ-style outlier demonstration.

Pure NumPy. torch appears only as a cross-check on the output-MSE computation.
Everything asserted is measured here; nothing is quoted from a paper.
"""
import numpy as np

rng = np.random.default_rng(0)

# ----------------------------------------------------------------------
# Setup: a "layer" y = W x with correlated calibration activations.
# X is (d_in, n): n calibration vectors as columns.  The layerwise
# objective GPTQ minimizes is ||W X - W_q X||_F^2, whose Hessian w.r.t.
# each row of W is H = 2 X X^T -- shared across rows.
# ----------------------------------------------------------------------
d_in, d_out, n = 256, 256, 2048
# Controlled covariance: decaying spectrum + random rotation, so H has
# strong off-diagonal structure (propagation only matters if it does).
U, _ = np.linalg.qr(rng.standard_normal((d_in, d_in)))
spec = (np.arange(1, d_in + 1) ** -0.7)            # power-law eigenvalues
A = U * np.sqrt(spec)                               # Sigma^(1/2) up to rotation
X = A @ rng.standard_normal((d_in, n))              # cov(X) ~ U diag(spec) U^T
W = rng.standard_normal((d_out, d_in)) / np.sqrt(d_in)
Y_ref = W @ X                                       # exact layer output


def out_mse(Wq, Xc=X):
    """Layer-output MSE: mean squared error of (W - Wq) Xc."""
    D = (W - Wq) @ Xc
    return float(np.mean(D * D))


# ----------------------------------------------------------------------
# Quantization grids.  Asymmetric affine: q = clip(round(w/s) + z, 0, 2^b-1),
# w_hat = s (q - z).  Scales/zeros are per-row ("per-channel"), per-group
# (contiguous input-dim blocks), or one pair for the whole tensor.
# ----------------------------------------------------------------------
def make_grid(Wm, bits, granularity="channel", group=32):
    qmax = 2 ** bits - 1
    if granularity == "tensor":
        lo, hi = Wm.min(), Wm.max()
        s = np.maximum((hi - lo) / qmax, 1e-12); z = np.round(-lo / s)
        return np.full((Wm.shape[0], Wm.shape[1]), s), np.full(Wm.shape, z)
    if granularity == "channel":
        lo = Wm.min(axis=1, keepdims=True); hi = Wm.max(axis=1, keepdims=True)
        s = np.maximum((hi - lo) / qmax, 1e-12); z = np.round(-lo / s)
        return np.broadcast_to(s, Wm.shape).copy(), np.broadcast_to(z, Wm.shape).copy()
    # per-group: independent (s, z) for each row x each block of `group` cols
    s = np.empty_like(Wm); z = np.empty_like(Wm)
    for g0 in range(0, Wm.shape[1], group):
        blk = Wm[:, g0:g0 + group]
        lo = blk.min(axis=1, keepdims=True); hi = blk.max(axis=1, keepdims=True)
        sg = np.maximum((hi - lo) / qmax, 1e-12)
        s[:, g0:g0 + group] = sg; z[:, g0:g0 + group] = np.round(-lo / sg)
    return s, z


def quant(Wm, s, z, bits):
    q = np.clip(np.round(Wm / s) + z, 0, 2 ** bits - 1)
    return s * (q - z)


def rtn(Wm, bits, granularity="channel", group=32):
    s, z = make_grid(Wm, bits, granularity, group)
    return quant(Wm, s, z, bits)


# ----------------------------------------------------------------------
# GPTQ, Cholesky form.  H = 2 X X^T + damping.  Let T = chol(H^{-1})
# (upper).  Quantize columns left to right; after fixing column j, subtract
# (w_j - q_j)/T_jj * T_{j,j+1:} from the not-yet-quantized columns -- the
# exact OBQ update, batched via the Cholesky factor.
# `propagate=False` uses the identical grid but skips the update:
# that is column-wise RTN on the same grid (ablation (b)).
# ----------------------------------------------------------------------
def gptq(Wm, Xc, bits, granularity="channel", group=32, propagate=True):
    H = 2.0 * (Xc @ Xc.T)
    H[np.diag_indices_from(H)] += 0.01 * np.mean(np.diag(H))  # damping
    Hinv = np.linalg.inv(H)
    T = np.linalg.cholesky(Hinv, upper=True)   # upper-triangular factor
    Wk = Wm.copy()
    s, z = make_grid(Wm, bits, granularity, group)  # grid fixed up front
    Q = np.empty_like(Wm)
    for j in range(Wm.shape[1]):
        w = Wk[:, j]
        Q[:, j] = quant(w, s[:, j], z[:, j], bits)
        if propagate and j + 1 < Wm.shape[1]:
            err = (w - Q[:, j]) / T[j, j]                 # scaled residual
            Wk[:, j + 1:] -= np.outer(err, T[j, j + 1:])  # push into the future
    return Q


# ======================================================================
if __name__ == "__main__":
    # ---------- (a) + (b): GPTQ vs RTN vs GPTQ-without-propagation ----
    print("bits |    RTN MSE   |  no-prop MSE |   GPTQ MSE   | GPTQ/RTN")
    for bits in (4, 3, 2):
        m_rtn = out_mse(rtn(W, bits))
        m_nop = out_mse(gptq(W, X, bits, propagate=False))
        m_gptq = out_mse(gptq(W, X, bits))
        print(f"  {bits}  | {m_rtn:.6e} | {m_nop:.6e} | {m_gptq:.6e} | {m_gptq/m_rtn:6.3f}")
        assert m_gptq < m_rtn,  f"(a) GPTQ !< RTN at {bits} bits"
        assert m_gptq < m_nop,  f"(b) propagation did not help at {bits} bits"
    # sanity: same grid, no propagation == plain per-channel RTN
    assert np.allclose(gptq(W, X, 4, propagate=False), rtn(W, 4))

    # ---------- (c): per-group vs per-tensor at 4 bits (RTN) ----------
    m_tensor = out_mse(rtn(W, 4, "tensor"))
    m_group = out_mse(rtn(W, 4, "group", 32))
    print(f"\n4-bit RTN  per-tensor MSE {m_tensor:.6e}  vs  per-group(32) "
          f"{m_group:.6e}  ({m_tensor/m_group:.2f}x better)")
    assert m_group < m_tensor, "(c) per-group !< per-tensor"

    # ---------- (d): outlier channels + AWQ-style rescue --------------
    Xo = X.copy()
    out_idx = rng.choice(d_in, 4, replace=False)
    Xo[out_idx] *= 50.0                       # 4 emergent-outlier channels
    Y_o = W @ Xo
    dW = W - rtn(W, 3)                        # 3-bit RTN weight error
    # per-input-channel share of output error energy: ||dW_col_j||^2 E[x_j^2]
    energy = (dW ** 2).sum(0) * (Xo ** 2).mean(1)
    frac = energy[out_idx].sum() / energy.sum()
    print(f"\n4/{d_in} channels are outliers (x50); they carry "
          f"{100*frac:.1f}% of weight-quantization output error energy")
    assert frac > 0.5, "(d) error did not concentrate on outlier channels"

    base = float(np.mean(((W - rtn(W, 3)) @ Xo) ** 2))
    best = (None, np.inf)
    for alpha in (0.25, 0.5, 0.75, 1.0):      # AWQ grid search on the scale
        s_ch = np.abs(Xo).mean(1) ** alpha    # per-channel activation scale
        s_ch /= s_ch.mean()
        Wq = rtn(W * s_ch, 3)                 # quantize scaled weights
        m = float(np.mean(((W * s_ch - Wq) @ (Xo / s_ch[:, None])) ** 2))
        if m < best[1]:
            best = (alpha, m)
    print(f"3-bit RTN output MSE on outlier data: {base:.6e} -> "
          f"{best[1]:.6e} with AWQ scale (alpha={best[0]}), "
          f"{base/best[1]:.1f}x reduction")
    assert best[1] < base, "(d) AWQ scaling did not reduce error"

    # ---------- torch cross-check on the MSE computation --------------
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        Wq4 = gptq(W, X, 4)
        ref = torch.mean(((torch.from_numpy(W) - torch.from_numpy(Wq4))
                          @ torch.from_numpy(X)) ** 2).item()
        np_val = out_mse(Wq4)
        print(f"\ntorch cross-check: |MSE_np - MSE_torch| = {abs(ref-np_val):.2e}")
        assert abs(ref - np_val) < 1e-10
    else:
        print("[skipped: torch not installed]")
    print("\nAll assertions passed.")
