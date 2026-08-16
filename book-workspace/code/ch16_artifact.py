"""Chapter 16 artifact: Conv2d forward + backward from scratch via im2col.

Core is pure NumPy. torch is used only as a cross-check and is import-guarded.
Everything is verified: forward against torch.nn.Conv2d, and BOTH gradients
(input and weight) against torch.autograd, over random stride/padding/dilation/
groups configurations. Finally we measure the receptive field of a deep stack
empirically -- which input pixels carry nonzero gradient -- and compare it to
the analytic recursion r_l = r_{l-1} + (k_l - 1) * d_l * j_{l-1}.
"""
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


# ----------------------------------------------------------------------------
# im2col: the index arithmetic that turns a sliding window into a matmul.
# ----------------------------------------------------------------------------
def _im2col_indices(Cin, H, W, KH, KW, stride, pad, dil):
    """Return (k, i, j) index arrays selecting patches out of the PADDED input.

    Shapes: each is (Cin*KH*KW, Hout*Wout) after broadcasting, so that
    Xpad[:, k, i, j] is (B, Cin*KH*KW, Hout*Wout) -- the column matrix.
    """
    sh, sw = stride
    ph, pw = pad
    dh, dw = dil
    Hout = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    Wout = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    # offsets within one kernel window (dilated)
    i0 = np.repeat(np.arange(KH), KW) * dh            # (KH*KW,)
    i0 = np.tile(i0, Cin)                             # (Cin*KH*KW,)
    j0 = np.tile(np.arange(KW) * dw, KH * Cin)
    # top-left corner of each output position (strided)
    i1 = sh * np.repeat(np.arange(Hout), Wout)        # (Hout*Wout,)
    j1 = sw * np.tile(np.arange(Wout), Hout)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(Cin), KH * KW).reshape(-1, 1)
    return k, i, j, Hout, Wout


def im2col(X, KH, KW, stride, pad, dil):
    """X: (B, Cin, H, W) -> cols: (B, Cin*KH*KW, Hout*Wout)."""
    B, Cin, H, W = X.shape
    ph, pw = pad
    Xp = np.pad(X, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    k, i, j, Hout, Wout = _im2col_indices(Cin, H, W, KH, KW, stride, pad, dil)
    return Xp[:, k, i, j], Hout, Wout


def col2im(cols, Xshape, KH, KW, stride, pad, dil):
    """Adjoint of im2col: scatter-add columns back into an input-shaped array."""
    B, Cin, H, W = Xshape
    ph, pw = pad
    Xp = np.zeros((B, Cin, H + 2 * ph, W + 2 * pw), dtype=cols.dtype)
    k, i, j, _, _ = _im2col_indices(Cin, H, W, KH, KW, stride, pad, dil)
    # np.add.at accumulates on repeated indices -- overlapping windows demand it.
    np.add.at(Xp, (slice(None), k, i, j), cols)
    return Xp[:, :, ph:H + ph, pw:W + pw]


# ----------------------------------------------------------------------------
# Conv2d forward / backward, grouped.
# ----------------------------------------------------------------------------
def conv2d_forward(X, Wt, b, stride=(1, 1), pad=(0, 0), dil=(1, 1), groups=1):
    """X:(B,Cin,H,W)  Wt:(Cout,Cin/g,KH,KW)  b:(Cout,) -> Y:(B,Cout,Hout,Wout)."""
    B, Cin, H, W = X.shape
    Cout, Cing, KH, KW = Wt.shape
    assert Cin == Cing * groups and Cout % groups == 0
    Og = Cout // groups
    outs, caches = [], []
    for g in range(groups):
        Xg = X[:, g * Cing:(g + 1) * Cing]
        cols, Hout, Wout = im2col(Xg, KH, KW, stride, pad, dil)   # (B, Cing*KH*KW, P)
        Wg = Wt[g * Og:(g + 1) * Og].reshape(Og, -1)              # (Og, Cing*KH*KW)
        Yg = np.einsum('ok,bkp->bop', Wg, cols)                   # (B, Og, P)
        outs.append(Yg.reshape(B, Og, Hout, Wout))
        caches.append(cols)
    Y = np.concatenate(outs, axis=1) + b.reshape(1, -1, 1, 1)
    return Y, (caches, X.shape, Wt, stride, pad, dil, groups)


def conv2d_backward(dY, cache):
    """Returns dX, dW, db. dW is a correlation of X with dY; dX is col2im of W^T dY."""
    caches, Xshape, Wt, stride, pad, dil, groups = cache
    B, Cin, H, W = Xshape
    Cout, Cing, KH, KW = Wt.shape
    Og = Cout // groups
    db = dY.sum(axis=(0, 2, 3))
    dX_parts, dW_parts = [], []
    for g in range(groups):
        cols = caches[g]
        dYg = dY[:, g * Og:(g + 1) * Og].reshape(B, Og, -1)        # (B, Og, P)
        # weight gradient: sum over batch and spatial positions
        dWg = np.einsum('bop,bkp->ok', dYg, cols)                  # (Og, Cing*KH*KW)
        dW_parts.append(dWg.reshape(Og, Cing, KH, KW))
        # input gradient: transpose of the matmul, then scatter back (col2im)
        Wg = Wt[g * Og:(g + 1) * Og].reshape(Og, -1)
        dcols = np.einsum('ok,bop->bkp', Wg, dYg)                  # (B, Cing*KH*KW, P)
        dX_parts.append(col2im(dcols, (B, Cing, H, W), KH, KW, stride, pad, dil))
    return np.concatenate(dX_parts, axis=1), np.concatenate(dW_parts, axis=0), db



def conv_flops_params(Cin, Cout, KH, KW, Hout, Wout, groups=1, bias=True):
    """Analytic counts. FLOPs counted as multiply-adds x2, per single image."""
    params = Cout * (Cin // groups) * KH * KW + (Cout if bias else 0)
    macs = Cout * (Cin // groups) * KH * KW * Hout * Wout
    return params, 2 * macs


def receptive_field(layers):
    """layers: list of (k, s, d). Returns (r, jump) after the whole stack."""
    r, j = 1, 1
    for (k, s, d) in layers:
        r = r + (k - 1) * d * j
        j = j * s
    return r, j


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------
CONFIGS = [
    # (B, Cin, Cout, H, W, KH, KW, stride, pad, dil, groups)
    (2, 3, 4, 9, 9, 3, 3, (1, 1), (1, 1), (1, 1), 1),
    (2, 4, 6, 11, 9, 3, 3, (2, 2), (1, 1), (1, 1), 1),
    (1, 4, 4, 13, 13, 3, 3, (1, 1), (2, 2), (2, 2), 1),
    (2, 6, 6, 10, 12, 3, 3, (2, 1), (1, 2), (1, 1), 3),
    (2, 8, 8, 9, 9, 3, 3, (1, 1), (1, 1), (1, 1), 8),   # depthwise
    (1, 3, 5, 8, 8, 5, 3, (2, 2), (0, 1), (1, 2), 1),
]


def check_config(cfg, rng):
    B, Cin, Cout, H, W, KH, KW, st, pd, dl, g = cfg
    X = rng.standard_normal((B, Cin, H, W))
    Wt = rng.standard_normal((Cout, Cin // g, KH, KW)) * 0.3
    b = rng.standard_normal(Cout) * 0.1
    Y, cache = conv2d_forward(X, Wt, b, st, pd, dl, g)
    dY = rng.standard_normal(Y.shape)
    dX, dW, db = conv2d_backward(dY, cache)
    if torch is None:
        return None
    tX = torch.tensor(X, requires_grad=True)
    tW = torch.tensor(Wt, requires_grad=True)
    tb = torch.tensor(b, requires_grad=True)
    tY = torch.nn.functional.conv2d(tX, tW, tb, stride=st, padding=pd,
                                    dilation=dl, groups=g)
    tY.backward(torch.tensor(dY))
    ef = np.abs(Y - tY.detach().numpy()).max()
    ex = np.abs(dX - tX.grad.numpy()).max()
    ew = np.abs(dW - tW.grad.numpy()).max()
    eb = np.abs(db - tb.grad.numpy()).max()
    return Y.shape, ef, ex, ew, eb


def main():
    rng = np.random.default_rng(0)
    print("=" * 72)
    print("Conv2d from scratch (im2col) vs torch.nn.functional.conv2d")
    print("=" * 72)
    if torch is None:
        print("[skipped: torch not installed] -- running forward/backward only")
    hdr = f"{'cfg (s,p,d,g)':>22} {'out shape':>16} {'|dY|f':>10} {'|dX|':>10} {'|dW|':>10} {'|db|':>10}"
    print(hdr)
    worst = 0.0
    for cfg in CONFIGS:
        res = check_config(cfg, rng)
        if res is None:
            continue
        shape, ef, ex, ew, eb = res
        tag = f"s{cfg[7]} p{cfg[8]} d{cfg[9]} g{cfg[10]}".replace(" ", "")
        print(f"{tag:>22} {str(shape):>16} {ef:10.2e} {ex:10.2e} {ew:10.2e} {eb:10.2e}")
        worst = max(worst, ef, ex, ew, eb)
        assert max(ef, ex, ew, eb) < 1e-5, f"config {cfg} failed"
    if torch is not None:
        print(f"\nworst residual over all {len(CONFIGS)} configs: {worst:.3e}  (tol 1e-5)")

    # ---- FLOP / parameter arithmetic on a concrete layer -------------------
    print("\n" + "-" * 72)
    print("FLOP and parameter arithmetic: 3x3, 256->256, 56x56 feature map")
    p_full, f_full = conv_flops_params(256, 256, 3, 3, 56, 56, groups=1)
    p_dw, f_dw = conv_flops_params(256, 256, 3, 3, 56, 56, groups=256)
    p_pw, f_pw = conv_flops_params(256, 256, 1, 1, 56, 56, groups=1)
    print(f"  dense 3x3      params={p_full:>10,d}  FLOPs={f_full:>14,d}")
    print(f"  depthwise 3x3  params={p_dw:>10,d}  FLOPs={f_dw:>14,d}")
    print(f"  pointwise 1x1  params={p_pw:>10,d}  FLOPs={f_pw:>14,d}")
    print(f"  separable/dense param ratio = {(p_dw+p_pw)/p_full:.4f}"
          f"   FLOP ratio = {(f_dw+f_pw)/f_full:.4f}")
    print(f"  closed form 1/Cout + 1/(KH*KW) = {1/256 + 1/9:.4f}")
    assert abs((f_dw + f_pw) / f_full - (1 / 256 + 1 / 9)) < 2e-3

    # ---- receptive field: analytic vs measured -----------------------------
    print("\n" + "-" * 72)
    print("Receptive field of a deep stack: analytic vs measured (nonzero grad)")
    # (k, s, d) per layer; no padding so the measured box is never clipped.
    layers = [(3, 1, 1), (3, 1, 1), (3, 2, 1), (3, 1, 2), (3, 1, 4), (3, 2, 1), (3, 1, 1)]
    r_an, j_an = receptive_field(layers)
    H = W = 2 * r_an + 8
    X = rng.random((1, 1, H, W)) + 1.0        # strictly positive input
    caches = []
    cur = X
    for (k, s, d) in layers:
        Wt = rng.random((1, 1, k, k)) + 0.5   # strictly positive: no cancellation,
        b = np.zeros(1)                       # so "nonzero grad" == "in the RF"
        cur, c = conv2d_forward(cur, Wt, b, (s, s), (0, 0), (d, d), 1)
        caches.append(c)
    Ho, Wo = cur.shape[2], cur.shape[3]
    dY = np.zeros_like(cur)
    dY[0, 0, Ho // 2, Wo // 2] = 1.0          # seed ONE output unit
    gr = dY
    for c in reversed(caches):                # backprop the whole stack
        gr, _, _ = conv2d_backward(gr, c)
    nz = np.argwhere(np.abs(gr[0, 0]) > 0)
    rh = nz[:, 0].max() - nz[:, 0].min() + 1
    rw = nz[:, 1].max() - nz[:, 1].min() + 1
    print(f"  layers (k,s,d): {layers}")
    print(f"  analytic receptive field r = {r_an}, jump j = {j_an}")
    print(f"  measured nonzero-gradient box = {rh} x {rw} pixels"
          f"  ({nz.shape[0]} of {H*W} input pixels touched)")
    assert rh == r_an and rw == r_an, (rh, rw, r_an)
    dense = (nz.shape[0] == rh * rw)
    print(f"  box fully dense (every pixel inside reached): {dense}")

    # a stride-only stack, to show jump != receptive field
    layers2 = [(3, 2, 1)] * 5
    r2, j2 = receptive_field(layers2)
    print(f"  five 3x3 stride-2 layers: r = {r2}, jump = {j2}"
          f"  (adjacent outputs share r-j = {r2-j2} of {r2} input pixels)")
    assert (r2, j2) == (63, 32)
    print("\nALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
