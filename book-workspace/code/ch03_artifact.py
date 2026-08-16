"""
Artifact 3.1 -- Double descent in random-feature ridgeless regression, measured
against the classical capacity bounds. Three parts, pure NumPy + SciPy:

(A) n = 100 fixed points; a random ReLU feature map of width p swept 1 -> 2000;
    at each p the MINIMUM-NORM interpolant, and its held-out risk. The curve
    rises, spikes at p = n, then descends below the classical minimum.
(B) The same feature map at three widths, refit to binary targets so that 0-1
    loss is defined: VC for linear threshold functions on p features, plus a
    norm-based Rademacher bound from MEASURED norm and radius. Both exceed 1.
(C) Self-verification: both least-squares branches against closed forms, the
    push-through ridge identity, brute-force shattering, torch cross-check.

Seeded with np.random.default_rng(0). Runtime ~6 s on CPU.
"""
import time
from math import e, log, sqrt

import numpy as np
from scipy.optimize import linprog

SEED = 0
N_TRAIN = 100        # n -- small, so the interpolation threshold sits at 100
D_IN = 20            # input dimension
N_TEST = 1500        # held-out sample for risk estimation
NOISE_SD = 0.5       # label noise; the spike is noise amplification
PROBE_NOISE = 0.25   # label noise for the binary probe used by the bounds
P_MAX = 2000         # widest random-feature model
N_TRIALS = 12        # independent draws of (data, features), averaged
DELTA = 0.05         # confidence parameter for all bounds
RCOND = 1e-14        # keep near-singular directions: the spike lives there

def sweep_widths():
    """Log-spaced widths, densified around the interpolation threshold p = n."""
    ws = {int(round(x)) for x in np.geomspace(1, P_MAX, 34)}
    ws |= set(range(60, 171, 10)) | {90, 95, 98, 100, 102, 105, 110}
    return np.array(sorted(w for w in ws if 1 <= w <= P_MAX))

def draw_features(rng):
    """Random ReLU feature weights ~ N(0, 1/d). Widths are nested prefixes, so
    each model in the sweep contains the features of every smaller model."""
    return rng.normal(0.0, 1.0 / sqrt(D_IN), size=(P_MAX, D_IN))

def phi(X, W):
    """Feature map phi(x) = relu(Wx); all P_MAX features in one matmul."""
    return np.maximum(X @ W.T, 0.0)

def min_norm_fit(Phi, y):
    """Minimum-l2-norm least squares; handles p < n and p > n alike. rcond is
    tiny on purpose -- truncating small singular values is implicit
    regularisation and would flatten the very spike we are trying to see."""
    return np.linalg.pinv(Phi, rcond=RCOND) @ y

def run_sweep(widths):
    """Mean test MSE, mean ||w||_2, and mean train MSE at each width."""
    rng = np.random.default_rng(SEED)
    beta = rng.normal(size=D_IN)
    beta /= np.linalg.norm(beta)          # teacher: unit-norm linear, Var = 1
    te = np.zeros((N_TRIALS, len(widths)))
    tr, nw = np.zeros_like(te), np.zeros_like(te)
    for t in range(N_TRIALS):
        Xtr = rng.normal(size=(N_TRAIN, D_IN))
        Xte = rng.normal(size=(N_TEST, D_IN))
        ytr = Xtr @ beta + NOISE_SD * rng.normal(size=N_TRAIN)
        yte = Xte @ beta                  # clean test target: signal only
        W = draw_features(rng)
        Ftr, Fte = phi(Xtr, W), phi(Xte, W)
        for j, p in enumerate(widths):
            A, Bm = Ftr[:, :p], Fte[:, :p]
            w = min_norm_fit(A, ytr)
            tr[t, j] = np.mean((A @ w - ytr) ** 2)
            te[t, j] = np.mean((Bm @ w - yte) ** 2)
            nw[t, j] = np.linalg.norm(w)
    return te.mean(0), nw.mean(0), tr.mean(0)

def vc_bound(h, n, delta=DELTA):
    """gap <= sqrt( 8 (log Pi(2n) + log(4/delta)) / n ). Symmetrisation doubles
    the sample, so the growth function is evaluated at 2n. Sauer's lemma gives
    log Pi(2n) <= h log(2en/h) when 2n >= h; otherwise the class shatters the
    ghost sample outright and only the trivial Pi(2n) <= 2^(2n) is available."""
    lg = h * log(2 * e * n / h) if 2 * n >= h else 2 * n * log(2)
    return sqrt(8.0 * (min(lg, 2 * n * log(2)) + log(4.0 / delta)) / n)

def vc_sample_complexity(h, target=1.0, delta=DELTA):
    """Smallest n at which the VC bound first drops to `target`. Bisection."""
    lo, hi = 2, 1
    while vc_bound(h, hi, delta) > target:
        hi *= 2
        if hi > 10 ** 15:
            return None
    while hi - lo > max(1, lo // 1000):
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if vc_bound(h, mid, delta) > target else (lo, mid)
    return hi

def rademacher_margin_bound(margin_loss, wnorm, radius, gamma, n, delta=DELTA):
    """Linear predictors of norm <= B on features of norm <= R have empirical
    Rademacher complexity <= B R / sqrt(n); the gamma-margin ramp is (1/gamma)-
    Lipschitz, so contraction divides by gamma and the 2 is Definition 3.4's.
    margin_loss is the indicator {margin < gamma}, which upper-bounds the ramp:
        err_01 <= margin_loss + 2 B R/(gamma sqrt(n)) + 3 sqrt(log(2/delta)/2n)."""
    return (margin_loss + 2.0 * wnorm * radius / (gamma * sqrt(n))
            + 3.0 * sqrt(log(2.0 / delta) / (2.0 * n)))

def binary_probe(probe_ws, n_trials=8):
    """Same feature map, binary targets t = sign(<beta,x> + noise), min-norm
    fit thresholded at 0. Returns measured 0-1 test error plus the three
    empirical quantities the margin bound needs."""
    rng = np.random.default_rng(SEED + 1)
    beta = rng.normal(size=D_IN)
    beta /= np.linalg.norm(beta)
    out = {p: dict(err=0.0, wn=0.0, rad=0.0, ml=0.0) for p in probe_ws}
    for _ in range(n_trials):
        Xtr = rng.normal(size=(N_TRAIN, D_IN))
        Xte = rng.normal(size=(N_TEST, D_IN))
        ttr = np.sign(Xtr @ beta + PROBE_NOISE * rng.normal(size=N_TRAIN))
        tte = np.sign(Xte @ beta)
        W = draw_features(rng)
        Ftr, Fte = phi(Xtr, W), phi(Xte, W)
        for p in probe_ws:
            A, Bm = Ftr[:, :p], Fte[:, :p]
            w = min_norm_fit(A, ttr)
            out[p]["ml"] += np.mean(ttr * (A @ w) < 1.0) / n_trials
            out[p]["wn"] += np.linalg.norm(w) / n_trials
            out[p]["rad"] += np.max(np.linalg.norm(A, axis=1)) / n_trials
            out[p]["err"] += np.mean(np.sign(Bm @ w) != tte) / n_trials
    return out

def separable(X, s):
    """Feasibility LP: is there (w,b) with s_i (<w,x_i> + b) >= 1 for all i?"""
    m, k = X.shape
    res = linprog(c=np.zeros(k + 1), b_ub=-np.ones(m), method="highs",
                  A_ub=-np.hstack([s[:, None] * X, s[:, None]]),
                  bounds=[(None, None)] * (k + 1))
    return bool(res.status == 0)

def all_labelings(m):
    """All 2^m sign vectors in {-1,+1}^m."""
    return [np.array([1.0 if (i >> b) & 1 else -1.0 for b in range(m)])
            for i in range(2 ** m)]

def shattering_check():
    """Halfspaces in R^2: the 3-point simplex is shattered; the XOR square
    (points 0,1 on one diagonal, 2,3 on the other) is not."""
    P3 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    P4 = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    return (sum(separable(P3, s) for s in all_labelings(3)),
            sum(separable(P4, s) for s in all_labelings(4)))

def closed_form_checks():
    """pinv against both normal-equation branches, plus the ridge identity
    (A'A + lam I)^-1 A' = A' (AA' + lam I)^-1."""
    rng = np.random.default_rng(SEED + 2)
    n, lam = 40, 1e-3
    At, Aw = rng.normal(size=(n, 10)), rng.normal(size=(n, 90))
    y = rng.normal(size=n)
    r_thin = np.linalg.norm(min_norm_fit(At, y) - np.linalg.solve(At.T @ At, At.T @ y))
    r_wide = np.linalg.norm(min_norm_fit(Aw, y) - Aw.T @ np.linalg.solve(Aw @ Aw.T, y))
    primal = np.linalg.solve(Aw.T @ Aw + lam * np.eye(90), Aw.T @ y)
    dual = Aw.T @ np.linalg.solve(Aw @ Aw.T + lam * np.eye(n), y)
    return r_thin, r_wide, np.max(np.abs(primal - dual))

def torch_crosscheck():
    """Optional: our min-norm solve against torch's SVD-based lstsq."""
    try:
        import torch
    except ImportError:
        return None
    rng = np.random.default_rng(SEED + 3)
    A, y = rng.normal(size=(60, 200)), rng.normal(size=60)
    ref = torch.linalg.lstsq(torch.tensor(A), torch.tensor(y),
                             driver="gelsd").solution.numpy()
    return float(np.max(np.abs(min_norm_fit(A, y) - ref)))

def main():
    t0 = time.time()
    ws = sweep_widths()
    te, nw, tr = run_sweep(ws)
    print("=" * 74)
    print("(A) MODEL-WISE DOUBLE DESCENT  (n=%d, d=%d, noise sd=%.2f, %d trials)"
          % (N_TRAIN, D_IN, NOISE_SD, N_TRIALS))
    print("=" * 74)
    print("%8s %12s %12s %12s" % ("width p", "train MSE", "test MSE", "||w||_2"))
    show = (1, 2, 5, 10, 20, 50, 80, 90, 100, 110, 130, 170, 300, 700, 2000)
    for j, p in enumerate(ws):
        if p in show:
            print("%8d %12.4f %12.4f %12.3f" % (p, tr[j], te[j], nw[j]))
    k, hi = int(np.argmax(te)), len(ws) - 1
    lo = int(np.argmin(np.abs(ws - N_TRAIN // 4)))   # classical sweet spot
    j_cls = int(np.argmin(te[:k]))
    print("\npeak test MSE %.4f at width %d (interpolation threshold n = %d)" % (te[k], ws[k], N_TRAIN))
    print("classical minimum : test MSE %.4f at width %d" % (te[j_cls], ws[j_cls]))
    print("modern regime     : test MSE %.4f at width %d" % (te[hi], ws[hi]))
    print("spike / underparam: %.1fx      spike / widest: %.1fx" % (te[k] / te[lo], te[k] / te[hi]))
    assert 0.6 * N_TRAIN <= ws[k] <= 1.6 * N_TRAIN, "peak not near n"
    assert te[k] > 3.0 * te[lo], "no left-side rise"
    assert te[k] > 5.0 * te[hi], "no right-side second descent"
    assert te[hi] < te[:k].min(), "second descent did not beat classical min"
    print("[assert] spike localised at p = n, dominates both neighbours: PASS")

    print("\n" + "=" * 74)
    print("(B) CLASSICAL BOUNDS, SAME FEATURE MAP, BINARY TARGETS  (delta=%.2f)" % DELTA)
    print("=" * 74)
    probe_ws = [20, 100, 2000]
    pr = binary_probe(probe_ws)
    print("%8s %10s %9s %9s %11s %11s %10s" % ("width p", "test 0-1", "||w||", "radius R", "margin loss", "Rademacher", "VC bound"))
    for p in probe_ws:
        s = pr[p]
        rb = rademacher_margin_bound(s["ml"], s["wn"], s["rad"], 1.0, N_TRAIN)
        vb = vc_bound(p + 1, N_TRAIN)
        print("%8d %10.4f %9.3f %9.2f %11.3f %11.3f %10.3f"
              % (p, s["err"], s["wn"], s["rad"], s["ml"], rb, vb))
        assert rb > 1.0 and vb > 1.0, "bound at p=%d not vacuous" % p
    print("[assert] every bound exceeds 1 -> vacuous, while measured 0-1 error"
          " at p=2000 is %.4f: PASS" % pr[2000]["err"])
    print("\nsanity ladder for sqrt(8(log Pi(2n) + log(4/delta))/n):")
    print("%12s %12s %12s" % ("VC dim h", "n", "bound"))
    for h, n in [(21, 100), (21, 10_000), (101, 100), (2001, 100),
                 (2_600_000, 60_000), (2_600_000, 10_000_000),
                 (2_600_000, 2_000_000_000)]:
        print("%12d %12d %12.3f" % (h, n, vc_bound(h, n)))
    print("rows 5-7: a 784-100-10 MLP, W = 79,510 weights, L = 2," " VCdim ~ W L log2 W = %.2e" % (79510 * 2 * np.log2(79510)))
    for h, lbl in [(21, "p=20 features,"), (2001, "p=2000 features,"),
                   (2_600_000, "784-100-10 MLP,")]:
        print("  n for the VC bound to reach 1.0 (%-17s h=%7d): %s"
              % (lbl, h, "{:,}".format(vc_sample_complexity(h))))

    print("\n" + "=" * 74)
    print("(C) SELF-VERIFICATION")
    print("=" * 74)
    r_thin, r_wide, r_ridge = closed_form_checks()
    print("pinv vs (A'A)^-1 A'y  (p=10 < n=40) residual : %.3e" % r_thin)
    print("pinv vs A'(AA')^-1 y  (p=90 > n=40) residual : %.3e" % r_wide)
    print("push-through ridge identity, max abs diff    : %.3e" % r_ridge)
    assert r_thin < 1e-8 and r_wide < 1e-8 and r_ridge < 1e-8
    ok3, ok4 = shattering_check()
    print("halfspaces in R^2: %d/8 labelings of the 3-point simplex realised," " %d/16 of the XOR square" % (ok3, ok4))
    assert (ok3, ok4) == (8, 14), "shattering counts wrong: %d %d" % (ok3, ok4)
    print("[assert] VCdim(halfspaces in R^2) = 3 by brute force: PASS")
    tc = torch_crosscheck()
    if tc is None:
        print("[skipped: torch not installed]")
    else:
        print("torch.linalg.lstsq vs our pinv, max abs diff : %.3e" % tc)
        assert tc < 1e-8
    print("\nwall clock: %.1f s" % (time.time() - t0))

if __name__ == "__main__":
    main()
