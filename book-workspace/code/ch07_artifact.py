"""
Artifact 7.1 -- SMO for the SVM dual, the scaling wall, and the empirical NTK.
Core is pure NumPy; sklearn and torch appear only as cross-checks.
  (1) SMO on a 3-point hand-solvable linear problem recovers alpha = (1/4, 1/4, 0);
  (2) SMO on 120 points with an RBF kernel matches sklearn.svm.SVC, and the KKT
      conditions are checked directly at the returned alpha;
  (3) SMO's iteration and kernel-evaluation counts are fitted against n, and the
      flop crossover against an N-parameter SGD model is solved exactly;
  (4) the empirical NTK of a width-m two-layer ReLU net, formed as the Jacobian
      outer product J J^T, converges to the arc-cosine limit at rate ~ m^-1/2.
Runtime ~1.5 s on CPU.
"""

import time
import numpy as np

try:
    from sklearn.svm import SVC
except ImportError:                      # cross-check only
    SVC = None
try:
    import torch
except ImportError:                      # cross-check only
    torch = None

# --- 1. Kernels --------------------------------------------------------------

def k_rbf(A, B, gamma):
    """exp(-gamma ||a-b||^2) via the expanded square; clamp to kill -1e-16 distances."""
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(d2, 0.0))

def k_linear(A, B):
    return A @ B.T

# --- 2. SMO for the soft-margin dual ----------------------------------------
# maximize  sum_i a_i - 1/2 sum_ij a_i a_j y_i y_j K_ij   s.t. sum_i a_i y_i = 0,
# 0 <= a_i <= C.  Working-set selection is the maximal-violating-pair rule
# (Keerthi/LIBSVM WSS1): with f_i = sum_j a_j y_j K_ij and E_i = f_i - y_i, the
# KKT system is equivalent to  max_{i in I_low} E_i  <=  min_{i in I_up} E_i.

def smo(K, y, C, tol=1e-10, max_iter=100_000):
    n = y.shape[0]
    a, f = np.zeros(n), np.zeros(n)      # f_i = sum_j a_j y_j K_ij  (no bias)
    tau = eps = 1e-12
    for it in range(max_iter):
        E = f - y
        # I_up: alpha can move so as to DECREASE  y_i * (margin); I_low: increase.
        up = ((a < C - eps) & (y > 0)) | ((a > eps) & (y < 0))
        low = ((a < C - eps) & (y < 0)) | ((a > eps) & (y > 0))
        if not up.any() or not low.any():
            break
        i = int(np.argmin(np.where(up, E, np.inf)))      # b_up  = min_{I_up} E
        j = int(np.argmax(np.where(low, E, -np.inf)))    # b_low = max_{I_low} E
        b_up, b_low = E[i], E[j]
        if b_low - b_up <= tol:                          # duality gap closed
            break
        # --- two-variable subproblem, Platt's box constraints -----------------
        y1, y2, a1, a2 = y[i], y[j], a[i], a[j]
        s = y1 * y2
        if s < 0:
            L, H = max(0.0, a2 - a1), min(C, C + a2 - a1)
        else:
            L, H = max(0.0, a1 + a2 - C), min(C, a1 + a2)
        if H - L < 1e-15:
            break
        eta = K[i, i] + K[j, j] - 2.0 * K[i, j]          # curvature along the line
        eta = max(eta, tau)                              # PSD-but-singular guard
        a2n = min(max(a2 + y2 * (E[i] - E[j]) / eta, L), H)
        a1n = min(max(a1 + s * (a2 - a2n), 0.0), C)
        # rank-2 update of f: only two alphas moved
        f += (a1n - a1) * y1 * K[i] + (a2n - a2) * y2 * K[j]
        a[i], a[j] = a1n, a2n
    E = f - y
    up = ((a < C - eps) & (y > 0)) | ((a > eps) & (y < 0))
    low = ((a < C - eps) & (y < 0)) | ((a > eps) & (y > 0))
    b_up = np.min(np.where(up, E, np.inf)) if up.any() else 0.0
    b_low = np.max(np.where(low, E, -np.inf)) if low.any() else 0.0
    b = -0.5 * (b_up + b_low)                            # midpoint of the free interval
    dual = a.sum() - 0.5 * float((a * y) @ K @ (a * y))
    return a, b, dual, it + 1

def kkt_violation(K, y, a, b, C):
    """Max violation of the complementary-slackness conditions, plus |sum a_i y_i|."""
    r = y * (K @ (a * y) + b)                            # functional margin y f(x)
    eps = 1e-8 * max(C, 1.0)
    v = np.where(a <= eps, np.maximum(0.0, 1.0 - r),
        np.where(a >= C - eps, np.maximum(0.0, r - 1.0), np.abs(r - 1.0)))
    return float(v.max()), float(abs((a * y).sum()))

# --- 3. Empirical NTK of a two-layer ReLU network ----------------------------
# f(x) = m^{-1/2} sum_i a_i relu(w_i . x),  w_i ~ N(0, I_d),  a_i ~ N(0, 1).
# Trainable: both layers.  The Jacobian is closed form, so no autodiff is needed.

def jacobian_two_layer(X, W, a):
    """Return J with J[k, :] = grad_theta f(x_k), theta = (vec W, a). Shape (n, m*d+m)."""
    n, d, m = X.shape[0], X.shape[1], W.shape[0]
    pre = X @ W.T                                        # (n, m)
    h = np.maximum(pre, 0.0)
    sgn = (pre > 0).astype(np.float64)
    inv = 1.0 / np.sqrt(m)
    J_W = inv * (sgn * a[None, :])[:, :, None] * X[:, None, :]   # (n, m, d)
    J_a = inv * h                                                # (n, m)
    return np.concatenate([J_W.reshape(n, m * d), J_a], axis=1)

def ntk_empirical(X, m, rng):
    W, a = rng.standard_normal((m, X.shape[1])), rng.standard_normal(m)
    J = jacobian_two_layer(X, W, a)
    return J @ J.T                                       # the Jacobian outer product

def ntk_analytic(X):
    """m -> infinity limit for ||x|| = 1:  (1/2) k1(u) + (1/2) k0(u) u,
    with k0(u) = (pi - t)/pi, k1(u) = (sin t + (pi - t) u)/pi, t = arccos u."""
    u = np.clip(X @ X.T, -1.0, 1.0)
    t = np.arccos(u)
    return (np.sin(t) + 2.0 * (np.pi - t) * u) / (2.0 * np.pi)

# --- 4. Demos ----------------------------------------------------------------

def demo_smo_tiny():
    print("[1] SMO on the 3-point hand-solvable problem")
    X = np.array([[1.0, 1.0], [-1.0, -1.0], [2.0, 2.0]])
    y = np.array([1.0, -1.0, 1.0])
    K = k_linear(X, X)
    a, b, dual, it = smo(K, y, C=10.0)
    w = (a * y) @ X
    print(f"    alpha      = {np.array2string(a, precision=6)}   (exact: [0.25 0.25 0])")
    print(f"    w = {np.array2string(w, precision=6)}, b = {b:+.6e}   (exact: [0.5 0.5], 0)")
    print(f"    margin 2/||w|| = {2/np.linalg.norm(w):.6f}   (exact: 2*sqrt(2) = {2*np.sqrt(2):.6f})")
    print(f"    dual objective = {dual:.6f}   (exact: 0.25),  iters = {it}")
    assert np.allclose(a, [0.25, 0.25, 0.0], atol=1e-9)
    assert abs(dual - 0.25) < 1e-9

def demo_smo_vs_sklearn():
    print("[2] SMO vs sklearn.svm.SVC, n=120, RBF(gamma=0.5), C=1.0")
    rng = np.random.default_rng(0)
    n, C, gamma = 120, 1.0, 0.5
    X = rng.standard_normal((n, 2))
    y = np.where(X[:, 0] ** 2 + 0.7 * X[:, 1] ** 2 + 0.3 * rng.standard_normal(n) > 1.2, 1.0, -1.0)
    K = k_rbf(X, X, gamma)
    t0 = time.perf_counter()
    a, b, dual, it = smo(K, y, C=C)
    t_smo = time.perf_counter() - t0
    viol, eqc = kkt_violation(K, y, a, b, C)
    sv = np.where(a > 1e-8)[0]
    print(f"    iters = {it}, time = {t_smo*1e3:.1f} ms, dual = {dual:.10f}")
    print(f"    |#SV| = {sv.size} ({np.sum(a > C-1e-8)} at the box bound C)")
    print(f"    max KKT violation = {viol:.3e},  |sum a_i y_i| = {eqc:.3e}")
    assert viol < 1e-6 and eqc < 1e-9
    if SVC is None:
        print("    [skipped: sklearn not installed]")
        return
    clf = SVC(C=C, kernel="precomputed", tol=1e-12).fit(K, y)
    a_sk = np.zeros(n)
    a_sk[clf.support_] = np.abs(clf.dual_coef_[0])
    dual_sk = a_sk.sum() - 0.5 * float((a_sk * y) @ K @ (a_sk * y))
    Xg = rng.standard_normal((400, 2))
    Kg = k_rbf(Xg, X, gamma)
    ours = Kg @ (a * y) + b
    theirs = clf.decision_function(Kg)
    d_alpha = float(np.abs(a - a_sk).max())
    d_dec = float(np.abs(ours - theirs).max())
    same_sv = set(sv.tolist()) == set(clf.support_.tolist())
    print(f"    sklearn dual = {dual_sk:.10f}   (ours - theirs = {dual - dual_sk:+.3e})")
    print(f"    max |alpha - alpha_sklearn| = {d_alpha:.3e}")
    print(f"    max |f(x) - f_sklearn(x)| over 400 fresh points = {d_dec:.3e}")
    print(f"    support-vector index sets identical: {same_sv}")
    assert d_dec < 1e-4 and d_alpha < 1e-4 and same_sv

def demo_cubic_wall():
    print("[3] How kernel training scales, and where a parametric model wins")
    # (a) SMO's own work: iteration count and kernel entries touched, deterministic
    #     given the fixed rng.  Wall clock on a shared VM is not; flops and counts are.
    ns, iters, kevals = (250, 500, 1000, 2000), [], []
    for n in ns:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, 20))
        yv = np.where(np.linalg.norm(X[:, :3], axis=1) > 1.6, 1.0, -1.0)
        K = k_rbf(X, X, 0.05)
        t0 = time.perf_counter()
        a, b, dual, it = smo(K, yv, C=1.0, tol=1e-9)
        dt = time.perf_counter() - t0
        viol, _ = kkt_violation(K, yv, a, b, 1.0)
        iters.append(it)
        kevals.append(n * n + 2 * n * it)      # Gram build + two rows per SMO step
        print(f"    n = {n:5d}   SMO iters = {it:6d}   kernel evals = {kevals[-1]:11.3e}"
              f"   #SV = {int((a>1e-8).sum()):4d}   KKT viol = {viol:.1e}   ({dt*1e3:6.1f} ms)")
        assert viol < 1e-5
    p_it = np.polyfit(np.log(ns), np.log(iters), 1)[0]
    p_ke = np.polyfit(np.log(ns), np.log(kevals), 1)[0]
    print(f"    fitted:  SMO iterations ~ n^{p_it:.2f},  kernel evaluations ~ n^{p_ke:.2f}")
    print("    (LIBSVM's empirical range is n^2 to n^3; the Gram matrix alone is n^2)")
    assert 0.8 < p_it < 2.6 and 1.6 < p_ke < 3.2
    # (b) deterministic crossover: (2/3)n^3 + 2 n^2 d flops  vs  6 N n E flops
    print(f"    Gram matrix at n = 1e6: {8*1e12/2**40:.1f} TiB;  exact solve there:"
          f" {2/3*1e18/1e18:.2f} EFLOP")
    print("    crossover n* where exact kernel training costs the same as SGD on N params:")
    for N, E, d in ((1e5, 100, 100), (1e6, 100, 100), (1e8, 10, 100), (7e9, 1, 100)):
        # (2/3)n^2 + 2 d n - 6 N E = 0
        nstar = (-2 * d + np.sqrt(4 * d * d + 4 * (2 / 3) * 6 * N * E)) / (2 * (2 / 3))
        k = (2 / 3) * nstar ** 3 + 2 * nstar ** 2 * d
        para = 6 * N * nstar * E
        assert abs(k - para) / para < 1e-6           # the root really equalises the two
        print(f"      N = {N:8.0e}, E = {E:3d} epochs  ->  n* = {nstar:10.3e}"
              f"   (Gram there = {8*nstar**2/2**30:8.2f} GiB)")

def demo_ntk():
    print("[4] Empirical NTK -> analytic arc-cosine kernel")
    rng = np.random.default_rng(0)
    n, d, R = 16, 8, 8
    X = rng.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)         # unit sphere: the limit above assumes it
    Kinf = ntk_analytic(X)
    if torch is not None:                                 # cross-check the closed-form Jacobian
        W = rng.standard_normal((50, d)); av = rng.standard_normal(50)
        Wt = torch.tensor(W, requires_grad=True); at = torch.tensor(av, requires_grad=True)
        Xt = torch.tensor(X)
        rows = []
        for k in range(n):
            out = (torch.relu(Xt[k] @ Wt.T) @ at) / np.sqrt(50.0)
            gW, ga = torch.autograd.grad(out, [Wt, at], retain_graph=False)
            rows.append(np.concatenate([gW.numpy().ravel(), ga.numpy()]))
        err = np.abs(np.array(rows) - jacobian_two_layer(X, W, av)).max()
        print(f"    torch autograd vs closed-form Jacobian (m=50): max abs diff = {err:.3e}")
        assert err < 1e-12
    else:
        print("    [skipped: torch not installed]")
    ms, errs = (10, 100, 1000, 10000), []
    scale = np.linalg.norm(Kinf)
    for m in ms:
        e = [np.linalg.norm(ntk_empirical(X, m, rng) - Kinf) / scale for _ in range(R)]
        errs.append(float(np.mean(e)))
        print(f"    m = {m:6d}   mean rel. Frobenius error = {errs[-1]:.5f}"
              f"   (sd over {R} seeds = {np.std(e):.5f})")
    p = np.polyfit(np.log(ms), np.log(errs), 1)[0]
    ratio = errs[0] / errs[-1]
    print(f"    fitted rate  err ~ m^p:  p = {p:.3f}   (O(m^-1/2) predicts -0.500)")
    print(f"    error ratio m=10 -> m=10000: {ratio:.1f}x   (sqrt(1000) = {np.sqrt(1000):.1f}x)")
    print(f"    diag: K_inf(x,x) = {Kinf[0,0]:.6f}   (exact 1/2 + 1/2 = 1.0)")
    assert -0.65 < p < -0.35
    assert abs(Kinf[0, 0] - 1.0) < 1e-7

if __name__ == "__main__":
    t0 = time.perf_counter()
    demo_smo_tiny(); print()
    demo_smo_vs_sklearn(); print()
    demo_cubic_wall(); print()
    demo_ntk()
    print(f"\nall assertions passed in {time.perf_counter()-t0:.1f} s")
