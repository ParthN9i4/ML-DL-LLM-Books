"""
Artifact 6.1 -- Logistic regression from scratch: IRLS (Newton) and gradient descent.

Self-verifying. Everything numerical is checked against an independent computation:
  (1) analytic gradient  vs central finite differences of the objective
  (2) analytic Hessian   vs central finite differences of the analytic gradient
  (3) IRLS and GD solutions vs sklearn.linear_model.LogisticRegression, regularization matched
  (4) separable data: ||w|| grows without bound while w/||w|| -> the max-margin direction
  (5) calibration: on a well-specified model, binned empirical frequency ~ mean predicted prob

Core math is pure NumPy. sklearn and scipy are cross-checks only and are import-guarded.
"""

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- core model
def sigmoid(z):
    """Numerically stable logistic sigmoid: no exp() of a large positive number."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])                      # z < 0, so exp(z) in (0, 1)
    out[~pos] = ez / (1.0 + ez)
    return out


def nll(w, X, y, lam=0.0):
    """Penalized negative log-likelihood, summed (not averaged) over the n rows.

    L(w) = sum_i [ log(1 + exp(z_i)) - y_i z_i ] + (lam/2)||w||^2,  z = Xw.
    logaddexp(0, z) is log(1+e^z) computed without overflow.
    """
    z = X @ w
    return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * w @ w)


def grad(w, X, y, lam=0.0):
    """X^T (sigma(Xw) - y) + lam*w  -- the residual form used by every classifier head."""
    return X.T @ (sigmoid(X @ w) - y) + lam * w


def hess(w, X, y, lam=0.0):
    """X^T diag(s(1-s)) X + lam*I, with s = sigma(Xw). Independent of y."""
    s = sigmoid(X @ w)
    return (X * (s * (1.0 - s))[:, None]).T @ X + lam * np.eye(X.shape[1])


# ---------------------------------------------------------------- optimizers
def fit_irls(X, y, lam=1e-8, tol=1e-12, max_iter=100):
    """Newton / IRLS. Each step solves (X^T S X + lam I) delta = grad, a weighted least squares.

    A ridge of lam keeps the Hessian invertible when s(1-s) underflows on confident points.
    """
    w = np.zeros(X.shape[1])
    for it in range(1, max_iter + 1):
        g = grad(w, X, y, lam)
        H = hess(w, X, y, lam)
        delta = np.linalg.solve(H, g)
        # Newton decrement squared: g^T H^-1 g, a bound on 2*(L(w) - L*) near the optimum.
        decrement = float(g @ delta)
        w = w - delta
        if decrement < tol:
            return w, it, decrement
    return w, max_iter, decrement


def fit_gd(X, y, lam=0.0, eta=None, n_steps=20000, track=None):
    """Plain full-batch gradient descent. eta defaults to 1/L with L the exact smoothness
    constant of the logistic loss: L = (1/4) * lambda_max(X^T X) + lam."""
    n, d = X.shape
    if eta is None:
        eta = 1.0 / (0.25 * np.linalg.eigvalsh(X.T @ X)[-1] + lam)
    w = np.zeros(d)
    log = []
    for t in range(1, n_steps + 1):
        w -= eta * grad(w, X, y, lam)
        if track is not None and t in track:
            log.append((t, w.copy()))
    return w, log


# ---------------------------------------------------------------- experiments
def check_derivatives():
    """Central differences: f'(x) ~ (f(x+h)-f(x-h))/2h, error O(h^2) + O(eps/h)."""
    n, d, lam = 200, 5, 0.7
    X = RNG.normal(size=(n, d))
    y = (RNG.random(n) < sigmoid(X @ RNG.normal(size=d))).astype(float)
    w = RNG.normal(size=d) * 0.5
    h = 1e-5

    g_fd = np.zeros(d)
    H_fd = np.zeros((d, d))
    for j in range(d):
        e = np.zeros(d)
        e[j] = h
        g_fd[j] = (nll(w + e, X, y, lam) - nll(w - e, X, y, lam)) / (2 * h)
        H_fd[:, j] = (grad(w + e, X, y, lam) - grad(w - e, X, y, lam)) / (2 * h)

    g_err = np.max(np.abs(grad(w, X, y, lam) - g_fd))
    H_err = np.max(np.abs(hess(w, X, y, lam) - H_fd))
    print(f"  max |analytic grad - central FD|    = {g_err:.3e}")
    print(f"  max |analytic Hess - central FD|    = {H_err:.3e}")
    print(f"  Hessian symmetry ||H - H^T||_max    = {np.max(np.abs(hess(w, X, y, lam) - hess(w, X, y, lam).T)):.3e}")
    print(f"  min eigenvalue of Hessian           = {np.linalg.eigvalsh(hess(w, X, y, lam))[0]:.6f}  (>0 => strictly convex)")
    assert g_err < 1e-6 and H_err < 1e-6


def check_against_sklearn():
    """sklearn minimizes 0.5||w||^2 + C * sum_i loss_i. Dividing by C, that is our objective
    with lam = 1/C. So C = 1/lam is the matched setting (fit_intercept=False; sklearn does
    not penalize an intercept, our objective would)."""
    n, d, lam = 500, 6, 2.0
    X = RNG.normal(size=(n, d))
    w_true = RNG.normal(size=d)
    y = (RNG.random(n) < sigmoid(X @ w_true)).astype(float)

    w_n, iters, dec = fit_irls(X, y, lam)
    w_g, _ = fit_gd(X, y, lam, n_steps=60000)
    print(f"  IRLS converged in {iters} Newton steps, final Newton decrement^2 = {dec:.3e}")
    print(f"  ||grad at IRLS solution||_2         = {np.linalg.norm(grad(w_n, X, y, lam)):.3e}")
    print(f"  ||w_IRLS - w_GD||_inf               = {np.max(np.abs(w_n - w_g)):.3e}")

    if LogisticRegression is None:
        print("  [skipped: sklearn not installed]")
        return
    sk = LogisticRegression(C=1.0 / lam, fit_intercept=False, tol=1e-10, max_iter=5000)
    sk.fit(X, y)
    w_sk = sk.coef_.ravel()
    err = np.max(np.abs(w_n - w_sk))
    print(f"  ||w_IRLS - w_sklearn||_inf          = {err:.3e}")
    print(f"  objective ours {nll(w_n, X, y, lam):.9f}  vs sklearn {nll(w_sk, X, y, lam):.9f}")
    assert err < 1e-5


def separable_run():
    """Separable through the origin: the minimizer is at infinity, but the *direction*
    converges to the hard-margin SVM solution (Soudry et al., 2018)."""
    rng = np.random.default_rng(0)            # own stream, so this run is reproducible alone
    n, d = 40, 2
    w_star = np.array([1.0, -0.6])
    scale = np.array([3.0, 1.0])              # anisotropy: the centroid-difference direction
    Xs = []                                   # (which GD chases first) is NOT the max-margin one
    while len(Xs) < n:                        # reject points inside the margin band
        x = rng.normal(size=d) * scale
        if abs(x @ w_star) > 0.35:
            Xs.append(x)
    X = np.array(Xs)
    s = np.sign(X @ w_star)                   # labels in {-1,+1}
    y = (s > 0).astype(float)                 # labels in {0,1} for the logistic fit

    # Hard-margin SVM (no intercept): min 0.5||u||^2 s.t. s_i x_i^T u >= 1.
    if minimize is None:
        print("  [skipped: scipy not installed]")
        return
    con = {"type": "ineq", "fun": lambda u: s * (X @ u) - 1.0}
    res = minimize(lambda u: 0.5 * u @ u, np.zeros(d), jac=lambda u: u,
                   constraints=[con], method="SLSQP", options={"maxiter": 500, "ftol": 1e-14})
    u_mm = res.x / np.linalg.norm(res.x)
    margin = float(np.min(s * (X @ u_mm)))
    n_sv = int(np.sum(s * (X @ res.x) < 1.0 + 1e-6))
    print(f"  max-margin direction {np.round(u_mm, 6)}, margin = {margin:.6f}, {n_sv} support vectors")

    ckpts = [10**k for k in range(3, 7)]
    _, log = fit_gd(X, y, lam=0.0, n_steps=ckpts[-1], track=set(ckpts))
    print("       step        ||w||     loss        cos(w, w_mm)     1 - cos")
    for t, w in log:
        c = float(w @ u_mm / np.linalg.norm(w))
        print(f"   {t:>9d}   {np.linalg.norm(w):8.3f}  {nll(w, X, y):.3e}   {c:.9f}   {1-c:.3e}")
    norms = [np.linalg.norm(w) for _, w in log]
    coss = [float(w @ u_mm / np.linalg.norm(w)) for _, w in log]
    assert all(np.diff(norms) > 0), "weight norm must grow monotonically"
    assert all(np.diff(coss) > 0) and coss[-1] > 0.999, "direction must converge to max margin"
    print(f"  ||w|| grew {norms[0]:.3f} -> {norms[-1]:.3f}; angle gap fell {1-coss[0]:.2e} -> {1-coss[-1]:.2e}")
    # Theory (Soudry et al.): w(t) ~ (u_mm/margin) * log t, so d||w||/d log t -> 1/margin.
    slope = (norms[-1] - norms[-2]) / np.log(ckpts[-1] / ckpts[-2])
    print(f"  measured d||w||/d log t = {slope:.3f};  predicted 1/margin = {1/margin:.3f}")


def calibration():
    """Well-specified logistic model => the fitted probabilities should be calibrated.
    Bin the predictions into deciles and compare mean predicted prob vs empirical frequency."""
    n, d = 20000, 4
    X = RNG.normal(size=(n, d))
    w_true = RNG.normal(size=d)
    y = (RNG.random(n) < sigmoid(X @ w_true)).astype(float)
    w_hat, _, _ = fit_irls(X, y, lam=1e-8)

    p = sigmoid(X @ w_hat)
    edges = np.quantile(p, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    print("    bin     count   mean p_hat   empirical      gap     gap/SE")
    ece, worst_z = 0.0, 0.0
    for b in range(10):
        m = idx == b
        cnt, pm, fm = int(m.sum()), float(p[m].mean()), float(y[m].mean())
        se = np.sqrt(max(pm * (1 - pm), 1e-12) / cnt)
        z = abs(fm - pm) / se
        ece += cnt / n * abs(fm - pm)
        worst_z = max(worst_z, z)
        print(f"     {b:>2d}   {cnt:>6d}    {pm:9.5f}   {fm:9.5f}  {fm-pm:+8.5f}   {z:6.2f}")
    print(f"  ECE (10 equal-mass bins) = {ece:.5f};  worst bin |gap|/SE = {worst_z:.2f}")

    # A deliberately miscalibrated model: shrink the logits (over-regularized => under-confident).
    p_bad = sigmoid(0.4 * (X @ w_hat))
    idxb = np.digitize(p_bad, np.quantile(p_bad, np.linspace(0, 1, 11))[1:-1])
    ece_bad = sum(np.mean(idxb == b) * abs(y[idxb == b].mean() - p_bad[idxb == b].mean()) for b in range(10))
    print(f"  ECE of the same model with logits scaled by 0.4 = {ece_bad:.5f}  ({ece_bad/ece:.1f}x worse)")
    assert ece < 0.01 and worst_z < 4.0 and ece_bad > 10 * ece


if __name__ == "__main__":
    print("=" * 78)
    print("[1] gradient and Hessian vs central finite differences")
    check_derivatives()
    print("\n[2] IRLS vs gradient descent vs sklearn (matched regularization lam = 1/C)")
    check_against_sklearn()
    print("\n[3] separable data: norm diverges, direction converges to max margin")
    separable_run()
    print("\n[4] calibration of a well-specified logistic model (n = 20000)")
    calibration()
    print("\nall assertions passed")
    print("=" * 78)
