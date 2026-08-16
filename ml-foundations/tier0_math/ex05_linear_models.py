"""
ex05 — Linear and logistic regression: the atom.  (Book: Chapter 6)

Every classifier in this book ends in a linear layer followed by a softmax, and
its gradient has the same shape as the one you derive here:

    grad_w = X^T (sigmoid(Xw) - y)          # (prediction - target), pulled back

That is it. The final layer of a 400-billion-parameter transformer computes the
same expression with a bigger X. Learn it once here where you can check it
against a closed form.

Three facts that keep mattering later:

  * Ridge is MAP estimation under a Gaussian prior. The regularizer is not a
    hack; it is a prior, and lambda is its inverse variance.
  * Logistic loss is convex but has NO closed-form minimizer, which is why
    everything after this chapter is iterative.
  * On separable data the weight norm diverges while the DIRECTION converges to
    the max-margin solution. Gradient descent has an implicit bias, and that
    bias is doing work no explicit regularizer was asked for (Chapter 14).

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Least squares and ridge
# ---------------------------------------------------------------------------

def ols(X, y):
    """Ordinary least squares via the normal equations: w = (X^T X)^-1 X^T y.

    Use np.linalg.solve, never an explicit inverse — forming the inverse costs
    more and is less accurate (Chapter 1).
    """
    # === YOUR CODE HERE ===
    return np.linalg.solve(X.T @ X, X.T @ y)


def ridge(X, y, lam):
    """Ridge regression: w = (X^T X + lam*I)^-1 X^T y.

    Equivalently the MAP estimate under w ~ N(0, sigma^2/lam * I). The ridge
    term also makes the system solvable when X^T X is singular, which is why it
    appears in IRLS below.
    """
    d = X.shape[1]
    # === YOUR CODE HERE ===
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


# ---------------------------------------------------------------------------
# 2. Logistic regression
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Numerically stable logistic sigmoid.

    exp(-z) overflows for very negative z, so branch: for z >= 0 use
    1/(1+exp(-z)); for z < 0 use exp(z)/(1+exp(z)). Both are exact and neither
    ever exponentiates a positive number.
    """
    z = np.asarray(z, dtype=np.float64)
    # === YOUR CODE HERE ===
    out = np.empty_like(z)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def logistic_loss(w, X, y, lam=0.0):
    """Mean negative log-likelihood, computed stably via log(1+exp(.)).

    Do NOT write -y*log(p) - (1-y)*log(1-p): p saturates to exactly 0 or 1 and
    the log is -inf. Use the log-sum-exp form:
        loss_i = log(1 + exp(z_i)) - y_i * z_i
    with log(1+exp(z)) evaluated as max(z,0) + log(1+exp(-|z|)).
    """
    z = X @ w
    # === YOUR CODE HERE ===
    softplus = np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))
    return float(np.mean(softplus - y * z) + 0.5 * lam * w @ w)


def logistic_grad(w, X, y, lam=0.0):
    """grad = X^T (sigmoid(Xw) - y) / n  + lam*w.  THE gradient of this book."""
    n = X.shape[0]
    # === YOUR CODE HERE ===
    return X.T @ (sigmoid(X @ w) - y) / n + lam * w


def logistic_hessian(w, X, y, lam=0.0):
    """H = X^T diag(s(1-s)) X / n + lam*I.  Positive semidefinite => convex."""
    n = X.shape[0]
    s = sigmoid(X @ w)
    # === YOUR CODE HERE ===
    return (X * (s * (1 - s))[:, None]).T @ X / n + lam * np.eye(X.shape[1])


def fit_irls(X, y, lam=1e-6, n_steps=100, tol=1e-12):
    """Newton's method (equivalently IRLS). Quadratic convergence near the optimum."""
    w = np.zeros(X.shape[1])
    # === YOUR CODE HERE ===
    for _ in range(n_steps):
        g = logistic_grad(w, X, y, lam)
        H = logistic_hessian(w, X, y, lam)
        step = np.linalg.solve(H, g)
        w = w - step
        if np.linalg.norm(step) < tol:
            break
    return w


def fit_gd(X, y, eta=0.5, n_steps=20000, lam=0.0):
    """Plain gradient descent, for comparison with Newton."""
    w = np.zeros(X.shape[1])
    # === YOUR CODE HERE ===
    for _ in range(n_steps):
        w = w - eta * logistic_grad(w, X, y, lam)
    return w


def finite_diff_grad(f, w, eps=1e-6):
    """Central-difference gradient — the reference every analytic gradient is checked against."""
    g = np.zeros_like(w)
    # === YOUR CODE HERE ===
    for i in range(len(w)):
        wp, wm = w.copy(), w.copy()
        wp[i] += eps
        wm[i] -= eps
        g[i] = (f(wp) - f(wm)) / (2 * eps)
    return g


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("\n--- least squares ---")
    n, d = 300, 8
    X = rng.standard_normal((n, d))
    w_true = rng.standard_normal(d)
    y = X @ w_true + 0.1 * rng.standard_normal(n)

    w_ols = ols(X, y)
    check("OLS recovers the planted weights", w_ols, w_true, tol=0.05)
    # The residual is orthogonal to the column space — the defining property.
    check("residual is orthogonal to every column of X",
          X.T @ (y - X @ w_ols), np.zeros(d), tol=1e-9)
    # Ridge shrinks toward zero, monotonically in lambda.
    norms = [float(np.linalg.norm(ridge(X, y, lam))) for lam in (0.0, 1.0, 10.0, 100.0)]
    print(f"      ||w|| vs lambda 0/1/10/100: {' '.join(f'{v:.3f}' for v in norms)}")
    check("ridge shrinks the norm monotonically",
          all(norms[i] > norms[i + 1] for i in range(len(norms) - 1)))
    check("ridge at lambda=0 equals OLS", ridge(X, y, 0.0), w_ols, tol=1e-8)

    print("\n--- sigmoid stability ---")
    check("sigmoid(0) = 0.5", float(sigmoid(0.0)), 0.5, tol=1e-15)
    check("sigmoid is finite at -1000", bool(np.all(np.isfinite(sigmoid([-1000.0, 1000.0])))))
    check("sigmoid(-1000) underflows to 0, not nan", float(sigmoid(np.array([-1000.0]))[0]), 0.0, tol=1e-300)
    check("sigmoid(z) + sigmoid(-z) = 1",
          sigmoid(np.array([-3.0, 0.0, 7.0])) + sigmoid(np.array([3.0, 0.0, -7.0])),
          np.ones(3), tol=1e-15)

    print("\n--- the gradient ---")
    Xc = np.column_stack([rng.standard_normal((400, 5)), np.ones(400)])
    w_gen = rng.standard_normal(6)
    yc = (rng.random(400) < sigmoid(Xc @ w_gen)).astype(float)
    w0 = rng.standard_normal(6) * 0.3

    g_analytic = logistic_grad(w0, Xc, yc, lam=0.1)
    g_numeric = finite_diff_grad(lambda w: logistic_loss(w, Xc, yc, lam=0.1), w0)
    check("analytic gradient matches central differences", g_analytic, g_numeric, tol=1e-7)

    # The Hessian too — this is what makes Newton's method possible.
    H = logistic_hessian(w0, Xc, yc, lam=0.1)
    H_num = np.column_stack([
        finite_diff_grad(lambda w: logistic_grad(w, Xc, yc, lam=0.1)[i], w0)
        for i in range(6)
    ])
    check("analytic Hessian matches differences of the gradient", H, H_num, tol=1e-6)
    check("Hessian is symmetric", H, H.T, tol=1e-12)
    check("Hessian is positive definite (the loss is convex)",
          bool(np.all(np.linalg.eigvalsh(H) > 0)))

    print("\n--- Newton versus gradient descent ---")
    w_newton = fit_irls(Xc, yc, lam=0.1)
    w_gd = fit_gd(Xc, yc, eta=0.5, n_steps=20000, lam=0.1)
    print(f"      IRLS  loss {logistic_loss(w_newton, Xc, yc, 0.1):.10f}  (a handful of steps)")
    print(f"      GD    loss {logistic_loss(w_gd,     Xc, yc, 0.1):.10f}  (20000 steps)")
    check("Newton reaches a stationary point",
          float(np.linalg.norm(logistic_grad(w_newton, Xc, yc, 0.1))) < 1e-10)
    check("gradient descent gets close but not as close",
          logistic_loss(w_gd, Xc, yc, 0.1) >= logistic_loss(w_newton, Xc, yc, 0.1) - 1e-12)

    try:
        from sklearn.linear_model import LogisticRegression
        # sklearn minimizes sum(loss) + 0.5*||w||^2/C, i.e. C = 1/(n*lam) here.
        n_c = Xc.shape[0]
        clf = LogisticRegression(C=1.0 / (n_c * 0.1), fit_intercept=False, tol=1e-10, max_iter=5000)
        clf.fit(Xc, yc)
        check("IRLS matches sklearn LogisticRegression", w_newton, clf.coef_.ravel(), tol=1e-4)
    except ImportError:
        print("  ....  [skipped: scikit-learn not installed] sklearn cross-check")

    print("\n--- implicit bias on separable data ---")
    # Two clearly separated blobs. There is no finite minimizer: the loss keeps
    # decreasing as ||w|| grows, but the DIRECTION converges to max-margin.
    Xs = np.vstack([rng.standard_normal((60, 2)) + [3, 3],
                    rng.standard_normal((60, 2)) - [3, 3]])
    ys = np.concatenate([np.ones(60), np.zeros(60)])

    prev_dir = None
    print(f"      {'steps':>8} {'||w||':>10} {'loss':>12} {'angle change':>14}")
    for steps in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6):
        w = fit_gd(Xs, ys, eta=1.0, n_steps=steps, lam=0.0)
        nrm = float(np.linalg.norm(w))
        direction = w / nrm
        ang = "-" if prev_dir is None else f"{np.arccos(np.clip(prev_dir @ direction, -1, 1)):.2e}"
        print(f"      {steps:8d} {nrm:10.3f} {logistic_loss(w, Xs, ys):12.3e} {ang:>14}")
        if prev_dir is not None:
            check(f"direction has nearly stopped moving by {steps} steps",
                  float(np.arccos(np.clip(prev_dir @ direction, -1, 1))) < 0.2)
        prev_dir = direction

    w_final = fit_gd(Xs, ys, eta=1.0, n_steps=10 ** 6, lam=0.0)
    check("weight norm grows without bound on separable data",
          float(np.linalg.norm(w_final)) > 5.0)
    check("loss keeps falling toward zero", logistic_loss(w_final, Xs, ys) < 1e-3)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) The naive loss. p saturates to exactly 1.0 in float64 at z ~ 37, so
    #     log(1-p) is log(0) = -inf, and the training run reports nan forever.
    z_big = np.array([40.0])
    p = sigmoid(z_big)
    with np.errstate(divide="ignore"):
        naive = float(-(0.0 * np.log(p) + 1.0 * np.log(1.0 - p))[0])
    stable = logistic_loss(np.array([40.0]), np.array([[1.0]]), np.array([0.0]))
    print(f"      sigmoid(40) rounds to {p[0]:.20f}")
    print(f"      naive -log(1-p) = {naive},  stable form = {stable:.6f}")
    check("the naive loss form blows up", np.isinf(naive))
    check("the log1p form stays finite and correct", stable, 40.0, tol=1e-6)

    # (b) Newton without any ridge term. On separable data every fitted
    #     probability saturates, so the weights s(1-s) all go to zero and the
    #     Hessian X^T diag(s(1-s)) X collapses toward the zero matrix. The
    #     mechanism to assert is the CONDITIONING, not the weight norm: how far
    #     ||w|| runs before the step size underflows depends on the iteration
    #     count and the data, so an absolute threshold on it is not a real test.
    try:
        w_bad = fit_irls(Xs, ys, lam=0.0, n_steps=50)
        bad_norm = float(np.linalg.norm(w_bad))
        H_bad = logistic_hessian(w_bad, Xs, ys, lam=0.0)
        H_ok = logistic_hessian(fit_irls(Xs, ys, lam=0.1), Xs, ys, lam=0.1)
        cond_bad = float(np.linalg.cond(H_bad))
        cond_ok = float(np.linalg.cond(H_ok))
        s_bad = sigmoid(Xs @ w_bad)
        print(f"      unregularized: ||w|| = {bad_norm:.3e}, "
              f"max s(1-s) = {float(np.max(s_bad*(1-s_bad))):.3e}, cond(H) = {cond_bad:.3e}")
        print(f"      regularized  : ||w|| = {np.linalg.norm(fit_irls(Xs, ys, lam=0.1)):.3e}, "
              f"cond(H) = {cond_ok:.3e}")
        degenerate = (not np.isfinite(cond_bad)) or cond_bad > 1e4 * cond_ok
    except np.linalg.LinAlgError:
        degenerate = True
        print("      unregularized Newton on separable data -> singular matrix")
    check("without ridge the Hessian conditioning collapses on separable data", degenerate)

    summary()
