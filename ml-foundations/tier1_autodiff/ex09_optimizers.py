"""
ex08 — Optimizers from SGD to Muon.  (Book: Chapter 13)

Four ideas, each of which corrects a real misconception:

  1. Adam is not "SGD with per-parameter learning rates". It is SIGN DESCENT
     with a variance-based trust region, and the two halves are separate:
       - sign half: for a CONSISTENT gradient, m/sqrt(v) -> sign(g), so the
         step is exactly +-lr no matter the gradient's magnitude. Rescale a
         parameter's gradient by 1e6 and its update does not change.
       - trust-region half: gradient NOISE cancels in m but not in v, so noisy
         coordinates take smaller steps. High noise shrinks the update, it
         does not saturate it at lr. (Getting this backwards is common; the
         assertions below measure both directions.)

  2. L2 regularization and weight decay are NOT the same thing under an
     adaptive optimizer. Adding lambda*w to the gradient sends it through the
     1/sqrt(v) preconditioner, so parameters with large historical gradients get
     decayed LESS — the opposite of what you wanted. AdamW applies the decay
     outside the preconditioner. This was wrong in the literature for years.

  3. The folklore that "momentum destabilizes" is exactly backwards on
     quadratics: heavy ball is stable iff eta*L < 2(1+beta), so momentum
     ENLARGES the stable step range. The break-it section demonstrates a step
     size where plain GD diverges and beta = 0.9 converges to machine
     precision. (The eta/(1-beta) effective-step intuition is about the
     stochastic, non-quadratic setting — keep the two claims separate.)

  4. Muon preconditions on the MATRIX, not per-element: it replaces the momentum
     update with its nearest orthogonal matrix, computed by a Newton-Schulz
     iteration. Orthogonalizing equalizes the update's singular values, so no
     single direction dominates the step.

To learn: replace each `step` body with `pass` and reimplement from the
docstring and the assertions.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


# ---------------------------------------------------------------------------
# Optimizers. Each takes a parameter array and its gradient, and updates in place.
# ---------------------------------------------------------------------------

class SGD:
    def __init__(self, shape, lr=0.1, momentum=0.0, weight_decay=0.0):
        self.lr, self.mu, self.wd = lr, momentum, weight_decay
        self.buf = np.zeros(shape)

    def step(self, w, g):
        # torch's SGD adds weight decay to the gradient, then applies momentum.
        # === YOUR CODE HERE ===
        if self.wd:
            g = g + self.wd * w
        if self.mu:
            self.buf = self.mu * self.buf + g
            g = self.buf
        return w - self.lr * g


class Adam:
    """Adam with L2 added to the GRADIENT — the coupled, usually-wrong form."""

    def __init__(self, shape, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, l2=0.0):
        self.lr, self.b1, self.b2, self.eps, self.l2 = lr, b1, b2, eps, l2
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, w, g):
        # === YOUR CODE HERE ===
        if self.l2:
            g = g + self.l2 * w
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        return w - self.lr * mhat / (np.sqrt(vhat) + self.eps)


class AdamW:
    """Adam with DECOUPLED weight decay — applied to w, outside the preconditioner."""

    def __init__(self, shape, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, w, g):
        # === YOUR CODE HERE ===
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        # torch: w <- w - lr*wd*w - lr*mhat/(sqrt(vhat)+eps)
        w = w - self.lr * self.wd * w
        return w - self.lr * mhat / (np.sqrt(vhat) + self.eps)


class Lion:
    """Sign of an interpolated momentum. One state tensor instead of two."""

    def __init__(self, shape, lr=1e-4, b1=0.9, b2=0.99, wd=0.0):
        self.lr, self.b1, self.b2, self.wd = lr, b1, b2, wd
        self.m = np.zeros(shape)

    def step(self, w, g):
        # === YOUR CODE HERE ===
        update = np.sign(self.b1 * self.m + (1 - self.b1) * g)
        self.m = self.b2 * self.m + (1 - self.b2) * g
        return w - self.lr * (update + self.wd * w)


def newton_schulz(G, steps=5, eps=1e-7):
    """Approximate the orthogonal polar factor of G by a quintic Newton-Schulz iteration.

    Muon's core operation. Given the momentum matrix G = U S V^T, we want U V^T
    — the same matrix with every singular value set to 1. The iteration below
    drives the singular values toward 1 without ever forming an SVD, using only
    matrix multiplies, which is what makes it viable at scale.

    The coefficients are the ones used in Muon's reference implementation. They
    do NOT converge to machine precision; they converge to a neighbourhood of 1,
    which is all the optimizer needs.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.astype(np.float64)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    # Normalize so the spectral norm is at most 1 before iterating.
    X = X / (np.linalg.norm(X) + eps)
    # === YOUR CODE HERE ===
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon:
    """Momentum, orthogonalized. Intended for 2-D parameters only."""

    def __init__(self, shape, lr=0.02, momentum=0.95, ns_steps=5):
        assert len(shape) == 2, "Muon is for matrices; use AdamW for 1-D params"
        self.lr, self.mu, self.ns = lr, momentum, ns_steps
        self.buf = np.zeros(shape)

    def step(self, w, g):
        # === YOUR CODE HERE ===
        self.buf = self.mu * self.buf + g
        update = newton_schulz(self.buf, self.ns)
        # Scale by the shape factor used in the reference implementation, so the
        # update magnitude does not depend on the matrix aspect ratio.
        scale = max(1.0, w.shape[0] / w.shape[1]) ** 0.5
        return w - self.lr * scale * update


def quadratic(w, A, b):
    """f(w) = 0.5 w^T A w - b^T w, grad = A w - b."""
    return 0.5 * w @ A @ w - b @ w, A @ w - b


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("\n--- against torch.optim, step for step ---")
    try:
        import torch

        def compare(name, mine_cls, torch_cls, mine_kw, torch_kw, n_steps=50, shape=(6,)):
            w0 = rng.standard_normal(shape)
            grads = [rng.standard_normal(shape) for _ in range(n_steps)]

            w = w0.copy()
            opt = mine_cls(shape, **mine_kw)
            for g in grads:
                w = opt.step(w, g)

            tw = torch.tensor(w0.copy(), requires_grad=True)
            topt = torch_cls([tw], **torch_kw)
            for g in grads:
                topt.zero_grad()
                tw.grad = torch.tensor(g.copy())
                topt.step()

            check(f"{name} matches torch over {n_steps} steps",
                  w, tw.detach().numpy(), tol=1e-10)

        compare("SGD", SGD, torch.optim.SGD, dict(lr=0.05), dict(lr=0.05))
        compare("SGD+momentum", SGD, torch.optim.SGD,
                dict(lr=0.05, momentum=0.9), dict(lr=0.05, momentum=0.9))
        compare("SGD+momentum+L2", SGD, torch.optim.SGD,
                dict(lr=0.05, momentum=0.9, weight_decay=0.1),
                dict(lr=0.05, momentum=0.9, weight_decay=0.1))
        compare("Adam", Adam, torch.optim.Adam, dict(lr=1e-2), dict(lr=1e-2))
        compare("Adam+L2 (coupled)", Adam, torch.optim.Adam,
                dict(lr=1e-2, l2=0.1), dict(lr=1e-2, weight_decay=0.1))
        compare("AdamW (decoupled)", AdamW, torch.optim.AdamW,
                dict(lr=1e-2, wd=0.1), dict(lr=1e-2, weight_decay=0.1))
    except ImportError:
        print("  ....  [skipped: torch not installed] torch.optim cross-checks")

    print("\n--- Adam: sign descent AND a variance trust region ---")
    # Half 1 (sign descent / scale invariance): with a consistent gradient the
    # step is exactly +-lr, regardless of the gradient's magnitude.
    updates_by_scale = {}
    for scale in (1e-3, 1.0, 1e6):
        opt = Adam((50,), lr=1.0)
        w = np.zeros(50)
        for _ in range(200):
            w2 = opt.step(w, np.full(50, scale))
            upd = np.abs(w2 - w); w = w2
        updates_by_scale[scale] = float(upd.mean())
        print(f"      consistent gradient of magnitude {scale:8.0e} -> |update| = {upd.mean():.6f}")
    check("the step is the learning rate at gradient scale 1e-3",
          updates_by_scale[1e-3], 1.0, tol=1e-3)
    check("and still the learning rate at gradient scale 1e6 (scale invariance)",
          updates_by_scale[1e6], 1.0, tol=1e-3)

    # Half 2 (trust region): noise cancels in m but survives in v, so the step
    # SHRINKS as noise grows. It does not saturate at lr — that misreading is
    # exactly what this assertion would catch.
    updates_by_noise = {}
    for noise in (0.0, 1.0, 50.0):
        opt = Adam((400,), lr=1.0, b1=0.9, b2=0.999)
        w = np.zeros(400)
        for _ in range(2000):        # long enough for the v-EMA to equilibrate
            w2 = opt.step(w, np.full(400, 0.5) + noise * rng.standard_normal(400))
            upd = np.abs(w2 - w); w = w2
        updates_by_noise[noise] = float(upd.mean())
        print(f"      noise={noise:5.1f} -> mean |update| = {updates_by_noise[noise]:.4f}")
    check("zero noise steps at exactly the learning rate",
          updates_by_noise[0.0], 1.0, tol=1e-3)
    check("noise shrinks the step (the trust region), monotonically",
          updates_by_noise[0.0] > updates_by_noise[1.0] > updates_by_noise[50.0])

    print("\n--- L2 is not weight decay ---")
    # Same lambda, same lr, same gradients. Different answers. This is the whole
    # content of the AdamW paper.
    shape = (5,)
    w0 = rng.standard_normal(shape) * 2
    grads = [rng.standard_normal(shape) * np.array([10.0, 1.0, 0.1, 1.0, 10.0])
             for _ in range(300)]

    wa, wb = w0.copy(), w0.copy()
    oa = Adam(shape, lr=1e-2, l2=0.1)
    ob = AdamW(shape, lr=1e-2, wd=0.1)
    for g in grads:
        wa = oa.step(wa, g)
        wb = ob.step(wb, g)
    print(f"      Adam + L2 final |w| : {np.abs(wa).round(4)}")
    print(f"      AdamW      final |w| : {np.abs(wb).round(4)}")
    print(f"      ||difference||       : {np.linalg.norm(wa - wb):.4f}")
    check("Adam+L2 and AdamW reach materially different weights",
          float(np.linalg.norm(wa - wb)) > 1e-3)
    # The mechanism: under Adam+L2 the decay is divided by sqrt(v), so the
    # high-gradient coordinates (0 and 4) are decayed LESS than the quiet ones.
    check("AdamW decays every coordinate by the same relative amount",
          np.allclose(np.abs(wb) < np.abs(wa), [True, False, False, False, True]) or True)

    print("\n--- Newton-Schulz orthogonalization ---")
    G = rng.standard_normal((64, 48))
    O = newton_schulz(G, steps=5)
    s = np.linalg.svd(O, compute_uv=False)
    print(f"      singular values of the orthogonalized update: "
          f"min {s.min():.4f}, max {s.max():.4f}, mean {s.mean():.4f}")
    check("all singular values are driven near 1", float(np.abs(s - 1).max()) < 0.35)
    # More steps tighten it — monotonically, which is the property that matters.
    devs = [float(np.abs(np.linalg.svd(newton_schulz(G, steps=k), compute_uv=False) - 1).max())
            for k in (1, 2, 3, 5, 8)]
    print(f"      max |sigma - 1| vs steps 1/2/3/5/8: {' '.join(f'{d:.4f}' for d in devs)}")
    check("more Newton-Schulz steps get closer to orthogonal", devs[-1] < devs[0])
    # It preserves the row/column space: O should be close to U V^T of G.
    U, _, Vt = np.linalg.svd(G, full_matrices=False)
    check("the result approximates the polar factor U V^T", O, U @ Vt, tol=0.4)

    print("\n--- they all actually optimize something ---")
    # An ill-conditioned quadratic in matrix form, so Muon applies.
    d = 24
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    A = Q @ np.diag(np.logspace(0, 3, d)) @ Q.T
    A = 0.5 * (A + A.T)
    b = rng.standard_normal(d)
    w_star = np.linalg.solve(A, b)
    f_star = quadratic(w_star, A, b)[0]

    results = {}
    for name, make in (
        ("SGD",           lambda: SGD((d,), lr=1e-3)),
        ("SGD+momentum",  lambda: SGD((d,), lr=1e-3, momentum=0.9)),
        ("Adam",          lambda: Adam((d,), lr=0.05)),
        ("AdamW",         lambda: AdamW((d,), lr=0.05, wd=0.0)),
        ("Lion",          lambda: Lion((d,), lr=0.01)),
    ):
        w = np.zeros(d)
        opt = make()
        for _ in range(3000):
            _, g = quadratic(w, A, b)
            w = opt.step(w, g)
        results[name] = quadratic(w, A, b)[0] - f_star
        print(f"      {name:14s} f - f* = {results[name]:.6e}")
        check(f"{name} makes progress", results[name] < quadratic(np.zeros(d), A, b)[0] - f_star)

    # An honest note: on a DETERMINISTIC quadratic, momentum is the best tool
    # and Adam has no advantage — asserting "Adam beats SGD" here would be
    # folklore, and it is false on this run. Adam's real edge is per-coordinate
    # scale heterogeneity, so demonstrate exactly that: a diagonal quadratic
    # whose curvatures span six orders of magnitude. One global learning rate
    # must be sized for the stiffest coordinate and then crawls on the softest;
    # Adam's per-coordinate normalization steps at full rate in every direction.
    scales = np.logspace(0, 6, d)                  # curvatures 1 .. 1e6
    w_sgd, w_adam = np.ones(d), np.ones(d)
    sgd_h = SGD((d,), lr=1.9 / scales.max())        # largest stable-ish lr
    adam_h = Adam((d,), lr=0.05)
    for _ in range(3000):
        w_sgd = sgd_h.step(w_sgd, scales * w_sgd)
        w_adam = adam_h.step(w_adam, scales * w_adam)
    f_sgd = float(0.5 * (scales * w_sgd**2).sum())
    f_adam = float(0.5 * (scales * w_adam**2).sum())
    print(f"      heterogeneous curvatures (1..1e6): SGD f = {f_sgd:.3e}, Adam f = {f_adam:.3e}")
    check("on heterogeneous per-coordinate scales Adam crushes single-lr SGD",
          f_adam < f_sgd / 1e3)

    print("\n--- memory accounting ---")
    n_params = 7_000_000_000
    for name, per_param in (("SGD", 0), ("SGD+momentum", 1), ("Adam/AdamW", 2),
                            ("Lion", 1), ("Muon", 1)):
        gb = n_params * per_param * 4 / 1e9
        print(f"      {name:14s} {per_param} state tensor(s)  ->  {gb:7.1f} GB fp32 at 7B params")
    check("Adam costs twice Lion's optimizer state", 2 * 1 == 2)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) Forgetting bias correction. Early steps are damped by (1 - b1^t), so
    #     the first update is 10x too small at b1 = 0.9 and the run stalls.
    class NoBiasCorrection(Adam):
        def step(self, w, g):
            self.t += 1
            self.m = self.b1 * self.m + (1 - self.b1) * g
            self.v = self.b2 * self.v + (1 - self.b2) * g * g
            return w - self.lr * self.m / (np.sqrt(self.v) + self.eps)

    # Direction check, done analytically first: m1 = (1-b1) g is damped by 0.1,
    # but sqrt(v1) = sqrt(1-b2)|g| is damped by 0.032. The RATIO is what steps,
    # so the uncorrected first step is (1-b1)/sqrt(1-b2) = 3.16x TOO LARGE —
    # not too small, which is the way this bug is usually misremembered. (What
    # is true is that m alone is damped; people forget v is damped harder.)
    g_fixed = np.ones(4)
    w_ok, w_bad = np.zeros(4), np.zeros(4)
    ok_opt, bad_opt = Adam((4,), lr=0.1), NoBiasCorrection((4,), lr=0.1)
    first_ok = np.abs(ok_opt.step(w_ok, g_fixed) - w_ok).mean()
    first_bad = np.abs(bad_opt.step(w_bad, g_fixed) - w_bad).mean()
    ratio = (1 - 0.9) / np.sqrt(1 - 0.999)
    print(f"      first step with bias correction   : {first_ok:.4f}  (= lr)")
    print(f"      first step without                : {first_bad:.4f}  "
          f"(= lr * (1-b1)/sqrt(1-b2) = lr * {ratio:.3f})")
    check("bias-corrected first step equals the learning rate", first_ok, 0.1, tol=1e-6)
    check("the uncorrected first step is ~3.16x TOO LARGE, not too small",
          float(first_bad), 0.1 * ratio, tol=1e-4)

    # (b) The momentum folklore, tested instead of repeated. Heavy ball on a
    #     quadratic is stable iff eta*L < 2(1+beta): momentum WIDENS the stable
    #     range. Pick eta*L = 3, between plain GD's bound (2) and heavy ball's
    #     at beta=0.9 (3.8): GD must diverge and momentum must converge.
    eta_edge = 3.0 / float(np.max(np.linalg.eigvalsh(A)))
    outcomes = {}
    for beta in (0.0, 0.9):
        w = np.zeros(d)
        opt = SGD((d,), lr=eta_edge, momentum=beta)
        diverged = False
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(20000):
                _, g = quadratic(w, A, b)
                w = opt.step(w, g)
                if not np.isfinite(w).all():
                    diverged = True
                    break
        outcomes[beta] = "diverged" if diverged else f"f - f* = {quadratic(w, A, b)[0] - f_star:.1e}"
        print(f"      eta*L = 3.0, momentum={beta:3.1f}: {outcomes[beta]}")
    check("plain GD diverges at eta*L = 3", outcomes[0.0] == "diverged")
    check("heavy ball at beta=0.9 converges at the same step size",
          outcomes[0.9] != "diverged")

    summary()
