"""Artifact 4.1 -- Convergence rates of GD, heavy ball, and Nesterov on quadratics.

Four experiments, all pure NumPy (torch is an optional cross-check only):

  (A) GD on f(w) = 1/2 w^T A w is exactly w_t = (I - eta A)^t w_0; verified to
      machine precision, so later disagreement with theory indicts the theory.
  (B) Measured vs predicted iteration counts for kappa in {10, 100, 10000}.  The
      prediction is not the loose O(kappa) / O(sqrt(kappa)) headline but the exact
      contraction factor rho -- the largest spectral radius over the per-eigenvalue
      2x2 companion matrices.  The headline scaling is recovered separately by a
      log-log fit of measured iterations against kappa.
  (C) Edge of stability: a quadratic at eta = 2.05/L, which must and does diverge
      with sign-alternating iterates; then log cosh at eta = 2.5 > 2/L(w*) = 2,
      which does NOT diverge but settles into a period-2 orbit whose *secant*
      curvature is exactly 2/eta, while a flat coordinate keeps descending.
  (D) SGD noise floor: measured against the closed form eta sigma_B^2/(lam(2-eta lam)),
      and shown to depend on eta/B rather than on B.
"""

import math
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - artifact must run anywhere
    torch = None

MU = 1.0          # smallest Hessian eigenvalue, fixed across all condition numbers
D = 50            # problem dimension (diagonal Hessian, so a step costs O(D) flops)
TOL = 1e-8        # target relative distance ||w_t - w*|| / ||w_0 - w*||
MAX_IT = 400_000  # hard cap so a diverging run cannot hang the artifact

# Problem construction
def spectrum(kappa, d=D, mu=MU):
    """Log-spaced eigenvalues of A filling [mu, kappa*mu], endpoints exact."""
    lam = mu * np.logspace(0.0, math.log10(kappa), d)
    lam[0], lam[-1] = mu, mu * kappa
    return lam

def start_point(lam, rng):
    """Unit-norm start with mass on every eigendirection (no accidental head start)."""
    w0 = rng.standard_normal(lam.size)
    return w0 / np.linalg.norm(w0)

# The three optimizers.  Each returns (iterations, diverged_flag).
# Iteration = one gradient evaluation, which is the honest unit of cost.
def run_gd(lam, w0, eta, tol=TOL, max_it=MAX_IT):
    w, n0 = w0.copy(), np.linalg.norm(w0)
    for t in range(1, max_it + 1):
        w = w - eta * (lam * w)                      # grad f(w) = lam * w
        n = np.linalg.norm(w)
        if not np.isfinite(n) or n > 1e12 * n0:
            return t, True
        if n <= tol * n0:
            return t, False
    return max_it, True

def run_heavy_ball(lam, w0, eta, beta, tol=TOL, max_it=MAX_IT):
    """w_{t+1} = w_t - eta g(w_t) + beta (w_t - w_{t-1}); w_{-1} = w_0."""
    n0 = np.linalg.norm(w0)
    w_prev, w = w0.copy(), w0 - eta * (lam * w0)
    for t in range(2, max_it + 1):
        w, w_prev = w - eta * (lam * w) + beta * (w - w_prev), w
        n = np.linalg.norm(w)
        if not np.isfinite(n) or n > 1e12 * n0:
            return t, True
        if n <= tol * n0:
            return t, False
    return max_it, True

def run_nesterov(lam, w0, eta, beta, tol=TOL, max_it=MAX_IT):
    """y_t = x_t + beta (x_t - x_{t-1});  x_{t+1} = y_t - eta g(y_t)."""
    n0 = np.linalg.norm(w0)
    x_prev, x = w0.copy(), w0.copy()
    for t in range(1, max_it + 1):
        y = x + beta * (x - x_prev)
        x, x_prev = y - eta * (lam * y), x
        n = np.linalg.norm(x)
        if not np.isfinite(n) or n > 1e12 * n0:
            return t, True
        if n <= tol * n0:
            return t, False
    return max_it, True

# Exact asymptotic rates: spectral radius of the per-eigenvalue iteration matrix
def rho_gd(lam, eta):
    return float(np.max(np.abs(1.0 - eta * lam)))

def rho_two_step(lam, eta, beta, nesterov):
    """max_i spectral-radius of the 2x2 companion matrix for eigenvalue lam_i."""
    best = 0.0
    for l in lam:
        s = 1.0 - eta * l
        a, b = ((1.0 + beta) * s, -beta * s) if nesterov else (1.0 + beta - eta * l, -beta)
        best = max(best, float(np.max(np.abs(np.linalg.eigvals(np.array([[a, b], [1.0, 0.0]]))))))
    return best

def predict(rho, tol=TOL):
    """Iterations for a contraction factor rho to shrink the error by tol."""
    return math.inf if rho >= 1.0 else math.ceil(math.log(tol) / math.log(rho))

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # ---- (A) closed form -------------------------------------------------- #
    lam = spectrum(100.0)
    w0 = start_point(lam, rng)
    eta_a, steps = 2.0 / (MU + 100.0 * MU), 300
    w = w0.copy()
    for _ in range(steps):
        w = w - eta_a * (lam * w)
    closed = ((1.0 - eta_a * lam) ** steps) * w0
    res = float(np.max(np.abs(w - closed)) / np.max(np.abs(closed)))
    print(f"(A) GD loop vs closed form (I-etaA)^t w0, kappa=100, t={steps}: "
          f"max rel. residual {res:.3e}")
    assert res < 1e-12, res

    # ---- (B) measured vs predicted ---------------------------------------- #
    print("\n(B) iterations to reach ||w_t||/||w_0|| <= 1e-8   [tolerance: "
          "measured/predicted in [0.55, 1.60]]")
    print(f"{'kappa':>7} {'method':>10} {'eta':>10} {'beta':>8} {'rho(exact)':>12}"
          f" {'predicted':>10} {'measured':>10} {'ratio':>7}")
    measured = {"GD": {}, "heavy-ball": {}, "Nesterov": {}}
    for kappa in (10.0, 100.0, 10000.0):
        lam = spectrum(kappa)
        L, sk = MU * kappa, math.sqrt(kappa)
        w0 = start_point(lam, rng)
        cfg = [
            ("GD", 2.0 / (MU + L), 0.0, rho_gd(lam, 2.0 / (MU + L)), run_gd),
            ("heavy-ball", 4.0 / (math.sqrt(L) + math.sqrt(MU)) ** 2,
             ((sk - 1.0) / (sk + 1.0)) ** 2, None, run_heavy_ball),
            ("Nesterov", 1.0 / L, (sk - 1.0) / (sk + 1.0), None, run_nesterov),
        ]
        for name, eta, beta, rho, fn in cfg:
            if rho is None:
                rho = rho_two_step(lam, eta, beta, nesterov=(name == "Nesterov"))
            pred = predict(rho)
            it, div = (fn(lam, w0, eta) if name == "GD" else fn(lam, w0, eta, beta))
            assert not div, f"{name} diverged at kappa={kappa}"
            ratio = it / pred
            measured[name][kappa] = it
            print(f"{kappa:>7.0f} {name:>10} {eta:>10.3e} {beta:>8.5f} {rho:>12.8f}"
                  f" {pred:>10d} {it:>10d} {ratio:>7.3f}")
            assert 0.55 <= ratio <= 1.60, (name, kappa, ratio)

    # log-log scaling exponent over the two largest condition numbers
    print("\n    empirical scaling exponent p in  iterations ~ kappa^p "
          "(fit on kappa = 100 -> 10000)")
    for name, target in (("GD", 1.0), ("heavy-ball", 0.5), ("Nesterov", 0.5)):
        p = math.log(measured[name][10000.0] / measured[name][100.0]) / math.log(100.0)
        print(f"      {name:>10}: p = {p:.4f}   (theory {target})")
        assert abs(p - target) < 0.05, (name, p)
    speedup = measured["GD"][10000.0] / measured["Nesterov"][10000.0]
    print(f"      GD/Nesterov iteration ratio at kappa=1e4: {speedup:.1f}x "
          f"(sqrt(kappa)/2 = {math.sqrt(10000.0)/2:.0f}x)")
    assert speedup > 20.0

    # heavy-ball cross-check: torch SGD(momentum=beta) is exactly this recursion
    lam = spectrum(100.0)
    w0 = start_point(lam, rng)
    eta, beta = 4.0 / (math.sqrt(100.0) + 1.0) ** 2, ((10.0 - 1.0) / (10.0 + 1.0)) ** 2
    NSTEP = 60
    w_prev, w_np = w0.copy(), w0 - eta * (lam * w0)
    for _ in range(NSTEP - 1):
        w_np, w_prev = w_np - eta * (lam * w_np) + beta * (w_np - w_prev), w_np
    if torch is not None:
        p = torch.tensor(w0, requires_grad=True)
        opt = torch.optim.SGD([p], lr=eta, momentum=beta)
        lt = torch.tensor(lam)
        for _ in range(NSTEP):
            opt.zero_grad()
            (0.5 * (lt * p * p).sum()).backward()
            opt.step()
        gap = float(np.max(np.abs(p.detach().numpy() - w_np)))
        rel = gap / float(np.max(np.abs(w_np)))
        print(f"\n    heavy-ball cross-check vs torch.optim.SGD(momentum=beta), "
              f"{NSTEP} steps: ||w||={np.linalg.norm(w_np):.3e}, max |diff| = "
              f"{gap:.3e} (relative {rel:.2e})")
        assert rel < 1e-12, rel
    else:
        print(f"\n    [skipped: torch not installed] numpy heavy-ball ||w_200|| = "
              f"{np.linalg.norm(w_np):.3e}")

    # ---- (C) edge of stability -------------------------------------------- #
    print("\n(C) edge of stability")
    lam = spectrum(100.0)
    L = MU * 100.0
    eta_big = 2.05 / L
    w = start_point(lam, rng)
    hist = [float(w[-1])]
    for _ in range(60):
        w = w - eta_big * (lam * w)
        hist.append(float(w[-1]))
    growth = abs(hist[-1] / hist[-2])
    signs = sum(hist[i] * hist[i + 1] < 0 for i in range(len(hist) - 1))
    print(f"    quadratic, eta = 2.05/L: sharp-coordinate growth factor per step "
          f"{growth:.6f} (predicted |1-eta*L| = {abs(1-eta_big*L):.6f}), "
          f"sign flips {signs}/{len(hist)-1}")
    assert abs(growth - abs(1 - eta_big * L)) < 1e-9
    assert signs == len(hist) - 1                    # textbook period-2 divergence

    # non-quadratic sharp direction: f1(x) = log cosh(x), f1''(0) = 1 = L, but
    # curvature DECAYS away from 0, so eta > 2/L need not diverge.
    eta_eos, mu_flat = 2.5, 0.02
    x, y = 0.01, 1.0
    for _ in range(4000):
        x = x - eta_eos * math.tanh(x)
        y = y - eta_eos * mu_flat * y
    x_orbit = abs(x)
    hess = 1.0 / math.cosh(x_orbit) ** 2             # pointwise sharpness at the orbit
    secant = math.tanh(x_orbit) / x_orbit            # curvature the update actually sees
    print(f"    log cosh, eta = 2.5 > 2/L = 2.0: bounded period-2 orbit at "
          f"|x| = {x_orbit:.6f}")
    print(f"      pointwise sharpness f''(x) = {hess:.6f}  <  2/eta = "
          f"{2/eta_eos:.6f}  <  f''(0) = 1.000000")
    print(f"      secant curvature (f'(x)-f'(-x))/(2x) = {secant:.10f} vs 2/eta = "
          f"{2/eta_eos:.10f}, |diff| = {abs(secant - 2/eta_eos):.2e}")
    print(f"      meanwhile the flat coordinate keeps descending: |y| = {abs(y):.3e}")
    assert abs(secant - 2.0 / eta_eos) < 1e-9        # the orbit sits exactly at 2/eta
    assert hess < 2.0 / eta_eos < 1.0                # local Hessian is NOT 2/eta
    assert abs(y) < 1e-8                             # unstable and training anyway

    # ---- (D) SGD noise floor ---------------------------------------------- #
    print("\n(D) SGD stationary noise floor, kappa = 100, sigma = 1")
    lam = spectrum(100.0, d=20)
    print(f"{'eta':>10} {'B':>6} {'eta/B':>10} {'measured E||w||^2':>19}"
          f" {'closed form':>13} {'rel.err':>9}")
    floors = []
    for eta, B in ((0.004, 1), (0.008, 2), (0.002, 1)):
        sig2 = 1.0 / B                               # minibatch gradient noise variance
        w = np.zeros(lam.size)
        acc, n = 0.0, 0
        for t in range(60_000):
            g = lam * w + math.sqrt(sig2) * rng.standard_normal(lam.size)
            w = w - eta * g
            if t >= 20_000:                          # discard burn-in
                acc += float(w @ w)
                n += 1
        meas = acc / n
        exact = float(np.sum(eta * sig2 / (lam * (2.0 - eta * lam))))
        err = abs(meas - exact) / exact
        floors.append(meas)
        print(f"{eta:>10.4f} {B:>6d} {eta/B:>10.4f} {meas:>19.6e} {exact:>13.6e}"
              f" {err:>8.2%}")
        assert err < 0.06, (eta, B, err)
    rel = abs(floors[0] - floors[1]) / floors[0]
    print(f"    (eta,B) = (0.004,1) and (0.008,2) share eta/B and agree to "
          f"{rel:.2%}; halving eta at fixed B cuts the floor "
          f"{floors[0]/floors[2]:.2f}x")
    assert rel < 0.05 and floors[0] / floors[2] > 1.8

    print("\nall assertions passed")
