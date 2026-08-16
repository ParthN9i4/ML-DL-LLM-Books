"""
ex03 — Optimization: convergence rates you can measure.  (Book: Chapter 4)

Convex theory gives EXACT rates, and the point of this exercise is that you can
watch a real optimizer obey them to two significant figures.  Everything later
in the book — why Adam helps, why warmup exists, why some runs diverge at step
4000 — is a statement about curvature, and curvature is what kappa measures.

On a quadratic f(x) = 0.5 x^T A x with eigenvalues in [mu, L], and kappa = L/mu:

    gradient descent, optimal step 2/(mu+L) :  error ~ ((kappa-1)/(kappa+1))^t
    heavy ball, optimally tuned             :  error ~ ((sqrt(k)-1)/(sqrt(k)+1))^t

So GD needs O(kappa) iterations and momentum needs O(sqrt(kappa)).  At kappa =
10^4 that is the difference between 10^4 and 10^2 steps — two orders of
magnitude, from one change of a single line.

The stability threshold is exact and worth memorizing: plain GD on a quadratic
converges if and only if eta < 2/L.  Above it, the largest-curvature mode
oscillates with growing amplitude.  Real networks are observed to hover just
past this edge and train anyway ("edge of stability"), which is a genuinely
open puzzle rather than a settled result.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


def make_quadratic(kappa, n=50, seed=0):
    """A quadratic with condition number exactly `kappa`.

    Returns (A, mu, L) where f(x) = 0.5 x^T A x, grad f(x) = A x, and the
    eigenvalues are log-spaced in [1, kappa] so mu = 1 and L = kappa.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.logspace(0, np.log10(kappa), n)
    # === YOUR CODE HERE ===
    A = (Q * eigs) @ Q.T
    A = 0.5 * (A + A.T)          # symmetrize away rounding asymmetry
    return A, float(eigs.min()), float(eigs.max())


def gd(A, x0, eta, n_steps, tol=1e-8, history=False):
    """Plain gradient descent. Returns (final x, steps to reach ||x|| < tol[, errs]).

    Steps is n_steps if the tolerance was never reached. The optimum is x = 0,
    so ||x|| IS the error.
    """
    x = x0.copy()
    errs = [float(np.linalg.norm(x))]
    # === YOUR CODE HERE ===
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(1, n_steps + 1):
            x = x - eta * (A @ x)
            if history:
                errs.append(float(np.linalg.norm(x)))
            if not np.isfinite(x).all():
                return (x, n_steps, errs) if history else (x, n_steps)
            if np.linalg.norm(x) < tol:
                return (x, t, errs) if history else (x, t)
    return (x, n_steps, errs) if history else (x, n_steps)


def heavy_ball(A, x0, eta, beta, n_steps, tol=1e-8, history=False):
    """Polyak heavy-ball momentum: x_{t+1} = x_t - eta*grad + beta*(x_t - x_{t-1})."""
    x, x_prev = x0.copy(), x0.copy()
    errs = [float(np.linalg.norm(x))]
    # === YOUR CODE HERE ===
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(1, n_steps + 1):
            x_new = x - eta * (A @ x) + beta * (x - x_prev)
            x_prev, x = x, x_new
            if history:
                errs.append(float(np.linalg.norm(x)))
            if not np.isfinite(x).all():
                return (x, n_steps, errs) if history else (x, n_steps)
            if np.linalg.norm(x) < tol:
                return (x, t, errs) if history else (x, t)
    return (x, n_steps, errs) if history else (x, n_steps)


def asymptotic_rate(errs, drop_front=0.4):
    """Per-step contraction factor fitted to the TAIL of an error trajectory.

    Fit log||x_t|| = log C + t log(rate) by least squares over the last portion
    of the run, discarding the transient. The slope, exponentiated, is the rate.
    """
    e = np.asarray(errs, dtype=np.float64)
    keep = (e > 0) & np.isfinite(e)
    idx = np.arange(len(e))[keep]
    e = e[keep]
    start = int(len(idx) * drop_front)
    idx, e = idx[start:], e[start:]
    if len(idx) < 5:
        return float("nan")
    slope = np.polyfit(idx, np.log(e), 1)[0]
    return float(np.exp(slope))


def optimal_gd_params(mu, L):
    """Step size minimizing the GD contraction factor, and that factor.

    eta* = 2/(mu+L) equalizes the contraction |1 - eta*mu| = |1 - eta*L|,
    giving rate (kappa-1)/(kappa+1).
    """
    # === YOUR CODE HERE ===
    eta = 2.0 / (mu + L)
    rate = (L / mu - 1) / (L / mu + 1)
    return eta, rate


def optimal_hb_params(mu, L):
    """Optimal heavy-ball step and momentum, and the resulting contraction factor."""
    # === YOUR CODE HERE ===
    sk = np.sqrt(L / mu)
    beta = ((sk - 1) / (sk + 1)) ** 2
    eta = (1 + np.sqrt(beta)) ** 2 / L
    return eta, beta, (sk - 1) / (sk + 1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("\n--- the stability threshold is exact ---")
    A, mu, L = make_quadratic(100.0, n=30)
    x0 = rng.standard_normal(30)
    # eta < 2/L converges; eta > 2/L diverges. There is no grey zone.
    x_ok, _ = gd(A, x0, 1.99 / L, 3000)
    x_bad, _ = gd(A, x0, 2.01 / L, 3000)
    print(f"      2/L = {2/L:.6f}")
    print(f"      eta = 1.99/L -> final ||x|| = {np.linalg.norm(x_ok):.3e}")
    print(f"      eta = 2.01/L -> final ||x|| = {np.linalg.norm(x_bad):.3e}")
    check("eta just below 2/L converges", np.linalg.norm(x_ok) < np.linalg.norm(x0))
    check("eta just above 2/L diverges",
          np.linalg.norm(x_bad) > 1e3 * np.linalg.norm(x0) or not np.isfinite(x_bad).all())

    print("\n--- measured rate matches theory ---")
    print(f"      {'kappa':>8} {'GD pred':>10} {'GD meas':>10} {'HB pred':>10} {'HB meas':>10} {'speedup':>9}")
    for kappa in (10.0, 100.0, 10000.0):
        n = 30
        A, mu, L = make_quadratic(kappa, n=n)
        x0 = rng.standard_normal(n)
        x0 /= np.linalg.norm(x0)

        eta_g, rate_g = optimal_gd_params(mu, L)
        eta_h, beta_h, rate_h = optimal_hb_params(mu, L)

        # Predicted steps to contract by 1e-8: rate^t = 1e-8.
        pred_g = int(np.ceil(np.log(1e-8) / np.log(rate_g)))
        pred_h = int(np.ceil(np.log(1e-8) / np.log(rate_h)))

        budget = 20 * max(pred_g, pred_h) + 1000
        _, steps_g, errs_g = gd(A, x0, eta_g, budget, history=True)
        _, steps_h, errs_h = heavy_ball(A, x0, eta_h, beta_h, budget, history=True)

        meas_rate_g = asymptotic_rate(errs_g)
        meas_rate_h = asymptotic_rate(errs_h)

        print(f"      {kappa:8.0f} {pred_g:10d} {steps_g:10d} {pred_h:10d} {steps_h:10d} "
              f"{steps_g/max(steps_h,1):8.1f}x")

        # GD: the contraction bound is tight and holds from the first step, so a
        # generic start comes in at or just under the predicted step count.
        check(f"kappa={kappa:.0f}: GD reaches tolerance within its bound",
              steps_g <= pred_g + 5)

        # HEAVY BALL: the step count OVERSHOOTS the naive bound (e.g. 113 steps
        # against 92 predicted at kappa=100), and that is not a bug. The formula
        # (sqrt(k)-1)/(sqrt(k)+1) is the ASYMPTOTIC rate. The heavy-ball
        # iteration matrix is non-normal, so ||x_t|| <= C * rate^t with C > 1:
        # there is a transient before the asymptotic rate takes hold, and
        # log(tol)/log(rate) silently assumes C = 1. Test the claim that is
        # actually being made — the rate, fitted to the tail — not the step count.
        check(f"kappa={kappa:.0f}: measured GD rate matches (k-1)/(k+1)",
              meas_rate_g, rate_g, tol=0.02 * rate_g)
        check(f"kappa={kappa:.0f}: measured heavy-ball rate matches (sqrt k -1)/(sqrt k +1)",
              meas_rate_h, rate_h, tol=0.05 * rate_h)
        # And the transient constant is real: bound it rather than pretend it is 1.
        check(f"kappa={kappa:.0f}: heavy-ball transient costs under 50% extra steps",
              steps_h <= 1.5 * pred_h)

        # The headline: momentum's advantage grows like sqrt(kappa).
        if kappa >= 100:
            check(f"kappa={kappa:.0f}: momentum is materially faster",
                  steps_h < steps_g * 0.6)

    print("\n--- O(kappa) vs O(sqrt kappa) ---")
    # Multiply kappa by 100 and GD's step count should grow ~100x while heavy
    # ball's grows ~10x. That contrast is the entire argument for momentum.
    def steps_for(kappa, method):
        n = 30
        A, mu, L = make_quadratic(kappa, n=n)
        x0 = rng.standard_normal(n)
        x0 /= np.linalg.norm(x0)
        if method == "gd":
            eta, _ = optimal_gd_params(mu, L)
            return gd(A, x0, eta, 2_000_000)[1]
        eta, beta, _ = optimal_hb_params(mu, L)
        return heavy_ball(A, x0, eta, beta, 2_000_000)[1]

    g_lo, g_hi = steps_for(100.0, "gd"), steps_for(10000.0, "gd")
    h_lo, h_hi = steps_for(100.0, "hb"), steps_for(10000.0, "hb")
    print(f"      GD:         kappa 100 -> {g_lo:6d} steps,  kappa 10^4 -> {g_hi:6d}  ({g_hi/g_lo:5.1f}x)")
    print(f"      heavy ball: kappa 100 -> {h_lo:6d} steps,  kappa 10^4 -> {h_hi:6d}  ({h_hi/h_lo:5.1f}x)")
    check("GD step count grows ~linearly in kappa (100x for 100x)", 60 < g_hi / g_lo < 160)
    check("heavy ball grows ~sqrt (10x for 100x)", 6 < h_hi / h_lo < 16)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    A, mu, L = make_quadratic(50.0, n=20)
    x0 = rng.standard_normal(20)

    # (a) Tuning eta on the AVERAGE curvature instead of the LARGEST is the
    #     classic divergence. The mean eigenvalue is far below L here.
    mean_eig = float(np.trace(A) / 20)
    eta_naive = 2.0 / mean_eig
    x_div, _ = gd(A, x0, eta_naive, 500)
    # The whole point is that this overflows, so do not let numpy warn about it.
    with np.errstate(over="ignore", invalid="ignore"):
        div_norm = float(np.linalg.norm(x_div))
    print(f"      mean eigenvalue {mean_eig:.2f} vs L = {L:.2f}")
    print(f"      eta = 2/mean -> ||x|| = {div_norm:.3e}  (stability needs eta < {2/L:.4f})")
    check("stepping on mean curvature diverges",
          not np.isfinite(x_div).all() or div_norm > 1e6)

    # (b) Momentum is not free: beta too close to 1 with an unadjusted step
    #     overshoots and oscillates.
    eta_h, beta_h, _ = optimal_hb_params(mu, L)
    x_osc, _ = heavy_ball(A, x0, eta_h, 0.999, 2000)
    x_good, _ = heavy_ball(A, x0, eta_h, beta_h, 2000)
    print(f"      beta = {beta_h:.4f} (optimal) -> ||x|| = {np.linalg.norm(x_good):.3e}")
    print(f"      beta = 0.999              -> ||x|| = {np.linalg.norm(x_osc):.3e}")
    check("over-aggressive momentum is much worse",
          np.linalg.norm(x_osc) > 1e3 * max(np.linalg.norm(x_good), 1e-12))

    summary()
