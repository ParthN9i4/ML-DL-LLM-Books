"""Artifact 24.1 -- Chinchilla parametric fit, compute-optimal allocation,
and the inference-aware correction.

Plan: (1) plant the published Chinchilla ground truth
L(N,D) = E + A/N^alpha + B/D^beta  with (E,A,B,alpha,beta) =
(1.69, 406.4, 410.7, 0.34, 0.28); (2) generate noisy synthetic loss
observations on a grid of (N, D), including 5% heavy-tailed outliers;
(3) recover the five parameters by minimizing a SUMMED Huber loss on
log-loss with an LSE parameterization, exactly as Hoffmann et al. (2022)
Approach 3 did -- and as the Besiroglu et al. (2024) replication showed
must be summed, not averaged, to avoid premature L-BFGS termination;
(4) derive the compute-optimal (N*, D*) analytically, cross-check by
brute force, and check the fit reproduces it; (5) re-solve under the
Sardana-Frankle inference-aware objective  6*N*D_tr + 2*N*D_inf  at a
fixed loss target and show N shrinks while D grows. Pure NumPy + scipy.
"""
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

rng = np.random.default_rng(0)

# ----- ground truth: Hoffmann et al. (2022), Table "Approach 3" (rounded) ---
TRUE = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)


def surface(N, D, p):
    """Chinchilla parametric loss surface L(N, D)."""
    return p["E"] + p["A"] * N ** -p["alpha"] + p["B"] * D ** -p["beta"]


# ---------------------------------------------------------------- synthetic
def make_data(n_side=18):
    """Grid of (N, D) runs, 1e7..1e11 params x 1e9..3e12 tokens, with
    multiplicative log-normal noise (sigma = 1%) plus 5% gross outliers
    (extra sigma = 8%) -- the case Huber loss exists to survive."""
    N, D = np.meshgrid(np.logspace(7, 11, n_side), np.logspace(9, 12.5, n_side))
    N, D = N.ravel(), D.ravel()
    eps = rng.normal(0.0, 0.01, N.size)
    out = rng.random(N.size) < 0.05
    eps[out] += rng.normal(0.0, 0.08, out.sum())
    return N, D, surface(N, D, TRUE) * np.exp(eps)


# ------------------------------------------------------------------ the fit
def huber_sum(r, delta=1e-3):
    """SUMMED Huber loss. Averaging instead of summing shrinks the objective
    scale ~300x and made L-BFGS-B stop early in the original paper's fit
    (Besiroglu et al. 2024) -- so we sum."""
    a = np.abs(r)
    return np.sum(np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta)))


def objective(x, logN, logD, logL):
    a, b, e, al, be = x  # A = exp(a), B = exp(b), E = exp(e)
    pred = logsumexp(np.stack([a - al * logN, b - be * logD,
                               np.full_like(logN, e)]), axis=0)
    return huber_sum(pred - logL)


def fit(N, D, L):
    """L-BFGS-B from a grid of inits (Hoffmann et al. style); keep the best."""
    logN, logD, logL = np.log(N), np.log(D), np.log(L)
    best = None
    for a0 in (0.0, 5.0, 10.0):
        for b0 in (0.0, 5.0, 10.0):
            for e0 in (-1.0, 0.0, 1.0):
                for al0 in (0.1, 0.3, 0.7):
                    for be0 in (0.1, 0.3, 0.7):
                        r = minimize(objective, x0=(a0, b0, e0, al0, be0),
                                     args=(logN, logD, logL), method="L-BFGS-B")
                        if best is None or r.fun < best.fun:
                            best = r
    a, b, e, al, be = best.x
    return dict(E=np.exp(e), A=np.exp(a), B=np.exp(b), alpha=al, beta=be)


# ------------------------------------------- compute-optimal allocation
def compute_optimal(p, C):
    """argmin_{6ND=C} L(N,D):  N* = G (C/6)^a with a = beta/(alpha+beta),
    G = (alpha A / (beta B))^{1/(alpha+beta)};  D* = (C/6)/N*."""
    al, be = p["alpha"], p["beta"]
    a = be / (al + be)
    G = (al * p["A"] / (be * p["B"])) ** (1.0 / (al + be))
    Nopt = G * (C / 6.0) ** a
    return Nopt, (C / 6.0) / Nopt


# --------------------------------------- inference-aware optimum (Sardana)
def inference_aware(p, ell, D_inf):
    """min_N  6 N D_tr(N) + 2 N D_inf   s.t.  L(N, D_tr) = ell, where
    D_tr(N) = (B / (ell - E - A N^-alpha))^{1/beta}. 1-D bounded search."""
    E, A, B, al, be = (p[k] for k in ("E", "A", "B", "alpha", "beta"))
    N_min = (A / (ell - E)) ** (1.0 / al)          # below this, ell unreachable

    def D_tr(N):
        return (B / (ell - E - A * N ** -al)) ** (1.0 / be)

    def total(logN):
        N = 10.0 ** logN
        return 6.0 * N * D_tr(N) + 2.0 * N * D_inf

    res = minimize_scalar(total, bounds=(np.log10(N_min) + 1e-3, 13.0),
                          method="bounded",
                          options={"xatol": 1e-10})
    N = 10.0 ** res.x
    return N, D_tr(N), res.fun


if __name__ == "__main__":
    t0 = time.time()
    N, D, L = make_data()
    print(f"synthetic runs: {N.size} (N in [1e7,1e11], D in [1e9,3e12])")

    # ---- 1. recover the planted parameters ------------------------------
    f = fit(N, D, L)
    print("\nparam    truth      fitted")
    for k in ("E", "A", "B", "alpha", "beta"):
        print(f"{k:5s} {TRUE[k]:9.4f} {f[k]:11.4f}")
    assert abs(f["alpha"] - TRUE["alpha"]) < 0.02, "alpha not recovered"
    assert abs(f["beta"] - TRUE["beta"]) < 0.02, "beta not recovered"
    assert abs(f["E"] - TRUE["E"]) / TRUE["E"] < 0.02, "E not recovered"
    assert abs(np.log(f["A"] / TRUE["A"])) < 0.25, "A not recovered"
    assert abs(np.log(f["B"] / TRUE["B"])) < 0.25, "B not recovered"

    # ---- 2. compute-optimal allocation, analytic vs brute force ---------
    C0 = 5.76e23                                   # the Gopher/Chinchilla budget
    a_true = TRUE["beta"] / (TRUE["alpha"] + TRUE["beta"])
    a_fit = f["beta"] / (f["alpha"] + f["beta"])
    Nc_t, Dc_t = compute_optimal(TRUE, C0)
    Nc_f, Dc_f = compute_optimal(f, C0)
    # brute force on the planted surface: scan N at fixed C, D = C/(6N)
    Ng = np.logspace(8, 12, 400_001)
    Nbf = Ng[np.argmin(surface(Ng, C0 / (6.0 * Ng), TRUE))]
    print(f"\nexponent a = beta/(alpha+beta): true {a_true:.4f}  fitted {a_fit:.4f}")
    print(f"N* at C={C0:.2e}: analytic {Nc_t:.3e}  brute-force {Nbf:.3e}  "
          f"fitted-law {Nc_f:.3e}")
    print(f"tokens/param at C0: true {Dc_t/Nc_t:.1f}  fitted {Dc_f/Nc_f:.1f}")
    assert abs(np.log(Nbf / Nc_t)) < 1e-3, "closed form disagrees w/ brute force"
    assert abs(a_fit - a_true) < 0.02, "allocation exponent not recovered"
    assert abs(np.log((Dc_f / Nc_f) / (Dc_t / Nc_t))) < 0.10, \
        "fitted tokens/param ratio off by >10%"

    # ---- 3. inference-aware re-solve (Sardana & Frankle objective) ------
    ell = surface(Nc_f, Dc_f, f)                   # fix the quality target
    D_inf = 5e12                                   # 5T served tokens over life
    Ni, Di, tot_i = inference_aware(f, ell, D_inf)
    tot_c = 6.0 * Nc_f * Dc_f + 2.0 * Nc_f * D_inf
    print(f"\nloss target {ell:.4f}, lifetime inference demand {D_inf:.1e} tokens")
    print("              Chinchilla-optimal   inference-aware")
    print(f"N            {Nc_f:15.3e} {Ni:17.3e}")
    print(f"D_train      {Dc_f:15.3e} {Di:17.3e}")
    print(f"tokens/param {Dc_f/Nc_f:15.1f} {Di/Ni:17.1f}")
    print(f"train FLOPs  {6*Nc_f*Dc_f:15.3e} {6*Ni*Di:17.3e}")
    print(f"infer FLOPs  {2*Nc_f*D_inf:15.3e} {2*Ni*D_inf:17.3e}")
    print(f"total FLOPs  {tot_c:15.3e} {tot_i:17.3e}"
          f"   (saves {100*(1-tot_i/tot_c):.1f}%)")
    assert Ni < Nc_f, "inference-aware N must be strictly smaller"
    assert Di > Dc_f, "inference-aware D must be strictly larger"
    assert tot_i < tot_c, "re-solved optimum must lower total FLOPs"
    # same quality, by construction: verify numerically
    assert abs(surface(Ni, Di, f) - ell) < 1e-9

    print(f"\nall assertions passed  ({time.time()-t0:.1f}s)")
