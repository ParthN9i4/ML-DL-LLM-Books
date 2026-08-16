"""
ex29 — Polynomial approximation and the multiplicative-depth ledger.
(Book: Chapters 38-39)

Under CKKS you get additions, multiplications, and rotations. Everything else —
exp, division, inverse square root, comparison — must become a polynomial, and
every multiplication spends a LEVEL from a budget fixed at key generation.
This exercise builds the two tools that turn "encrypt a transformer" from a
slogan into an engineering estimate:

  1. Chebyshev approximation on a chosen interval, with the error-versus-degree
     table that decides what degree you must pay for.

  2. Depth accounting. Evaluating a degree-d polynomial costs ceil(log2 d)
     levels with a Paterson-Stockmeyer/baby-step-giant-step schedule — NOT d
     levels (Horner) and NOT free. The difference decides whether a network
     fits in a leveled scheme or needs bootstrapping.

The punchline is the final ledger: one attention head's depth, line by line.
No FHE library is needed — the arithmetic constraints are what is being
learned, and they are scheme facts, not library facts.

[VERIFY] Depth costs stated here are the standard textbook accounting
(one level per ciphertext-ciphertext multiplication; plaintext multiplies
also consume scale in CKKS). Library-specific optimizations (lazy rescaling,
hoisting) change constants, not shape - check your OpenFHE version's behavior.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


def cheb_fit(f, a, b, degree, n_nodes=None):
    """Chebyshev coefficients of f on [a, b].

    Sample f at Chebyshev nodes mapped to [a, b], take the DCT-like sum.
    Returns coefficients c_0..c_degree for sum c_k T_k(t), t in [-1, 1].
    """
    n = n_nodes or (degree + 1)
    # === YOUR CODE HERE ===
    k = np.arange(n)
    t = np.cos(np.pi * (k + 0.5) / n)                 # nodes in [-1, 1]
    x = 0.5 * (b - a) * t + 0.5 * (b + a)
    fx = f(x)
    c = np.zeros(degree + 1)
    for j in range(degree + 1):
        c[j] = (2.0 / n) * np.sum(fx * np.cos(np.pi * j * (k + 0.5) / n))
    c[0] *= 0.5
    return c


def cheb_eval(c, x, a, b):
    """Evaluate the Chebyshev series at x (Clenshaw recurrence)."""
    t = (2.0 * np.asarray(x, dtype=np.float64) - (a + b)) / (b - a)
    # === YOUR CODE HERE ===
    b1 = np.zeros_like(t)
    b2 = np.zeros_like(t)
    for ck in c[:0:-1]:
        b1, b2 = 2 * t * b1 - b2 + ck, b1
    return t * b1 - b2 + c[0]


def max_err(f, c, a, b, n=20001):
    x = np.linspace(a, b, n)
    return float(np.max(np.abs(f(x) - cheb_eval(c, x, a, b))))


def poly_eval_depth(degree):
    """Multiplicative depth to evaluate a degree-d polynomial on a ciphertext.

    Powers x, x^2, ..., x^d can be built by repeated squaring/combination in
    ceil(log2 d) ciphertext-ciphertext multiplications of DEPTH; the
    baby-step-giant-step recombination adds plaintext multiplies (scale, not
    depth-critical in this simple accounting). Horner would cost d.
    """
    # === YOUR CODE HERE ===
    if degree <= 1:
        return 0 if degree < 1 else 0   # linear = plaintext mult + add only
    return int(np.ceil(np.log2(degree)))


def newton_inv_sqrt(y, x0, iters):
    """Newton-Raphson for 1/sqrt(y): x <- x * (1.5 - 0.5 * y * x^2).

    Each iteration costs exactly 2 ciphertext multiplications of depth
    (y*x^2 needs x^2 then y*(x^2); the final x*(...) is the second depth level
    on the running variable — total 2 levels per iteration in the standard
    accounting).
    """
    x = np.full_like(np.asarray(y, dtype=np.float64), x0)
    # === YOUR CODE HERE ===
    for _ in range(iters):
        x = x * (1.5 - 0.5 * y * x * x)
    return x


NEWTON_DEPTH_PER_ITER = 2


def attention_head_depth_ledger(softmax_degree, invsqrt_iters, gelu_degree):
    """The ledger: multiplicative depth of ONE pre-norm attention block under
    CKKS, line by line, using the exp-approximation softmax
    (exp via polynomial, then normalization via Newton inverse — here we
    charge the common goldschmidt/newton division at the same 2-levels/iter).

    Returns (total, list of (operation, levels)).
    """
    # === YOUR CODE HERE ===
    lines = [
        ("RMSNorm: square + mean (x^2)",              1),
        ("RMSNorm: Newton 1/sqrt, %d iters" % invsqrt_iters, NEWTON_DEPTH_PER_ITER * invsqrt_iters),
        ("RMSNorm: multiply x by inv-std",            1),
        ("QKV projections (plaintext weights)",       0),   # plaintext mult: scale only
        ("Q @ K^T (ciphertext x ciphertext)",         1),
        ("softmax: exp via degree-%d Chebyshev" % softmax_degree, poly_eval_depth(softmax_degree)),
        ("softmax: Newton division, %d iters" % invsqrt_iters, NEWTON_DEPTH_PER_ITER * invsqrt_iters),
        ("scores @ V (ciphertext x ciphertext)",      1),
        ("output projection (plaintext weights)",     0),
        ("MLP up-projection (plaintext)",             0),
        ("GELU via degree-%d Chebyshev" % gelu_degree, poly_eval_depth(gelu_degree)),
        ("MLP down-projection (plaintext)",           0),
    ]
    return sum(v for _, v in lines), lines


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("\n--- Chebyshev machinery sanity ---")
    # A polynomial of degree <= d is reproduced exactly.
    c = cheb_fit(lambda x: 3 * x ** 3 - x + 0.5, -1, 1, 3)
    check("degree-3 polynomial reproduced exactly",
          max_err(lambda x: 3 * x ** 3 - x + 0.5, c, -1, 1), 0.0, tol=1e-12)
    # Interval mapping works.
    c2 = cheb_fit(np.exp, 0, 4, 12)
    check("exp on [0,4] at degree 12 is accurate", max_err(np.exp, c2, 0, 4) < 1e-8)

    print("\n--- error versus degree: the table you pay for ---")
    print(f"      {'fn':>10} {'interval':>12} " + " ".join(f"d={d:>2}" for d in (3, 7, 15, 31)))
    tables = {}
    for name, f, a, b in (
        ("exp", np.exp, -8.0, 2.0),                       # softmax numerator range
        ("inv-sqrt", lambda x: 1 / np.sqrt(x), 0.1, 4.0), # RMSNorm range
        ("gelu", lambda x: x * 0.5 * (1 + np.vectorize(lambda t: np.math.erf(t / np.sqrt(2)) if hasattr(np.math, "erf") else 0)(x)) if False else 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))), -6.0, 6.0),
    ):
        errs = [max_err(f, cheb_fit(f, a, b, d), a, b) for d in (3, 7, 15, 31)]
        tables[name] = errs
        print(f"      {name:>10} [{a:5.1f},{b:4.1f}] " + " ".join(f"{e:8.1e}" for e in errs))
    check("error decreases monotonically with degree (exp)",
          all(tables["exp"][i] > tables["exp"][i + 1] for i in range(3)))
    check("degree 15 exp on [-8,2] is below 1e-4", tables["exp"][2] < 1e-4)
    # Domain restriction is worth more than degree: exp on [-4,2] at d=7
    # beats exp on [-8,2] at d=7 by a wide margin.
    e_wide = max_err(np.exp, cheb_fit(np.exp, -8, 2, 7), -8, 2)
    e_narrow = max_err(np.exp, cheb_fit(np.exp, -4, 2, 7), -4, 2)
    print(f"      exp d=7: [-8,2] err {e_wide:.1e} vs [-4,2] err {e_narrow:.1e} "
          f"({e_wide/e_narrow:.0f}x better from the narrower domain)")
    check("halving the domain beats doubling nothing (>10x at same degree)",
          e_wide > 10 * e_narrow)

    print("\n--- Newton inverse square root ---")
    y = np.linspace(0.2, 3.5, 1000)
    errs_by_iter = {}
    for iters in (2, 4, 6):
        x = newton_inv_sqrt(y, x0=0.7, iters=iters)
        errs_by_iter[iters] = float(np.max(np.abs(x - 1 / np.sqrt(y))))
        depth = NEWTON_DEPTH_PER_ITER * iters
        print(f"      {iters} iters (depth {depth:2d}): max err {errs_by_iter[iters]:.2e}")
    # A first draft asserted 6 iterations reach 1e-6; measured 2.7e-5 on this
    # wide interval with a single fixed x0 — the edges of the basin converge
    # slower. The honest claims are the measured accuracy and the QUADRATIC
    # convergence that defines Newton: the error at 6 iters is far below the
    # SQUARE of the error at 4 (9.78e-2^2 = 9.6e-3).
    check("6 Newton iterations reach ~3e-5 on [0.2, 3.5]", errs_by_iter[6] < 1e-4)
    check("convergence is (at least) quadratic between 4 and 6 iterations",
          errs_by_iter[6] < errs_by_iter[4] ** 2)
    # The catch: Newton diverges if the initial guess is bad relative to the
    # input range — the reason encrypted RMSNorm needs input normalization.
    with np.errstate(over="ignore", invalid="ignore"):
        x_div = newton_inv_sqrt(np.array([50.0]), x0=0.7, iters=6)
    print(f"      same x0 on y=50 -> {x_div[0]:.3e}  (diverged; 1/sqrt(50)={1/np.sqrt(50):.3f})")
    check("Newton diverges outside its basin (why domain control is not optional)",
          not np.isfinite(x_div[0]) or abs(x_div[0] - 1 / np.sqrt(50)) > 1.0)

    print("\n--- depth of polynomial evaluation ---")
    check("degree 3 costs 2 levels", poly_eval_depth(3), 2)
    check("degree 15 costs 4 levels", poly_eval_depth(15), 4)
    check("degree 31 costs 5 levels", poly_eval_depth(31), 5)
    print("      (Horner would cost d levels: 31 versus 5 at degree 31 — the whole"
          " reason Paterson-Stockmeyer exists)")

    print("\n--- THE LEDGER: one attention block under CKKS ---")
    total, lines = attention_head_depth_ledger(softmax_degree=15, invsqrt_iters=4, gelu_degree=15)
    for op, lv in lines:
        print(f"      {lv:3d}  {op}")
    print(f"      ---  total: {total} levels")
    check("one block at practical settings costs 20-40 levels", 20 <= total <= 40)
    # The consequence, stated as arithmetic: a 32-layer model at this cost.
    per_block = total
    layers_32 = 32 * per_block
    print(f"      32 layers x {per_block} = {layers_32} levels -> far beyond any practical")
    print(f"      leveled parameter set (~20-40 usable levels) -> bootstrapping is not")
    print(f"      optional for deep encrypted transformers; it is the architecture.")
    check("a 32-layer model exceeds any leveled budget by an order of magnitude",
          layers_32 > 10 * 40)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")
    # Use the approximation outside its fitted interval and it does not degrade
    # gracefully — a Chebyshev fit EXPLODES off-interval. This is the single
    # most common failure in encrypted inference: an activation drifts outside
    # the assumed range and the "approximate" result is not approximately
    # anything.
    c_exp = cheb_fit(np.exp, -8, 2, 15)
    inside = float(np.abs(np.exp(2.0) - cheb_eval(c_exp, 2.0, -8, 2)))
    outside = float(np.abs(np.exp(4.0) - cheb_eval(c_exp, 4.0, -8, 2)))
    print(f"      |error| at x=2 (edge of interval): {inside:.2e}")
    print(f"      |error| at x=4 (just outside)    : {outside:.2e}")
    check("off-interval evaluation explodes (>1e4x the on-interval error)",
          outside > 1e4 * max(inside, 1e-12))

    summary()
