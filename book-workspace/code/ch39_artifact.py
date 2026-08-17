# Artifact 39.1 -- Approximating the non-polynomial core, measured.
# Three toolkits: (1) Chebyshev vs Remez-minimax for exp on an interval;
# (2) composite sign iteration f_4 (Cheon-Kim-Kim, ASIACRYPT 2020);
# (3) accuracy-vs-depth Pareto table for a CKKS-style softmax
#     (minimax exp + squaring + Newton reciprocal). Depth numbers are a
#     COUNTING model (multiplicative levels consumed), not an FHE timing.
import numpy as np
from numpy.polynomial import chebyshev as C

rng = np.random.default_rng(0)
GRID = np.linspace(-1, 1, 200_001)

# ---------- shared machinery: Chebyshev interpolation and Remez ----------
def cheb_interp(f, a, b, deg):
    """Interpolant of f at the deg+1 Chebyshev nodes of [a,b] (cheb basis)."""
    t = np.cos((2 * np.arange(deg + 1) + 1) * np.pi / (2 * deg + 2))
    x = (a + b) / 2 + (b - a) / 2 * t
    return C.chebfit(t, f(x), deg)

def remez(f, a, b, deg, iters=20):
    """Minimax fit on [a,b] by Remez exchange, Chebyshev basis on [-1,1].
    Solve p(t_i) + (-1)^i E = f(x(t_i)) on deg+2 references, then move the
    references to the error extrema (one per sign-change segment)."""
    to_x = lambda t: (a + b) / 2 + (b - a) / 2 * t
    ref = np.cos(np.pi * np.arange(deg + 2) / (deg + 1))[::-1]
    for _ in range(iters):
        V = C.chebvander(ref, deg)
        M = np.hstack([V, ((-1.0) ** np.arange(deg + 2))[:, None]])
        sol = np.linalg.solve(M, f(to_x(ref)))
        c = sol[:-1]
        err = f(to_x(GRID)) - C.chebval(GRID, c)
        s = np.where(np.diff(np.sign(err)) != 0)[0]      # sign changes
        bnd = np.concatenate([[0], s + 1, [len(GRID)]])  # -> segments
        new = [b0 + np.argmax(np.abs(err[b0:b1]))
               for b0, b1 in zip(bnd[:-1], bnd[1:])]
        if len(new) != deg + 2:                          # defensive only
            break
        ref = GRID[new]
    return c, err

# ---------- (1) Chebyshev vs minimax for exp on [-4,0], degree 7 ----------
A, B, DEG = -4.0, 0.0, 7
cc = cheb_interp(np.exp, A, B, DEG)
cheb_err = np.max(np.abs(np.exp((A + B) / 2 + (B - A) / 2 * GRID)
                         - C.chebval(GRID, cc)))
c_mm, err_mm = remez(np.exp, A, B, DEG)
mm_err = np.max(np.abs(err_mm))
# equioscillation audit: alternating extrema of near-equal magnitude
s = np.where(np.diff(np.sign(err_mm)) != 0)[0]
bnd = np.concatenate([[0], s + 1, [len(GRID)]])
ext = np.array([err_mm[b0 + np.argmax(np.abs(err_mm[b0:b1]))]
                for b0, b1 in zip(bnd[:-1], bnd[1:])])
alt_ok = np.all(np.sign(ext[1:]) != np.sign(ext[:-1]))
ratio = np.min(np.abs(ext)) / np.max(np.abs(ext))
print(f"[1] exp on [{A:g},{B:g}], degree {DEG}")
print(f"    Chebyshev-interp max err : {cheb_err:.3e}")
print(f"    Remez minimax   max err : {mm_err:.3e}  "
      f"(improvement x{cheb_err / mm_err:.3f})")
print(f"    equioscillation: {len(ext)} alternating extrema (need {DEG + 2}), "
      f"min/max magnitude ratio {ratio:.6f}")
assert mm_err < cheb_err, "minimax must beat interpolation at equal degree"
assert len(ext) == DEG + 2 and alt_ok and ratio > 0.999

# ---------- (2) composite sign iteration ----------
# f4 and g4 are exactly the odd degree-9 polynomials shipped in NEXUS's
# CKKS evaluator (F4_COEFFS/128, G4_COEFFS/1024); f4 is f_n (n=4) from
# Cheon-Kim-Kim ASIACRYPT 2020, g4 the acceleration polynomial.
f4 = lambda x: np.polyval(
    np.array([35, 0, -180, 0, 378, 0, -420, 0, 315, 0]) / 128.0, x)
g4 = lambda x: np.polyval(
    np.array([46623, 0, -113492, 0, 97015, 0, -34974, 0, 5850, 0]) / 1024.0, x)

def trans_width(k, tol=1e-2, first_g=0):
    """Half-width w s.t. |F(x) - sgn(x)| <= tol for all w <= |x| <= 1,
    where F = (k - first_g) rounds of f4 after first_g rounds of g4."""
    x = np.linspace(-1, 1, 400_001)
    y = x.copy()
    for i in range(k):
        y = g4(y) if i < first_g else f4(y)
    bad = np.abs(y - np.sign(x)) > tol
    return np.max(np.abs(x[bad])) if bad.any() else 0.0

print("[2] composite sign: rounds k, depth 4k (deg-9 poly = 4 levels/round)")
widths = [trans_width(k) for k in range(1, 6)]
for k, w in enumerate(widths, 1):
    print(f"    k={k}  depth={4 * k:2d}  width(|err|<=1e-2) = {w:.6f}")
w_nexus = trans_width(4, first_g=2)
print(f"    NEXUS g4,g4,f4,f4 (k=4, depth 16): width = {w_nexus:.6f} "
      f"vs f4^4 = {widths[3]:.6f}")
assert all(widths[i + 1] < widths[i] for i in range(4)), "k must sharpen"
# domain-drift hazard, measured: composing f4 outside [-1,1] diverges
print(f"    f4(1.5) = {f4(1.5):.3f}, f4(f4(2.0)) = {f4(f4(2.0)):.3e}")

# ---------- (3) softmax accuracy-vs-depth Pareto (counting model) ----------
# Pipeline (n=8 slots, logits U[-4,4], exact max-subtraction stand-in, so
# inputs live in [-8,0]):
#   exp: minimax degree-d fit of e^x on [-1,0] (toolkit 1), evaluated at
#        x/8, then squared j=3 times    -> depth ceil(log2 d) + 3 + 1
#   sum: rotations+adds, free in levels; scale by 1/n so v=sum/n in [1/8,1]
#   1/v: Newton y<-y(2-v*y), y0 = a+b*v the minimax linear init on [m,1],
#        m=1/8: equioscillation of q(v)=1-v*y0 at {m,(m+1)/2,1} gives
#        e0=(1-m)^2/(1+6m+m^2), error e0^(2^t) -> depth 2t
#   recombine: 1 ct-ct mult + 1 plain mult -> depth 2
n, ntrials, J = 8, 200, 3
Z = rng.uniform(-4, 4, size=(ntrials, n))
Zs = Z - Z.max(axis=1, keepdims=True)
ref = np.exp(Zs) / np.exp(Zs).sum(axis=1, keepdims=True)
m = 1.0 / n
e0 = (1 - m) ** 2 / (1 + 6 * m + m * m)      # 0.4336 for m=1/8
b_i = -8 * e0 / (1 - m) ** 2                 # y0 = a_i + b_i*v (derived
a_i = -b_i * (1 + m)                         #  from the equioscillation)

def softmax_poly(zs, d, t):
    cd, _ = remez(np.exp, -1.0, 0.0, d)      # minimax exp on [-1,0]
    e = C.chebval(2 * (zs / 8.0) + 1, cd)    # eval at x/8, map to [-1,1]
    for _ in range(J):
        e = e * e                            # back to range [-8,0]
    v = e.sum(axis=1, keepdims=True) / n
    y = a_i + b_i * v
    for _ in range(t):
        y = y * (2.0 - v * y)
    return e * y / n

rows = []
for d in (3, 5, 7):
    for t in (2, 3, 4, 5):
        depth = int(np.ceil(np.log2(d))) + J + 1 + 2 * t + 2
        err = np.max(np.abs(softmax_poly(Zs, d, t) - ref))
        rows.append((depth, d, t, err))
rows.sort()
print(f"[3] softmax Pareto: depth | d (exp) | t (Newton) | max |err| "
      f"({ntrials} draws, n={n}, logits U[-4,4])")
best, frontier = np.inf, []
for depth, d, t, e in rows:
    star = " *" if e < best else ""
    best = min(best, e)
    frontier.append(best)
    print(f"    {depth:2d}   |  {d}  |  {t}  | {e:.3e}{star}")
assert all(frontier[i + 1] <= frontier[i] for i in range(len(frontier) - 1)), \
    "frontier must be monotone: more depth never hurts best-achievable error"
assert frontier[-1] < 1e-6, "deep end of the table must reach 1e-6"

# cross-check the plaintext reference against torch, if present
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    dev = np.max(np.abs(torch.softmax(torch.tensor(Z), dim=1).numpy() - ref))
    print(f"[x] torch.softmax cross-check: max deviation {dev:.2e}")
    assert dev < 1e-12
else:
    print("[x] [skipped: torch not installed]")

if __name__ == "__main__":
    print("all assertions passed")
