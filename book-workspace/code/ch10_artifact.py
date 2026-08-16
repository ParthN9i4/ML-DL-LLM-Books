"""Artifact 10.1 - A 3-layer MLP with a hand-derived backward pass, plus an
empirical count of linear regions as a function of depth and width.

Part 1: forward/backward for d0 -> h1 -> h2 -> C with ReLU and softmax CE.
        Every gradient is derived by hand (no autodiff) and checked against
        central finite differences; torch is an optional cross-check only.
Part 2: linear regions of a 2-D ReLU net, counted by hashing the binary
        activation pattern of every hidden unit over a dense input grid, for
        (a) an explicit depth-L "sawtooth" attaining 4^L regions at width 4,
        (b) a random-init net of the same shape, (c) a random-init single
        hidden layer with the SAME parameter count.
Part 3: parameter / FLOP accounting and the 6ND rule.
"""

import numpy as np
from math import comb

try:
    import torch
except ImportError:  # the artifact must run without torch
    torch = None


# ---------------- Part 1 - MLP forward / backward by hand -------------------

def init_mlp(dims, rng):
    """He-scaled init.  Column convention: A has shape (features, batch)."""
    ps = {}
    for i in range(len(dims) - 1):
        ps[f"W{i+1}"] = rng.normal(0.0, np.sqrt(2.0 / dims[i]), (dims[i + 1], dims[i]))
        ps[f"b{i+1}"] = rng.normal(0.0, 0.1, (dims[i + 1], 1))
    return ps


def forward(ps, X, y):
    """Z^(l) = W^(l) A^(l-1) + b^(l);  A = relu(Z);  loss = mean softmax CE."""
    Z1 = ps["W1"] @ X + ps["b1"]
    A1 = np.maximum(Z1, 0.0)
    Z2 = ps["W2"] @ A1 + ps["b2"]
    A2 = np.maximum(Z2, 0.0)
    Z3 = ps["W3"] @ A2 + ps["b3"]
    # stable log-softmax: subtract the column max before exponentiating
    m = Z3.max(axis=0, keepdims=True)
    logZ = m + np.log(np.exp(Z3 - m).sum(axis=0, keepdims=True))
    logp = Z3 - logZ
    B = X.shape[1]
    loss = -logp[y, np.arange(B)].mean()
    cache = (X, Z1, A1, Z2, A2, np.exp(logp), y, B)
    return loss, cache


def backward(ps, cache):
    """Hand-derived gradients.  The only non-obvious step is the first:
    d(mean CE)/dZ3 = (P - Y) / B, where P is the softmax and Y the one-hot
    target.  Everything after it is the chain rule with two facts:
    d/dW (W A) contracts as dW = dZ A^T, and relu'(z) = 1[z > 0]."""
    X, Z1, A1, Z2, A2, P, y, B = cache
    g = {}
    dZ3 = P.copy()
    dZ3[y, np.arange(B)] -= 1.0
    dZ3 /= B
    g["W3"] = dZ3 @ A2.T
    g["b3"] = dZ3.sum(axis=1, keepdims=True)
    dZ2 = (ps["W3"].T @ dZ3) * (Z2 > 0)      # backprop through relu
    g["W2"] = dZ2 @ A1.T
    g["b2"] = dZ2.sum(axis=1, keepdims=True)
    dZ1 = (ps["W2"].T @ dZ2) * (Z1 > 0)
    g["W1"] = dZ1 @ X.T
    g["b1"] = dZ1.sum(axis=1, keepdims=True)
    return g


def finite_difference_check(ps, X, y, eps=1e-5):
    """Central differences on every scalar parameter.  Returns (max relative
    error, count of parameters checked)."""
    g = backward(ps, forward(ps, X, y)[1])
    worst, n = 0.0, 0
    for k in sorted(ps):
        flat, gflat = ps[k].reshape(-1), g[k].reshape(-1)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            lp = forward(ps, X, y)[0]
            flat[i] = orig - eps
            lm = forward(ps, X, y)[0]
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            denom = max(1.0, abs(num), abs(gflat[i]))
            worst = max(worst, abs(num - gflat[i]) / denom)
            n += 1
    return worst, n


def torch_cross_check(ps, X, y):
    if torch is None:
        print("  [skipped: torch not installed]")
        return
    tp = {k: torch.tensor(v, requires_grad=True) for k, v in ps.items()}
    tX = torch.tensor(X)
    Z1 = tp["W1"] @ tX + tp["b1"]
    Z2 = tp["W2"] @ torch.relu(Z1) + tp["b2"]
    Z3 = tp["W3"] @ torch.relu(Z2) + tp["b3"]
    loss = torch.nn.functional.cross_entropy(Z3.T, torch.tensor(y))
    loss.backward()
    ours = backward(ps, forward(ps, X, y)[1])
    d = max(float(np.abs(ours[k] - tp[k].grad.numpy()).max()) for k in ps)
    print(f"  torch autograd vs hand-derived: max |dg| = {d:.3e}")
    assert d < 1e-12, d


# ------------- Part 2 - regions by hashing activation patterns --------------

def count_regions(layers, X):
    """layers: list of (W, b) for the HIDDEN layers only (the output layer is
    affine and creates no new regions).  X: (2, n) grid of inputs.  Two inputs
    lie in the same linear region iff every hidden unit has the same sign, so
    hash the sign vector and count distinct hashes."""
    A, bits, shift = X, np.zeros(X.shape[1], dtype=np.int64), 0
    for W, b in layers:
        Z = W @ A + b
        for row in (Z > 0):
            bits |= row.astype(np.int64) << shift
            shift += 1
        A = np.maximum(Z, 0.0)
    assert shift <= 62, "pattern does not fit in an int64 hash"
    return int(np.unique(bits).size)


def sawtooth_layers(L):
    """Width-4 construction: two copies (one per input coordinate) of the hat
    map g(t) = 2*relu(t) - 4*relu(t - 1/2), which folds [0,1] onto itself.  L
    compositions give 2^L pieces per coordinate, 4^L regions in the plane, at
    20 parameters per layer -- exponential regions for linear parameter cost."""
    W1 = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    b = np.array([[0.0], [-0.5], [0.0], [-0.5]])
    Wk = np.array([[2.0, -4.0, 0.0, 0.0], [2.0, -4.0, 0.0, 0.0],
                   [0.0, 0.0, 2.0, -4.0], [0.0, 0.0, 2.0, -4.0]])
    return [(W1, b)] + [(Wk, b.copy()) for _ in range(L - 1)]


def random_layers(widths, rng, d_in=2):
    """Random hidden layers.  Each first-layer hyperplane is made to pass
    through a uniformly random point of the unit square -- generous to the
    shallow net, since a hyperplane missing the box contributes no region."""
    layers, fan = [], d_in
    for i, h in enumerate(widths):
        W = rng.normal(0.0, np.sqrt(2.0 / fan), (h, fan))
        if i == 0:
            P = rng.uniform(0.0, 1.0, (2, h))
            b = -(W * P.T).sum(axis=1, keepdims=True)
        else:
            b = rng.normal(0.0, 0.3, (h, 1))
        layers.append((W, b))
        fan = h
    return layers


def best_of(widths, X, seeds=5):
    """Max region count over several random draws (favourable to the baseline)."""
    return max(count_regions(random_layers(widths, np.random.default_rng(100 + s)), X)
               for s in range(seeds))


# ---------------- Part 3 - parameter and FLOP accounting --------------------

def mlp_params_flops(dims):
    """Parameters and forward FLOPs per example (2 FLOPs per MAC; the O(width)
    elementwise activation is ignored, as it should be)."""
    p = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
    mac = sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))
    return p, 2 * mac


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=" * 72)
    print("Part 1: hand-derived backward pass, 3-layer MLP (4 -> 6 -> 5 -> 3)")
    print("=" * 72)
    dims, B = [4, 6, 5, 3], 7
    ps = init_mlp(dims, rng)
    X = rng.normal(size=(dims[0], B))
    y = rng.integers(0, dims[-1], size=B)
    loss, _ = forward(ps, X, y)
    print(f"  loss = {loss:.10f}   (ln 3 = {np.log(3):.10f} at chance)")
    err, npar = finite_difference_check(ps, X, y)
    print(f"  central-difference check over {npar} parameters:"
          f" max relative error = {err:.3e}")
    assert err < 1e-7, err
    torch_cross_check(ps, X, y)

    print()
    print("=" * 72)
    print("Part 2: linear regions on a dense grid of the unit square")
    print("=" * 72)
    n = 600
    # Offset by 1/pi rather than 1/2: a dyadic offset lands exactly on a fold of
    # the sawtooth at depth >= 4, where relu' is undefined and the pattern hash
    # picks up spurious singleton "regions".
    t = (np.arange(n) + 1.0 / np.pi) / n
    G = np.stack(np.meshgrid(t, t, indexing="ij"), 0).reshape(2, -1)
    print(f"  grid: {n} x {n} = {G.shape[1]} points in [0,1]^2")
    print()
    hdr = ("  L  built |  params  | regions(built) | rand same shape |"
           " shallow h | params | regions(rand) | shallow bound")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    built_counts = {}
    for L in range(2, 7):
        deep = sawtooth_layers(L)
        p_deep = 12 + 20 * (L - 1) + 5                 # +5 for the output layer
        r_built = count_regions(deep, G)
        r_rand_deep = best_of([4] * L, G)
        h = round((p_deep - 1) / 4)                    # 4h+1 params, matched
        p_shal = 4 * h + 1
        r_shal = best_of([h], G)
        bound = 1 + h + comb(h, 2)                     # Zaslavsky bound in 2-D
        built_counts[L] = r_built
        print(f"  {L}  w=4   |  {p_deep:5d}   |     {r_built:6d}     |"
              f"      {r_rand_deep:5d}      |    {h:4d}   | {p_shal:5d}  |"
              f"     {r_shal:6d}    |    {bound:6d}")
        assert r_built == 4 ** L, (L, r_built)
        assert r_shal <= bound, (r_shal, bound)
        if L >= 4:
            assert r_built > r_shal, (L, r_built, r_shal)
    ratios = [built_counts[L] / built_counts[L - 1] for L in range(3, 7)]
    print(f"\n  region-count ratio between consecutive depths: "
          f"{['%.1f' % r for r in ratios]}  (exactly 4 = 2 per coordinate)")
    r6_rand = best_of([4] * 6, G)
    print(f"  random init, L=6 w=4: {r6_rand} regions vs {built_counts[6]} for the"
          f" same architecture with chosen weights"
          f"  ({built_counts[6] / r6_rand:.0f}x)")
    assert r6_rand < built_counts[6] / 4

    print()
    print("=" * 72)
    print("Part 3: parameter and FLOP accounting")
    print("=" * 72)
    for d in (1024, 4096):
        dff = int(round(8 * d / 3 / 64) * 64)          # 2/3 of 4d, rounded to 64
        p_gelu, f_gelu = mlp_params_flops([d, 4 * d, d])
        p_swi = 3 * d * dff
        print(f"  d={d:5d}: dense 4d MLP  params={p_gelu:>12,d}  fwd FLOPs/token="
              f"{f_gelu:>13,d}")
        print(f"          : SwiGLU d_ff={dff:<5d} params={p_swi:>12,d}  fwd FLOPs/token="
              f"{2 * p_swi:>13,d}")
        assert abs(p_swi / (4 * d * d * 2) - 1.0) < 0.02   # matched to the dense MLP
    N, D = 7e9, 2e12
    print(f"  6ND rule: N={N:.0e} params, D={D:.0e} tokens -> C = {6 * N * D:.2e} FLOPs")
    print(f"            at 4e14 FLOP/s sustained: {6 * N * D / 4e14 / 86400:.1f} GPU-days")

    print("\nAll assertions passed.")
