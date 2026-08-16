"""
Artifact 12.1 -- Signal-propagation analyzer.  A 50-layer network reporting the
variance of forward activations and backward gradients at every depth, swept over
initialization {naive N(0,1), Xavier, He} x normalization {none, LayerNorm,
RMSNorm} x placement {pre-norm, post-norm}.  Pure NumPy float64; torch is used
only to cross-check the hand-written LayerNorm VJP.

Row-vector convention: activations are (B, d) and a layer is  a @ W  with W of
shape (fan_in, fan_out) -- the transpose of the book's  y = W x.
"""

import time
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

EPS = 1e-5

# ----------------------------------------------------------------------------- init
def make_weight(kind, fan_in, fan_out, rng):
    """Draw W; s is the per-entry standard deviation the scheme prescribes."""
    s = {"naive1": 1.0,                                  # no fan-in correction at all
         "xavier": np.sqrt(2.0 / (fan_in + fan_out)),    # Var = 2/(fan_in+fan_out)
         "he": np.sqrt(2.0 / fan_in),                    # Var = 2/fan_in, ReLU gain
         "lecun": np.sqrt(1.0 / fan_in)}[kind]           # Var = 1/fan_in, linear gain
    return rng.standard_normal((fan_in, fan_out)) * s

# ------------------------------------------------------------------ normalizers + VJP
def layernorm_fwd(x):
    """y = (x - mu) / sqrt(var + eps), per row.  Returns y and a cache."""
    xc = x - x.mean(1, keepdims=True)
    inv = 1.0 / np.sqrt((xc * xc).mean(1, keepdims=True) + EPS)
    y = xc * inv
    return y, (y, inv)

def layernorm_bwd(g, cache):
    """dx = inv * (g - mean(g) - y * mean(g*y)): two rank-one projections removed."""
    y, inv = cache
    return inv * (g - g.mean(1, keepdims=True) - y * (g * y).mean(1, keepdims=True))

def rmsnorm_fwd(x):
    """y = x / sqrt(mean(x^2) + eps).  No mean subtraction: one fewer statistic."""
    inv = 1.0 / np.sqrt((x * x).mean(1, keepdims=True) + EPS)
    return x * inv, (x * inv, inv)

def rmsnorm_bwd(g, cache):
    """dx = inv * (g - y * mean(g*y)): one rank-one projection, not two."""
    y, inv = cache
    return inv * (g - y * (g * y).mean(1, keepdims=True))

NORMS = {"none": (lambda x: (x, None), lambda g, c: g),
         "ln": (layernorm_fwd, layernorm_bwd),
         "rms": (rmsnorm_fwd, rmsnorm_bwd)}

# --------------------------------------------------------------------- plain deep MLP
def plain_probe(init, L, d, B, rng):
    """Depth-L ReLU stack, no residual, no norm.  Returns (fwd_var, bwd_var)."""
    Ws = [make_weight(init, d, d, rng) for _ in range(L)]
    a = rng.standard_normal((B, d))                    # unit-variance input
    fwd, masks = [a.var()], []
    for W in Ws:
        z = a @ W
        m = z > 0
        a = z * m
        masks.append(m)
        fwd.append(a.var())
    g = rng.standard_normal((B, d))                    # unit cotangent at the top
    bwd = [g.var()]
    for W, m in zip(reversed(Ws), reversed(masks)):
        g = (g * m) @ W.T
        bwd.append(g.var())
    return np.array(fwd), np.array(bwd[::-1])          # index l = state at depth l

# ------------------------------------------------------- residual + normalized network
def residual_probe(init, norm, placement, L, d, B, rng, final_norm=True):
    """Depth-L residual net with branch F(u) = relu(u @ W1) @ W2.

    pre-norm h <- h + F(norm(h)) gets ONE final norm on the output; post-norm
    h <- norm(h + F(h)) is already normalized there.  That final norm is not
    cosmetic -- it divides the pre-norm backward pass by the grown stream scale.
    Returns (stream var, stream-gradient var, Var(branch)/Var(stream), ||dW2||_F).
    """
    nf, nb = NORMS[norm]
    W1 = [make_weight(init, d, d, rng) for _ in range(L)]
    # Output projection at 1/fan_in so one branch contributes ~1 unit of variance.
    W2 = [make_weight("lecun", d, d, rng) for _ in range(L)]
    h = rng.standard_normal((B, d))
    stream, ratio, tape = [h.var()], [], []
    for l in range(L):
        u, c_pre = (nf(h) if placement == "pre" else (h, None))
        z = u @ W1[l]
        m = z > 0
        a = z * m                                      # branch hidden activation
        r = a @ W2[l]
        ratio.append(r.var() / h.var())
        s = h + r
        c_post = None
        if placement == "post":
            s, c_post = nf(s)
        tape.append((a, m, c_pre, c_post, l))
        h = s
        stream.append(h.var())
    c_fin = None
    if placement == "pre" and final_norm:
        h, c_fin = nf(h)

    g = rng.standard_normal((B, d))                    # unit cotangent on the output
    if c_fin is not None:
        g = nb(g, c_fin)
    bwd, gw2 = [g.var()], []
    for (a, m, c_pre, c_post, l) in reversed(tape):
        if placement == "post":                        # identity path goes THROUGH norm
            g = nb(g, c_post)
        gw2.append(np.linalg.norm(a.T @ g))            # parameter gradient for W2[l]
        gb = ((g @ W2[l].T) * m) @ W1[l].T
        if placement == "pre":                         # norm sits only on the branch
            gb = nb(gb, c_pre)
        g = g + gb
        bwd.append(g.var())
    return (np.array(stream), np.array(bwd[::-1]),
            np.array(ratio), np.array(gw2[::-1]))

# ------------------------------------------------------------------------ verification
def check_vjps(rng):
    """Finite-difference both normalizer VJPs against a random scalar functional."""
    x = rng.standard_normal((4, 16)) * 1.7 + 0.3
    w = rng.standard_normal((4, 16))
    worst = 0.0
    for name in ("ln", "rms"):
        f, b = NORMS[name]
        ana = b(w, f(x)[1])                            # d/dx of sum(w * f(x))
        num, h = np.zeros_like(x), 1e-6
        for i in np.ndindex(x.shape):
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            num[i] = ((w * f(xp)[0]).sum() - (w * f(xm)[0]).sum()) / (2 * h)
        rel = np.abs(ana - num).max() / np.abs(num).max()
        worst = max(worst, rel)
        print(f"  {name:<4s} VJP vs central differences : max rel err {rel:.3e}")
    assert worst < 1e-7, worst
    if torch is None:
        print("  [skipped: torch not installed]")
        return
    xt = torch.tensor(x, requires_grad=True)
    (torch.nn.functional.layer_norm(xt, (16,), eps=EPS) * torch.tensor(w)).sum().backward()
    err = np.abs(xt.grad.numpy() - layernorm_bwd(w, layernorm_fwd(x)[1])).max()
    print(f"  ln   VJP vs torch F.layer_norm       : max abs err {err:.3e}")
    assert err < 1e-10, err

def table(title, header, rows, idx):
    print(f"\n{title}")
    print("  layer " + "".join(f"{h:>14s}" for h in header))
    for l in idx:
        print(f"  {l:5d} " + "".join(f"{r[l]:14.3e}" for r in rows))

# --------------------------------------------------------------------------------- main
if __name__ == "__main__":
    t0 = time.time()
    L, d, B = 50, 256, 64
    idx = [0, 10, 20, 30, 40, 50]
    print("=" * 76)
    print("Artifact 12.1 -- signal propagation, L=50, d=256, B=64, float64")
    print("=" * 76)

    print("\n[1] normalizer VJP verification")
    check_vjps(np.random.default_rng(1))

    print("\n[2] Plain ReLU stack, no residual, no norm.  Predicted per-layer")
    print("    second-moment gain is d*Var(W)/2 = 128 / 0.5 / 1.0.")
    inits = ["naive1", "xavier", "he"]
    fwd, bwd = {}, {}
    for k in inits:
        fwd[k], bwd[k] = plain_probe(k, L, d, B, np.random.default_rng(0))
    table("  forward activation variance", inits, [fwd[k] for k in inits], idx)
    table("  backward gradient variance", inits, [bwd[k] for k in inits], idx)

    print("\n  geometric mean per-layer forward gain (layers 1..50):")
    for k in inits:
        g = np.exp(np.log(fwd[k][L] / fwd[k][1]) / (L - 1))
        pred = {"naive1": d / 2, "xavier": 0.5, "he": 1.0}[k]
        print(f"    {k:<7s} measured {g:10.4f}   predicted {pred:10.4f}"
              f"   ratio {g / pred:6.3f}")

    he_ratio = bwd["he"].max() / bwd["he"].min()
    blow = bwd["naive1"][0] / bwd["naive1"][L]
    xav = bwd["xavier"][0] / bwd["xavier"][L]
    print(f"\n  He     : max/min backward variance over all 51 depths = {he_ratio:.3f}")
    print(f"  N(0,1) : backward variance at input / at output       = {blow:.3e}")
    print(f"  Xavier : backward variance at input / at output       = {xav:.3e}"
          f"  (2^-50 = {2.0 ** -50:.3e})")
    assert he_ratio < 5.0, he_ratio                    # He: bounded across 50 layers
    assert blow > 1e100, blow                          # naive: 100+ orders of magnitude
    assert 0.2 < xav / 2.0 ** -50 < 5.0, xav           # Xavier on ReLU loses 2x per layer

    print("\n[3] Residual net, He branches: normalization x placement")
    cfgs = [("none", "pre"), ("ln", "pre"), ("rms", "pre"),
            ("ln", "post"), ("rms", "post")]
    names = [f"{n}/{p}" for n, p in cfgs]
    st, bw, rt, gp = {}, {}, {}, {}
    for c in cfgs:
        st[c], bw[c], rt[c], gp[c] = residual_probe("he", c[0], c[1], L, d, B,
                                                    np.random.default_rng(0))
    table("  residual-stream variance", names, [st[c] for c in cfgs], idx)
    table("  backward gradient variance", names, [bw[c] for c in cfgs], idx)

    s = st[("ln", "pre")]
    slope, intercept = np.polyfit(np.arange(L + 1), s, 1)
    print(f"\n  pre-norm stream variance: linear fit {slope:.3f}*l + {intercept:.3f};"
          f"  Var(h_50)/Var(h_0) = {s[L] / s[0]:.2f}")
    r49 = rt[("ln", "pre")][L - 1]
    print(f"  pre-norm branch/stream variance ratio at l=49: {r49:.4f}"
          f"   (1/L = {1.0 / L:.4f})")
    assert 0.7 < slope < 1.6, slope                    # linear, not exponential
    assert s[L] / s[0] > 20.0, s[L] / s[0]
    assert 0.4 < r49 * L < 2.5, r49                    # branch contribution dies as 1/l

    print("\n  depth non-uniformity of the backward pass (max/min over the 51")
    print("  depths), worst of 3 seeds:")
    for c in cfgs:
        m = max(b.max() / b.min() for b in
                [residual_probe("he", c[0], c[1], L, d, B, np.random.default_rng(sd))[1]
                 for sd in range(3)])
        print(f"    {c[0] + '/' + c[1]:<10s} max/min = {m:14.4g}")
        # unnormalized residual stack is exponential in depth; every normalized
        # variant stays polynomial at initialization.
        assert (m > 1e6) if c[0] == "none" else (m < 5e2), (c, m)

    print("\n[4] The warmup argument.  ||dW2||_F for the TOP block, which is the step")
    print("    size the optimizer takes there at iteration 0:")
    top = {}
    for c in [("ln", "pre"), ("rms", "pre"), ("ln", "post"), ("rms", "post")]:
        top[c] = gp[c][L - 1]
        print(f"    {c[0] + '/' + c[1]:<10s} ||dW2^(50)||_F = {top[c]:.4f}"
              f"   bottom block {gp[c][0]:.4f}")
    r = top[("ln", "pre")] / top[("ln", "post")]
    print(f"\n  pre/post top-block gradient ratio = {r:.4f}"
          f"   (1/sqrt(L) = {L ** -0.5:.4f})")
    assert 0.5 < r * np.sqrt(L) < 2.5, r     # pre-norm's final LN divides by sqrt(L)

    print("\n  Same measurement with the final norm removed from the pre-norm net:")
    g_nofin = residual_probe("he", "ln", "pre", L, d, B, np.random.default_rng(0),
                             final_norm=False)[3]
    print(f"    ln/pre (no final norm) ||dW2^(50)||_F = {g_nofin[L - 1]:.4f}"
          f"   ratio to post-norm = {g_nofin[L - 1] / top[('ln', 'post')]:.3f}")
    assert g_nofin[L - 1] / top[("ln", "post")] > 3.0 * r, "final norm is the cause"

    print(f"\nall assertions passed in {time.time() - t0:.1f}s")
