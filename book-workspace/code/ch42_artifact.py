"""Artifact 42.1 -- the research pitch as arithmetic: an attention block and a
selective-SSM block, priced in CKKS depth and rotations at matched width.

COUNTING MODEL in the sense of Chapter 38: nothing is encrypted, no FHE
library is imported, every number is a longest-path count in a labelled
circuit -- not a measurement.  Two extensions of the Chapter 38 ledger:
depth becomes a longest path through an explicit DAG (so parallel branches
are charged max(parents)+cost, not a sum), and a selective-SSM block is
added -- associative scan of depth ceil(log2 T), input-dependent gating
polynomials, no softmax.  Self-checks: DAG longest path vs. brute-force
path enumeration; exact reproduction of Chapter 38's hand ledger; and a
NumPy Kogge-Stone scan (np.roll for slot rotation) checked against the
sequential recurrence with multiplicative depth tracked per slot.
Conventions (Chapter 38): pt-ct = ct-ct mult = 1 level; rotation and
addition = 0; degree-k polynomial = ceil(log2(k+1)) levels; Goldschmidt
reciprocal and rsqrt = 2 levels per iteration.
"""
import numpy as np
from math import ceil, log2, isqrt

def poly_depth(k):                 # Paterson-Stockmeyer depth, degree k
    return ceil(log2(k + 1))

GOLD = 2                           # Goldschmidt levels per iteration

# ------------------------------------------------------------- DAG machinery
# A block is a dict: name -> (levels, [parents], category).  Sources: no parents.
def dag_depth(nodes):
    memo = {}
    def d(n):
        if n not in memo:
            lv, par, _ = nodes[n]
            memo[n] = lv + (max(d(p) for p in par) if par else 0)
        return memo[n]
    return max(d(n) for n in nodes)

def dag_depth_bruteforce(nodes):
    """Every root-to-node path, enumerated. Exponential; these graphs are tiny."""
    best = 0
    def walk(n, acc):
        nonlocal best
        acc += nodes[n][0]
        best = max(best, acc)
        for p in nodes[n][1]:
            walk(p, acc)
    for n in nodes:
        walk(n, 0)
    return best

def cat_levels(nodes, cat):
    return sum(lv for lv, _, c in nodes.values() if c == cat)

# ---------------------------------------------------------- block circuits
def _norm(r):                      # LayerNorm / RMSNorm: square, rsqrt, apply
    return {"ln_var": (1, [], "norm"), "ln_rsqrt": (GOLD * r, ["ln_var"], "norm"),
            "ln_apply": (1, ["ln_rsqrt"], "norm")}

def attention_block(cfg):
    r, kx, ri = cfg["norm_iters"], cfg["exp_deg"], cfg["recip_iters"]
    return dict(_norm(r), **{
        "Wq":       (1, ["ln_apply"], "matmul"),
        "Wk":       (1, ["ln_apply"], "matmul"),
        "Wv":       (1, ["ln_apply"], "matmul"),
        "scale":    (1, ["Wq"], "matmul"),              # 1/sqrt(d_k)
        "QKt":      (1, ["scale", "Wk"], "matmul"),     # ct-ct: acts x acts
        "sm_exp":   (poly_depth(kx),  ["QKt"],    "softmax"),
        "sm_recip": (GOLD * ri,       ["sm_exp"], "softmax"),
        "sm_norm":  (1, ["sm_recip", "sm_exp"],   "softmax"),
        "AV":       (1, ["sm_norm", "Wv"], "matmul"),
        "Wo":       (1, ["AV"], "matmul")})

def mlp_block(cfg):
    return dict(_norm(cfg["norm_iters"]), **{
        "W_up":   (1, ["ln_apply"], "matmul"),
        "gelu":   (poly_depth(cfg["gelu_deg"]), ["W_up"], "activation"),
        "W_down": (1, ["gelu"], "matmul")})

def ssm_block(cfg, T):
    """Selective (Mamba-style) block; one such block replaces attention AND MLP.
    RMSNorm -> in_proj -> depthwise causal conv -> SiLU -> selective Delta,B,C
    -> discretise -> parallel scan -> output gate -> out_proj."""
    kg = cfg["ssm_gate_deg"]
    scan = ceil(log2(T)) * (1 + cfg["scan_mask_levels"])
    return dict(_norm(cfg["norm_iters"]), **{
        "in_proj_x": (1, ["ln_apply"], "matmul"),
        "in_proj_z": (1, ["ln_apply"], "matmul"),       # gate branch, parallel
        "conv1d":    (1, ["in_proj_x"], "matmul"),      # plaintext depthwise kernel
        "silu_x":    (poly_depth(kg), ["conv1d"], "activation"),
        "silu_z":    (poly_depth(kg), ["in_proj_z"], "activation"),
        "dt_proj":   (1, ["silu_x"], "matmul"),
        "softplus":  (poly_depth(cfg["softplus_deg"]), ["dt_proj"], "activation"),
        "dt_A":      (1, ["softplus"], "scan"),         # Delta * diag(A), pt-ct
        "Abar_exp":  (poly_depth(cfg["ssm_exp_deg"]), ["dt_A"], "activation"),
        "B_proj":    (1, ["silu_x"], "matmul"),
        "C_proj":    (1, ["silu_x"], "matmul"),
        "Bbar":      (1, ["softplus", "B_proj"], "scan"),   # Delta (.) B
        "Bx":        (1, ["Bbar", "silu_x"], "scan"),       # (.) x_t
        "scan":      (scan, ["Abar_exp", "Bx"], "scan"),    # Kogge-Stone
        "y_C":       (1, ["scan", "C_proj"], "scan"),
        "gate":      (1, ["y_C", "silu_z"], "activation"),
        "out_proj":  (1, ["gate"], "matmul")})

# ----------------------------------------------------- rotations (Chapter 38)
def rot_bsgs(d):
    n1 = isqrt(d)
    while d % n1:
        n1 -= 1
    return n1 + d // n1 - 2

def rotations(cfg, T, kind):
    d, H, s = cfg["d"], cfg["heads"], cfg["slots"]
    n_x = ceil(T * d / s)
    if kind == "attention":                       # 12 dxd matmul equivalents
        w, cc = 12 * n_x * rot_bsgs(d), 2 * H * T * (ceil(log2(T)) +
                                                     ceil(log2(d // H)))
    else:                                         # in_proj x2 + out_proj
        w, cc = 3 * n_x * rot_bsgs(d), 2 * n_x * ceil(log2(T))
    return w + cc, w, cc

# ------------------------------------- simulate the scan, and its depth, in NumPy
def kogge_stone_scan(A, b):
    """Inclusive scan of the affine monoid (A1,b1)o(A2,b2) = (A2 A1, A2 b1 + b2),
    np.roll standing in for slot rotation.  Slots [T,2T) hold the monoid identity
    (1,0), so the cyclic wrap brings in identities and no masking multiply is
    needed (cost: 2x the slot footprint).  Returns (h, rotations, depth)."""
    T = len(A)
    Ap = np.concatenate([A, np.ones(T)])
    bp = np.concatenate([b, np.zeros(T)])
    dep, rots = np.zeros(2 * T), 0
    for j in range(ceil(log2(T))):
        sh = 1 << j
        Ash, bsh = np.roll(Ap, sh), np.roll(bp, sh)
        rots += 2                                        # one rotation per lane
        dep = np.maximum(dep, np.roll(dep, sh)) + 1      # one ct-ct mult / step
        Ap, bp = Ap * Ash, Ap * bsh + bp
    return bp[:T], rots, int(dep[:T].max())

def sequential_scan(A, b):
    h, out = 0.0, []
    for t in range(len(A)):
        h = A[t] * h + b[t]
        out.append(h)
    return np.array(out)

# --------------------------------------------------------------------- pricing
BASE = dict(d=768, heads=12, slots=2 ** 15, norm_iters=4, recip_iters=4,
            exp_deg=15, gelu_deg=15, ssm_gate_deg=15, softplus_deg=15,
            ssm_exp_deg=15, scan_mask_levels=0)

def attn_depth(cfg):
    return dag_depth(attention_block(cfg)) + dag_depth(mlp_block(cfg))

def ssm_depth(cfg, T):
    return dag_depth(ssm_block(cfg, T))

def crossover(cfg):
    """Largest power-of-two T at which the SSM block is still no deeper."""
    k = 1 + cfg["scan_mask_levels"]
    const = ssm_depth(cfg, 2) - k                   # depth = const + k*ceil(log2 T)
    return 2 ** ((attn_depth(cfg) - const) // k), const

if __name__ == "__main__":
    cfg, Ts = dict(BASE), (128, 512, 2048, 8192)
    print("=== Assumptions (all of them; nothing below is measured) ===")
    print("  " + " | ".join("%s=%s" % (k, cfg[k]) for k in sorted(cfg)))
    print("  pt-ct mult = ct-ct mult = 1 level; rotation and addition = 0 levels")
    print("  degree-k polynomial = ceil(log2(k+1)) levels (Paterson-Stockmeyer)")
    print("  Goldschmidt reciprocal and rsqrt = 2 levels per iteration")
    print("  scan = Kogge-Stone, ceil(log2 T) ct-ct steps, identity-padded (no mask)")

    a_n, m_n, s_n = attention_block(cfg), mlp_block(cfg), ssm_block(cfg, 2048)
    print("\n=== Check 1: longest path, memoised vs. brute-force enumeration ===")
    for nm, nd in (("attention", a_n), ("mlp", m_n), ("ssm(T=2048)", s_n)):
        f, g = dag_depth(nd), dag_depth_bruteforce(nd)
        print("  %-12s memoised %3d   brute force %3d   agree %s" % (nm, f, g, f == g))
        assert f == g

    a_d, m_d = dag_depth(a_n), dag_depth(m_n)
    print("\n=== Check 2: reproduces Chapter 38's hand ledger ===")
    print("  attention %d (Ch.38: 28) | MLP %d (16) | block %d (44)"
          % (a_d, m_d, a_d + m_d))
    assert (a_d, m_d) == (28, 16)

    print("\n=== Check 3: simulated Kogge-Stone scan (np.roll for rotations) ===")
    rng = np.random.default_rng(0)
    for T in (128, 512, 2048):
        A, b = 0.5 + 0.4 * rng.random(T), rng.standard_normal(T)
        h, rots, dep = kogge_stone_scan(A, b)
        res = float(np.max(np.abs(h - sequential_scan(A, b))))
        print("  T=%5d  residual vs. sequential %.3e | rotations %2d | depth %2d "
              "(ceil(log2 T)=%d)" % (T, res, rots, dep, ceil(log2(T))))
        assert res < 1e-9 and dep == ceil(log2(T)) and rots == 2 * ceil(log2(T))

    print("\n=== Depth and rotations, matched width d=768, 12 heads ===")
    print("  (a Mamba-style block replaces attention AND the MLP, so the "
          "attention column is attn+MLP)")
    print("  %6s %10s %8s %9s %12s %11s %8s"
          % ("T", "attn+MLP", "SSM", "SSM-attn", "attn rot", "SSM rot", "rot cut"))
    for T in Ts:
        ad, sd = attn_depth(cfg), ssm_depth(cfg, T)
        at, st = rotations(cfg, T, "attention")[0], rotations(cfg, T, "ssm")[0]
        print("  %6d %10d %8d %+9d %12d %11d %7.0fx"
              % (T, ad, sd, sd - ad, at, st, at / st))

    print("\n=== Assertion 1: SSM depth = const + ceil(log2 T); attention is T-free ===")
    dp = [ssm_depth(cfg, 2 ** i) for i in range(1, 9)]
    print("  SSM depth, T=2..256: %s -> first differences %s"
          % (dp, [dp[i + 1] - dp[i] for i in range(7)]))
    assert all(dp[i + 1] - dp[i] == 1 for i in range(7))
    const = dp[0] - 1
    assert all(ssm_depth(cfg, T) == const + ceil(log2(T)) for T in Ts)
    assert len({attn_depth(dict(cfg, **{})) for _ in Ts}) == 1
    print("  closed form: SSM = %d + ceil(log2 T); attention = %d for every T  [OK]"
          % (const, attn_depth(cfg)))

    print("\n=== Assertion 2: the crossover interval ===")
    Tstar, c2 = crossover(cfg)
    print("  SSM no deeper than attention for T <= %d; attention wins for T > %d"
          % (Tstar, Tstar))
    assert c2 == const
    assert ssm_depth(cfg, Tstar) <= attn_depth(cfg) < ssm_depth(cfg, 2 * Tstar)
    assert all(ssm_depth(cfg, T) < attn_depth(cfg) for T in Ts)
    cm = dict(cfg, scan_mask_levels=1)               # if masking is unavoidable
    Tm = crossover(cm)[0]
    print("  if each scan step also needs a masking multiply, crossover falls to "
          "T = %d (SSM depth at T=2048: %d)" % (Tm, ssm_depth(cm, 2048)))
    assert 128 <= Tm < Tstar

    print("\n=== Assertion 3: the SSM block spends zero softmax levels ===")
    sm = cat_levels(a_n, "softmax")
    print("  attention: exp %d + reciprocal %d + normalise 1 = %d of %d critical-path "
          "levels" % (poly_depth(cfg["exp_deg"]), GOLD * cfg["recip_iters"], sm, a_d))
    print("  SSM: %d  (no node of category 'softmax' exists in the circuit)"
          % cat_levels(s_n, "softmax"))
    assert cat_levels(s_n, "softmax") == 0 and sm == 13 and sm / a_d > 0.45

    print("\n=== The crossover depends on the softmax budget, not on ideology ===")
    print("  %8s %9s %11s %10s %12s"
          % ("exp deg", "recip it", "attn+MLP", "SSM const", "crossover T"))
    for kx, ri in ((3, 2), (7, 2), (15, 4), (31, 4), (63, 6)):
        c = dict(cfg, exp_deg=kx, recip_iters=ri)
        ts, cc = crossover(c)
        print("  %8d %9d %11d %10d %12d" % (kx, ri, attn_depth(c), cc, ts))
        assert ts >= 512

    print("\n=== Three candidate problems, and the chapters that equip you ===")
    for i, (t, ch, why) in enumerate([
        ("Non-interactive CKKS selective-SSM block, benchmarked", "21, 38, 39, 41",
         "published encrypted SSMs use PUBLIC (input-independent) decay; the "
         "selective gate is the open half"),
        ("Routing-oblivious MoE under pure FHE, no MPC fallback", "22, 39, 40",
         "published private-MoE work is an HE+MPC hybrid; non-interactive top-k "
         "is unsolved"),
        ("Depth-aware scheme-switch placement as an optimisation problem",
         "34, 39, 40",
         "CKKS<->FHEW switching ships in OpenFHE; WHERE to switch is chosen by "
         "hand everywhere")]):
        print("  %d. %s\n     chapters %s -- %s" % (i + 1, t, ch, why))

    print("\nAll assertions passed. Every number above is a longest-path count in a\n"
          "labelled circuit; nothing was encrypted and nothing was timed.")
