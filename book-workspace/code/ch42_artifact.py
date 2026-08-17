"""Artifact 42.1 -- The research pitch as arithmetic.

Extends the Chapter 38 CKKS COUNTING MODEL (no ciphertext is ever created)
with a selective-SSM block and compares it against the attention sub-block
at matched width across T in {128, 512, 2048, 8192}.

Conventions inherited from Artifact 38.1: pt-ct mult = 1 level, ct-ct mult
= 1 level, rotations/additions = 0 levels; a degree-k polynomial via
Paterson-Stockmeyer costs ceil(log2(k+1)) levels; Goldschmidt reciprocal
and rsqrt cost 2 levels per iteration.  Depth here is the CRITICAL PATH
(parallel branches do not add), which is what CKKS actually charges.

Self-checks: (1) a NumPy Hillis-Steele scan for h_t = a_t h_{t-1} + b_t is
verified against the sequential recurrence, and a per-element depth tag is
tracked through the scan to confirm max depth == ceil(log2 T); (2) the
attention ledger reproduces the 28-level figure of Artifact 38.1; (3) the
crossover interval is computed and asserted, not eyeballed.
"""
import numpy as np
from math import ceil, log2, isqrt

# ------------------------------------------------------------- depth atoms
def poly_depth(k):   # Paterson-Stockmeyer depth of a degree-k polynomial
    return ceil(log2(k + 1))

def poly_mults(k):   # PS non-scalar mult count ~ sqrt(2k) + log2 k
    return ceil((2 * k) ** 0.5) + max(0, ceil(log2(k + 1)) - 1)

RSQ_D, RSQ_M = 2, 3          # Goldschmidt rsqrt: 2 levels, 3 mults / iter
REC_D, REC_M = 2, 2          # Goldschmidt reciprocal: 2 levels, 2 mults / iter

def rot_bsgs(d):             # BSGS rotations for one d x d pt-ct matvec
    n1 = isqrt(d)
    while d % n1:
        n1 -= 1
    return n1 + d // n1 - 2

# ---------------------------------------------------- attention sub-block
def attn_ledger(kx, r_ln, r_sm):
    """(label, levels) chain; the attention path is a single serial chain."""
    return [("norm variance (ct-ct square)",             1),
            ("norm rsqrt, %d Goldschmidt iters" % r_ln,  RSQ_D * r_ln),
            ("norm apply (ct-ct)",                       1),
            ("Q,K,V projections (pt-ct)",                1),
            ("scale by 1/sqrt(d_k) (pt)",                1),
            ("Q K^T (ct-ct matmul)",                     1),
            ("softmax exp poly deg %d" % kx,             poly_depth(kx)),
            ("softmax reciprocal, %d iters" % r_sm,      REC_D * r_sm),
            ("softmax normalize (ct-ct)",                1),
            ("attention @ V (ct-ct matmul)",             1),
            ("output projection W_O (pt-ct)",            1)]

def attn_depth(kx, r_ln, r_sm):
    return sum(v for _, v in attn_ledger(kx, r_ln, r_sm))

def softmax_levels(ledger):
    return sum(v for name, v in ledger if name.startswith("softmax"))

# ---------------------------------------------------- selective-SSM block
def ssm_ledger(T, kg, r_ln):
    """Two parallel branches after the shared trunk; depth = critical path.
    State path:  norm -> in_proj -> conv -> gating poly (decay a_t and
                 input gate, deg kg) -> scan (ceil(log2 T)) -> C (.) h.
    Gate path:   norm -> in_proj -> SiLU poly (deg kg) on z.
    Then y (.) gate (1 ct-ct) and out_proj (1 pt-ct).  NO softmax lines
    exist by construction: no exp over scores, no reciprocal, no normalize."""
    trunk = [("norm variance (ct-ct square)",            1),
             ("norm rsqrt, %d Goldschmidt iters" % r_ln, RSQ_D * r_ln),
             ("norm apply (ct-ct)",                      1),
             ("in_proj to (x, z) (pt-ct)",               1)]
    state = [("depthwise conv1d (pt-ct)",                1),
             ("gating poly (decay+input), deg %d" % kg,  poly_depth(kg)),
             ("parallel scan, ceil(log2 T) stages",      ceil(log2(T))),
             ("readout C (.) h (ct-ct)",                 1)]
    gate = [("SiLU poly on z, deg %d" % kg,              poly_depth(kg))]
    tail = [("gated product y (.) SiLU(z) (ct-ct)",      1),
            ("out_proj (pt-ct)",                         1)]
    return trunk, state, gate, tail

def ssm_depth(T, kg, r_ln):
    trunk, state, gate, tail = ssm_ledger(T, kg, r_ln)
    s = lambda part: sum(v for _, v in part)
    return s(trunk) + max(s(state), s(gate)) + s(tail)

# ----------------------------------- operation counts (rotations & mults)
def attn_ops(T, d, H, kx, r_ln, r_sm, slots):
    dk = d // H
    n_x = ceil(T * d / slots)               # activation ciphertexts
    n_s = ceil(H * T * T / slots)           # T x T score ciphertexts (!)
    n_kv = ceil(T * dk / slots)             # one head's K (or V)
    rot = 4 * n_x * rot_bsgs(d) + 2 * H * T * (ceil(log2(T)) + ceil(log2(dk)))
    mults = (2 * n_x + RSQ_M * r_ln                     # one norm
             + 2 * H * T * n_kv                         # QK^T and attn@V
             + poly_mults(kx) * n_s                     # exp on every score ct
             + REC_M * r_sm + n_s)                      # reciprocal + normalize
    return rot, mults, n_s

def ssm_ops(T, d, kg, r_ln, slots, d_state=16, conv_k=4):
    d_in = 2 * d                            # Mamba-style expansion factor 2
    n_x = ceil(T * d / slots)
    n_g = ceil(T * d_in / slots)            # gate/channel sequences
    n_h = ceil(T * d_in * d_state / slots)  # the scanned state sequence
    S = ceil(log2(T))
    rot = (5 * n_x * rot_bsgs(d)            # in/out/x-dependent projections
           + (conv_k - 1) * n_g             # depthwise conv shifts
           + S * n_h)                       # one rotation per ct per stage
    mults = (2 * n_x + RSQ_M * r_ln                     # norm
             + poly_mults(kg) * (2 * n_g)               # gating + SiLU polys
             + n_h                                      # b_t = Bbar (.) x write
             + 2 * S * n_h                              # scan: 2 mults / stage
             + n_h + n_g)                               # readout + gated product
    return rot, mults, n_h

# --------------------------------------------- scan verification (NumPy)
def scan_check(T, lanes=4, rng=None):
    """Hillis-Steele scan of f_t(h) = a_t h + b_t with per-element depth tags.
    Returns (max residual vs sequential loop, number of stages, max depth)."""
    a = 0.5 + 0.5 * rng.random((T, lanes))              # decays in (0.5, 1)
    b = rng.standard_normal((T, lanes)) * 0.1
    A, B, depth, s, stages = a.copy(), b.copy(), np.zeros(T, int), 1, 0
    while s < T:
        A2, B2, d2 = A.copy(), B.copy(), depth.copy()
        A2[s:] = A[s:] * A[:-s]                         # compose F_t o F_{t-s}
        B2[s:] = A[s:] * B[:-s] + B[s:]                 # (2 mults, in parallel
        d2[s:] = np.maximum(depth[s:], depth[:-s]) + 1  #  => +1 level)
        A, B, depth, s, stages = A2, B2, d2, 2 * s, stages + 1
    h_seq, h = np.zeros(lanes), np.zeros((T, lanes))    # sequential reference
    for t in range(T):
        h_seq = a[t] * h_seq + b[t]
        h[t] = h_seq
    return np.abs(B - h).max(), stages, depth.max()     # h0 = 0 => h_t = B_t

# ------------------------------------------------------------------- main
if __name__ == "__main__":
    d, H, slots, r_ln, kg = 768, 12, 2 ** 15, 4, 15
    grid = (128, 512, 2048, 8192)
    scenarios = [("modest softmax  (exp deg 15, 4 recip iters)", 15, 4),
                 ("faithful softmax (exp deg 63, 6 recip iters)", 63, 6)]

    print("ASSUMPTIONS (every number below follows from these):")
    for line in [
        "counting model only -- no FHE library, nothing is encrypted",
        "pt-ct mult / ct-ct mult = 1 level; rotations and adds = 0 levels",
        "degree-k poly = ceil(log2(k+1)) levels (Paterson-Stockmeyer)",
        "Goldschmidt: rsqrt & reciprocal = 2 levels/iter; LayerNorm/RMSNorm "
        "use %d iters" % r_ln,
        "depth = critical path; parallel branches meet at max(.)+1",
        "matched width d=%d, H=%d heads, slots=%d; SSM: expand 2x, "
        "d_state=16, conv 4, gate polys deg %d" % (d, H, slots, kg),
        "attention softmax degree held FIXED across T (Ch. 39: the honest "
        "degree grows with T; that shifts the crossover toward the SSM)"]:
        print("  - " + line)

    print("\n=== Scan self-check: parallel == sequential, depth == log2 T ===")
    rng = np.random.default_rng(0)
    for T in grid:
        res, stages, dmax = scan_check(T, rng=rng)
        print("T=%5d  stages=%2d  max depth tag=%2d  residual=%.2e"
              % (T, stages, dmax, res))
        assert stages == dmax == ceil(log2(T)) and res < 1e-9

    a28 = attn_depth(15, r_ln, 4)
    assert a28 == 28, a28                     # reproduces Artifact 38.1 ledger
    ssm_const = ssm_depth(128, kg, r_ln) - ceil(log2(128))
    for T in grid:                            # HARD ASSERTION 1: shape of both
        assert ssm_depth(T, kg, r_ln) == ssm_const + ceil(log2(T))
        assert attn_depth(15, r_ln, 4) == a28 # T-independent
    print("\nattention sub-block depth (modest): %d levels, T-independent"
          % a28)
    print("selective-SSM block depth: %d + ceil(log2 T) levels" % ssm_const)

    # HARD ASSERTION 2: zero softmax levels in the SSM, by construction.
    trunk, state, gate, tail = ssm_ledger(2048, kg, r_ln)
    ssm_sm = softmax_levels(trunk + state + gate + tail)
    at_sm = softmax_levels(attn_ledger(15, r_ln, 4))
    print("softmax-related levels: attention %d, SSM %d" % (at_sm, ssm_sm))
    assert ssm_sm == 0 and at_sm == poly_depth(15) + REC_D * 4 + 1 == 13

    print("\n=== Comparison table (depth | rotations | ct-ct mults) ===")
    hdr = ("%6s %9s %9s | %11s %11s | %11s %11s | %8s %8s"
           % ("T", "attnDep", "ssmDep", "attnRot", "ssmRot",
              "attnMult", "ssmMult", "scoreCt", "stateCt"))
    for label, kx, r_sm in scenarios:
        print("\n-- scenario: %s --\n%s" % (label, hdr))
        ad = attn_depth(kx, r_ln, r_sm)
        for T in grid:
            sd = ssm_depth(T, kg, r_ln)
            ar, am, ns = attn_ops(T, d, H, kx, r_ln, r_sm, slots)
            sr, sm, nh = ssm_ops(T, d, kg, r_ln, slots)
            print("%6d %9d %9d | %11d %11d | %11d %11d | %8d %8d"
                  % (T, ad, sd, ar, sr, am, sm, ns, nh))

    # HARD ASSERTION 3: the crossover interval, per scenario, computed.
    def crossover(kx, r_sm):
        ad = attn_depth(kx, r_ln, r_sm)
        c = ad - ssm_const                    # SSM wins iff ceil(log2 T) < c
        return 2 ** (c - 1), 2 ** c           # strict-win bound, tie bound
    w1, t1 = crossover(15, 4)
    w2, t2 = crossover(63, 6)
    print("\ncrossover (modest): SSM wins depth for T <= %d, tie through "
          "T = %d, attention wins for T >= %d" % (w1, t1, t1 + 1))
    print("crossover (faithful): SSM wins depth for T <= %d, tie through "
          "T = %d, attention wins for T >= %d" % (w2, t2, t2 + 1))
    assert (w1, t1) == (256, 512) and (w2, t2) == (16384, 32768)
    assert ssm_depth(512, kg, r_ln) == attn_depth(15, r_ln, 4) == 28
    assert ssm_depth(128, kg, r_ln) < 28 < ssm_depth(2048, kg, r_ln)
    for T in grid:                            # faithful softmax: SSM sweeps
        assert ssm_depth(T, kg, r_ln) <= attn_depth(63, r_ln, 6)
    # Op counts: the SSM advantage grows with T (no T x T score object).
    ratios = [attn_ops(T, d, H, 15, r_ln, 4, slots)[1]
              / ssm_ops(T, d, kg, r_ln, slots)[1] for T in grid]
    print("ct-ct mult ratio attention/SSM across T: "
          + ", ".join("%.1f" % r for r in ratios))
    # ceil() packing effects make small-T ratios lumpy; from T=512 on the
    # ratio grows monotonically and exceeds 10x at T=8192.
    assert all(x < y for x, y in zip(ratios[1:], ratios[2:])) and ratios[-1] > 10

    print("\n=== Three concrete candidate problems (and the chapters that "
          "equip you) ===")
    for c in ["1. Build the encrypted selective-SSM block: a Mamba-style "
              "block under RNS-CKKS,\n   benchmarked against a polynomial "
              "attention block at matched width.\n   [Chapters 17, 21, 38, 39]",
              "2. Calibrate the crossover: turn this counting model into a "
              "measured cost atlas\n   over (T, d, degrees) with FHE-library "
              "microbenchmarks. [Chapters 24, 30, 38, 40, 41]",
              "3. Encrypted PEFT on an HE-friendly backbone: LoRA updates on "
              "the SSM block,\n   gradients under encryption. [Chapters 11, "
              "26, 34, 38, 41]"]:
        print(c)
    print("\nHonest conclusion: which block wins on DEPTH depends on T and "
          "on the softmax\ndegree you believe (Ch. 39); the SSM wins on "
          "operations and memory at long T\nby construction. The point is "
          "that the comparison is computable -- and the\nencrypted-SSM "
          "column of it has not been published at transformer scale.")
    print("All assertions passed.")
