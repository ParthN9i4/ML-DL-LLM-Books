"""Artifact 38.1 -- A CKKS cost model for transformer inference.

This is a COUNTING MODEL, not an FHE implementation: it never encrypts
anything.  It charges every operation in the depth/rotation/ciphertext
currency of Chapter 38 and checks itself three ways:
  (1) the block depth matches a hand-computed ledger,
  (2) the BSGS rotation formula matches an actual simulated Halevi-Shoup
      matrix-vector product in NumPy (rotations counted one by one),
  (3) scaling laws (linear in L, T-independent depth) hold exactly.
Conventions (stated in Section 38.4): pt-ct mult = 1 level, ct-ct mult
= 1 level, rotation/addition = 0 levels; degree-k poly via
Paterson-Stockmeyer = ceil(log2(k+1)) levels; Goldschmidt reciprocal and
inverse-sqrt = 2 levels per iteration (2 resp. 3 ct-ct mults per iter).
"""
import numpy as np
from math import ceil, log2, isqrt

# ---------------------------------------------------------------- depth atoms
def poly_depth(k):        # Paterson-Stockmeyer depth for a degree-k poly
    return ceil(log2(k + 1))

def poly_mults(k):        # PS non-scalar mult count, ~ sqrt(2k) + log2 k
    return ceil((2 * k) ** 0.5) + max(0, ceil(log2(k + 1)) - 1)

REC_D, REC_M = 2, 2       # Goldschmidt reciprocal: depth 2, 2 muls / iter
RSQ_D, RSQ_M = 2, 3       # coupled Goldschmidt rsqrt: depth 2, 3 muls / iter

# ------------------------------------------------------------------ the ledger
def block_ledger(cfg):
    """Per-block (label, levels) list for one pre-norm transformer block."""
    r, kx, kg = cfg["newton_iters"], cfg["exp_deg"], cfg["gelu_deg"]
    ln = [("  LayerNorm: variance (ct-ct square)", 1),
          ("  LayerNorm: rsqrt, %d Goldschmidt iters" % r, RSQ_D * r),
          ("  LayerNorm: apply (x-mu) * rsqrt (ct-ct)", 1)]
    attn = ln + [
        ("  Q,K,V projections (pt-ct matmul)", 1),
        ("  scale by 1/sqrt(d_k) (pt mult, foldable)", 1),
        ("  Q K^T (ct-ct matmul)", 1),
        ("  softmax: exp poly, degree %d" % kx, poly_depth(kx)),
        ("  softmax: reciprocal, %d Goldschmidt iters" % r, REC_D * r),
        ("  softmax: normalize (ct-ct)", 1),
        ("  attention @ V (ct-ct matmul)", 1),
        ("  output projection W_O (pt-ct matmul)", 1)]
    mlp = ln + [
        ("  W_up (pt-ct matmul)", 1),
        ("  GELU poly, degree %d" % kg, poly_depth(kg)),
        ("  W_down (pt-ct matmul)", 1)]
    return attn, mlp

def block_depth(cfg):
    attn, mlp = block_ledger(cfg)
    return sum(v for _, v in attn), sum(v for _, v in mlp)

def total_depth(cfg):                      # residuals/rotations cost 0 levels
    a, m = block_depth(cfg)
    return cfg["L"] * (a + m)

# ------------------------------------------------------- rotations & ciphertexts
def rot_naive(d):                          # Halevi-Shoup: one rot per diagonal
    return d - 1

def rot_bsgs(d):                           # n1 baby + n2 giant, n1*n2 = d
    n1 = isqrt(d)
    while d % n1:                          # largest factor <= sqrt(d)
        n1 -= 1
    return n1 + d // n1 - 2                # = 2*sqrt(d) - 2 when d is square

def block_costs(cfg, packing="bsgs"):
    """Rotations, ct-ct mults, and ciphertext counts for ONE block."""
    d, H, T, s = cfg["d"], cfg["heads"], cfg["T"], cfg["slots"]
    dk, r = d // H, cfg["newton_iters"]
    n_x = ceil(T * d / s)                  # activation ciphertexts
    n_h = ceil(T * 4 * d / s)              # MLP hidden ciphertexts
    n_s = ceil(H * T * T / s)              # attention-score ciphertexts
    n_kv = ceil(T * dk / s)                # one head's K (or V) ciphertexts
    rot1 = {"naive": rot_naive, "bsgs": rot_bsgs}[packing](d)
    # 12 d x d pt-ct matmul equivalents: Wq,Wk,Wv,Wo + 4 for W_up + 4 for W_down
    rot_w = 12 * n_x * rot1
    # ct-ct matmuls, naive per-row model: each of T query rows per head needs a
    # log2(T) broadcast and a log2(dk) rotate-reduce; same under either packing.
    rot_cc = 2 * H * T * (ceil(log2(T)) + ceil(log2(dk)))
    mults = (2 * (n_x + RSQ_M * r + n_x)             # two LayerNorms
             + H * T * n_kv                          # Q K^T
             + poly_mults(cfg["exp_deg"]) * n_s      # exp on scores
             + REC_M * r + n_s                       # reciprocal + normalize
             + H * T * n_kv                          # attn @ V
             + poly_mults(cfg["gelu_deg"]) * n_h)    # GELU on hidden
    return dict(rot=rot_w + rot_cc, rot_weights=rot_w, rot_cc=rot_cc,
                ctct_mults=mults, cts=dict(acts=n_x, scores=n_s, hidden=n_h))

# --------------------------------------- simulate Halevi-Shoup to verify counts
def hs_matvec(A, v, bsgs=False):
    """Diagonal-method A @ v with np.roll standing in for slot rotation.
    Returns (result, rotations actually performed). Diagonal extraction and
    plaintext pre-rotations are free (they happen on plaintext)."""
    d = len(v); rots = 0
    diag = lambda i: np.array([A[t, (t + i) % d] for t in range(d)])
    if not bsgs:
        acc = diag(0) * v
        for i in range(1, d):
            rots += 1                              # rotate v by i (hoisted: 1 ea.)
            acc = acc + diag(i) * np.roll(v, -i)
        return acc, rots
    n1 = isqrt(d); n2 = d // n1; assert n1 * n2 == d
    baby = [v]
    for k in range(1, n1):
        rots += 1
        baby.append(np.roll(v, -k))                # n1 - 1 baby rotations
    acc = np.zeros(d)
    for j in range(n2):
        inner = sum(np.roll(diag(j * n1 + k), j * n1) * baby[k]
                    for k in range(n1))            # plaintext pre-rotation: free
        if j:
            rots += 1                              # n2 - 1 giant rotations
            inner = np.roll(inner, -j * n1)
        acc = acc + inner
    return acc, rots

# ------------------------------------------------------------------------ main
if __name__ == "__main__":
    cfg = dict(L=12, d=768, heads=12, T=128, exp_deg=15, gelu_deg=15,
               newton_iters=4, slots=2 ** 15)      # slots = N/2 at N = 2^16
    attn, mlp = block_ledger(cfg)
    a_d, m_d = block_depth(cfg)

    print("=== Depth ledger: one pre-norm block (deg-15 polys, 4 iters) ===")
    run = 0
    for name, lv in attn + mlp:
        run += lv
        print("%-52s %3d  (cum %3d)" % (name, lv, run))
    print("attention sub-block: %d levels | MLP sub-block: %d | block: %d"
          % (a_d, m_d, a_d + m_d))
    HAND = [1, 8, 1, 1, 1, 1, 4, 8, 1, 1, 1]       # hand ledger, attention
    assert a_d == sum(HAND) == 28 and 26 <= a_d <= 30, a_d
    assert a_d + m_d == 44

    print("\n=== Depth scaling (independent of T; linear in L) ===")
    print("%6s %12s %18s %s" % ("L", "depth", "vs 40-lvl budget",
                                "bootstraps @12 lvl/refresh"))
    for L in (1, 2, 12, 32):
        dep = total_depth({**cfg, "L": L})
        boots = 0 if dep <= 40 else ceil((dep - 40) / 12)
        print("%6d %12d %17.1fx %d" % (L, dep, dep / 40, boots))
    assert total_depth({**cfg, "L": 32}) == 32 * total_depth({**cfg, "L": 1})
    assert total_depth(cfg) == total_depth({**cfg, "T": 2048})   # T-free
    assert total_depth({**cfg, "L": 32}) >= 10 * 40              # >=10x budget

    print("\n=== Rotations: d x d pt-ct matmul, per packed matvec ===")
    for d in (768, 4096):
        n, b = rot_naive(d), rot_bsgs(d)
        print("d=%5d  naive %5d   BSGS %4d   ratio %5.1fx" % (d, n, b, n / b))
    assert rot_bsgs(4096) == 2 * 64 - 2 == 126 and rot_naive(4096) == 4095
    assert rot_naive(4096) / rot_bsgs(4096) > 0.9 * (4096 ** 0.5) / 2

    print("\n=== Simulated Halevi-Shoup matvec (d=256): counts vs formula ===")
    rng = np.random.default_rng(0)
    A, v = rng.standard_normal((256, 256)), rng.standard_normal(256)
    for flag, fn in ((False, rot_naive), (True, rot_bsgs)):
        y, rots = hs_matvec(A, v, bsgs=flag)
        res = np.max(np.abs(y - A @ v))
        print("%s: %3d rotations (formula %3d), residual %.2e"
              % ("BSGS " if flag else "naive", rots, fn(256), res))
        assert rots == fn(256) and res < 1e-9

    print("\n=== Per-block operation counts (BERT-base shape, T=128) ===")
    for p in ("naive", "bsgs"):
        c = block_costs(cfg, p)
        print("%-5s rotations %7d (weights %6d + ct-ct matmul %5d)"
              % (p, c["rot"], c["rot_weights"], c["rot_cc"]))
    c = block_costs(cfg, "bsgs")
    print("ct-ct mults per block: %d | ciphertexts: %s"
          % (c["ctct_mults"], c["cts"]))
    print("\nAll assertions passed. This is a counting model; no ciphertext "
          "was harmed (or created).")
