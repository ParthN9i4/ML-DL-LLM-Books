"""
ex30 — An encrypted linear layer: packing, rotations, and depth.
(Book: Chapters 38 and 41)

There is NO FHE library here. This is pure numpy, and it models the SLOT ALGEBRA
and the DEPTH LEDGER of a CKKS-style scheme — not the cryptography. No RLWE
sampling, no noise growth, no rescaling of moduli, no security of any kind. What
is faithful is the interface, and the interface is the whole engineering problem:
a ciphertext is a vector of n slots, and you are allowed exactly three moves.

    add          elementwise, free
    multiply     elementwise, costs ONE LEVEL from a budget fixed at keygen
    rotate       cyclic shift of the slots by k, costs no level but is expensive

Everything else must be built from those three. You cannot index a slot. You
cannot compare. You cannot sum a vector — you must synthesize the sum out of
rotations. That single constraint is what makes an encrypted matrix-vector
product an interesting problem rather than a call to `@`.

This exercise builds the same d x d matrix-vector product three ways and counts
what each one actually spends. The naive row-wise version pays log2(n) rotations
per inner product, so d*log2(n) in total; the diagonal (Halevi-Shoup) method pays
d-1; baby-step/giant-step pays about 2*sqrt(d). All three agree with numpy to
machine precision, and the whole lesson is in the counters, because rotations —
not multiplications — are the scarce resource in a real deployment.

Then the ledger, in the style of ex29: every level a full layer consumes,
including the level a PLAINTEXT multiply costs, which is widely and wrongly
assumed free. Scheduling that plaintext multiply one step earlier buys a whole
extra layer out of the same budget, and the file proves it by running out.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


# ---------------------------------------------------------------------------
# The machine: a ciphertext, three operations, and a meter that watches.
# ---------------------------------------------------------------------------

class DepthExhausted(RuntimeError):
    """Raised when an operation needs a level and the budget has none left."""

    def __init__(self, op):
        super().__init__(f"no levels left for: {op}")
        self.op = op


class Ctxt:
    """n slots of float64 plus a remaining-level counter. That is all a
    ciphertext is, as far as the arithmetic of a leveled scheme is concerned."""

    __slots__ = ("slots", "level")

    def __init__(self, slots, level):
        self.slots = np.asarray(slots, dtype=np.float64)
        self.level = int(level)

    @property
    def n(self):
        return self.slots.size


class Meter:
    """Counts rotations, ciphertext-ciphertext and plaintext multiplications,
    and records a ledger line for every labelled stage."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.rotations = 0
        self.ct_mults = 0
        self.pt_mults = 0
        self.ledger = []          # (label, levels_charged, level_remaining)

    def spend(self, label, level_after):
        if label:
            self.ledger.append((label, 1, level_after))

    def free(self, label, level_now):
        if label:
            self.ledger.append((label, 0, level_now))


METER = Meter()


def encrypt(vec, level, n=None):
    """Pack a plaintext vector into the first slots of a fresh ciphertext."""
    v = np.asarray(vec, dtype=np.float64)
    n = n or v.size
    slots = np.zeros(n)
    slots[:v.size] = v
    return Ctxt(slots, level)


def decrypt(ct):
    return ct.slots.copy()


def ct_add(a, b):
    """Elementwise add. Free. Operands at different levels are mod-switched
    down to the lower one, which is also free — hence the min()."""
    # === YOUR CODE HERE ===
    return Ctxt(a.slots + b.slots, min(a.level, b.level))


def ct_add_plain(a, v):
    """Add a plaintext vector. Free, and does not touch the level."""
    # === YOUR CODE HERE ===
    return Ctxt(a.slots + np.asarray(v, dtype=np.float64), a.level)


def ct_mul(a, b, label=None):
    """Ciphertext x ciphertext, elementwise. Costs one level."""
    # === YOUR CODE HERE ===
    lvl = min(a.level, b.level)
    if lvl <= 0:
        raise DepthExhausted(label or "ciphertext x ciphertext multiply")
    METER.ct_mults += 1
    out = Ctxt(a.slots * b.slots, lvl - 1)
    METER.spend(label, out.level)
    return out


def ct_mul_plain(a, v, label=None):
    """Ciphertext x plaintext, elementwise. ALSO costs one level: the product
    doubles the scale and CKKS must rescale, which drops a modulus. Treating
    this as free is the single most common mistake in a depth estimate."""
    # === YOUR CODE HERE ===
    if a.level <= 0:
        raise DepthExhausted(label or "plaintext multiply")
    METER.pt_mults += 1
    out = Ctxt(a.slots * np.asarray(v, dtype=np.float64), a.level - 1)
    METER.spend(label, out.level)
    return out


def ct_rot(a, k):
    """Cyclic rotation: result[i] = a[(i + k) mod n]. No level cost, but each
    one needs a key-switch, which is the dominant runtime cost in practice."""
    # === YOUR CODE HERE ===
    METER.rotations += 1
    return Ctxt(np.roll(a.slots, -(k % a.n)), a.level)


def pt_rot(v, k):
    """The same rotation applied to a PLAINTEXT. Free — it happens at encode
    time, before anything is encrypted. Baby-step/giant-step depends on this."""
    # === YOUR CODE HERE ===
    return np.roll(np.asarray(v, dtype=np.float64), -(k % np.asarray(v).size))


# ---------------------------------------------------------------------------
# The reduction primitive: summing slots you are not allowed to index.
# ---------------------------------------------------------------------------

def rotate_and_sum(ct):
    """Doubling reduction: ct <- ct + rot(ct, 1), then +rot(ct, 2), +rot(ct, 4)...

    After ceil(log2 n) steps every slot holds the sum of all n slots. This is
    exact precisely because n is a power of two — the doubling windows tile the
    ring perfectly. CKKS always gives you a power-of-two slot count (n = N/2),
    which is why nobody ever notices the assumption.
    """
    # === YOUR CODE HERE ===
    out = ct
    shift = 1
    while shift < ct.n:
        out = ct_add(out, ct_rot(out, shift))
        shift *= 2
    return out


# ---------------------------------------------------------------------------
# Matrix-vector product, three ways.
# ---------------------------------------------------------------------------

def diagonals(M):
    """The Halevi-Shoup generalized diagonals: diag_k[i] = M[i, (i + k) % d].

    Then  M @ v  ==  sum_k diag_k * rot(v, k), which is d plaintext multiplies
    and d-1 rotations — and, crucially, ONE level of depth.
    """
    d = M.shape[0]
    # === YOUR CODE HERE ===
    return np.stack([np.array([M[i, (i + k) % d] for i in range(d)])
                     for k in range(d)])


def matvec_naive(ct, M):
    """Row-wise: mask row i, reduce with rotate-and-sum, mask slot i, accumulate.

    Costs d * log2(n) rotations and TWO levels (one plaintext multiply for the
    row, one more for the output mask). This is what everyone writes first.
    """
    d = M.shape[0]
    acc = None
    # === YOUR CODE HERE ===
    for i in range(d):
        t = ct_mul_plain(ct, M[i])          # level 1: elementwise row product
        t = rotate_and_sum(t)               # log2(n) rotations, no level
        e = np.zeros(ct.n)
        e[i] = 1.0
        t = ct_mul_plain(t, e)              # level 2: keep only slot i
        acc = t if acc is None else ct_add(acc, t)
    return acc


def matvec_diagonal(ct, M, label=None):
    """Halevi-Shoup: sum_k diag_k * rot(v, k). d-1 rotations, ONE level.

    If the ciphertext is wider than d it must hold the record REPLICATED
    n/d times, so that a rotation modulo n coincides with a rotation modulo d;
    the diagonals are tiled to match. Replication is what makes the plain
    rotation legal here — and it is also what wastes the slots.
    """
    d = M.shape[0]
    reps = ct.n // d
    D = diagonals(M)
    acc = None
    # === YOUR CODE HERE ===
    if label:
        METER.free(f"rotate x by k=1..{d - 1}  ({d - 1} rotations, no level cost)",
                   ct.level)
    for k in range(d):
        rk = ct if k == 0 else ct_rot(ct, k)
        Dk = np.tile(D[k], reps) if reps > 1 else D[k]
        t = ct_mul_plain(rk, Dk, label if k == 0 else None)
        acc = t if acc is None else ct_add(acc, t)
    return acc


def bsgs_split(d):
    """Factor d = d1 * d2 with d1 + d2 as small as possible (powers of two)."""
    best = None
    p = 1
    while p <= d:
        if d % p == 0:
            cand = (p, d // p)
            if best is None or sum(cand) < sum(best):
                best = cand
        p *= 2
    return best


def matvec_bsgs(ct, M):
    """Baby-step / giant-step over the diagonals.

        sum_k diag_k rot(v,k) = sum_j rot( sum_i pt_rot(diag_{j d1+i}, -j d1)
                                            * rot(v, i),  j d1 )

    using rot(a,s)*rot(b,s) == rot(a*b, s). The d1-1 baby rotations of v are
    computed once and reused for every giant step, so the total is
    (d1-1) + (d2-1) rotations instead of d-1 — still ONE level.
    """
    d = M.shape[0]
    d1, d2 = bsgs_split(d)
    D = diagonals(M)
    # === YOUR CODE HERE ===
    baby = [ct] + [ct_rot(ct, i) for i in range(1, d1)]
    acc = None
    for j in range(d2):
        inner = None
        for i in range(d1):
            dk = pt_rot(D[j * d1 + i], -(j * d1))      # plaintext, free
            t = ct_mul_plain(baby[i], dk)
            inner = t if inner is None else ct_add(inner, t)
        outer = inner if j == 0 else ct_rot(inner, j * d1)
        acc = outer if acc is None else ct_add(acc, outer)
    return acc


# ---------------------------------------------------------------------------
# Packing layouts.
# ---------------------------------------------------------------------------

def matvec_blockpacked(ct, M, d, buggy=False):
    """Apply the same d x d matrix to EVERY record in a ciphertext that holds
    n/d records packed back to back.

    The rotation the diagonal method wants is cyclic MODULO d, inside each
    block. The rotation the machine gives you is cyclic modulo n. They differ,
    and the fix is two masked rotations per diagonal:

        blockrot(x, k) = head_mask * rot(x, k) + tail_mask * rot(x, k - d)

    with the masks folded into the diagonals so the fix costs rotations, not
    depth. `buggy=True` skips the fix and rotates by k alone — right modulo d,
    wrong modulo n.
    """
    n = ct.n
    nrec = n // d
    D = diagonals(M)
    j = np.arange(n) % d
    acc = None
    # === YOUR CODE HERE ===
    for k in range(d):
        Dk = np.tile(D[k], nrec)
        if k == 0:
            acc = ct_mul_plain(ct, Dk)
            continue
        if buggy:
            t = ct_mul_plain(ct_rot(ct, k), Dk)        # the bug, in one line
            acc = ct_add(acc, t)
            continue
        head = (j + k < d).astype(np.float64)
        t1 = ct_mul_plain(ct_rot(ct, k), Dk * head)
        t2 = ct_mul_plain(ct_rot(ct, k - d), Dk * (1.0 - head))
        acc = ct_add(acc, ct_add(t1, t2))
    return acc


def matvec_columnpacked(cts, M):
    """Column (SIMD) packing: cts[f] holds feature f of every record, one
    record per slot. The matrix-vector product becomes pure scalar broadcast —
    ZERO rotations — at the cost of d ciphertexts instead of one."""
    d = M.shape[0]
    n = cts[0].n
    out = []
    # === YOUR CODE HERE ===
    for i in range(d):
        acc = None
        for f in range(d):
            t = ct_mul_plain(cts[f], np.full(n, M[i, f]))
            acc = t if acc is None else ct_add(acc, t)
        out.append(acc)
    return out


# ---------------------------------------------------------------------------
# One encrypted layer: matvec + bias + degree-3 polynomial activation.
# ---------------------------------------------------------------------------

def poly_act_naive(y, c, tag=""):
    """p(x) = c0 + c1 x + c2 x^2 + c3 x^3, scheduled the obvious way.

    Critical path: y -> x^2 -> x^3 -> c3*x^3. The plaintext multiply by c3
    lands AFTER the last ciphertext multiply, so it adds a third level.
    """
    n = y.n
    # === YOUR CODE HERE ===
    x2 = ct_mul(y, y, f"{tag}act: x^2  (ct x ct)")
    x3 = ct_mul(x2, y, f"{tag}act: x^3 = x^2 * x  (ct x ct)")
    t3 = ct_mul_plain(x3, np.full(n, c[3]),
                      f"{tag}act: c3 * x^3  (PLAINTEXT mult -- NOT free)")
    t2 = ct_mul_plain(x2, np.full(n, c[2]), f"{tag}act: c2 * x^2  (plaintext mult)")
    t1 = ct_mul_plain(y, np.full(n, c[1]), f"{tag}act: c1 * x    (plaintext mult)")
    out = ct_add(ct_add(t1, t2), t3)
    METER.free(f"{tag}act: sum terms, add c0  (adds are free)", out.level)
    return ct_add_plain(out, np.full(n, c[0]))


def poly_act_folded(y, c, tag=""):
    """The same polynomial, one level cheaper: fold c3 into x BEFORE squaring,
    so c3*x^3 = x^2 * (c3 x) and the plaintext multiply rides a branch that was
    going to cost a level anyway. Identical output, critical path of 2."""
    n = y.n
    # === YOUR CODE HERE ===
    x2 = ct_mul(y, y, f"{tag}act: x^2  (ct x ct)")
    u = ct_mul_plain(y, np.full(n, c[3]),
                     f"{tag}act: c3 * x   (plaintext mult, FOLDED EARLY)")
    t3 = ct_mul(x2, u, f"{tag}act: c3 x^3 = x^2 * (c3 x)  (ct x ct)")
    t2 = ct_mul_plain(x2, np.full(n, c[2]), f"{tag}act: c2 * x^2  (plaintext mult)")
    t1 = ct_mul_plain(y, np.full(n, c[1]), f"{tag}act: c1 * x    (plaintext mult)")
    out = ct_add(ct_add(t1, t2), t3)
    METER.free(f"{tag}act: sum terms, add c0  (adds are free)", out.level)
    return ct_add_plain(out, np.full(n, c[0]))


def encrypted_layer(ct, W, b, c, folded=True, tag=""):
    """W @ x + b, then the polynomial activation. Levels charged by the meter."""
    # === YOUR CODE HERE ===
    y = matvec_diagonal(ct, W, label=f"{tag}W @ x  ({W.shape[0]} plaintext diagonals)")
    y = ct_add_plain(y, b)
    METER.free(f"{tag}+ bias  (plaintext add)", y.level)
    return (poly_act_folded if folded else poly_act_naive)(y, c, tag)


def plain_layer(x, W, b, c):
    """The numpy reference the encrypted version must reproduce."""
    y = W @ x + b
    return c[0] + c[1] * y + c[2] * y ** 2 + c[3] * y ** 3


def run_until_exhausted(x, W, b, c, start_level, folded):
    """Stack layers until the level budget runs out. Returns the number of
    COMPLETE layers, the label of the operation that could not be paid for,
    and the level left at that moment."""
    ct = encrypt(x, start_level)
    layers = 0
    # === YOUR CODE HERE ===
    while True:
        try:
            ct = encrypted_layer(ct, W, b, c, folded=folded, tag=f"L{layers + 1} ")
        except DepthExhausted as e:
            return layers, e.op, ct.level
        layers += 1


if __name__ == "__main__":
    t_start = time.perf_counter()
    rng = np.random.default_rng(0)

    print("\n--- the three operations, and what each one costs ---")
    n = 16
    x = rng.standard_normal(n)
    ct = encrypt(x, level=10)
    METER.reset()
    a = ct_add(ct, ct)
    check("add is exact and free", decrypt(a), 2 * x, tol=1e-15)
    check("add consumes no level", a.level, 10)
    m = ct_mul(ct, ct)
    check("multiply is elementwise", decrypt(m), x * x, tol=1e-15)
    check("multiply consumes exactly one level", m.level, 9)
    p = ct_mul_plain(ct, np.arange(n, dtype=float))
    check("PLAINTEXT multiply also consumes a level (rescaling)", p.level, 9)
    r = ct_rot(ct, 3)
    check("rot(x, k)[i] == x[(i+k) mod n]", decrypt(r), np.roll(x, -3), tol=1e-15)
    check("rotation consumes no level", r.level, 10)
    # The group law is what makes baby-step/giant-step legal at all.
    check("rot(rot(x,a),b) == rot(x,a+b)",
          decrypt(ct_rot(ct_rot(ct, 5), 6)), decrypt(ct_rot(ct, 11)), tol=1e-15)
    print(f"      after {METER.rotations + METER.ct_mults + METER.pt_mults} metered ops "
          f"on one 16-slot ciphertext: "
          f"{METER.rotations} rotations, {METER.ct_mults} ct-mults, {METER.pt_mults} pt-mults")

    print("\n--- rotate-and-sum: a total you are not allowed to index ---")
    METER.reset()
    s = rotate_and_sum(encrypt(x, 10))
    print(f"      n={n}: {METER.rotations} rotations, every slot now holds the total")
    check("log-depth reduction uses log2(n) rotations", METER.rotations, int(np.log2(n)))
    check("all n slots hold the full sum", decrypt(s), np.full(n, x.sum()), tol=1e-12)
    check("the reduction costs no levels", s.level, 10)

    # Now break the power-of-two assumption. n = 6 needs ceil(log2 6) = 3 steps,
    # which sweep 8 offsets over a ring of 6 -- so offsets 0 and 1 are counted
    # TWICE. The result is not the sum, and nothing raises.
    n6 = 6
    x6 = rng.standard_normal(n6)
    METER.reset()
    s6 = decrypt(rotate_and_sum(encrypt(x6, 10)))
    predicted = np.array([x6.sum() + x6[i] + x6[(i + 1) % n6] for i in range(n6)])
    print(f"      n=6:  {METER.rotations} rotations; true sum {x6.sum():+.4f}, "
          f"slot 0 got {s6[0]:+.4f}")
    print(f"      the doubling sweeps 2^3 = 8 offsets over a ring of 6, so each slot")
    print(f"      double-counts exactly x[i] and x[i+1] -- an overcount, not a crash")
    check("n=6 does NOT give the sum", abs(s6[0] - x6.sum()) > 1e-3)
    check("and the error is exactly the double-counted offsets", s6, predicted, tol=1e-12)

    print("\n--- matrix-vector product, three ways ---")
    d = 16
    M = rng.standard_normal((d, d)) / np.sqrt(d)
    v = rng.standard_normal(d)
    ref = M @ v
    d1, d2 = bsgs_split(d)
    results = {}
    print(f"      {'method':<34} {'rots':>5} {'pt-mul':>7} {'depth':>6} {'max err':>10}")
    for name, fn in (("naive row-wise (rotate & sum)", matvec_naive),
                     ("diagonal (Halevi-Shoup)", matvec_diagonal),
                     (f"diagonal + BSGS ({d1}x{d2})", matvec_bsgs)):
        METER.reset()
        out = fn(encrypt(v, 10), M)
        err = float(np.max(np.abs(decrypt(out)[:d] - ref)))
        results[name] = (METER.rotations, METER.pt_mults, 10 - out.level, err)
        print(f"      {name:<34} {METER.rotations:5d} {METER.pt_mults:7d} "
              f"{10 - out.level:6d} {err:10.2e}")

    naive_rot, naive_pt, naive_depth, naive_err = results["naive row-wise (rotate & sum)"]
    diag_rot, diag_pt, diag_depth, diag_err = results["diagonal (Halevi-Shoup)"]
    bsgs_rot, bsgs_pt, bsgs_depth, bsgs_err = results[f"diagonal + BSGS ({d1}x{d2})"]

    check("naive matches plaintext matmul", naive_err < 1e-12)
    check("diagonal matches plaintext matmul", diag_err < 1e-12)
    check("BSGS matches plaintext matmul", bsgs_err < 1e-12)
    # Mechanism, not magnitude: each count is its closed form, exactly.
    check("naive costs exactly d*log2(n) rotations (n = d = 16 here)",
          naive_rot, d * int(np.log2(d)))
    check("diagonal costs exactly d-1 rotations", diag_rot, d - 1)
    check("BSGS costs exactly (d1-1)+(d2-1) rotations", bsgs_rot, (d1 - 1) + (d2 - 1))
    check("naive needs 2 levels (row mask, then output mask)", naive_depth, 2)
    check("both diagonal forms need 1", diag_depth == 1 and bsgs_depth == 1)
    print(f"      the diagonal method is not more accurate -- all three land at ~1e-16.")
    print(f"      it is {naive_rot / diag_rot:.2f}x fewer rotations, and BSGS "
          f"{naive_rot / bsgs_rot:.2f}x, for the SAME answer.")

    print("\n--- how the rotation count scales (this is the whole argument) ---")
    print(f"      {'d':>4} {'naive d*log2 d':>15} {'diagonal d-1':>14} {'BSGS':>7} {'naive/BSGS':>11}")
    scaling = []
    for dd in (8, 16, 32, 64):
        Md = rng.standard_normal((dd, dd)) / np.sqrt(dd)
        vd = rng.standard_normal(dd)
        counts = []
        worst = 0.0
        for fn in (matvec_naive, matvec_diagonal, matvec_bsgs):
            METER.reset()
            o = fn(encrypt(vd, 10), Md)
            counts.append(METER.rotations)
            worst = max(worst, float(np.max(np.abs(decrypt(o)[:dd] - Md @ vd))))
        scaling.append((dd, counts, worst))
        print(f"      {dd:4d} {counts[0]:15d} {counts[1]:14d} {counts[2]:7d} "
              f"{counts[0] / counts[2]:10.1f}x")
    check("every size, every method, agrees with numpy",
          max(w for _, _, w in scaling) < 1e-12)
    check("naive rotations grow as d*log2(d)",
          all(c[0] == dd * int(np.log2(dd)) for dd, c, _ in scaling))
    check("diagonal rotations grow linearly, as d-1",
          all(c[1] == dd - 1 for dd, c, _ in scaling))
    check("the naive/BSGS gap widens with d",
          scaling[-1][1][0] / scaling[-1][1][2] > scaling[0][1][0] / scaling[0][1][2])

    print("\n--- THE LEDGER: one encrypted linear layer + polynomial activation ---")
    W = rng.standard_normal((d, d)) / np.sqrt(d)
    bvec = rng.standard_normal(d) * 0.1
    coeffs = (0.05, 0.5, 0.15, -0.02)          # a bounded cubic ReLU surrogate
    x0 = rng.standard_normal(d)

    ledgers = {}
    for sched, folded in (("naive schedule", False), ("folded schedule", True)):
        METER.reset()
        out = encrypted_layer(encrypt(x0, 10), W, bvec, coeffs, folded=folded)
        consumed = 10 - out.level
        # Count OPERATIONS that dropped a level, from the meter -- not ledger
        # lines. The ledger's "W @ x" line collapses all d plaintext-diagonal
        # multiplies (each on its own parallel branch) into one labelled stage.
        n_paid = METER.ct_mults + METER.pt_mults
        ledgers[sched] = (consumed, n_paid, decrypt(out)[:d])
        print(f"\n      [{sched}]")
        print(f"      {'stage':<52} {'lvl':>4} {'left':>5}")
        for lab, cost, left in METER.ledger:
            print(f"      {lab:<52} {cost:4d} {left:5d}")
        print(f"      {'':<52} {'---':>4}")
        print(f"      {'critical-path depth of the layer':<52} {consumed:4d}")
        print(f"      ({n_paid} multiplies each charge a level -- the W @ x line alone")
        print(f"       is {d} of them -- but they sit on parallel branches:")
        print(f"       depth is the LONGEST path, not the count)")

    ref_layer = plain_layer(x0, W, bvec, coeffs)
    check("encrypted layer (naive schedule) matches numpy",
          ledgers["naive schedule"][2], ref_layer, tol=1e-12)
    check("encrypted layer (folded schedule) matches numpy",
          ledgers["folded schedule"][2], ref_layer, tol=1e-12)
    check("the two schedules give bit-comparable answers",
          ledgers["naive schedule"][2], ledgers["folded schedule"][2], tol=1e-12)
    check("naive schedule costs 4 levels", ledgers["naive schedule"][0], 4)
    check("folding the plaintext coefficient saves exactly one level",
          ledgers["folded schedule"][0], ledgers["naive schedule"][0] - 1)
    # Closed form: the matvec pays d plaintext multiplies, the cubic pays
    # 3 plaintext + 2 ct x ct = 5 more, under EITHER schedule -- folding
    # re-orders the multiplies, it does not remove any.
    check("both schedules pay exactly d+5 level-charging multiplies",
          ledgers["naive schedule"][1] == d + 5
          and ledgers["folded schedule"][1] == d + 5)
    check("depth is the critical path, not the number of paid operations",
          ledgers["folded schedule"][1] > ledgers["folded schedule"][0])

    print("\n--- the packing decision, in slots actually used ---")
    dp, npack = 8, 64                      # 8 features, 64 slots per ciphertext
    Mp = rng.standard_normal((dp, dp)) / np.sqrt(dp)

    def slot_census(cts_list):
        """MEASURE a layout's footprint from the ciphertext objects themselves:
        how many ciphertexts, how many slots they occupy in total, and how many
        DISTINCT payload values they hold across the whole layout. A slot
        counts as useful iff its value is nonzero (the payload is continuous
        random data, never exactly 0.0 -- zeros are padding) and not a repeat
        of any other slot anywhere in the layout (replication is redundancy,
        not information). Nothing here is a hardcoded claim about what the
        layout SHOULD cost."""
        ncts = len(cts_list)
        total = sum(c.n for c in cts_list)
        payload = np.concatenate([c.slots[c.slots != 0.0] for c in cts_list])
        useful = int(np.unique(payload).size)
        return ncts, total, useful

    def pack_report(B):
        """Cost and utilisation of three layouts for B records of dp features."""
        X = rng.standard_normal((B, dp))
        ref = X @ Mp.T
        rows = []

        # (1) row packing: one record per ciphertext, replicated to fill so the
        #     cyclic rotation stays inside the record.
        cts_row = [encrypt(np.tile(X[r], npack // dp), 10) for r in range(B)]
        METER.reset()
        worst = 0.0
        for r in range(B):
            o = matvec_diagonal(cts_row[r], Mp)
            worst = max(worst, float(np.max(np.abs(decrypt(o)[:dp] - ref[r]))))
        ncts, total, useful = slot_census(cts_row)
        rows.append(("row: one record per ciphertext", ncts, useful, total,
                     METER.rotations, METER.pt_mults, worst))

        # (2) diagonal/block packing: npack//dp records per ciphertext.
        per = npack // dp
        cts_blk = [encrypt(X[c0:c0 + per].ravel(), 10, n=npack)
                   for c0 in range(0, B, per)]
        METER.reset()
        worst = 0.0
        for i, c0 in enumerate(range(0, B, per)):
            blk = X[c0:c0 + per]
            o = matvec_blockpacked(cts_blk[i], Mp, dp)
            worst = max(worst, float(np.max(np.abs(
                decrypt(o).reshape(per, dp)[:len(blk)] - ref[c0:c0 + per]))))
        ncts, total, useful = slot_census(cts_blk)
        rows.append(("diagonal: %d records per ciphertext" % per,
                     ncts, useful, total, METER.rotations, METER.pt_mults, worst))

        # (3) column packing: one ciphertext per feature, records across slots.
        cts_col = [encrypt(X[:, f], 10, n=npack) for f in range(dp)]
        METER.reset()
        outs = matvec_columnpacked(cts_col, Mp)
        got = np.stack([decrypt(o)[:B] for o in outs], axis=1)
        ncts, total, useful = slot_census(cts_col)
        rows.append(("column: one ciphertext per feature", ncts, useful, total,
                     METER.rotations, METER.pt_mults,
                     float(np.max(np.abs(got - ref)))))
        return rows

    all_rows = {}
    for B in (8, 64):
        print(f"\n      batch B={B} records, d={dp} features, n={npack} slots")
        print(f"      {'layout':<36} {'cts':>4} {'util':>7} {'rots':>6} "
              f"{'pt-mul':>7} {'rot/rec':>8} {'max err':>9}")
        rows = pack_report(B)
        all_rows[B] = rows
        for name, ncts, useful, total, rots, ptm, err in rows:
            util = 100.0 * useful / total
            print(f"      {name:<36} {ncts:4d} {util:6.1f}% {rots:6d} {ptm:7d} "
                  f"{rots / B:8.2f} {err:9.1e}")

    check("all three layouts compute the same matmul",
          max(r[6] for B in (8, 64) for r in all_rows[B]) < 1e-12)
    # Utilisation is the mechanism: a layout wastes slots exactly in proportion
    # to how much of the ciphertext holds distinct useful values. These four
    # numbers are MEASURED by slot_census from the encrypted slots, then
    # compared to each layout's closed form -- a layout that pads, replicates,
    # or allocates ciphertexts differently moves the measured side.
    util_row_8 = 100.0 * all_rows[8][0][2] / all_rows[8][0][3]
    util_diag_8 = 100.0 * all_rows[8][1][2] / all_rows[8][1][3]
    util_col_8 = 100.0 * all_rows[8][2][2] / all_rows[8][2][3]
    util_col_64 = 100.0 * all_rows[64][2][2] / all_rows[64][2][3]
    print(f"\n      row packing holds {dp} distinct values in {npack} slots: "
          f"{util_row_8:.1f}% used, {100 - util_row_8:.1f}% wasted")
    check("row packing wastes n-d of every n slots",
          util_row_8, 100.0 * dp / npack, tol=1e-9)
    check("diagonal packing fills the ciphertext", util_diag_8, 100.0, tol=1e-9)
    # Column packing's utilisation is exactly B/n: it is a THROUGHPUT layout,
    # dead weight at small batch and perfect once the batch fills the slots.
    check("column-packing utilisation is exactly 100*B/n at B=8",
          util_col_8, 100.0 * 8 / npack, tol=1e-9)
    check("column-packing utilisation reaches 100% exactly at B=n",
          util_col_64, 100.0, tol=1e-9)
    check("column packing needs ZERO rotations", all_rows[64][2][4], 0)
    # And the honest caveat: packing 8 records per ciphertext does NOT make it
    # 8x cheaper, because the masked block rotation costs 2 rotations per
    # diagonal instead of 1. Measure the real factor.
    speedup = all_rows[64][0][4] / all_rows[64][1][4]
    print(f"      packing {npack // dp} records per ciphertext cuts rotations "
          f"{speedup:.1f}x, not {npack // dp}x --")
    print(f"      the masked block rotation costs 2 rotations per diagonal, not 1.")
    # The exact mechanism: row packing spends (d-1) rotations per record,
    # block packing 2(d-1) per (n/d) records, so the saving is (n/d)/2 -- half
    # the packing factor, every time, not a number to be tuned.
    check("the saving is exactly half the packing factor",
          speedup, (npack / dp) / 2.0, tol=1e-9)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) The rotation offset that is right modulo d and wrong modulo n. The
    #     diagonal method wants rot_d(x, k) inside each packed record; the
    #     machine gives rot_n(x, k) across the whole ciphertext. Nothing
    #     raises, nothing is NaN -- the tail of every record silently reads the
    #     head of the NEXT record.
    per = npack // dp
    Xb = rng.standard_normal((per, dp))
    ref_b = Xb @ Mp.T
    good = decrypt(matvec_blockpacked(encrypt(Xb.ravel(), 10, n=npack), Mp, dp)).reshape(per, dp)
    bad = decrypt(matvec_blockpacked(encrypt(Xb.ravel(), 10, n=npack), Mp, dp,
                                     buggy=True)).reshape(per, dp)
    err_good = float(np.max(np.abs(good - ref_b)))
    err_bad = float(np.max(np.abs(bad - ref_b)))
    wrong = int(np.sum(np.abs(bad - ref_b) > 1e-9))
    print(f"      masked block rotation : max err {err_good:.2e}")
    print(f"      rotate by k alone     : max err {err_bad:.2e}  "
          f"({wrong}/{bad.size} slots wrong = {100 * wrong / bad.size:.1f}%)")
    rel = float(np.linalg.norm(bad - ref_b) / np.linalg.norm(ref_b))
    print(f"      no NaN, no overflow: correct output lives in "
          f"[{ref_b.min():+.3f}, {ref_b.max():+.3f}], buggy output in "
          f"[{bad.min():+.3f}, {bad.max():+.3f}]")
    print(f"      relative error {100 * rel:.1f}% -- wrong, but wrong in the range "
          f"a plausible answer occupies.")
    check("the masked version is exact", err_good < 1e-12)
    check("the unmasked version is wrong by a visible margin",
          err_bad > 1e6 * max(err_good, 1e-16))
    # The mechanism, exactly: output row i of record r is the matrix applied to
    # a HYBRID vector -- record r for features m >= i, record r+1 for m < i.
    hybrid = np.empty_like(bad)
    for r in range(per):
        for i in range(dp):
            z = np.where(np.arange(dp) >= i, Xb[r], Xb[(r + 1) % per])
            hybrid[r, i] = Mp[i] @ z
    check("the bug is exactly 'record r+1 bleeds into the tail of record r'",
          bad, hybrid, tol=1e-12)
    # Row 0 of every record has no tail to corrupt, so it is exactly right --
    # which is precisely why this bug survives a spot-check.
    check("output slot 0 of every record is still exactly correct",
          bad[:, 0], ref_b[:, 0], tol=1e-12)
    print(f"      slot 0 of every record is EXACTLY right ("
          f"max err {float(np.max(np.abs(bad[:, 0] - ref_b[:, 0]))):.1e}); "
          f"spot-checking one value")
    print(f"      would pass. The mean error over the wrong slots is "
          f"{float(np.abs(bad - ref_b)[np.abs(bad - ref_b) > 1e-9].mean()):.3f}.")

    # (b) Exceeding the depth budget. Stack layers until the meter cannot pay.
    print()
    budget = 10
    for sched, folded in (("naive (4 levels/layer)", False),
                          ("folded (3 levels/layer)", True)):
        METER.reset()
        layers, op, left = run_until_exhausted(x0, W, bvec, coeffs, budget, folded)
        cost = ledgers["naive schedule" if not folded else "folded schedule"][0]
        print(f"      budget {budget} levels, {sched}: {layers} complete layers, "
              f"{left} level(s) left")
        print(f"        -> ran out at: {op}")
        check(f"{sched}: layers completed == budget // depth-per-layer",
              layers, budget // cost)
        # WHICH op runs out is forced by the leftover budget, so pin it exactly.
        # Naive leaves 10 - 2*4 = 2 levels: the matvec takes one, x^2 the other,
        # and x^3 finds none. Folded leaves 10 - 3*3 = 1: the matvec takes it,
        # and x^2 finds none. An off-by-one in either depth guard shifts the
        # failing op by one step and this comparison catches it.
        want = (f"L{layers + 1} act: x^2  (ct x ct)" if folded
                else f"L{layers + 1} act: x^3 = x^2 * x  (ct x ct)")
        check(f"{sched}: the failure names the exact operation that ran out",
              op, want)
    lf, _, _ = run_until_exhausted(x0, W, bvec, coeffs, budget, True)
    ln, _, _ = run_until_exhausted(x0, W, bvec, coeffs, budget, False)
    print(f"      one level of scheduling buys a whole extra layer: {lf} vs {ln}")
    check("the folded schedule fits one more layer from the same budget", lf, ln + 1)
    # The boundary itself: level 1 is still alive -- exactly one multiply left.
    check("a level-1 ciphertext can still multiply exactly once",
          ct_mul(encrypt(x0, 1), encrypt(x0, 1)).level, 0)
    # Once levels are gone, no multiplication is possible at all -- that is the
    # hard wall a leveled scheme has, and the reason bootstrapping exists.
    dead = encrypt(x0, 0)
    raised = False
    try:
        ct_mul(dead, dead, "post-mortem multiply")
    except DepthExhausted:
        raised = True
    check("at level 0 no further multiplication is possible", raised)
    check("but addition and rotation still work at level 0",
          decrypt(ct_rot(ct_add(dead, dead), 1)), np.roll(2 * x0, -1), tol=1e-15)

    print(f"\n      (runtime {time.perf_counter() - t_start:.2f}s)")
    summary()
