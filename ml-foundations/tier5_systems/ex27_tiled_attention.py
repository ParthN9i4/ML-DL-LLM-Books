"""
ex27 — Tiled attention and the online softmax.  (Book: Chapter 31)

Attention is not compute-bound. On any modern accelerator the matmuls in
softmax(Q K^T / sqrt(d)) V are cheap; what costs is writing a T x T score
matrix out to HBM, reading it back to normalize it, writing the probabilities
out again, and reading them back to multiply by V. That is 4 T^2 elements of
memory traffic to do 4 T^2 d flops of work — a hopeless arithmetic intensity.
FlashAttention's contribution is not a faster matmul, it is never storing the
T x T matrix at all.

The obstacle is the softmax. It needs a normalizer that depends on every score
in the row, so the naive kernel must see the whole row before it can emit
anything. The online softmax dissolves that obstacle. Keep a running maximum m
and a running exponential sum l; when a new block arrives with a larger
maximum, rescale what you already have by exp(m_old - m_new) and add the new
block's contribution. After the last block, (m, l) are exactly the statistics
the two-pass algorithm would have produced — algebraically exact, not an
approximation — and the same rescaling trick applied to a running output
accumulator turns attention into a streaming algorithm over K/V tiles.

The one insight this file is built to make undeniable is that the rescaling
must hit the ACCUMULATOR, not just the denominator. Rescaling only the
denominator is the single most common way to write a broken flash kernel, and
how visible it is depends entirely on the input: the error is proportional to
exp(running-max drift) - 1, so on gently-scaled scores it hides below 1e-4 and
on a query that strongly prefers one late key block it lands at 13.0. The
break-it section pins that proportionality constant to four digits, and shows
the bug being exactly zero on a test vector many people would call fair.

Everything here is numpy. The I/O argument is made by counting real bytes and
by reading a real peak-allocation high-water mark, not by asserting an O().

To learn: replace each function body with `pass` and reimplement.
"""

import gc
import os
import sys
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


# ---------------------------------------------------------------------------
# Stage 1 — the online softmax
# ---------------------------------------------------------------------------

def softmax_two_pass(x):
    """The standard safe softmax: one pass for the max, one for the sum.

    This is the reference. It is numerically bulletproof and it is exactly
    what the streaming version has to reproduce.
    """
    # === YOUR CODE HERE ===
    m = np.max(x)
    e = np.exp(x - m)
    return e / np.sum(e)


def softmax_naive_one_pass(x):
    """exp first, normalize second — one pass, and one overflow away from nan.

    The reason nobody ships this: exp(x) overflows the format's exponent range
    long before x is anything unusual for a logit, and inf/inf is nan.
    """
    # === YOUR CODE HERE ===
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(x)
        return e / np.sum(e)


def online_softmax_stats(x, block):
    """Stream x in blocks, returning (m, l) = (max, sum of exp(x - max)).

    The recurrence, which is the whole of FlashAttention in two lines:

        m_new = max(m_old, max(block))
        l_new = l_old * exp(m_old - m_new) + sum(exp(block - m_new))

    The exp(m_old - m_new) factor retroactively re-bases everything already
    accumulated onto the new maximum. It is an identity, not an approximation:
    l_old is a sum of exp(x_i - m_old), and multiplying by exp(m_old - m_new)
    turns every term into exp(x_i - m_new).
    """
    m = -np.inf
    l = 0.0
    # === YOUR CODE HERE ===
    for i in range(0, len(x), block):
        b = x[i:i + block]
        m_new = max(m, float(np.max(b)))
        l = l * np.exp(m - m_new) + float(np.sum(np.exp(b - m_new)))
        m = m_new
    return m, l


def online_softmax(x, block):
    """Full softmax built from the streaming statistics."""
    m, l = online_softmax_stats(x, block)
    # === YOUR CODE HERE ===
    return np.exp(x - m) / l


# ---------------------------------------------------------------------------
# Stage 2 — attention, quadratic and tiled
# ---------------------------------------------------------------------------

def attention_naive(Q, K, V, causal=False):
    """The textbook kernel. Materializes the T x T score matrix in full."""
    T, d = Q.shape
    # === YOUR CODE HERE ===
    S = Q @ K.T / np.sqrt(d)
    if causal:
        S = np.where(np.tril(np.ones((T, T), dtype=bool)), S, -np.inf)
    S = S - np.max(S, axis=1, keepdims=True)
    P = np.exp(S)
    return (P / np.sum(P, axis=1, keepdims=True)) @ V


def attention_tiled(Q, K, V, block_q, block_k, causal=False):
    """FlashAttention's forward pass, in numpy. Never builds a T x T array.

    For each query tile, walk the K/V tiles keeping three small running things:
    the row maxima m, the row exponential sums l, and an unnormalized output
    accumulator acc. When a tile pushes the maximum up by exp(m_old - m_new),
    BOTH l and acc are rescaled by that factor — acc because every term already
    inside it was exponentiated against the old maximum.

    Rows that are entirely masked out have m = -inf; the arithmetic below routes
    around inf - inf (which is nan) by rebasing on a finite surrogate.
    """
    T, d = Q.shape
    d_v = V.shape[1]
    out = np.empty((T, d_v))
    # === YOUR CODE HERE ===
    for q_start in range(0, T, block_q):
        q_end = min(q_start + block_q, T)
        Qb = Q[q_start:q_end]
        m = np.full(q_end - q_start, -np.inf)
        l = np.zeros(q_end - q_start)
        acc = np.zeros((q_end - q_start, d_v))
        for k_start in range(0, T, block_k):
            if causal and k_start > q_end - 1:
                break                      # whole tile lies in the future
            k_end = min(k_start + block_k, T)
            s = Qb @ K[k_start:k_end].T / np.sqrt(d)
            if causal:
                qi = np.arange(q_start, q_end)[:, None]
                kj = np.arange(k_start, k_end)[None, :]
                s = np.where(qi >= kj, s, -np.inf)   # ABSOLUTE positions
            m_new = np.maximum(m, np.max(s, axis=1))
            m_safe = np.where(np.isneginf(m_new), 0.0, m_new)
            p = np.exp(s - m_safe[:, None])
            alpha = np.exp(m - m_safe)               # 0 when m was -inf
            l = l * alpha + np.sum(p, axis=1)
            acc = acc * alpha[:, None] + p @ V[k_start:k_end]
            m = m_new
        out[q_start:q_end] = acc / l[:, None]
    return out


# ---------------------------------------------------------------------------
# The I/O argument, counted in real bytes
# ---------------------------------------------------------------------------

def hbm_bytes_naive(Q, K, V):
    """Bytes crossing the HBM boundary for the three-kernel naive pipeline.

    Every term is the true .nbytes of an array the kernel really allocates:
      1. S = Q K^T   : read Q, read K, write S
      2. P = softmax : read S, write P
      3. O = P V     : read P, read V, write O
    """
    T = Q.shape[0]
    S = np.empty((T, T), dtype=Q.dtype)
    O = np.empty((T, V.shape[1]), dtype=Q.dtype)
    # === YOUR CODE HERE ===
    io = Q.nbytes + K.nbytes + S.nbytes          # kernel 1
    io += S.nbytes + S.nbytes                    # kernel 2 (read S, write P)
    io += S.nbytes + V.nbytes + O.nbytes         # kernel 3 (read P, read V, write O)
    return io


def hbm_bytes_tiled(Q, K, V, block_q, block_k, causal=False):
    """Bytes crossing the HBM boundary for the tiled kernel.

    Mirrors attention_tiled's loop exactly and charges the true .nbytes of each
    tile it loads. Each query tile is read once and each output tile written
    once; the K/V tiles are re-read once per query tile, which is the entire
    O(T^2 d / block) term.
    """
    T = Q.shape[0]
    io = 0
    # === YOUR CODE HERE ===
    for q_start in range(0, T, block_q):
        q_end = min(q_start + block_q, T)
        io += Q[q_start:q_end].nbytes                    # load Q tile
        for k_start in range(0, T, block_k):
            if causal and k_start > q_end - 1:
                break
            k_end = min(k_start + block_k, T)
            io += K[k_start:k_end].nbytes + V[k_start:k_end].nbytes
        io += V[q_start:q_end].nbytes                    # write O tile (T x d_v)
    return io


def peak_bytes(fn):
    """Real peak allocation of fn(), measured with tracemalloc.

    numpy reports its data allocations to tracemalloc, so this is the honest
    high-water mark of the call and not an estimate. Arrays created before the
    tracer starts are invisible to it, which is what we want: the number below
    is the memory the ALGORITHM needs on top of its inputs.
    """
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    result = fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    del result
    return peak - base


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # -----------------------------------------------------------------------
    print("\n--- stage 1: the online softmax is exact, the naive one is not ---")
    n = 4096
    print(f"      {'dtype':>8} {'range':>12} {'online rel.err':>15} "
          f"{'naive finite?':>14} {'naive nan/inf':>14}")
    online_errs = {}
    extreme32 = None
    for dtype in (np.float64, np.float32):
        for lo, hi in ((-3.0, 3.0), (-100.0, 100.0)):
            x = rng.uniform(lo, hi, n).astype(dtype)
            ref = softmax_two_pass(x.astype(np.float64))
            y_on = online_softmax(x, block=256).astype(np.float64)
            y_naive = softmax_naive_one_pass(x)
            rel = float(np.max(np.abs(y_on - ref)) / np.max(ref))
            n_bad = int((~np.isfinite(y_naive)).sum())
            online_errs[(np.dtype(dtype).name, hi)] = rel
            if dtype is np.float32 and hi == 100.0:
                extreme32 = (x, y_naive, online_softmax(x, block=256))
            print(f"      {np.dtype(dtype).name:>8} {f'+-{hi:g}':>12} {rel:15.3e} "
                  f"{str(n_bad == 0):>14} {n_bad:14d}")

    # The streaming recurrence reproduces the two-pass answer to the last bit
    # the format can carry: float64 lands at ~3e-16, float32 at ~1e-7 (which is
    # its own eps, 1.2e-7), and the extreme range does not degrade it.
    check("online == two-pass in float64, ordinary range",
          online_errs[("float64", 3.0)] < 1e-12)
    check("online == two-pass in float64, values spanning +-100",
          online_errs[("float64", 100.0)] < 1e-12)
    check("online survives float32 at +-100 (error is float32 eps, not blow-up)",
          online_errs[("float32", 100.0)] < 1e-6)

    # And the naive one-pass dies exactly where the exponent range runs out:
    # exp(100) is 2.7e43, comfortably past float32's largest finite value 3.4e38,
    # so the numerator is inf and inf/inf is nan.
    x32, y_naive32, y_on32 = extreme32
    print(f"      float32, +-100: naive gives {int(np.isnan(y_naive32).sum())} nan and "
          f"{int(np.isinf(y_naive32).sum())} inf out of {n} entries")
    print(f"                      online gives {int((~np.isfinite(y_on32)).sum())} non-finite "
          f"entries, and they sum to {float(np.sum(y_on32.astype(np.float64))):.12f}")
    check("naive one-pass is destroyed by float32 at +-100",
          int((~np.isfinite(y_naive32)).sum()) > 0)
    check("online one-pass is entirely finite there", np.isfinite(y_on32).all())
    check("and still normalized", float(np.sum(y_on32.astype(np.float64))), 1.0, tol=1e-6)
    # float64 has a wider exponent, so the same demonstration needs a wider
    # range — the failure is about exp overflowing the format, not about fp32.
    x64 = rng.uniform(-800, 800, n)
    check("in float64 the naive version needs +-800 to break, and does",
          int((~np.isfinite(softmax_naive_one_pass(x64))).sum()) > 0)
    check("online is fine at +-800 in float64",
          np.isfinite(online_softmax(x64, block=256)).all())

    # The mechanism, stated as a prefix property: after k blocks the running
    # (m, l) are EXACTLY the two-pass statistics of the first k blocks.
    x = rng.uniform(-100, 100, 4096)
    worst_m = worst_l = 0.0
    for k in (1, 2, 7, 16):
        m_k, l_k = online_softmax_stats(x[:k * 256], block=256)
        pre = x[:k * 256]
        worst_m = max(worst_m, abs(m_k - float(pre.max())))
        worst_l = max(worst_l, abs(l_k - float(np.exp(pre - pre.max()).sum())))
    print(f"      prefix statistics after 1/2/7/16 blocks: max |dm| = {worst_m:.1e}, "
          f"max |dl| = {worst_l:.1e}")
    check("running max is bit-exact against the prefix max", worst_m, 0.0, tol=0.0)
    check("running sum matches the prefix two-pass sum", worst_l < 1e-13)

    # -----------------------------------------------------------------------
    print("\n--- stage 2: tiled attention == naive attention ---")
    print(f"      {'T':>5} {'d':>4} {'causal':>7} {'Bq':>5} {'Bk':>5} {'max|diff|':>12}")
    worst = 0.0
    for T, d, d_v in ((256, 64, 64), (300, 32, 48)):
        Q = rng.standard_normal((T, d))
        K = rng.standard_normal((T, d))
        V = rng.standard_normal((T, d_v))
        for causal in (False, True):
            ref = attention_naive(Q, K, V, causal)
            for bq, bk in ((64, 64), (32, 128), (37, 53), (T, T)):
                got = attention_tiled(Q, K, V, bq, bk, causal)
                err = float(np.abs(got - ref).max())
                worst = max(worst, err)
                print(f"      {T:5d} {d:4d} {str(causal):>7} {bq:5d} {bk:5d} {err:12.3e}")
    check("tiled == naive for every (T, d, causal, block) above", worst < 1e-10)
    # Ragged blocks matter: 37 and 53 do not divide 256 or 300, so the last
    # tile of each loop is short. A kernel that assumes even division passes
    # the (64, 64) row and fails these.
    check("agreement is at fp64 round-off, not merely 'close'", worst < 1e-13)

    if HAVE_TORCH:
        Qt = torch.tensor(Q)[None, None]
        Kt = torch.tensor(K)[None, None]
        Vt = torch.tensor(V)[None, None]
        for causal in (False, True):
            ref_t = torch.nn.functional.scaled_dot_product_attention(
                Qt, Kt, Vt, is_causal=causal).numpy()[0, 0]
            err = float(np.abs(attention_tiled(Q, K, V, 64, 64, causal) - ref_t).max())
            print(f"      vs torch SDPA (causal={causal}): max|diff| = {err:.3e}")
            check(f"tiled == torch scaled_dot_product_attention (causal={causal})",
                  err < 1e-12)
    else:
        print("      (torch unavailable — skipping the SDPA cross-check)")

    # -----------------------------------------------------------------------
    print("\n--- the I/O argument, in bytes actually moved ---")
    print("      naive  : 2Td + 2Td_v + 4T^2 elements   (write S, read S, write P, read P)")
    print("      tiled  : Td + Td_v + ceil(T/Bq)(Td + Td_v) elements")
    print(f"      {'T':>5} {'d':>4} {'Bq=Bk':>6} {'naive MB':>10} {'tiled MB':>10} "
          f"{'ratio':>7} {'4Bq/(d+dv)':>11}")
    io_rows = []
    for T, d, B in ((1024, 64, 64), (1024, 64, 128), (1024, 64, 256),
                    (4096, 64, 128), (4096, 128, 128)):
        Q = np.zeros((T, d)); K = np.zeros((T, d)); V = np.zeros((T, d))
        bn = hbm_bytes_naive(Q, K, V)
        bt = hbm_bytes_tiled(Q, K, V, B, B)
        io_rows.append((T, d, B, bn, bt))
        print(f"      {T:5d} {d:4d} {B:6d} {bn/1e6:10.1f} {bt/1e6:10.1f} "
              f"{bn/bt:7.2f} {4*B/(2*d):11.2f}")
        # The counted bytes must equal the closed form, or the count is a story.
        elems = T*d + T*d + int(np.ceil(T/B)) * (T*d + T*d)
        check(f"counted tiled bytes match the closed form (T={T}, d={d}, B={B})",
              bt, elems * 8, tol=0.0)
        elems_n = 2*T*d + 2*T*d + 4*T*T
        check(f"counted naive bytes match the closed form (T={T}, d={d})",
              bn, elems_n * 8, tol=0.0)

    # Mechanism, not magnitude: the tiled traffic splits into a term paid once
    # (read every Q tile, write every O tile: 2Td elements, block-independent)
    # and the K/V re-reading term T^2(d + d_v)/Bq, which is the whole cost.
    # Doubling the block halves THAT term exactly — and that is the O(T^2 d / M)
    # claim, since the block size is what the SRAM budget M buys you: M elements
    # hold four B x d tiles, so B = M/4d and T^2 d / B becomes 4 T^2 d^2 / M.
    b64, b128, b256 = io_rows[0][4], io_rows[1][4], io_rows[2][4]
    once = 2 * 1024 * 64 * 8                     # Q read + O write, in bytes
    print(f"      T=1024,d=64: traffic at Bq=64/128/256 = "
          f"{b64/1e6:.1f} / {b128/1e6:.1f} / {b256/1e6:.1f} MB "
          f"(of which {once/1e6:.1f} MB is the block-independent Q/O term)")
    check("doubling the block halves the K/V re-reading term (64 -> 128)",
          (b64 - once) / (b128 - once), 2.0, tol=0.0)
    check("and again (128 -> 256)", (b128 - once) / (b256 - once), 2.0, tol=0.0)
    # The end-to-end ratio therefore climbs toward its asymptote 4Bq/(d+d_v)
    # from below as T grows and the once-paid term becomes negligible: at
    # Bq=128, d=d_v=64 it is 3.78 at T=1024 and 3.94 at T=4096, never 4.00.
    r_1024 = io_rows[1][3] / io_rows[1][4]
    r_4096 = io_rows[3][3] / io_rows[3][4]
    asymptote = 4 * 128 / (64 + 64)
    check("the naive/tiled ratio approaches 4Bq/(d+d_v) from below as T grows",
          r_1024 < r_4096 < asymptote)
    # Naive traffic is 4T^2 + O(Td) regardless of any block choice.
    check("naive traffic does not depend on the block size at all",
          io_rows[0][3] == io_rows[1][3] == io_rows[2][3])
    # Causal masking lets the tiled kernel skip whole future tiles.
    Q = np.zeros((4096, 64)); K = np.zeros((4096, 64)); V = np.zeros((4096, 64))
    bt_full = hbm_bytes_tiled(Q, K, V, 128, 128, causal=False)
    bt_causal = hbm_bytes_tiled(Q, K, V, 128, 128, causal=True)
    print(f"      T=4096 causal skipping: {bt_full/1e6:.1f} MB -> {bt_causal/1e6:.1f} MB")
    check("causal tiling skips roughly half the K/V tiles",
          bt_full / bt_causal > 1.8)

    # -----------------------------------------------------------------------
    print("\n--- peak memory: measured, not estimated ---")
    print(f"      {'T':>5} {'one T x T':>10} {'naive peak':>11} {'tiled peak':>11} "
          f"{'ratio':>7}")
    peaks_n, peaks_t = [], []
    for T in (512, 1024, 2048):
        d = 64
        Q = rng.standard_normal((T, d))
        K = rng.standard_normal((T, d))
        V = rng.standard_normal((T, d))
        pn = peak_bytes(lambda: attention_naive(Q, K, V))
        pt = peak_bytes(lambda: attention_tiled(Q, K, V, 128, 128))
        peaks_n.append(pn); peaks_t.append(pt)
        print(f"      {T:5d} {T*T*8/1e6:9.1f}M {pn/1e6:10.1f}M {pt/1e6:10.2f}M "
              f"{pn/pt:7.1f}")

    # The predicted mechanism: naive peak is O(T^2) and tiled peak is
    # O(T d + block^2). Assert the SCALING, which is the claim, rather than any
    # particular megabyte count.
    r_naive = [peaks_n[i+1] / peaks_n[i] for i in range(2)]
    r_tiled = [peaks_t[i+1] / peaks_t[i] for i in range(2)]
    print(f"      peak growth per doubling of T:  naive {r_naive[0]:.2f}x, {r_naive[1]:.2f}x"
          f"   tiled {r_tiled[0]:.2f}x, {r_tiled[1]:.2f}x")
    check("naive peak quadruples when T doubles (quadratic)",
          all(3.5 < r < 4.5 for r in r_naive))
    check("tiled peak grows sub-quadratically", all(r < 2.0 for r in r_tiled))
    # The sharpest statement: the tiled kernel's entire working set is smaller
    # than ONE copy of the score matrix it refuses to build.
    for T, pt in zip((512, 1024, 2048), peaks_t):
        check(f"tiled peak < one T x T score matrix (T={T})", pt < T * T * 8)
    # The naive pipeline must hold at least two T x T arrays live at once (the
    # scores and their exponentials), so its peak cannot come in under 2 T^2
    # elements no matter how carefully it is written. Measured: 3.0-3.1x.
    print("      naive peak / (T x T): " + ", ".join(
        f"{pn/(T*T*8):.2f}" for T, pn in zip((512, 1024, 2048), peaks_n)))
    check("naive peak holds at least two T x T arrays simultaneously",
          all(pn > 2 * T * T * 8 for T, pn in zip((512, 1024, 2048), peaks_n)))

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    def attention_tiled_no_acc_rescale(Q, K, V, block_q, block_k):
        """BUG (a): rescale the denominator l but forget the accumulator acc."""
        T, d = Q.shape
        d_v = V.shape[1]
        out = np.empty((T, d_v))
        for q_start in range(0, T, block_q):
            q_end = min(q_start + block_q, T)
            Qb = Q[q_start:q_end]
            m = np.full(q_end - q_start, -np.inf)
            l = np.zeros(q_end - q_start)
            acc = np.zeros((q_end - q_start, d_v))
            for k_start in range(0, T, block_k):
                k_end = min(k_start + block_k, T)
                s = Qb @ K[k_start:k_end].T / np.sqrt(d)
                m_new = np.maximum(m, np.max(s, axis=1))
                p = np.exp(s - m_new[:, None])
                alpha = np.exp(m - m_new)
                l = l * alpha + np.sum(p, axis=1)
                acc = acc + p @ V[k_start:k_end]       # <-- alpha missing
                m = m_new
            out[q_start:q_end] = acc / l[:, None]
        return out

    # A dedicated stream, so the numbers quoted in this section do not move when
    # anything above it changes how much randomness it consumes.
    rng_b = np.random.default_rng(27)
    T, d, B = 256, 64, 64
    Q = rng_b.standard_normal((T, d))
    K = rng_b.standard_normal((T, d))
    V = rng_b.standard_normal((T, d))

    def max_running_drift(Q_, K_, bk):
        """How far the running max climbs after each block — the quantity the
        missing alpha fails to correct for."""
        S = Q_ @ K_.T / np.sqrt(Q_.shape[1])
        bm = np.stack([S[:, i:i + bk].max(1) for i in range(0, S.shape[1], bk)])
        run = np.maximum.accumulate(bm, axis=0)
        return float((run[-1] - run).max())

    # (a1) The weak test vector: make every block maximum identical (all keys
    #      equal), so alpha == 1 and the bug is EXACTLY zero. A test suite built
    #      on this input certifies a broken kernel.
    K_tied = np.tile(K[0], (T, 1))
    ref_tied = attention_naive(Q, K_tied, V)
    bad_tied = attention_tiled_no_acc_rescale(Q, K_tied, V, B, B)
    err_tied = float(np.abs(bad_tied - ref_tied).max())
    print(f"      all keys identical (no max drift): broken kernel err = {err_tied:.2e}")
    check("with zero running-max drift the bug is invisible", err_tied < 1e-14)

    # (a2) The proportionality. Scale Q and K down and the score spread, hence
    #      the running-max drift, shrinks with it. Measure error against
    #      expm1(drift) — the exact factor the accumulator is off by.
    print(f"      {'score scale':>12} {'max drift':>10} {'expm1(drift)':>13} "
          f"{'broken err':>11} {'err/expm1':>10} {'correct err':>12}")
    scales = (1.0, 0.3, 0.1, 0.05, 0.02)
    ratios, errs, bounds, good_errs = [], [], [], []
    for scale in scales:
        Qs, Ks = Q * scale, K * scale
        ref = attention_naive(Qs, Ks, V)
        bad = attention_tiled_no_acc_rescale(Qs, Ks, V, B, B)
        good = attention_tiled(Qs, Ks, V, B, B)
        drift = max_running_drift(Qs, Ks, B)
        e_bad = float(np.abs(bad - ref).max())
        errs.append(e_bad)
        bounds.append(float(np.expm1(drift)))
        good_errs.append(float(np.abs(good - ref).max()))
        ratios.append(e_bad / np.expm1(drift))
        print(f"      {scale:12.2f} {drift:10.5f} {np.expm1(drift):13.3e} "
              f"{e_bad:11.3e} {ratios[-1]:10.4f} {good_errs[-1]:12.3e}")
    # The relation err ~ C * expm1(drift) is a LINEAR-RESPONSE statement, so it
    # is exact only as the drift goes to zero. In the small-drift rows (scale
    # <= 0.1, drift <= 0.02) C settles to 0.0926 and stays there to four digits;
    # at scale 1.0 the drift is 2.0 and the first-order picture degrades into a
    # mere bound, which is exactly why constancy is asserted where it applies.
    small = ratios[2:]
    print(f"      err/expm1(drift) -> [{min(small):.4f}, {max(small):.4f}] once the "
          f"drift is small; the whole sweep spans [{min(ratios):.4f}, {max(ratios):.4f}]")
    check("in the small-drift limit the error is exactly proportional to expm1(drift)",
          max(small) / min(small) < 1.01)
    check("and expm1(drift) bounds the error at every scale, drift = 2.0 included",
          all(e < b for e, b in zip(errs, bounds)))
    check("the error shrinks monotonically with the drift",
          all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)))
    check("the correct kernel is at round-off for every one of those scales",
          max(good_errs) < 1e-14)
    # And the practical consequence: the SAME broken kernel is 1e-4-accurate on
    # tame data and off by more than an absolute 1 on a sharply-peaked one.
    ref_tame = attention_naive(Q * 0.02, K * 0.02, V)
    err_tame = float(np.abs(attention_tiled_no_acc_rescale(Q * 0.02, K * 0.02, V, B, B)
                            - ref_tame).max())
    K_spike = K.copy()
    K_spike[3 * B:] *= 6.0                   # the LAST key block dominates
    ref_spike = attention_naive(Q, K_spike, V)
    err_spike = float(np.abs(attention_tiled_no_acc_rescale(Q, K_spike, V, B, B)
                             - ref_spike).max())
    good_spike = float(np.abs(attention_tiled(Q, K_spike, V, B, B) - ref_spike).max())
    print(f"      same broken kernel: tame input err = {err_tame:.2e}, "
          f"spiked-last-block err = {err_spike:.3f}")
    print(f"      the correct kernel on the spiked input: {good_spike:.2e}")
    check("a 1e-3 tolerance would pass the broken kernel on tame data",
          err_tame < 1e-3)
    check("the same kernel is off by more than 1.0 when one key block dominates",
          err_spike > 1.0)
    check("the input-dependence spans more than four orders of magnitude",
          err_spike / err_tame > 1e4)
    check("while the correct kernel is at round-off on BOTH inputs",
          good_spike < 1e-13)

    def attention_tiled_local_mask(Q, K, V, block_q, block_k):
        """BUG (b): apply the causal mask with LOCAL tile indices, so every tile
        gets its own lower triangle and the tile's absolute offset is lost."""
        T, d = Q.shape
        d_v = V.shape[1]
        out = np.empty((T, d_v))
        for q_start in range(0, T, block_q):
            q_end = min(q_start + block_q, T)
            Qb = Q[q_start:q_end]
            m = np.full(q_end - q_start, -np.inf)
            l = np.zeros(q_end - q_start)
            acc = np.zeros((q_end - q_start, d_v))
            for k_start in range(0, T, block_k):
                k_end = min(k_start + block_k, T)
                s = Qb @ K[k_start:k_end].T / np.sqrt(d)
                li = np.arange(q_end - q_start)[:, None]
                lj = np.arange(k_end - k_start)[None, :]
                s = np.where(li >= lj, s, -np.inf)     # <-- local, not absolute
                m_new = np.maximum(m, np.max(s, axis=1))
                m_safe = np.where(np.isneginf(m_new), 0.0, m_new)
                p = np.exp(s - m_safe[:, None])
                alpha = np.exp(m - m_safe)
                l = l * alpha + np.sum(p, axis=1)
                acc = acc * alpha[:, None] + p @ V[k_start:k_end]
                m = m_new
            out[q_start:q_end] = acc / l[:, None]
        return out

    T, d, B = 128, 32, 32
    Q = rng_b.standard_normal((T, d))
    K = rng_b.standard_normal((T, d))
    V = rng_b.standard_normal((T, d))
    ref = attention_naive(Q, K, V, causal=True)
    good = attention_tiled(Q, K, V, B, B, causal=True)
    bad = attention_tiled_local_mask(Q, K, V, B, B)
    print(f"      causal, T={T}, B={B}: correct tiling err = "
          f"{float(np.abs(good-ref).max()):.2e}, local-mask err = "
          f"{float(np.abs(bad-ref).max()):.3f}")
    check("the local-mask kernel does not compute causal attention",
          float(np.abs(bad - ref).max()) > 1.0)

    # Reconstruct the mask the buggy kernel actually implements.
    M_bad = np.zeros((T, T), dtype=bool)
    for q_start in range(0, T, B):
        for k_start in range(0, T, B):
            li = np.arange(min(B, T - q_start))[:, None]
            lj = np.arange(min(B, T - k_start))[None, :]
            M_bad[q_start:q_start + B, k_start:k_start + B] = li >= lj
    M_true = np.tril(np.ones((T, T), dtype=bool))
    leaked = int((M_bad & ~M_true).sum())
    blocked = int((~M_bad & M_true).sum())
    print(f"      implied mask: {leaked} future positions allowed, "
          f"{blocked} past positions blocked, out of {T*T}")
    print(f"      query 0 attends to keys {np.where(M_bad[0])[0].tolist()} "
          f"(should be [0])")
    check("the tiled-local mask is block-triangular, not triangular", leaked > 0)
    check("and it also throws away legitimate past keys", blocked > 0)

    # The consequence that actually matters: measurable future leakage. Perturb
    # a value vector at position B (which the local mask lets query 0 see) and
    # watch output[0] move.
    V2 = V.copy()
    V2[B] += 10.0
    leak_bad = float(np.abs(attention_tiled_local_mask(Q, K, V2, B, B)[0] - bad[0]).max())
    leak_good = float(np.abs(attention_tiled(Q, K, V2, B, B, causal=True)[0] - good[0]).max())
    print(f"      perturbing V[{B}] moves output[0] by {leak_bad:.3f} (local mask) "
          f"vs {leak_good:.1e} (absolute mask)")
    check("the local-mask kernel leaks the future into output[0]", leak_bad > 1e-3)
    check("the correct kernel's output[0] is bit-identical", leak_good, 0.0, tol=0.0)

    summary()
