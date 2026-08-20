"""
ex28 — Ring all-reduce and the bandwidth argument.  (Book: Chapter 32)

Data-parallel training is a sum. Every rank computes a gradient over its own
shard of the batch, and every rank must end the step holding the same total.
The obvious way to do that is to ship everything to a root, add it up there,
and send the answer back — and it is the reason nobody could scale past a
handful of GPUs for years, because the root's link carries (P-1) N bytes and
that grows without bound.

Ring all-reduce removes P from the cost. Lay the ranks out in a circle, split
the tensor into P chunks, and run P-1 reduce-scatter steps in which each rank
sends one chunk forward and adds one chunk it receives, followed by P-1
all-gather steps that circulate the finished chunks. Every rank sends exactly
2 (P-1)/P * N bytes, which is under 2N for every P and converges to 2N: the
per-rank volume counted here rises by only a factor 1.1384 going from P=8 to
P=256, while the naive root's rises by 36.43, and the ratio of the two
per-rank VOLUMES is exactly P/2 at every P. That is the whole argument, and
this file makes it by COUNTING bytes inside the transfer loop — every P in
the volume tables, 256 included, is actually simulated and counted, never
extrapolated from a big-O.

What the ring buys in bandwidth it pays for in latency: 2(P-1) strictly
sequential hops. Under a stated 5 us/hop and 25 GB/s, the message size at
which the latency term equals the bandwidth term is alpha * bw * P — 1.00 MB
at P=8, 32 MB at P=256 — and below ~286 KB at P=8 the naive two-hop scheme is
genuinely faster. That is why DDP buckets gradients (200 small tensors cost
14.06 ms one at a time and 0.13 ms bucketed, 110x) and why tree algorithms
exist at all (at P=256 a ring pays 510 hops, a binary tree 16).

The one insight the break-it section is built to make undeniable is that an
off-by-one in the ring index moves exactly the same bytes and leaves every
rank holding bit-identical data, so byte counters and cross-rank consistency
checks pass — and because each (destination, chunk) pair receives at most one
message per reduce-scatter, it can only DROP contributions, never duplicate
them. WHERE the off-by-one sits decides how loud it is. In the reduce-scatter
it drops P-1 of the P contributions per chunk and any gradient-norm dashboard
catches it instantly (norm ratio 0.360). In the all-gather it drops exactly
ONE of P, so the NORM moves by only 6.6% while the gradient itself is 36%
wrong per element — that is the variant no check people actually run can see.
Bit-exact agreement between ranks proves consensus, not correctness.

To learn: replace each function body with `pass` and reimplement.
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


# ---------------------------------------------------------------------------
# Stage 1 — the ring, simulated rank by rank, with every byte counted
# ---------------------------------------------------------------------------

def ring_allreduce(x, rs_offset=0, ag_offset=0):
    """Full ring all-reduce over P simulated ranks. x is (P, N), row r is rank r.

    Reduce-scatter (P-1 steps): at step k rank r sends chunk (r - k) mod P to
    rank (r+1) mod P, which ADDS it into its own copy of that same chunk. The
    chunk index each rank sends walks BACKWARDS while the data walks forwards;
    after P-1 steps rank r owns the fully reduced chunk (r+1) mod P.

    All-gather (P-1 steps): at step k rank r sends chunk (r + 1 - k) mod P to
    rank (r+1) mod P, which OVERWRITES. After P-1 steps every rank has every
    reduced chunk, bit-for-bit identical, because the all-gather copies.

    rs_offset / ag_offset shift the chunk index by a constant; 0/0 is correct
    and the nonzero settings are the off-by-ones dissected in the break-it
    section. The data still flows to rank r+1 in every case — which is exactly
    why the bug looks right when you read the code.

    Returns (out, stats). stats counts BYTES ACTUALLY MOVED, per rank,
    accumulated inside the transfer loop; nothing in it is a formula.
    """
    P, N = x.shape
    if N % P:
        raise ValueError("N must be divisible by P for an exact ring")
    C = N // P
    itemsize = x.dtype.itemsize
    buf = x.copy()
    sent = np.zeros(P, dtype=np.int64)
    recv = np.zeros(P, dtype=np.int64)
    steps = 0

    def sl(i):
        return slice((i % P) * C, (i % P) * C + C)

    # === YOUR CODE HERE ===
    for k in range(P - 1):                                   # reduce-scatter
        idx = [(r - k + rs_offset) % P for r in range(P)]
        wire = [buf[r, sl(idx[r])].copy() for r in range(P)]  # all sends concurrent
        for r in range(P):
            dst = (r + 1) % P
            buf[dst, sl(idx[r])] += wire[r]
            sent[r] += C * itemsize
            recv[dst] += C * itemsize
        steps += 1
    for k in range(P - 1):                                   # all-gather
        idx = [(r + 1 - k + ag_offset) % P for r in range(P)]
        wire = [buf[r, sl(idx[r])].copy() for r in range(P)]
        for r in range(P):
            dst = (r + 1) % P
            buf[dst, sl(idx[r])] = wire[r]
            sent[r] += C * itemsize
            recv[dst] += C * itemsize
        steps += 1

    return buf, {"sent": sent, "recv": recv, "steps": steps, "chunk_bytes": C * itemsize}


def naive_allreduce(x):
    """Gather everything to rank 0, sum there, broadcast the result back.

    The obvious algorithm, and the one whose cost is proportional to P: the
    root both receives and sends (P-1) whole tensors while every other rank
    moves exactly one. Instrumented identically, so the two algorithms can be
    compared by counted bytes rather than by an asserted big-O.
    """
    P, N = x.shape
    itemsize = x.dtype.itemsize
    sent = np.zeros(P, dtype=np.int64)
    recv = np.zeros(P, dtype=np.int64)
    # === YOUR CODE HERE ===
    acc = x[0].copy()
    for r in range(1, P):                       # gather
        acc += x[r]
        sent[r] += N * itemsize
        recv[0] += N * itemsize
    out = np.empty_like(x)
    for r in range(P):                          # broadcast
        out[r] = acc
        if r != 0:
            sent[0] += N * itemsize
            recv[r] += N * itemsize
    return out, {"sent": sent, "recv": recv, "steps": 2, "chunk_bytes": N * itemsize}


def sequential_sum(x, start):
    """Sum the rows of x in the exact order start, start+1, ... (mod P).

    The bit-exact reference for one ring chunk: the ring reduces chunk c by
    walking the ranks from c onwards, so a bit-exact reference must use that
    same order. Float addition is commutative but NOT associative, so the
    order is the only thing that has to match — and the only thing that can
    make two mathematically identical reductions differ.
    """
    P = x.shape[0]
    # === YOUR CODE HERE ===
    acc = x[start % P].copy()
    for j in range(1, P):
        acc = acc + x[(start + j) % P]
    return acc


def contribution_multiplicity(P, N, **kw):
    """How many times does rank r's data land in chunk c of the final answer?

    Run the ring on one-hot inputs (rank r holds all-ones, everyone else
    zeros) and read the answer off. A correct all-reduce gives the all-ones
    P x P matrix; anything else localizes the damage exactly. This is the
    instrument that turns "the answer is wrong" into "rank 3 never reaches
    chunk 5".
    """
    C = N // P
    M = np.zeros((P, P), dtype=int)
    # === YOUR CODE HERE ===
    for r in range(P):
        probe = np.zeros((P, N))
        probe[r] = 1.0
        out, _ = ring_allreduce(probe, **kw)
        for c in range(P):
            M[c, r] = int(round(float(out[0, c * C])))
    return M


# ---------------------------------------------------------------------------
# Stage 2 — the cost model
# ---------------------------------------------------------------------------

def ring_bytes_per_rank(P, nbytes):
    """The textbook formula: 2 (P-1)/P * N bytes sent per rank. It is checked
    against the counted bytes at every P the volume tables print (256
    included); in the alpha-beta TIME model further down it is a model input,
    not a measurement."""
    # === YOUR CODE HERE ===
    return 2.0 * (P - 1) / P * nbytes


def naive_bytes_at_root(P, nbytes):
    """(P-1) N sent by the root, and (P-1) N received by it. Linear in P."""
    # === YOUR CODE HERE ===
    return (P - 1) * float(nbytes)


def ring_time(P, nbytes, alpha, bw):
    """alpha-beta model: 2(P-1) sequential hops, each paying latency alpha,
    plus 2(P-1)/P * N bytes down a link of bandwidth bw."""
    # === YOUR CODE HERE ===
    return 2 * (P - 1) * alpha + ring_bytes_per_rank(P, nbytes) / bw


def naive_time(P, nbytes, alpha, bw):
    """Two logical hops of latency (gather, then broadcast) — the most
    GENEROUS possible model for the naive scheme, assuming the root overlaps
    all P-1 receives — but the root's own link still carries (P-1) N bytes."""
    # === YOUR CODE HERE ===
    return 2 * alpha + naive_bytes_at_root(P, nbytes) / bw


def latency_dominant_bytes(P, alpha, bw):
    """Message size at which the ring's latency term equals its bandwidth term.

        2(P-1) alpha  ==  (2(P-1)/P) N / bw     =>     N = alpha * bw * P

    The (P-1) cancels: break-even grows linearly in P, not quadratically.
    Below this size the all-reduce is latency-bound and the ring is the wrong
    algorithm; well above it the ring is effectively free of P.
    """
    # === YOUR CODE HERE ===
    return alpha * bw * P


def ring_naive_crossover_bytes(P, alpha, bw):
    """Message size at which ring_time == naive_time under the same model.

        2(P-1)a + 2(P-1)/P * N/bw  =  2a + (P-1) N/bw
        2(P-2)a = (P-1)(1 - 2/P) N/bw = (P-1)(P-2)/P * N/bw
        N = 2 a bw P / (P-1)                                    (P > 2)

    Below it the ring's 2(P-1) latency hops cost more than the root's extra
    bandwidth; above it the ring wins and keeps winning.
    """
    # === YOUR CODE HERE ===
    if P <= 2:
        return float("inf")
    return 2.0 * alpha * bw * P / (P - 1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("\n--- the ring computes the exact sum, and every rank agrees bit-for-bit ---")
    P, N = 8, 2400
    x = rng.standard_normal((P, N))
    out, st = ring_allreduce(x)
    C = N // P

    check("all ranks bit-identical after all-gather",
          out, np.repeat(out[:1], P, axis=0), tol=0.0)

    worst = 0.0
    for c in range(P):
        ref = sequential_sum(x[:, c * C:(c + 1) * C], start=c)
        worst = max(worst, float(np.abs(out[0, c * C:(c + 1) * C] - ref).max()))
    print(f"      max |ring - same-order sequential sum| = {worst:.1e}")
    check("each chunk is bit-exactly the sum in the ring's own order", worst, 0.0, tol=0.0)

    ref_np = x.sum(axis=0)
    err_np = float(np.abs(out[0] - ref_np).max())
    print(f"      max |ring - np.sum| = {err_np:.2e}   (different order, same value)")
    check("ring == np.sum to float64 rounding", out[0], ref_np, tol=1e-12)

    out_n, st_n = naive_allreduce(x)
    check("naive gather/broadcast agrees with the ring", out_n[0], out[0], tol=1e-12)
    # Chunk 0 is reduced by the ring in the order 0,1,...,P-1 — the same order
    # the naive root uses — so those two agree to the last bit and no others do.
    check("chunk 0 agrees BIT-exactly (identical reduction order)",
          out_n[0, :C], out[0, :C], tol=0.0)
    later = float(np.abs(out_n[0, C:] - out[0, C:]).max())
    print(f"      chunks 1..7 differ from the naive order by up to {later:.2e} "
          f"(pure re-association)")
    check("chunks reduced in a different order do NOT agree bit-exactly", later > 0.0)

    print("\n--- counted bytes vs the 2(P-1)/P N formula ---")
    print(f"      {'P':>4} {'N bytes':>10} {'sent/rank':>12} {'2(P-1)/P N':>12} "
          f"{'recv/rank':>12} {'steps':>7}")
    for P_ in (2, 3, 4, 6, 8, 12):
        xs = rng.standard_normal((P_, 2400))
        nbytes = xs[0].nbytes
        _, s = ring_allreduce(xs)
        formula = ring_bytes_per_rank(P_, nbytes)
        print(f"      {P_:4d} {nbytes:10d} {s['sent'][0]:12d} {formula:12.0f} "
              f"{s['recv'][0]:12d} {s['steps']:7d}")
        check(f"P={P_}: counted sent bytes == 2(P-1)/P N exactly",
              float(s['sent'][0]), formula, tol=0.0)
        check(f"P={P_}: every rank sends the same amount (no hot spot)",
              float(s['sent'].max() - s['sent'].min()), 0.0, tol=0.0)
        # Rank r's inbound link IS rank r-1's outbound link, so recv must be
        # sent rolled one position around the ring. (In a healthy ring both
        # arrays are uniform, so this coincides with plain sent == recv;
        # crediting a receive to the sender instead of the destination is
        # invisible to any aggregate per-rank counter here. The attribution
        # only becomes falsifiable where the traffic is ASYMMETRIC — the
        # naive scheme below — which is why its recv side gets its own
        # checks.)
        check(f"P={P_}: rank r receives exactly what rank r-1 sent",
              s['recv'], np.roll(s['sent'], 1), tol=0.0)
        check(f"P={P_}: every byte sent is received (conservation)",
              int(s['sent'].sum()), int(s['recv'].sum()))
        check(f"P={P_}: 2(P-1) sequential steps", s['steps'], 2 * (P_ - 1))
    # The naive scheme's counted bytes, same instrument, for contrast.
    _, sn = naive_allreduce(rng.standard_normal((8, 2400)))
    print(f"      naive at P=8: root sends {sn['sent'][0]}, leaf sends {sn['sent'][1]}, "
          f"root/ring = {sn['sent'][0] / ring_bytes_per_rank(8, 2400*8):.1f}x")
    check("naive root's counted bytes == (P-1) N exactly",
          float(sn['sent'][0]), naive_bytes_at_root(8, 2400 * 8), tol=0.0)
    check("the naive root is the hot spot: it moves (P-1)x a leaf",
          float(sn['sent'][0]) / float(sn['sent'][1]), 7.0, tol=1e-12)
    # The naive traffic is asymmetric, so here the RECEIVE attribution is
    # actually falsifiable: the root's inbound link carries (P-1) N (the
    # gather), every leaf's carries exactly N (its broadcast copy).
    check("naive root receives (P-1) N — its inbound link is the hot spot too",
          float(sn['recv'][0]), naive_bytes_at_root(8, 2400 * 8), tol=0.0)
    check("each naive leaf receives exactly N (one broadcast copy)",
          sn['recv'][1:], np.full(7, 2400 * 8, dtype=np.int64), tol=0.0)

    print("\n--- why P barely matters for the ring, and matters entirely for the naive ---")
    # Every row of this table is COUNTED by running the transfer loops at that
    # P — including P=256 — and only then checked against the closed forms.
    # N = 2304 = 2^8 * 3^2 elements so that every P below divides it exactly.
    N_scale = 2304
    nbytes = N_scale * 8
    # Byte counts are data-independent; a local generator keeps the main rng
    # stream (and every number quoted from the sections below) untouched.
    rng_scale = np.random.default_rng(32)
    print(f"      {'P':>5} {'ring sent/rank':>16} {'/N':>7} {'naive sent@root':>17} {'/N':>8}")
    ring_tab, naive_tab = {}, {}
    for P_ in (2, 4, 8, 16, 64, 256):
        _, s_r = ring_allreduce(rng_scale.standard_normal((P_, N_scale)))
        _, s_n = naive_allreduce(rng_scale.standard_normal((P_, N_scale)))
        rb, nb = int(s_r['sent'][0]), int(s_n['sent'][0])
        check(f"P={P_}: counted ring bytes/rank == 2(P-1)/P N exactly",
              float(rb), ring_bytes_per_rank(P_, nbytes), tol=0.0)
        check(f"P={P_}: counted naive root bytes == (P-1) N exactly",
              float(nb), naive_bytes_at_root(P_, nbytes), tol=0.0)
        ring_tab[P_], naive_tab[P_] = rb, nb
        print(f"      {P_:5d} {rb:16d} {rb/nbytes:7.3f} {nb:17d} {nb/nbytes:8.1f}")
    check("ring per-rank volume is bounded by 2N for every P",
          all(v < 2.0 * nbytes for v in ring_tab.values()))
    grow_ring = ring_tab[256] / ring_tab[8]
    grow_naive = naive_tab[256] / naive_tab[8]
    print(f"      P=8 -> P=256 (32x the ranks): ring x{grow_ring:.4f}, naive x{grow_naive:.2f}")
    check("naive volume grows exactly as (P-1)", grow_naive, 255 / 7, tol=1e-12)
    check("ring volume grows by under 15% across that same 32x in P", grow_ring < 1.15)
    for P_ in (2, 4, 8, 64):
        check(f"P={P_}: naive/ring per-rank volume == P/2 exactly",
              naive_tab[P_] / ring_tab[P_], P_ / 2.0, tol=1e-12)

    print("\n--- the latency term: bandwidth-optimal, latency-LINEAR ---")
    ALPHA = 5e-6        # stated model input: 5 us per hop
    BW = 25e9           # stated model input: 25 GB/s per link, per direction
    print(f"      model inputs: alpha = {ALPHA*1e6:.0f} us/hop, bw = {BW/1e9:.0f} GB/s")
    print(f"      {'P':>5} {'lat hops':>9} {'lat time':>10} {'N* (lat==bw)':>14} "
          f"{'ring>naive above':>17}")
    for P_ in (4, 8, 16, 64, 256):
        lat = 2 * (P_ - 1) * ALPHA
        nstar = latency_dominant_bytes(P_, ALPHA, BW)
        ncross = ring_naive_crossover_bytes(P_, ALPHA, BW)
        print(f"      {P_:5d} {2*(P_-1):9d} {lat*1e6:9.0f}us {nstar/1e6:12.3f}MB "
              f"{ncross/1e3:15.1f}KB")
    for P_ in (4, 8, 64):
        nstar = latency_dominant_bytes(P_, ALPHA, BW)
        check(f"P={P_}: at N* the latency and bandwidth terms are equal",
              ring_bytes_per_rank(P_, nstar) / BW, 2 * (P_ - 1) * ALPHA, tol=1e-15)
        ncross = ring_naive_crossover_bytes(P_, ALPHA, BW)
        check(f"P={P_}: at the crossover N, ring_time == naive_time",
              ring_time(P_, ncross, ALPHA, BW), naive_time(P_, ncross, ALPHA, BW),
              tol=1e-15)

    # Locate the crossover by scanning the model, not by trusting the algebra.
    P_ = 8
    sizes = np.logspace(2, 9, 20001)
    ring_t = np.array([ring_time(P_, n, ALPHA, BW) for n in sizes])
    naive_t = np.array([naive_time(P_, n, ALPHA, BW) for n in sizes])
    scanned = sizes[np.where(ring_t < naive_t)[0][0]]
    closed = ring_naive_crossover_bytes(P_, ALPHA, BW)
    print(f"      P=8: scanned crossover {scanned/1e3:.1f} KB, "
          f"closed form {closed/1e3:.1f} KB")
    check("scanned crossover matches the closed form to grid resolution",
          scanned, closed, tol=0.002 * closed)
    check("below the crossover the naive scheme is genuinely faster",
          naive_time(P_, closed / 10, ALPHA, BW) < ring_time(P_, closed / 10, ALPHA, BW))
    check("above it the ring wins", ring_time(P_, 100 * closed, ALPHA, BW)
          < naive_time(P_, 100 * closed, ALPHA, BW))

    # Which is exactly why DDP buckets gradients before all-reducing them.
    n_small, n_tensors = 4096, 200
    unbucketed = n_tensors * ring_time(P_, n_small, ALPHA, BW)
    bucketed = ring_time(P_, n_tensors * n_small, ALPHA, BW)
    print(f"      {n_tensors} tensors of {n_small} B: one-at-a-time "
          f"{unbucketed*1e3:.2f} ms, bucketed {bucketed*1e3:.2f} ms "
          f"({unbucketed/bucketed:.0f}x)")
    check("bucketing wins because 2(P-1)alpha is paid once, not 200 times",
          unbucketed / bucketed > 10.0)
    # And the reason tree/hierarchical algorithms exist at all: their latency
    # is O(log P), so in the latency-bound regime they beat the ring outright.
    tree_hops = 2 * int(np.ceil(np.log2(256)))
    print(f"      at P=256 a ring pays {2*255} latency hops; a binary tree pays "
          f"{tree_hops}")
    check("tree latency is logarithmic, ring latency is linear",
          2 * 255 / tree_hops > 15.0)

    print("\n--- what this costs a real training step ---")
    PARAMS = 7e9
    STEP_COMPUTE = 2.0          # stated model input: seconds of compute per step
    print(f"      {PARAMS/1e9:.0f}B params, P=8 ranks, {STEP_COMPUTE:.1f} s compute/step")
    print(f"      {'dtype':>6} {'grads':>9} {'moved/rank':>12} "
          f"{'12.5 GB/s':>11} {'25 GB/s':>10} {'50 GB/s':>10}")
    tt = {}
    for name, itemsize in (("fp32", 4), ("bf16", 2)):
        gb = PARAMS * itemsize
        row = []
        for bw in (12.5e9, 25e9, 50e9):
            t = ring_time(8, gb, ALPHA, bw)
            tt[(name, bw)] = t
            row.append(t)
        print(f"      {name:>6} {gb/1e9:8.1f}G {ring_bytes_per_rank(8, gb)/1e9:11.1f}G "
              f"{row[0]:10.2f}s {row[1]:9.2f}s {row[2]:9.2f}s")
    for bw in (12.5e9, 25e9, 50e9):
        exp32 = max(0.0, tt[("fp32", bw)] - STEP_COMPUTE)
        exp16 = max(0.0, tt[("bf16", bw)] - STEP_COMPUTE)
        print(f"      at {bw/1e9:4.1f} GB/s, exposed after perfect overlap: "
              f"fp32 {exp32:.2f} s, bf16 {exp16:.2f} s "
              f"(step becomes {(STEP_COMPUTE+exp32)/STEP_COMPUTE:.2f}x / "
              f"{(STEP_COMPUTE+exp16)/STEP_COMPUTE:.2f}x longer)")
    check("halving the dtype halves the all-reduce time exactly",
          tt[("fp32", BW)] / tt[("bf16", BW)], 2.0, tol=1e-3)
    check("doubling the bandwidth halves it too",
          tt[("fp32", 12.5e9)] / tt[("fp32", 25e9)], 2.0, tol=1e-3)
    # The regime boundary, measured rather than guessed: at 12.5 GB/s the fp32
    # all-reduce is longer than the whole compute step and cannot be hidden;
    # at 25 GB/s it fits under it, but only barely.
    check("at 12.5 GB/s the fp32 all-reduce exceeds the compute step",
          tt[("fp32", 12.5e9)] > STEP_COMPUTE)
    check("at 25 GB/s it fits under it", tt[("fp32", BW)] < STEP_COMPUTE)
    occupancy = tt[("fp32", BW)] / STEP_COMPUTE
    print(f"      at 25 GB/s the fp32 all-reduce occupies {occupancy*100:.0f}% of the "
          f"step: it fits, with {100*(1-occupancy):.0f}% slack")
    check("but it occupies over 90% of the step, so there is no headroom",
          occupancy > 0.9)
    lat_frac = 2 * 7 * ALPHA / tt[("fp32", BW)]
    print(f"      latency is {lat_frac*100:.5f}% of that time — at 28 GB it is noise")
    check("at 28 GB the latency term is below 1e-4 of the total", lat_frac < 1e-4)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) Off-by-one in the ring index. The data still flows to rank+1; only
    #     the chunk index is shifted. Every message is sent, every byte is
    #     counted, nothing raises — and the contribution graph is wrong.
    P, N = 8, 2400
    C = N // P
    x = rng.standard_normal((P, N))
    truth = x.sum(axis=0)
    good, sg = ring_allreduce(x)
    M_good = contribution_multiplicity(P, N)
    check("the correct ring counts every (chunk, rank) contribution exactly once",
          M_good.tolist(), np.ones((P, P), dtype=int).tolist(), tol=0.0)

    for label, kw in (("reduce-scatter index +1", dict(rs_offset=1)),
                      ("all-gather index +1", dict(ag_offset=1))):
        bad, sb = ring_allreduce(x, **kw)
        M = contribution_multiplicity(P, N, **kw)
        per_chunk = M.sum(axis=1)
        rel = float(np.abs(bad[0] - truth).max() / np.abs(truth).max())
        rms_ratio = float(np.sqrt((bad[0] ** 2).mean()) / np.sqrt((truth ** 2).mean()))
        print(f"      {label}: contributions per chunk {per_chunk.tolist()} "
              f"(correct: {P})")
        print(f"        max relative elementwise error {rel:.3f}; "
              f"gradient-norm ratio {rms_ratio:.3f}")
        check(f"{label}: moves exactly the same bytes as the correct ring",
              float(np.abs(sg['sent'] - sb['sent']).max()), 0.0, tol=0.0)
        check(f"{label}: all ranks still agree bit-for-bit (consistency check passes)",
              bad, np.repeat(bad[:1], P, axis=0), tol=0.0)
        check(f"{label}: drops contributions", int((M == 0).sum()) > 0)
        check(f"{label}: the answer is wrong by a large relative margin", rel > 0.1)
        # Ground M in the data, not just in symmetry. Every constant-offset M
        # is a circulant, so its row sums (printed above) cannot tell a wrong
        # chunk index inside contribution_multiplicity from a right one.
        # Rebuilding each chunk from the raw shards M says survived, and
        # matching it against the broken tensor, can: reading the wrong probe
        # element or transposing M fails HERE (verified by mutation).
        recon = np.concatenate([(M[c][:, None] * x[:, c * C:(c + 1) * C]).sum(0)
                                for c in range(P)])
        recon_err = float(np.abs(bad[0] - recon).max())
        print(f"        rebuilt from the shards M says survived: "
              f"max |bad - rebuild| = {recon_err:.1e}")
        check(f"{label}: M reconstructs the broken tensor from raw shards",
              recon_err, 0.0, tol=1e-12)

    # The structural fact behind all of it: in a ring, each (destination,
    # chunk) pair receives AT MOST ONE message per reduce-scatter, whatever
    # constant the index is shifted by. So an index off-by-one can only ever
    # DROP contributions — it can never double-count them. Sweep every
    # single-offset variant and confirm.
    n_variants = worst_missing = 0
    any_double = False
    worst_recon = 0.0
    for rs_o, ag_o in itertools.product((-1, 0, 1), repeat=2):
        if (rs_o, ag_o) == (0, 0):
            continue
        M = contribution_multiplicity(P, N, rs_offset=rs_o, ag_offset=ag_o)
        n_variants += 1
        any_double |= bool((M > 1).any())
        worst_missing = max(worst_missing, int((M == 0).sum()))
        # Same data-grounding as above, for every variant in the sweep: the
        # multiplicity table must reproduce each variant's actual output.
        bad_v, _ = ring_allreduce(x, rs_offset=rs_o, ag_offset=ag_o)
        recon = np.concatenate([(M[c][:, None] * x[:, c * C:(c + 1) * C]).sum(0)
                                for c in range(P)])
        worst_recon = max(worst_recon, float(np.abs(bad_v[0] - recon).max()))
    print(f"      swept {n_variants} index-offset variants: double-counting seen in "
          f"{'some' if any_double else 'NONE'}; worst case {worst_missing}/{P*P} "
          f"contributions dropped; worst rebuild error {worst_recon:.1e}")
    check("no index offset can ever double-count a contribution", not any_double)
    check("but they all drop contributions", worst_missing > 0)
    check("every variant's M reconstructs that variant's output from raw shards",
          worst_recon, 0.0, tol=1e-12)

    # And the plausibility trap: the all-gather off-by-one loses exactly ONE of
    # the P contributions per chunk, so the gradient norm — the thing people
    # actually watch on a dashboard — barely moves.
    bad_ag, _ = ring_allreduce(x, ag_offset=1)
    M_ag = contribution_multiplicity(P, N, ag_offset=1)
    norm_ratio = float(np.linalg.norm(bad_ag[0]) / np.linalg.norm(truth))
    per_elem = float(np.sqrt(((bad_ag[0] - truth) ** 2).mean())
                     / np.sqrt((truth ** 2).mean()))
    print(f"      all-gather off-by-one: exactly {P - M_ag.sum(1)[0]} of {P} "
          f"contributions lost per chunk")
    print(f"        grad-norm ratio {norm_ratio:.4f} (a {abs(1-norm_ratio)*100:.1f}% "
          f"dashboard wobble) but per-element RMS error {per_elem*100:.1f}%")
    check("the norm-based sanity check is fooled (within 10% of correct)",
          abs(1.0 - norm_ratio) < 0.10)
    check("while the actual gradient is wrong by more than 30% RMS", per_elem > 0.30)
    # sqrt(1/P) is the mechanism: drop one of P i.i.d. contributions and the
    # error is one contribution's worth against P of them.
    check("the RMS error is one contribution out of P, i.e. ~1/sqrt(P)",
          per_elem, 1.0 / np.sqrt(P), tol=0.05)

    # (b) Floating-point non-determinism. Rotating the ring (relabelling which
    #     physical rank is the ring's origin) changes the ORDER of every
    #     chunk's reduction. Float addition is not associative, so the bits
    #     change — with the same seed, the same data and the same code.
    print("      ---")
    P, N = 8, 4096
    xf = (rng.standard_normal((P, N))
          * np.exp(rng.standard_normal((P, N)) * 2.0)).astype(np.float32)

    def ring_rotated(a, shift):
        o, _ = ring_allreduce(np.roll(a, shift, axis=0))
        return np.roll(o, -shift, axis=0)

    r0 = ring_rotated(xf, 0)[0]
    r3 = ring_rotated(xf, 3)[0]
    ndiff = int((r0 != r3).sum())
    d0, d3 = r0.astype(np.float64), r3.astype(np.float64)
    absdiff = float(np.abs(d0 - d3).max())
    scale = float(np.abs(d0).max())
    eps32 = float(np.finfo(np.float32).eps)
    truth64 = xf.astype(np.float64).sum(axis=0)
    print(f"      rotating the ring changes {ndiff}/{N} float32 words "
          f"({100*ndiff/N:.1f}%)")
    print(f"      max |difference| = {absdiff:.3e} on a value scale of {scale:.3e} "
          f"= {absdiff/scale/eps32:.2f} float32 eps")
    check("the two orders are NOT bit-identical", ndiff > 0)
    check("both are equally 'correct' against the float64 truth",
          float(np.abs(d0 - truth64).max()) < 1e-4 * scale
          and float(np.abs(d3 - truth64).max()) < 1e-4 * scale)
    check("the discrepancy is rounding-scale, not a logic error",
          absdiff / scale < 100 * eps32)
    check("yet it is strictly larger than zero — reproducibility is gone",
          absdiff > 0.0)
    # It is not that one order is wrong; NEITHER is exactly right, and which
    # ring NCCL builds depends on topology and process launch order.
    check("neither order equals the float64 sum exactly",
          float(np.abs(d0 - truth64).max()) > 0.0
          and float(np.abs(d3 - truth64).max()) > 0.0)

    # Why a last-bit difference ends a reproducibility guarantee: training is a
    # nonlinear dynamical system, and nonlinear dynamical systems amplify.
    # Take the coordinate where the two orders actually disagree (54% of them
    # do; picking element 0 blindly would land on a bit-identical one), and
    # scale it down to a plausible gradient contribution.
    j = int(np.argmax(np.abs(d0 - d3)))
    w_a = w_b = np.float64(0.4)
    g_a, g_b = d0[j] * 1e-9, d3[j] * 1e-9
    print(f"      coordinate {j}: the two orders give {g_a:.17e} vs {g_b:.17e}")
    gaps = []
    for _ in range(60):
        w_a = 3.9 * w_a * (1 - w_a) + g_a
        w_b = 3.9 * w_b * (1 - w_b) + g_b
        gaps.append(abs(w_a - w_b))
    print(f"      feeding the two into a chaotic update: gap {gaps[0]:.2e} after "
          f"1 step, {gaps[-1]:.2e} after 60")
    check("a last-bit gradient difference is amplified by >1e6 in 60 steps",
          gaps[-1] / max(gaps[0], 1e-300) > 1e6)
    check("and becomes macroscopic — order of the parameter itself",
          gaps[-1] > 1e-3)

    summary()
