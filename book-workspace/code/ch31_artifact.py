"""Artifact 31.1 -- Tiled online-softmax attention (FlashAttention forward) in NumPy.

Verifies four claims from the chapter:
  (a) tiled output == naive attention to ~1e-6, across tile sizes incl. ragged edges;
  (b) running-rescale online softmax == batch softmax to ~1e-12 on a chunked row;
  (c) peak intermediate memory is O(N), not O(N^2)  (explicit allocation accounting);
  (d) HBM-equivalent reads of K/V blocks scale as N^2 * d / Br  (inverse in tile size),
      matching the paper's O(N^2 d^2 / M) analysis with Br ~ M/d.
No GPU is used or simulated for timing; "HBM reads" here means counted array elements
that a real kernel would have to fetch from off-chip memory. Everything else is exact.
"""
import numpy as np

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- reference path
def naive_attention(Q, K, V):
    """Standard attention: materializes the full N x N score matrix.

    Returns (output, peak_intermediate_elements). The peak intermediate is the
    N x N score matrix plus its N x N softmax -- what a framework without a fused
    kernel actually allocates (scores and probs coexist during softmax)."""
    N, d = Q.shape
    S = (Q @ K.T) / np.sqrt(d)                    # N x N   <- the quadratic buffer
    m = S.max(axis=1, keepdims=True)              # stabilize
    P = np.exp(S - m)
    P /= P.sum(axis=1, keepdims=True)             # N x N softmax
    return P @ V, 2 * N * N                        # S and P both live: 2*N^2 elements


def batch_softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------------------------------------------- (b) online softmax, one row
def online_softmax(x, chunk):
    """Streaming softmax over a 1-D array seen in chunks of size `chunk`.

    Maintains running max m and running normalizer l; every time a new chunk
    raises the max, the old normalizer is rescaled by exp(m_old - m_new).
    Algebraically identical to batch softmax (Theorem 31.1)."""
    m, l = -np.inf, 0.0
    for j in range(0, len(x), chunk):
        xj = x[j:j + chunk]
        m_new = max(m, xj.max())
        l = l * np.exp(m - m_new) + np.exp(xj - m_new).sum()
        m = m_new
    # second pass just to emit probabilities (attention never needs this pass:
    # it folds the division into the running output accumulator instead)
    return np.exp(x - m) / l


# ------------------------------------------------- (a,c,d) tiled forward pass
def flash_forward(Q, K, V, Br, Bc):
    """FlashAttention forward pass: tile Q into Br-row blocks, K/V into Bc-row
    blocks; never materialize more than a Br x Bc score tile.

    Returns (O, logsumexp L, stats) where stats counts
      kv_read_elems : elements of K and V blocks loaded across the whole loop
                      (the HBM traffic a real kernel pays), and
      peak_elems    : max intermediate elements alive inside the tile loops
                      (excludes inputs Q,K,V and the returned O,L)."""
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    O = np.zeros((N, d))
    L = np.zeros(N)                                # logsumexp per row, kept for backward
    kv_read_elems = 0
    peak_elems = 0
    for i in range(0, N, Br):                      # outer loop over query blocks
        Qi = Q[i:i + Br]                           # Br x d (ragged at the edge)
        bi = Qi.shape[0]
        mi = np.full(bi, -np.inf)                  # running row max
        li = np.zeros(bi)                          # running normalizer
        Oi = np.zeros((bi, d))                     # unnormalized output accumulator
        for j in range(0, N, Bc):                  # inner loop over key/value blocks
            Kj, Vj = K[j:j + Bc], V[j:j + Bc]      # each "loaded from HBM"
            bj = Kj.shape[0]
            kv_read_elems += 2 * bj * d            # count the load: K block + V block
            Sij = (Qi @ Kj.T) * scale              # bi x bj score TILE -- the only
            m_new = np.maximum(mi, Sij.max(axis=1))  # quadratic-shaped object, tiny
            Pij = np.exp(Sij - m_new[:, None])     # bi x bj
            alpha = np.exp(mi - m_new)             # rescale factor for stale state
            li = li * alpha + Pij.sum(axis=1)
            Oi = Oi * alpha[:, None] + Pij @ Vj    # rescale old accumulator, add new
            mi = m_new
            # everything alive right now inside the loops, in elements:
            live = (bi * d          # Qi
                    + 2 * bj * d    # Kj, Vj
                    + 2 * bi * bj   # Sij, Pij
                    + 3 * bi        # mi, li, alpha (m_new aliases into mi next)
                    + bi            # m_new
                    + bi * d)       # Oi
            peak_elems = max(peak_elems, live)
        O[i:i + Br] = Oi / li[:, None]             # single deferred division per row
        L[i:i + Br] = mi + np.log(li)              # logsumexp: all backward needs
    return O, L, {"kv_read_elems": kv_read_elems, "peak_elems": peak_elems}


# --------------------------------------------------------------------- checks
if __name__ == "__main__":
    N, d = 257, 64                                 # deliberately not a tile multiple
    Q = rng.standard_normal((N, d))
    K = rng.standard_normal((N, d))
    V = rng.standard_normal((N, d))
    O_ref, naive_peak = naive_attention(Q, K, V)

    # (b) online softmax == batch softmax to ~1e-12, chunked ragged
    x = rng.standard_normal(1000) * 5              # spread ~ +-15: overflow-prone naively
    worst_b = 0.0
    for chunk in (1, 7, 100, 333, 1000):
        err = np.abs(online_softmax(x, chunk) - batch_softmax(x)).max()
        worst_b = max(worst_b, err)
        assert err < 1e-12, (chunk, err)
    print(f"(b) online softmax vs batch, worst over chunk sizes {{1,7,100,333,1000}}: "
          f"max|diff| = {worst_b:.2e}  (< 1e-12)")

    # (a) tiled forward == naive across tile shapes, incl. ragged edges
    print("\n(a) tiled forward vs naive (N=257, d=64):")
    worst = 0.0
    for (Br, Bc) in [(16, 16), (32, 64), (64, 32), (64, 53), (128, 128), (257, 257), (100, 17)]:
        O, L, st = flash_forward(Q, K, V, Br, Bc)
        err = np.abs(O - O_ref).max()
        worst = max(worst, err)
        assert err < 1e-6, (Br, Bc, err)
        print(f"    Br={Br:>3} Bc={Bc:>3}  max|diff| = {err:.2e}")
    print(f"    worst over all tilings: {worst:.2e}  (< 1e-6)")

    # (c) peak intermediate memory: O(N) vs the naive O(N^2)
    _, _, st = flash_forward(Q, K, V, 64, 64)
    print(f"\n(c) peak intermediates, Br=Bc=64: tiled {st['peak_elems']:,} elements "
          f"vs naive {naive_peak:,} elements "
          f"(ratio {naive_peak / st['peak_elems']:.1f}x); tiled peak is independent of N "
          f"given fixed tiles (plus the N x d output, which naive also needs)")
    assert st["peak_elems"] < 6 * N * d            # O(N*d) bound, comfortably
    assert st["peak_elems"] * 4 < naive_peak       # far below the N^2 buffers
    # peak must not grow with N at fixed tile size (pure O(1) inner state):
    N2 = 2 * N
    Q2, K2, V2 = (rng.standard_normal((N2, d)) for _ in range(3))
    _, _, st2 = flash_forward(Q2, K2, V2, 64, 64)
    assert st2["peak_elems"] == st["peak_elems"], "inner-loop state grew with N"
    print(f"    doubling N to {N2}: tiled peak unchanged at {st2['peak_elems']:,} elements; "
          f"naive would need {2 * N2 * N2:,}")

    # (d) K/V HBM reads vs tile size: predicted = ceil(N/Br) * N * 2d
    print("\n(d) K/V read accounting (N=257, d=64):  reads = ceil(N/Br) * N * 2d")
    print(f"    {'Br':>4} {'measured':>12} {'predicted':>12} {'vs one pass over K,V':>22}")
    for Br in (16, 32, 64, 128, 257):
        _, _, st = flash_forward(Q, K, V, Br, 64)
        Tr = -(-N // Br)                           # ceil
        pred = Tr * N * 2 * d
        assert st["kv_read_elems"] == pred, (Br, st["kv_read_elems"], pred)
        print(f"    {Br:>4} {st['kv_read_elems']:>12,} {pred:>12,} {Tr:>20}x")
    print("    reads scale as 1/Br exactly: bigger query tiles (more SRAM) => fewer "
          "passes over K,V. With Br ~ M/d this is the O(N^2 d^2 / M) of Dao et al. 2022.")

    # cross-check against torch SDPA if available
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        ref = torch.nn.functional.scaled_dot_product_attention(
            torch.from_numpy(Q)[None], torch.from_numpy(K)[None], torch.from_numpy(V)[None]
        )[0].numpy()
        O, _, _ = flash_forward(Q, K, V, 64, 64)
        err = np.abs(O - ref).max()
        assert err < 1e-6
        print(f"\ncross-check vs torch SDPA: max|diff| = {err:.2e}")
    else:
        print("\n[skipped: torch not installed]")
    print("\nall assertions passed")
