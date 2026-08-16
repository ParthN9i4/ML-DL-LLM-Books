"""
ex18 — Selective state space models: recurrence, parallel scan, and duality.
(Book: Chapter 21)

A selective (input-dependent, diagonal) SSM computes, per channel:

    h_t = a_t * h_{t-1} + b_t * x_t          a_t, b_t depend on the INPUT
    y_t = c_t * h_t

Three ways to compute exactly the same thing, and the whole point of the
architecture is that all three exist:

  1. SEQUENTIAL: the obvious loop. O(T) time, O(T) sequential depth. This is
     what inference runs — constant memory per step, no KV cache.

  2. PARALLEL SCAN: the recurrence is an associative operation on pairs
     (a, b): (a2, b2) o (a1, b1) = (a1*a2, a2*b1 + b2). Associativity is what
     lets a Blelchley/Blelloch-style scan compute all prefixes in O(log T)
     parallel depth. This is what training runs.

  3. THE QUADRATIC (SSD) FORM: unroll the recurrence and y = M @ x where
     M[t, s] = c_t * (a_{t-1+1} * ... * a_t products) * b_s for s <= t — a
     masked matmul with a semiseparable mask. This is Mamba-2's structured
     state-space duality: the SSM *is* a form of masked linear attention.
     One matmul, tensor cores, no scan — at O(T^2) cost.

The checks force all three to agree to near machine precision, which is the
statement that "SSMs are linear attention with a structured mask" made
executable.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


def scan_sequential(a, b, x, c, h0=0.0):
    """The reference loop. a, b, x, c: (T,) each. Returns y: (T,)."""
    T = len(x)
    y = np.empty(T)
    h = h0
    # === YOUR CODE HERE ===
    for t in range(T):
        h = a[t] * h + b[t] * x[t]
        y[t] = c[t] * h
    return y


def scan_parallel(a, b, x, c, h0=0.0):
    """Blelloch-style associative scan over pairs (A, B) meaning h -> A*h + B.

    Combine rule (first u, then v):  (A_u, B_u) then (A_v, B_v)
        = (A_v * A_u,  A_v * B_u + B_v)
    After the scan, prefix t holds the map from h0 to h_t.

    Implemented with the recursive-doubling (Hillis-Steele) formulation, which
    is O(T log T) work but O(log T) DEPTH — depth is what matters both on a GPU
    and (Chapter 42) for multiplicative depth under encryption.
    """
    A = a.astype(np.float64).copy()
    B = (b * x).astype(np.float64)
    # === YOUR CODE HERE ===
    T = len(A)
    step = 1
    while step < T:
        # combine element t with element t-step (the prefix ending there)
        A_prev = A[:-step]
        B_prev = B[:-step]
        A_new = A.copy()
        B_new = B.copy()
        A_new[step:] = A[step:] * A_prev
        B_new[step:] = A[step:] * B_prev + B[step:]
        A, B = A_new, B_new
        step *= 2
    h = A * h0 + B
    return c * h


def ssd_matrix(a, b, c):
    """The semiseparable matrix M with y = M @ x equal to the recurrence.

    M[t, s] = c_t * prod(a[s+1..t]) * b_s   for s <= t, else 0.

    Built via cumulative log-products would underflow signs; build it with a
    cumulative-product ratio instead, guarding a=0. O(T^2) memory — the point
    is the equivalence, not efficiency.
    """
    T = len(a)
    # === YOUR CODE HERE ===
    M = np.zeros((T, T))
    # prod(a[s+1..t]) = P[t] / P[s], with P[t] = prod(a[1..t]) (P[0] = 1).
    # Stable enough here because |a| is bounded away from 0 in the tests;
    # production kernels use segment products instead.
    P = np.concatenate([[1.0], np.cumprod(a[1:])])
    for t in range(T):
        for s in range(t + 1):
            M[t, s] = c[t] * (P[t] / P[s]) * b[s]
    return M


def scan_ssd(a, b, x, c):
    """y via the quadratic masked-matmul form."""
    # === YOUR CODE HERE ===
    return ssd_matrix(a, b, c) @ x


def s4_kernel_conv(a_bar, b_bar, c_bar, x):
    """The LTI special case: constant (a, b, c) turns the SSM into a CONVOLUTION
    with kernel K = (c*b, c*a*b, c*a^2*b, ...), computable by FFT in O(T log T).

    Selectivity (input-dependent a_t, b_t) is precisely what destroys this:
    a time-varying system has no single convolution kernel. This function
    exists to make that boundary checkable.
    """
    T = len(x)
    # === YOUR CODE HERE ===
    K = c_bar * b_bar * a_bar ** np.arange(T)
    n = 1
    while n < 2 * T:
        n *= 2
    y = np.fft.irfft(np.fft.rfft(K, n) * np.fft.rfft(x, n), n)[:T]
    return y


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 512

    # Input-dependent parameters, as in Mamba: a in (0, 1) via sigmoid-like
    # gating, b and c bounded. Kept away from 0 so the SSD ratio form is stable.
    x = rng.standard_normal(T)
    a = 0.5 + 0.45 * (1 / (1 + np.exp(-rng.standard_normal(T))))   # (0.5, 0.95)
    b = 0.5 + rng.random(T)
    c = 0.5 + rng.random(T)

    print("\n--- three forms, one computation ---")
    t0 = time.perf_counter(); y_seq = scan_sequential(a, b, x, c); t_seq = time.perf_counter() - t0
    t0 = time.perf_counter(); y_par = scan_parallel(a, b, x, c); t_par = time.perf_counter() - t0
    t0 = time.perf_counter(); y_ssd = scan_ssd(a, b, x, c); t_ssd = time.perf_counter() - t0

    check("parallel scan == sequential recurrence", y_par, y_seq, tol=1e-9)
    check("SSD masked-matmul == sequential recurrence", y_ssd, y_seq, tol=1e-8)
    print(f"      wall-clock at T={T}: sequential {t_seq*1e3:.2f} ms, "
          f"parallel-scan {t_par*1e3:.2f} ms, SSD matmul {t_ssd*1e3:.2f} ms")
    print("      (in numpy the loop-free forms win by vectorization; on a GPU the "
          "scan wins by DEPTH — log2(512) = 9 steps versus 512)")

    # Nonzero initial state must thread through the scan correctly too.
    y_seq_h0 = scan_sequential(a, b, x, c, h0=3.0)
    y_par_h0 = scan_parallel(a, b, x, c, h0=3.0)
    check("nonzero initial state handled identically", y_par_h0, y_seq_h0, tol=1e-9)

    print("\n--- the associativity that makes the scan legal ---")
    # (u o v) o w == u o (v o w) for the combine rule — checked numerically over
    # many random triples. Without this, the parallel scan would be WRONG, not slow.
    ok = True
    for _ in range(500):
        Au, Bu, Av, Bv, Aw, Bw = rng.standard_normal(6)
        uv = (Av * Au, Av * Bu + Bv)
        left = (Aw * uv[0], Aw * uv[1] + Bw)
        vw = (Aw * Av, Aw * Bv + Bw)
        right = (vw[0] * Au, vw[0] * Bu + vw[1])
        ok &= abs(left[0] - right[0]) < 1e-12 and abs(left[1] - right[1]) < 1e-9
    check("the (A, B) combine rule is associative (500 random triples)", ok)

    print("\n--- the semiseparable structure ---")
    M = ssd_matrix(a, b, c)
    check("M is lower triangular (causality)",
          float(np.abs(np.triu(M, k=1)).max()), 0.0, tol=0.0)
    # The defining property of a (rank-1) semiseparable matrix: every submatrix
    # taken from the lower-triangular part has rank 1. Check a few.
    ok = True
    for _ in range(20):
        t_lo = int(rng.integers(T // 2, T - 8))
        s_hi = int(rng.integers(8, T // 2))
        sub = M[t_lo:t_lo + 8, s_hi - 8:s_hi]
        s_vals = np.linalg.svd(sub, compute_uv=False)
        ok &= s_vals[1] < 1e-8 * max(s_vals[0], 1e-30)
    check("off-diagonal blocks have numerical rank 1 (semiseparable)", ok)

    print("\n--- the LTI boundary: constant params = convolution ---")
    a0, b0, c0 = 0.9, 0.7, 1.3
    aa, bb, cc = np.full(T, a0), np.full(T, b0), np.full(T, c0)
    y_lti_scan = scan_sequential(aa, bb, x, cc)
    y_lti_conv = s4_kernel_conv(a0, b0, c0, x)
    check("constant-parameter SSM == FFT convolution", y_lti_conv, y_lti_scan, tol=1e-9)

    # And the destruction: for the SELECTIVE system, the best any fixed kernel
    # can do is measurably worse — fit the least-squares optimal Toeplitz kernel
    # and show real residual remains.
    from numpy.lib.stride_tricks import sliding_window_view
    x_pad = np.concatenate([np.zeros(T - 1), x])
    X_toep = sliding_window_view(x_pad, T)[:, ::-1]     # row t = [x_t, x_{t-1}, ...]
    k_best, *_ = np.linalg.lstsq(X_toep, y_seq, rcond=None)
    resid = np.linalg.norm(X_toep @ k_best - y_seq) / np.linalg.norm(y_seq)
    # Same fit on the LTI output recovers the kernel essentially exactly — the
    # honest comparison is against that near-zero floor, not an arbitrary
    # percentage (a mildly selective system with T free kernel parameters can
    # be fit surprisingly well; the first draft asserted >5% and measured 3.1%).
    k_lti, *_ = np.linalg.lstsq(X_toep, y_lti_scan, rcond=None)
    resid_lti = np.linalg.norm(X_toep @ k_lti - y_lti_scan) / np.linalg.norm(y_lti_scan)
    print(f"      best fixed kernel residual: selective {resid*100:.2f}%  vs  LTI {resid_lti:.2e}")
    check("a fixed kernel fits the LTI system to near machine precision", resid_lti < 1e-8)
    check("the selective system leaves a residual orders of magnitude above that floor",
          resid > 1e4 * max(resid_lti, 1e-12))

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) The combine-order bug: writing the rule as (A_u*A_v, A_v*B_u + B_v) is
    #     right, but swapping which operand's A multiplies which B — i.e.
    #     (A_u*A_v, A_u*B_v + B_u) — composes in the WRONG ORDER. When is it
    #     invisible? The first draft of this exercise claimed "on constant
    #     parameters" — false, and the test said so: B_t = b*x_t still varies
    #     with the input, so reversed composition differs (measured max error
    #     6.9 on constant params with real input). It is invisible exactly when
    #     the B values being combined are equal — i.e. constant parameters AND
    #     constant input. Which is precisely the weak all-ones test vector that
    #     lets this bug ship.
    def scan_wrong_order(a_, b_, x_, c_):
        A = a_.astype(np.float64).copy()
        B = (b_ * x_).astype(np.float64)
        Tn = len(A)
        step = 1
        while step < Tn:
            A_new, B_new = A.copy(), B.copy()
            A_new[step:] = A[step:] * A[:-step]
            B_new[step:] = A[:-step] * B[step:] + B[:-step]   # wrong order
            A, B = A_new, B_new
            step *= 2
        return c_ * B

    ones = np.ones(T)
    y_wrong_allconst = scan_wrong_order(aa, bb, ones, cc)
    y_ref_allconst = scan_sequential(aa, bb, ones, cc)
    y_wrong_realx = scan_wrong_order(aa, bb, x, cc)
    y_wrong_sel = scan_wrong_order(a, b, x, c)
    e_allconst = float(np.abs(y_wrong_allconst - y_ref_allconst).max())
    e_realx = float(np.abs(y_wrong_realx - y_lti_scan).max())
    e_sel = float(np.abs(y_wrong_sel - y_seq).max() / np.abs(y_seq).max())
    print(f"      wrong-order scan, constant params + constant input: max error {e_allconst:.2e}  (invisible)")
    print(f"      wrong-order scan, constant params + REAL input:     max error {e_realx:.2e}")
    print(f"      wrong-order scan, selective params + real input:    relative error {e_sel:.2f}")
    check("the order bug is invisible on the all-constant test vector", e_allconst < 1e-9)
    check("a real input exposes it even with constant parameters", e_realx > 1e-3)
    check("...and it is catastrophic on selective ones", e_sel > 0.1)

    # (b) Depth accounting: why the naive encrypted scan is unaffordable. The
    #     sequential form consumes one multiplicative level per step; the
    #     parallel scan consumes one per ROUND. At T=512 that is 512 versus 9 —
    #     count them explicitly.
    seq_depth = T
    par_depth = int(np.ceil(np.log2(T)))
    print(f"      multiplicative depth at T={T}: sequential {seq_depth}, parallel scan {par_depth}")
    check("parallel scan needs exponentially less depth", par_depth <= 2 * np.log2(T))

    summary()
