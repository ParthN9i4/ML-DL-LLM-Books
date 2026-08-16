"""
ex01 — Linear algebra as computation.  (Book: Chapter 1)

Three ideas, in order of how much they will matter to you later:

  1. FLOP counting.  An (m x k) @ (k x n) matmul costs 2*m*n*k floating-point
     operations: one multiply and one add per (i, j, l) triple.  The factor of 2
     is not a detail — it is the difference between quoting 6ND and 3ND for
     training compute in Chapter 24.

  2. Arithmetic intensity = FLOPs / bytes moved.  This single number predicts
     whether an operation is limited by the processor or by memory.  A large
     matmul reuses each loaded element O(n) times and is compute-bound; a
     matrix-vector product touches each matrix element exactly once and is
     memory-bound.  Chapter 30 builds the roofline model on top of this.

  3. The SVD and Eckart-Young-Mirsky: the truncated SVD is the *optimal*
     rank-k approximation in both Frobenius and spectral norm.  This is why
     LoRA (Chapter 26) is a reasonable thing to do at all.

To learn: replace each function body below with `pass` and reimplement from the
docstring and the assertions at the bottom.  The assertions ARE the spec.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Cost model
# ---------------------------------------------------------------------------

def matmul_flops(m, k, n):
    """FLOPs for an (m x k) @ (k x n) matrix product.

    Each of the m*n output entries is a length-k dot product: k multiplies and
    k-1 adds.  The universal convention rounds that to 2*k, giving 2*m*n*k.
    """
    # === YOUR CODE HERE ===
    return 2 * m * n * k


def arithmetic_intensity(m, k, n, dtype_bytes=4):
    """FLOPs per byte moved, assuming each matrix crosses the memory boundary once.

    Bytes = (A: m*k) + (B: k*n) + (C: m*n), all times dtype_bytes.  This is the
    *ideal* a well-blocked kernel approaches, not what a naive loop moves.
    """
    # === YOUR CODE HERE ===
    flops = matmul_flops(m, k, n)
    bytes_moved = dtype_bytes * (m * k + k * n + m * n)
    return flops / bytes_moved


# ---------------------------------------------------------------------------
# 2. Blocked matrix multiplication
# ---------------------------------------------------------------------------

def matmul_blocked(A, B, block=64):
    """C = A @ B, computed in `block`-sized tiles.

    Mathematically identical to the naive triple loop; the point is locality.
    A tile of A and a tile of B are loaded once and reused `block` times, which
    is what turns a memory-bound computation into a compute-bound one.
    """
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, f"inner dimensions disagree: {k} vs {k2}"
    C = np.zeros((m, n), dtype=A.dtype)
    # === YOUR CODE HERE ===
    for i0 in range(0, m, block):
        i1 = min(i0 + block, m)
        for l0 in range(0, k, block):
            l1 = min(l0 + block, k)
            A_tile = A[i0:i1, l0:l1]
            for j0 in range(0, n, block):
                j1 = min(j0 + block, n)
                C[i0:i1, j0:j1] += A_tile @ B[l0:l1, j0:j1]
    return C


# ---------------------------------------------------------------------------
# 3. SVD by one-sided Jacobi
# ---------------------------------------------------------------------------

def svd_jacobi(A, tol=1e-12, max_sweeps=60):
    """Thin SVD A = U @ diag(S) @ Vt via one-sided Jacobi rotations.

    Orthogonalize the columns of a working copy of A by repeatedly rotating
    column pairs (p, q) to kill their inner product.  When every pair is
    orthogonal, the column norms are the singular values and the normalized
    columns are U; V accumulates the rotations.

    Do NOT call np.linalg.svd here — that is the reference we check against.
    """
    W = A.astype(np.float64, copy=True)
    m, n = W.shape
    V = np.eye(n)

    # === YOUR CODE HERE ===
    for _ in range(max_sweeps):
        off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                alpha = W[:, p] @ W[:, p]
                beta = W[:, q] @ W[:, q]
                gamma = W[:, p] @ W[:, q]
                if abs(gamma) <= tol * np.sqrt(alpha * beta):
                    continue
                off = max(off, abs(gamma) / np.sqrt(alpha * beta))
                # Rotation angle that annihilates the (p, q) inner product.
                zeta = (beta - alpha) / (2.0 * gamma)
                t = np.sign(zeta) / (abs(zeta) + np.sqrt(1.0 + zeta * zeta))
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = c * t
                Wp, Wq = W[:, p].copy(), W[:, q].copy()
                W[:, p] = c * Wp - s * Wq
                W[:, q] = s * Wp + c * Wq
                Vp, Vq = V[:, p].copy(), V[:, q].copy()
                V[:, p] = c * Vp - s * Vq
                V[:, q] = s * Vp + c * Vq
        if off == 0.0:
            break

    S = np.linalg.norm(W, axis=0)
    order = np.argsort(-S)
    S = S[order]
    U = np.zeros((m, n))
    nz = S > 1e-300
    U[:, nz] = W[:, order][:, nz] / S[nz]
    return U, S, V[:, order].T


def low_rank_approx(U, S, Vt, k):
    """Best rank-k approximation: keep the top k singular triplets."""
    # === YOUR CODE HERE ===
    return (U[:, :k] * S[:k]) @ Vt[:k, :]


# ---------------------------------------------------------------------------
# Checks — these ARE the specification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("\n--- cost model ---")
    check("matmul FLOPs 2mnk", matmul_flops(2, 3, 4), 48)
    # A 4096-cube fp32 matmul: 2*4096^3 FLOPs over 3*4096^2*4 bytes.
    ai_mm = arithmetic_intensity(4096, 4096, 4096)
    check("square matmul intensity ~682 FLOP/byte", ai_mm, 682.67, tol=0.5)
    # The same matrix applied to ONE vector reuses nothing.
    ai_mv = arithmetic_intensity(4096, 4096, 1)
    check("matvec intensity ~0.5 FLOP/byte", ai_mv, 0.5, tol=0.01)
    check("matmul is >1000x more intense than matvec", ai_mm / ai_mv > 1000)

    print("\n--- blocked matmul ---")
    A = rng.standard_normal((256, 192))
    B = rng.standard_normal((192, 224))
    ref = A @ B
    for blk in (16, 64, 256):
        got = matmul_blocked(A, B, block=blk)
        check(f"blocked(block={blk}) matches BLAS", got, ref, tol=1e-10)
    # Non-multiple block sizes must still handle the ragged edge correctly.
    check("ragged edge (block=50)", matmul_blocked(A, B, block=50), ref, tol=1e-10)

    print("\n--- SVD ---")
    # Plant a known spectrum so the test does not depend on random conditioning.
    m, n = 60, 40
    Uk, _ = np.linalg.qr(rng.standard_normal((m, n)))
    Vk, _ = np.linalg.qr(rng.standard_normal((n, n)))
    true_S = np.array([0.8 ** i for i in range(n)])
    A = (Uk * true_S) @ Vk.T

    t0 = time.time()
    U, S, Vt = svd_jacobi(A)
    jac_ms = 1e3 * (time.time() - t0)

    ref_S = np.linalg.svd(A, compute_uv=False)
    check("singular values match np.linalg.svd", S, ref_S, tol=1e-10)
    check("reconstruction U diag(S) Vt == A", (U * S) @ Vt, A, tol=1e-10)
    check("U has orthonormal columns", U.T @ U, np.eye(n), tol=1e-10)
    check("V has orthonormal columns", Vt @ Vt.T, np.eye(n), tol=1e-10)
    check("singular values are non-increasing", bool(np.all(np.diff(S) <= 1e-12)))

    print("\n--- Eckart-Young-Mirsky ---")
    for k in (1, 4, 12):
        Ak = low_rank_approx(U, S, Vt, k)
        err_f = np.linalg.norm(A - Ak, "fro")
        # The theorem: the error equals the root-sum-square of the DISCARDED
        # singular values, in Frobenius norm...
        check(f"k={k}: Frobenius error == sqrt(sum of dropped sigma^2)",
              err_f, np.sqrt(np.sum(S[k:] ** 2)), tol=1e-10)
        # ...and exactly sigma_{k+1} in spectral norm.
        check(f"k={k}: spectral error == sigma_{{k+1}}",
              np.linalg.norm(A - Ak, 2), S[k], tol=1e-10)
        # And no other rank-k matrix does better. Sample the alternatives.
        best_random = min(
            np.linalg.norm(A - (A @ Q) @ Q.T, "fro")
            for Q in (np.linalg.qr(rng.standard_normal((n, k)))[0] for _ in range(40))
        )
        check(f"k={k}: no sampled rank-{k} projection beats the SVD",
              err_f <= best_random + 1e-12)

    # -----------------------------------------------------------------------
    # BREAK IT — the failure modes worth seeing once
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) Blocking changes locality, NOT arithmetic. If a "speedup" changes the
    #     answer, you have a bug, not an optimization.
    check("blocking is numerically neutral",
          float(np.max(np.abs(matmul_blocked(A, Vk, 8) - matmul_blocked(A, Vk, 64)))) < 1e-12)

    # (b) Conditioning. Solving with a badly conditioned matrix loses roughly
    #     log10(kappa) digits — here the spectrum spans 0.8^39, so kappa is huge.
    kappa = S[0] / S[-1]
    digits_lost = np.log10(kappa)
    print(f"      kappa_2 = {kappa:.3e}, so expect to lose ~{digits_lost:.1f} decimal digits")
    check("planted spectrum really is ill-conditioned", kappa > 1e3)

    # (c) The classic: ||a-b||^2 computed as ||a||^2 + ||b||^2 - 2<a,b> suffers
    #     catastrophic cancellation when the two terms nearly agree. In float32
    #     the result goes NEGATIVE, and np.sqrt of it is nan — which is how this
    #     bug usually announces itself, several layers downstream in a k-NN or
    #     an RBF kernel. The error scales with ||x||^2, so push the magnitude up.
    worst = 0.0
    for _ in range(200):
        x = (rng.standard_normal((8, 3)) * 1000).astype(np.float32)
        sq = (x * x).sum(1)
        d2 = sq[:, None] + sq[None, :] - 2 * x @ x.T
        worst = min(worst, float(d2.min()))
    print(f"      most negative squared distance seen over 200 trials: {worst:.3e}")
    print(f"      np.sqrt of that is {np.sqrt(np.array(worst, dtype=np.float32)) if worst >= 0 else 'nan'}"
          " — clip at 0 before taking the root")
    check("Gram-trick distances really do go negative in float32", worst < 0.0)

    summary()
