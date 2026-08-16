"""
Artifact 1.1 -- Blocked matmul + from-scratch one-sided Jacobi SVD.  No np.linalg.svd
inside any implementation; it appears only in verification.  Four things are checked:
(1) blocked matmul matches np.dot, and GFLOP/s depends strongly on block size;
(2) Jacobi recovers singular values to ~1e-13 relative error and reconstructs A to
machine precision; (3) Eckart-Young-Mirsky survives a brute-force search over 5,000
random rank-k approximants per k, each optimally fitted inside its own random column
space (a far stronger competitor than a random product UV^T); (4) measured arithmetic
intensity, matmul versus matvec.  Runtime ~9 s on 4 CPU cores.
"""

import time
import numpy as np

try:
    import torch
except ImportError:  # cross-check only; the artifact must run without it
    torch = None

# --- 1. Blocked matrix multiplication ---------------------------------------

def matmul_blocked(A, B, bs=64):
    """C = A @ B computed in bs x bs tiles.  The three outer loops walk tiles; the
    inner statement is a dense product on tiles that (for small bs) fit in cache
    together with the output tile.  Loop order i-p-j keeps the A tile resident across
    the whole j sweep, so it is loaded once per (i, p) pair, not once per triple."""
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, "inner dimensions must agree"
    C = np.zeros((m, n), dtype=A.dtype)
    for i0 in range(0, m, bs):
        i1 = min(i0 + bs, m)
        for p0 in range(0, k, bs):
            p1 = min(p0 + bs, k)
            A_tile = A[i0:i1, p0:p1]          # stays in cache across the j loop
            for j0 in range(0, n, bs):
                j1 = min(j0 + bs, n)
                C[i0:i1, j0:j1] += A_tile @ B[p0:p1, j0:j1]
    return C


def bench_matmul(n=768, block_sizes=(16, 32, 64, 128, 256, 768), reps=3):
    """Time matmul_blocked at several block sizes.  2*n^3 FLOPs per call."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    ref = A @ B
    flops = 2.0 * n ** 3
    rows = []
    for bs in block_sizes:
        C = matmul_blocked(A, B, bs)          # warm-up + correctness
        res = np.max(np.abs(C - ref)) / np.max(np.abs(ref))
        t0 = time.perf_counter()
        for _ in range(reps):
            matmul_blocked(A, B, bs)
        dt = (time.perf_counter() - t0) / reps
        rows.append((bs, dt, flops / dt / 1e9, res))
    t0 = time.perf_counter()
    for _ in range(reps):
        A @ B
    dt_blas = (time.perf_counter() - t0) / reps
    return rows, dt_blas, flops

# --- 2. One-sided Jacobi SVD -------------------------------------------------

def jacobi_svd(A, tol=1e-15, max_sweeps=60):
    """Thin SVD A = U diag(s) V^T by one-sided Jacobi.

    Orthogonalise the *columns* of A by plane rotations G(p, q, theta) applied on the
    right.  When every column pair is orthogonal, W = A V has orthogonal columns: their
    norms are the singular values, the normalised columns are the left singular vectors.
    For a pair (p, q) with alpha = w_p.w_p, beta = w_q.w_q, gamma = w_p.w_q, choose theta
    to annihilate gamma.  With zeta = (beta - alpha)/(2 gamma), the stable root of
    t^2 + 2 zeta t - 1 = 0 is t = sign(zeta)/(|zeta| + sqrt(1+zeta^2)); then
    c = 1/sqrt(1+t^2), s = c t.  This form avoids cancellation when |zeta| is large."""
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    transposed = m < n
    if transposed:                      # algorithm wants tall-and-skinny
        A = A.T
        m, n = A.shape
    W = A.copy()
    V = np.eye(n)
    for sweep in range(max_sweeps):
        off = 0.0                       # largest relative |gamma| seen this sweep
        for p in range(n - 1):
            for q in range(p + 1, n):
                alpha = W[:, p] @ W[:, p]
                beta = W[:, q] @ W[:, q]
                gamma = W[:, p] @ W[:, q]
                if gamma == 0.0 or alpha == 0.0 or beta == 0.0:
                    continue
                rel = abs(gamma) / np.sqrt(alpha * beta)
                off = max(off, rel)
                if rel <= tol:
                    continue
                zeta = (beta - alpha) / (2.0 * gamma)
                t = np.sign(zeta) / (abs(zeta) + np.sqrt(1.0 + zeta * zeta))
                if zeta == 0.0:
                    t = 1.0
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = c * t
                wp, wq = W[:, p].copy(), W[:, q].copy()
                W[:, p] = c * wp - s * wq
                W[:, q] = s * wp + c * wq
                vp, vq = V[:, p].copy(), V[:, q].copy()
                V[:, p] = c * vp - s * vq
                V[:, q] = s * vp + c * vq
        if off <= tol:
            break
    s = np.sqrt(np.sum(W * W, axis=0))          # column norms = singular values
    order = np.argsort(s)[::-1]                 # descending, by convention
    s = s[order]
    W = W[:, order]
    V = V[:, order]
    U = np.zeros_like(W)
    nz = s > 0
    U[:, nz] = W[:, nz] / s[nz]                 # normalise; null directions left at 0
    if transposed:
        return V, s, U                          # (A^T = U S V^T)  =>  A = V S U^T
    return U, s, V


def truncate(U, s, V, k):
    """Rank-k truncated SVD reconstruction."""
    return (U[:, :k] * s[:k]) @ V[:, :k].T

# --- 3. Brute-force check of Eckart-Young-Mirsky -----------------------------

def brute_force_rank_k(A, k, trials, rng):
    """Best Frobenius error over `trials` random rank-k approximants.  Each competitor
    is the orthogonal projection of A onto a random k-dimensional column space S:
    B = P_S A minimises ||A - B||_F over all rank-<=k matrices with column space in S.
    So this searches over subspaces, not over random factor pairs."""
    m, _ = A.shape
    best = np.inf
    for _ in range(trials):
        S = rng.standard_normal((m, k))
        Q, _ = np.linalg.qr(S)                  # orthonormal basis, not an SVD call
        err = np.linalg.norm(A - Q @ (Q.T @ A), "fro")
        best = min(best, err)
    return best

# --- 4. Arithmetic intensity -------------------------------------------------

def intensity_report(n=2048):
    """Measure matmul (n x n)(n x n) against matvec (n x n)(n x 1), fp64."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    x = rng.standard_normal((n, 1))
    A @ B; A @ x                                 # warm caches / threads
    t0 = time.perf_counter(); A @ B; t_mm = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        A @ x
    t_mv = (time.perf_counter() - t0) / 20
    f_mm, f_mv = 2.0 * n ** 3, 2.0 * n * n
    b_mm, b_mv = 8.0 * 3 * n * n, 8.0 * (n * n + 2 * n)
    return dict(n=n, t_mm=t_mm, t_mv=t_mv,
                ai_mm=f_mm / b_mm, ai_mv=f_mv / b_mv,
                gf_mm=f_mm / t_mm / 1e9, gf_mv=f_mv / t_mv / 1e9,
                bw_mv=b_mv / t_mv / 1e9)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("=" * 72)
    print("PART 1  Blocked matmul, 768 x 768 fp64, 2*768^3 = %.3f GFLOP" % (2 * 768 ** 3 / 1e9))
    print("=" * 72)
    rows, dt_blas, flops = bench_matmul()
    print(f"{'block':>7} {'seconds':>10} {'GFLOP/s':>10} {'rel.residual':>14}")
    for bs, dt, gf, res in rows:
        print(f"{bs:>7} {dt:>10.4f} {gf:>10.2f} {res:>14.3e}")
        assert res < 1e-13, "blocked matmul disagrees with np.dot"
    print(f"{'BLAS':>7} {dt_blas:>10.4f} {flops / dt_blas / 1e9:>10.2f}")

    print()
    print("=" * 72)
    print("PART 2  One-sided Jacobi SVD, A is 60 x 40 with a planted spectrum")
    print("=" * 72)
    m, n, r = 60, 40, 40
    Ua, _ = np.linalg.qr(rng.standard_normal((m, r)))
    Va, _ = np.linalg.qr(rng.standard_normal((n, r)))
    sig = np.array([2.0 ** (-i / 3.0) for i in range(r)])   # decays over ~5 decades
    A = (Ua * sig) @ Va.T
    t0 = time.perf_counter()
    U, s, V = jacobi_svd(A)
    t_jac = time.perf_counter() - t0
    s_ref = np.linalg.svd(A, compute_uv=False)              # verification only
    rel_s = np.max(np.abs(s - s_ref) / s_ref)
    recon = np.max(np.abs((U * s) @ V.T - A))
    orth_u = np.max(np.abs(U.T @ U - np.eye(n)))
    orth_v = np.max(np.abs(V.T @ V - np.eye(n)))
    print(f"jacobi wall clock         {t_jac * 1e3:.1f} ms")
    print(f"max rel. error in sigma   {rel_s:.3e}   (target < 1e-10)")
    print(f"max |U S V^T - A|         {recon:.3e}")
    print(f"max |U^T U - I|           {orth_u:.3e}")
    print(f"max |V^T V - I|           {orth_v:.3e}")
    print(f"sigma_1 = {s[0]:.6f}   sigma_40 = {s[-1]:.6e}   kappa_2 = {s[0] / s[-1]:.4e}")
    assert rel_s < 1e-10 and recon < 1e-12 and max(orth_u, orth_v) < 1e-12

    if torch is not None:
        s_t = torch.linalg.svdvals(torch.from_numpy(A)).numpy()
        print(f"torch cross-check         max|s - torch| = {np.max(np.abs(s - s_t)):.3e}")
    else:
        print("torch cross-check         [skipped: torch not installed]")

    print()
    print("=" * 72)
    print("PART 3  Eckart-Young-Mirsky by brute force")
    print("=" * 72)
    print(f"{'k':>3} {'||A-A_k||_F':>13} {'sqrt(sum s^2)':>14} {'best random':>13} "
          f"{'||A-A_k||_2':>13} {'sigma_{k+1}':>13}")
    for k in (1, 2, 4, 8):
        Ak = truncate(U, s, V, k)
        e_f = np.linalg.norm(A - Ak, "fro")
        tail = np.sqrt(np.sum(s[k:] ** 2))
        e_2 = np.linalg.norm(A - Ak, 2)
        best = brute_force_rank_k(A, k, 5000, rng)
        print(f"{k:>3} {e_f:>13.9f} {tail:>14.9f} {best:>13.9f} {e_2:>13.9f} {s[k]:>13.9f}")
        assert abs(e_f - tail) < 1e-12, "Frobenius identity failed"
        assert abs(e_2 - s[k]) < 1e-10, "spectral identity failed"
        assert best >= e_f - 1e-12, "brute force beat the SVD -- Eckart-Young violated"

    # Local check: perturb the optimal subspace and watch the error rise (2nd order).
    k = 4
    base = np.linalg.norm(A - truncate(U, s, V, k), "fro")
    for eps in (1e-1, 1e-2, 1e-3):
        Q, _ = np.linalg.qr(U[:, :k] + eps * rng.standard_normal((m, k)))
        e = np.linalg.norm(A - Q @ (Q.T @ A), "fro")
        print(f"  perturb optimal subspace by eps={eps:<6} -> excess error {e - base:.3e}")
        assert e >= base - 1e-13

    print()
    print("=" * 72)
    print("PART 4  Arithmetic intensity, fp64")
    print("=" * 72)
    rep = intensity_report()
    print(f"n = {rep['n']}")
    print(f"matmul  {rep['t_mm'] * 1e3:9.2f} ms   {rep['gf_mm']:8.2f} GFLOP/s   "
          f"AI = {rep['ai_mm']:8.1f} FLOP/byte")
    print(f"matvec  {rep['t_mv'] * 1e3:9.3f} ms   {rep['gf_mv']:8.2f} GFLOP/s   "
          f"AI = {rep['ai_mv']:8.3f} FLOP/byte")
    print(f"ratio of arithmetic intensities: {rep['ai_mm'] / rep['ai_mv']:.1f}x")
    print(f"matvec effective bandwidth: {rep['bw_mv']:.1f} GB/s")
    print(f"matmul is {rep['gf_mm'] / rep['gf_mv']:.1f}x faster per FLOP delivered")
    print()
    print("all assertions passed")
