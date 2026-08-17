"""
ex17 — Linear attention: the associativity trick and what it costs.
(Book: Chapters 18, 21)

Softmax attention is quadratic because the softmax sits between Q K^T and V,
forcing you to materialize the T x T score matrix:

    softmax(Q K^T / sqrt(d)) V          <- the softmax blocks reassociation

Remove the softmax and replace it with a feature map phi, and matrix
multiplication becomes associative again:

    (phi(Q) phi(K)^T) V   ==   phi(Q) (phi(K)^T V)
     \\_____________/            \\______________/
      T x T, O(T^2 d)             d x d, O(T d^2)

The right-hand grouping never builds anything of size T x T. For T >> d that
is the whole ballgame — and for causal masking it becomes a running sum, i.e.
a recurrent state of fixed size d x d, which is exactly the bridge to state
space models in ex18.

What it costs is the thing this exercise measures rather than asserts: the
softmax's sharpness. A softmax can concentrate essentially all its mass on one
key; a positive-feature-map linear attention cannot, and the checks below
quantify that gap. That is why pure linear-attention models lose on tasks that
need exact recall, and why 2024-2026 architectures are hybrids.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def softmax_attention(Q, K, V, causal=False):
    """The quadratic reference. Materializes the T x T score matrix."""
    T, d = Q.shape
    # === YOUR CODE HERE ===
    s = Q @ K.T / np.sqrt(d)
    if causal:
        s = np.where(np.tril(np.ones((T, T), dtype=bool)), s, -1e30)
    return softmax(s) @ V


def phi_elu(x):
    """The elu+1 feature map (Katharopoulos et al.): strictly positive, which
    is what keeps the denominator from vanishing or changing sign."""
    # === YOUR CODE HERE ===
    return np.where(x > 0, x + 1.0, np.exp(np.clip(x, -60, 60)))


def linear_attention_quadratic(Q, K, V, phi=phi_elu):
    """Left-grouped: (phi(Q) phi(K)^T) V, normalized. O(T^2 d). The reference
    the linear form must match."""
    Qp, Kp = phi(Q), phi(K)
    # === YOUR CODE HERE ===
    s = Qp @ Kp.T
    return (s @ V) / (s.sum(axis=-1, keepdims=True) + 1e-12)


def linear_attention_linear(Q, K, V, phi=phi_elu):
    """Right-grouped: phi(Q) (phi(K)^T V). O(T d^2), never builds T x T."""
    Qp, Kp = phi(Q), phi(K)
    # === YOUR CODE HERE ===
    kv = Kp.T @ V                       # (d, d_v) — the whole "memory"
    k_sum = Kp.sum(axis=0)              # (d,)
    num = Qp @ kv
    den = (Qp @ k_sum)[:, None] + 1e-12
    return num / den


def linear_attention_recurrent(Q, K, V, phi=phi_elu):
    """Causal linear attention as a RECURRENCE with fixed-size state.

        S_t = S_{t-1} + phi(k_t) v_t^T        (d x d_v state)
        z_t = z_{t-1} + phi(k_t)              (d normalizer)
        y_t = phi(q_t) S_t / (phi(q_t) . z_t)

    Constant memory per step, no KV cache that grows with T. This is the
    inference story that makes linear attention interesting, and the same
    shape as the SSM recurrence in ex18.
    """
    T, d = Q.shape
    d_v = V.shape[1]
    Qp, Kp = phi(Q), phi(K)
    S = np.zeros((d, d_v))
    z = np.zeros(d)
    out = np.empty((T, d_v))
    # === YOUR CODE HERE ===
    for t in range(T):
        S = S + np.outer(Kp[t], V[t])
        z = z + Kp[t]
        out[t] = (Qp[t] @ S) / (Qp[t] @ z + 1e-12)
    return out


def linear_attention_causal_quadratic(Q, K, V, phi=phi_elu):
    """Causal linear attention the slow way, for cross-checking the recurrence."""
    T = Q.shape[0]
    Qp, Kp = phi(Q), phi(K)
    s = np.tril(Qp @ Kp.T)
    return (s @ V) / (s.sum(axis=-1, keepdims=True) + 1e-12)


def max_attention_mass(weights):
    """The largest single weight per query — how sharply attention can point."""
    return float(np.max(weights, axis=-1).mean())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, d, d_v = 256, 32, 32
    Q = rng.standard_normal((T, d))
    K = rng.standard_normal((T, d))
    V = rng.standard_normal((T, d_v))

    print("\n--- associativity: same answer, different cost ---")
    y_quad = linear_attention_quadratic(Q, K, V)
    y_lin = linear_attention_linear(Q, K, V)
    check("left-grouped == right-grouped (associativity)", y_lin, y_quad, tol=1e-9)

    # And the timing crossover: the linear form wins once T exceeds ~d.
    print(f"      {'T':>7} {'quadratic ms':>14} {'linear ms':>12} {'ratio':>8}")
    for T_big in (128, 1024, 4096):
        Qb = rng.standard_normal((T_big, d))
        Kb = rng.standard_normal((T_big, d))
        Vb = rng.standard_normal((T_big, d_v))
        t0 = time.perf_counter(); linear_attention_quadratic(Qb, Kb, Vb); tq = time.perf_counter() - t0
        t0 = time.perf_counter(); linear_attention_linear(Qb, Kb, Vb); tl = time.perf_counter() - t0
        print(f"      {T_big:7d} {tq*1e3:14.2f} {tl*1e3:12.2f} {tq/max(tl,1e-9):7.1f}x")
    # Assert the ASYMPTOTIC claim rather than a wall-clock threshold: the
    # quadratic form's work grows ~T^2 while the linear form's grows ~T.
    def work_ratio(T_):
        return (T_ * T_ * d) / (T_ * d * d)      # = T/d
    check("analytic work ratio is T/d", work_ratio(4096), 4096 / d, tol=1e-9)
    check("at T=4096, d=32 the linear form does 128x less work", work_ratio(4096), 128.0, tol=1e-9)

    print("\n--- causal: the recurrence has fixed-size state ---")
    y_rec = linear_attention_recurrent(Q, K, V)
    y_causal_quad = linear_attention_causal_quadratic(Q, K, V)
    check("recurrent form == causal quadratic form", y_rec, y_causal_quad, tol=1e-9)
    # The state is d x d_v regardless of T — that is the whole point.
    state_elems = d * d_v
    kv_cache_elems = 2 * T * d
    print(f"      linear-attention state: {state_elems} numbers, independent of T")
    print(f"      softmax KV cache at T={T}: {kv_cache_elems} numbers, and growing")
    check("state size does not depend on T", state_elems == d * d_v)
    check(f"at T={T} the KV cache already exceeds the recurrent state",
          kv_cache_elems > state_elems)

    print("\n--- what it costs: sharpness ---")
    # Softmax can put ~all mass on one key; positive-feature linear attention
    # cannot. Build a deliberately "retrieval-like" case: one key aligned with
    # the query, the rest random.
    q = rng.standard_normal((1, d))
    K_r = rng.standard_normal((64, d))
    K_r[7] = q[0] * 3.0                       # one strongly matching key
    s_soft = softmax(q @ K_r.T / np.sqrt(d))
    s_lin = phi_elu(q) @ phi_elu(K_r).T
    s_lin = s_lin / s_lin.sum(axis=-1, keepdims=True)
    print(f"      softmax puts {s_soft[0, 7]*100:5.1f}% of its mass on the matching key")
    print(f"      linear  puts {s_lin[0, 7]*100:5.1f}% of its mass on the matching key")
    check("softmax concentrates far more sharply than linear attention",
          float(s_soft[0, 7]) > 3 * float(s_lin[0, 7]))
    # And the structural reason: a positive feature map of dimension d gives an
    # attention matrix of rank at most d, so it CANNOT be an arbitrary
    # near-one-hot T x T matrix once T > d.
    Qp, Kp = phi_elu(Q), phi_elu(K)
    A_lin = Qp @ Kp.T
    rank = int(np.linalg.matrix_rank(A_lin, tol=1e-8))
    print(f"      linear attention matrix is {T}x{T} with rank {rank} (feature dim {d})")
    check("linear attention scores are rank-limited by the feature dimension", rank <= d)
    A_soft = softmax(Q @ K.T / np.sqrt(d))
    rank_soft = int(np.linalg.matrix_rank(A_soft, tol=1e-8))
    print(f"      softmax attention matrix has rank {rank_soft} (full)")
    check("softmax attention is not rank-limited", rank_soft > d)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) A feature map that is not strictly positive. Use identity (no phi) and
    #     the denominator sum(q . k) can pass through zero — the output blows up
    #     or flips sign. This is why every linear-attention paper insists on
    #     positivity, and the reason elu+1 exists at all.
    def phi_identity(x):
        return x

    y_bad = linear_attention_linear(Q, K, V, phi=phi_identity)
    dens = (phi_identity(Q) @ phi_identity(K).sum(axis=0))
    print(f"      identity feature map: denominator range [{dens.min():.2f}, {dens.max():.2f}]")
    print(f"      -> output magnitude up to {np.abs(y_bad).max():.1f} "
          f"(positive map: {np.abs(linear_attention_linear(Q, K, V)).max():.2f})")
    check("a sign-changing denominator is reachable", float(dens.min()) < 0.0)
    check("and it produces wildly larger outputs",
          float(np.abs(y_bad).max()) > 10 * float(np.abs(linear_attention_linear(Q, K, V)).max()))

    # (b) The causal shortcut that silently is not causal: applying the LINEAR
    #     (right-grouped) form and then masking the output does not restore
    #     causality — the state already absorbed every key. Show that a future
    #     token changes an earlier output.
    V2 = V.copy()
    V2[-1] += 10.0                                   # perturb the LAST value
    y_a = linear_attention_linear(Q, K, V)
    y_b = linear_attention_linear(Q, K, V2)
    leak = float(np.abs(y_a[0] - y_b[0]).max())
    y_ra = linear_attention_recurrent(Q, K, V)
    y_rb = linear_attention_recurrent(Q, K, V2)
    leak_rec = float(np.abs(y_ra[0] - y_rb[0]).max())
    print(f"      perturbing the last value changes output[0] by {leak:.4f} (non-causal form)")
    print(f"      the recurrent causal form changes output[0] by {leak_rec:.2e}")
    check("the non-causal linear form leaks the future", leak > 1e-3)
    check("the recurrent form does not", leak_rec < 1e-12)

    summary()
