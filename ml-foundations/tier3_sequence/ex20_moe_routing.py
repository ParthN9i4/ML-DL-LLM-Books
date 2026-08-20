"""
ex20 — Mixture-of-experts routing, load balancing, and capacity.  (Book: Chapter 22)

A mixture-of-experts layer replaces one feed-forward block with E of them and a
tiny router that sends each token to only k. The router emits logits over the
experts, the top k are selected, a softmax is taken OVER THOSE k ONLY, and the
outputs are combined with those weights. Two facts follow immediately and both
are checked below: at k = E the whole thing collapses back into a dense weighted
ensemble (agreement here: 1.7e-16), and at k < E it is not an approximation of
that ensemble but a genuinely different function.

The bargain is parameters without FLOPs. For d_model=1024, d_ff=4096, E=8, k=2
the layer holds 67.1M parameters instead of 8.4M — 8x — while spending 33.6
MFLOPs per token instead of the 134.2 it would cost to run all eight, i.e.
25.01%, of which the router itself is 0.0122%. That is the entire architectural
argument, and it is arithmetic, not an empirical claim.

The insight this file is built to make undeniable is that the bargain is not
free: it is contingent on the router staying balanced, and nothing in the task
loss wants it to be. Training router and experts together here — with a target
every expert could learn equally well, so there is no "best expert" to find —
utilization entropy still falls from 2.054 to 1.681 nats against a ceiling of
log 8 = 2.079, the effective number of experts in use (exp of that entropy)
falls from 7.80 to 5.37, and four of the eight end below half their fair share.
That is rich-get-richer: whichever expert is dispatched slightly more improves
slightly faster and is dispatched more still. Add the standard auxiliary loss,
E * sum_i f_i * P_i — hard dispatch fraction times mean router probability —
and entropy returns to 2.078, every expert is un-starved, and the task loss is
four orders of magnitude BETTER (3.8e-9 vs 1.2e-4), because the collapsed run
had been wasting its parameters. That loss is often described as having a floor
of 1.0. It does not. Because f is detached and both vectors sum to 1, it equals
exactly 1 whenever f is uniform — for ANY P at all, so uniform f is a flat
plateau rather than a minimum — and it drops strictly below 1 when a non-uniform
f is anti-aligned with P, which the balanced run here does on its way in,
bottoming out at 0.998897. Both halves of that are asserted below.

Capacity is where the imbalance becomes a wrong answer rather than an
inefficiency. Give each expert a buffer of capacity_factor * N * k / E slots and
the overflow is simply dropped. The collapsed router loses 40.7% of its dispatch
slots at capacity_factor 1.0 and returns the zero vector for 179 of 512 tokens;
the balanced router loses 2.7%. Drops vanish exactly at cf* = E * max_i f_i —
1.867 for the collapsed router, 1.109 for the balanced one — which is the whole
relationship between balance and capacity in one number.

The break-it section covers the two bugs that survive review. Taking the softmax
over all E experts and then slicing the top k leaves the weights summing to 0.51
on average instead of 1, which rescales every token's output by exactly that
missing mass. And computing the auxiliary loss on the one-hot dispatch rather
than the probabilities gives a loss that logs a plausible, drifting number and
has EXACTLY zero gradient — central finite differences return 0.000e+00 in all
64 router entries at every checkpoint probed. (Only an eps that straddles a
top-k flip reports anything else: at eps=1e-4 the same probe returns cliff
heights up to 32.5, which is how this bug passes a careless gradient check.)
torch drops the term out of the graph entirely, and a run that applies that
measured gradient comes out bit-identical to one that never had the loss at all.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402

try:
    import torch
    HAVE_TORCH = True
except ImportError:                                    # pragma: no cover
    HAVE_TORCH = False


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------

def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def top_k_route(logits, k):
    """Top-k routing: pick the k largest logits, softmax OVER THOSE k ONLY.

    Returns (idx, gates) with shapes (N, k). The gates sum to exactly 1 per
    token because the softmax is taken after the selection, not before.
    """
    # === YOUR CODE HERE ===
    idx = np.argsort(-logits, axis=-1, kind="stable")[:, :k]
    sel = np.take_along_axis(logits, idx, axis=-1)
    return idx, softmax(sel, axis=-1)


def moe_forward(x, W_r, W_e, k, capacity=None):
    """Sparse MoE layer. Experts are linear maps: expert_i(x) = x @ W_e[i].

    W_r is (d, E), W_e is (E, d, d_out). Returns (y, idx, gates, kept) where
    `kept` is the boolean dispatch mask after the capacity buffer (all True
    when capacity is None).
    """
    logits = x @ W_r
    idx, gates = top_k_route(logits, k)
    if capacity is not None:
        kept = capacity_mask(idx, W_e.shape[0], capacity)
        gates = gates * kept
    else:
        kept = np.ones_like(gates, dtype=bool)
    # === YOUR CODE HERE ===
    expert_out = np.einsum("nd,nkdo->nko", x, W_e[idx])   # (N, k, d_out)
    y = np.einsum("nk,nko->no", gates, expert_out)
    return y, idx, gates, kept


def dense_ensemble(x, W_r, W_e):
    """The dense baseline: every expert runs, weighted by the FULL softmax.

    This is what top-k routing must reduce to when k = E.
    """
    p = softmax(x @ W_r, axis=-1)                         # (N, E)
    # === YOUR CODE HERE ===
    all_out = np.einsum("nd,edo->neo", x, W_e)            # (N, E, d_out)
    return np.einsum("ne,neo->no", p, all_out)


# ---------------------------------------------------------------------------
# Balance: the measurement, and the loss
# ---------------------------------------------------------------------------

def dispatch_fractions(idx, E):
    """f_i — the fraction of dispatch slots that went to expert i. Sums to 1."""
    # === YOUR CODE HERE ===
    return np.bincount(idx.ravel(), minlength=E) / idx.size


def routing_entropy(f):
    """Shannon entropy (nats) of the utilization histogram.

    exp(entropy) is the "effective number of experts in use": E when perfectly
    balanced, 1 when everything lands on one expert.
    """
    # === YOUR CODE HERE ===
    nz = f[f > 0]
    return float(-(nz * np.log(nz)).sum())


def load_balance_loss(logits, idx, E):
    """The standard auxiliary loss:  E * sum_i f_i * P_i.

    f_i is the HARD dispatch fraction (no gradient — it is a counting operation
    on argmax output) and P_i is the MEAN ROUTER PROBABILITY (differentiable).
    It grows as the two co-concentrate, which is the whole point of it. What it
    is NOT is bounded below by 1: with f detached and sum_i f_i = sum_i P_i = 1,
    a uniform f gives E * (1/E) * sum_i P_i = 1 exactly for ANY P — a flat
    plateau, not a minimum — and a non-uniform f that is anti-aligned with P
    gives strictly less than 1. All of the gradient comes through P_i; f_i only
    says which experts to push away from. Returns (loss, f, P).
    """
    # === YOUR CODE HERE ===
    f = dispatch_fractions(idx, E)
    P = softmax(logits, axis=-1).mean(axis=0)
    return float(E * np.dot(f, P)), f, P


def load_balance_loss_onehot(logits, idx, E):
    """THE BUG (see break-it): E * sum_i f_i * f_i.

    Someone reaches for the dispatch mask twice because "that is what the load
    actually is". It is a perfectly good *diagnostic* — but f is a histogram of
    argmax outputs, piecewise constant in the router weights, so this scalar has
    zero gradient almost everywhere.
    """
    f = dispatch_fractions(idx, E)
    return float(E * np.dot(f, f)), f


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------

def expert_capacity(N, k, E, capacity_factor):
    """Per-expert buffer size. N*k/E is the perfectly balanced load."""
    # === YOUR CODE HERE ===
    return int(np.floor(capacity_factor * N * k / E))


def capacity_mask(idx, E, capacity):
    """First-come-first-served dispatch into finite per-expert buffers.

    Tokens are offered in sequence order; once an expert's buffer is full every
    later token routed to it is DROPPED (its gate is zeroed, so that slot
    contributes nothing to the output). Returns a (N, k) boolean keep-mask.
    """
    N, k = idx.shape
    kept = np.ones((N, k), dtype=bool)
    used = np.zeros(E, dtype=int)
    # === YOUR CODE HERE ===
    for n in range(N):
        for j in range(k):
            e = idx[n, j]
            if used[e] < capacity:
                used[e] += 1
            else:
                kept[n, j] = False
    return kept


def predicted_drop_rate(f, N, k, capacity):
    """Closed form: the overflow is sum_i max(0, n_i - C), independent of order."""
    n_i = f * N * k
    # === YOUR CODE HERE ===
    return float(np.maximum(0.0, n_i - capacity).sum() / (N * k))


# ---------------------------------------------------------------------------
# Gradients (numpy, hand-written; torch cross-checks them below)
# ---------------------------------------------------------------------------

def moe_grads(x, y, W_r, W_e, k, aux_alpha=0.0):
    """Analytic gradients of  mean_n ||moe(x_n) - y_n||^2  +  alpha * aux.

    Returns (task_loss, aux_loss, dW_r, dW_e, f).
    """
    N, d = x.shape
    d_out = y.shape[1]
    E = W_e.shape[0]
    logits = x @ W_r
    idx, g = top_k_route(logits, k)
    expert_out = np.einsum("nd,nkdo->nko", x, W_e[idx])
    yhat = np.einsum("nk,nko->no", g, expert_out)
    r = yhat - y
    task = float((r ** 2).sum(axis=1).mean())
    # === YOUR CODE HERE ===
    dyhat = 2.0 * r / N                                   # (N, d_out)
    dg = np.einsum("no,nko->nk", dyhat, expert_out)       # through the gates
    dz = g * (dg - (g * dg).sum(axis=1, keepdims=True))   # softmax over k
    dlogits = np.zeros_like(logits)
    dlogits[np.arange(N)[:, None], idx] = dz
    dW_e = np.zeros_like(W_e)
    np.add.at(dW_e, idx, np.einsum("nk,nd,no->nkdo", g, x, dyhat))
    aux, f, P = load_balance_loss(logits, idx, E)
    if aux_alpha:
        # d/dlogits of  E * sum_i f_i * P_i , with f detached: dL/dp_ni = E*f_i/N
        probs = softmax(logits, axis=-1)
        dp = np.broadcast_to(E * f / N, (N, E))
        dlogits = dlogits + aux_alpha * probs * (dp - (probs * dp).sum(axis=1, keepdims=True))
    dW_r = x.T @ dlogits
    return task, aux, dW_r, dW_e, f


def numeric_grad_Wr(fn, W_r, eps=1e-5):
    """Central finite differences of a scalar fn(W_r) — the honest referee for
    'does this loss actually have a gradient?'."""
    g = np.zeros_like(W_r)
    for i in range(W_r.shape[0]):
        for j in range(W_r.shape[1]):
            hi = W_r.copy(); hi[i, j] += eps
            lo = W_r.copy(); lo[i, j] -= eps
            g[i, j] = (fn(hi) - fn(lo)) / (2 * eps)
    return g


def train_moe(x, y, W_r0, W_e0, k, steps, lr, aux_alpha=0.0,
              wrong_aux=False, record_every=50):
    """Plain SGD on router AND experts. Returns (W_r, W_e, history, aux_grad_max).

    `wrong_aux=True` swaps in the one-hot version of the balance loss and
    actually applies its gradient — MEASURED, by central finite differences at
    the starting router, rather than assumed. Once, not per step, and on
    purpose: f is a histogram of argmax outputs, piecewise constant in W_r, so
    its true derivative is zero wherever it exists, and the only thing a
    re-measurement along the trajectory can add is noise — whenever eps happens
    to straddle a top-k flip, the difference quotient reports the cliff height
    instead of a gradient (the break-it section measures exactly that at
    eps=1e-4). The measurement comes back the exact zero matrix, so the run is
    bit-for-bit a run that never had the loss — but that identity is asserted
    below as a consequence of the measurement, and an implementation of this
    loss with a real gradient (e.g. one that quietly used probabilities) would
    fail it. The loss is still computed and logged every step, exactly as a
    real training script would.

    `aux_grad_max` is the largest |entry| of that measured gradient under
    `wrong_aux`, and None otherwise (nothing was measured).
    """
    W_r, W_e = W_r0.copy(), W_e0.copy()
    E = W_e.shape[0]
    hist = []
    aux_grad_max = None
    # === YOUR CODE HERE ===
    g_aux = None
    if wrong_aux:
        g_aux = numeric_grad_Wr(                              # the whole bug, measured
            lambda W: load_balance_loss_onehot(x @ W, top_k_route(x @ W, k)[0], E)[0],
            W_r, 1e-6)
        aux_grad_max = float(np.abs(g_aux).max())
    for s in range(steps):
        task, aux, dW_r, dW_e, f = moe_grads(x, y, W_r, W_e, k,
                                             aux_alpha=0.0 if wrong_aux else aux_alpha)
        if wrong_aux:
            aux = load_balance_loss_onehot(x @ W_r, top_k_route(x @ W_r, k)[0], E)[0]
            dW_r = dW_r + aux_alpha * g_aux
        if s % record_every == 0:
            hist.append((s, task, aux, routing_entropy(f), float(f.max())))
        W_r -= lr * dW_r
        W_e -= lr * dW_e
    task, aux, _, _, f = moe_grads(x, y, W_r, W_e, k)
    hist.append((steps, task, aux, routing_entropy(f), float(f.max())))
    return W_r, W_e, hist, aux_grad_max


# ---------------------------------------------------------------------------
# The FLOP argument
# ---------------------------------------------------------------------------

def moe_flop_budget(d_model, d_ff, E, k):
    """FLOPs per token and parameter counts for one FFN vs an E-expert MoE.

    One FFN is two matmuls (d_model x d_ff and d_ff x d_model): 2*d_model*d_ff
    multiply-accumulates = 4*d_model*d_ff FLOPs. An MoE layer holds E of them
    but runs only k, plus a d_model x E router.
    """
    # === YOUR CODE HERE ===
    per_expert = 4 * d_model * d_ff
    router = 2 * d_model * E
    return {
        "params_one_ffn": 2 * d_model * d_ff,
        "params_moe": E * 2 * d_model * d_ff,
        "flops_all_experts": E * per_expert,
        "flops_moe": k * per_expert + router,
        "flops_router": router,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, d, d_out, E, k = 512, 8, 8, 8, 2
    x = rng.standard_normal((N, d))
    A = rng.standard_normal((d, d_out)) / np.sqrt(d)
    y_target = x @ A                       # every expert can learn this equally well
    W_r0 = rng.standard_normal((d, E)) * 0.1
    W_e0 = rng.standard_normal((E, d, d_out)) * 0.1

    print("\n--- the routing primitive ---")
    idx, gates = top_k_route(x @ W_r0, k)
    print(f"      N={N} tokens, E={E} experts, top-k={k}")
    check("gates sum to exactly 1 per token",
          gates.sum(axis=1), np.ones(N), tol=1e-12)
    check("selected experts are the k largest logits",
          bool(np.all(np.take_along_axis(x @ W_r0, idx, axis=1)
                      >= np.sort(x @ W_r0, axis=1)[:, -k][:, None] - 1e-12)))
    check("dispatch fractions sum to 1",
          float(dispatch_fractions(idx, E).sum()), 1.0, tol=1e-12)

    print("\n--- dense equivalence: k = E must reproduce the weighted ensemble ---")
    y_full, idx_full, g_full, _ = moe_forward(x, W_r0, W_e0, k=E)
    y_dense = dense_ensemble(x, W_r0, W_e0)
    err = float(np.abs(y_full - y_dense).max())
    print(f"      max |moe(k=E) - dense ensemble| = {err:.3e}")
    check("top-k with k=E IS the dense weighted ensemble", y_full, y_dense, tol=1e-12)
    # and the reason: scattered back to expert order, the k=E gates ARE softmax(logits)
    g_scattered = np.zeros((N, E))
    g_scattered[np.arange(N)[:, None], idx_full] = g_full
    check("its gates are the full softmax, merely permuted",
          g_scattered, softmax(x @ W_r0, axis=-1), tol=1e-15)
    # k < E is genuinely a different function, not an approximation of the dense
    # one: the dense ensemble averages 8 unrelated experts and largely cancels,
    # while top-2 keeps two of them at full strength.
    y_k, _, _, _ = moe_forward(x, W_r0, W_e0, k=k)
    print(f"      at k={k}, mean |y| = {np.abs(y_k).mean():.4f} vs dense "
          f"{np.abs(y_dense).mean():.4f}; |diff| is {float(np.abs(y_k - y_dense).mean() / np.abs(y_dense).mean()):.2f}x "
          f"the dense magnitude — sparse routing is a different function, not an approximation")

    print("\n--- routing collapse, without a balance loss ---")
    steps, lr = 400, 2.5
    W_r_a, W_e_a, hist_a, _ = train_moe(x, y_target, W_r0, W_e0, k, steps, lr, aux_alpha=0.0)
    print(f"      {'step':>6} {'task loss':>12} {'aux':>8} {'entropy':>9} {'exp(H)':>8} {'max f_i':>9}")
    for s, t, a, H, fm in hist_a:
        print(f"      {s:6d} {t:12.6f} {a:8.4f} {H:9.4f} {np.exp(H):8.3f} {fm:9.3f}")
    H_max = float(np.log(E))
    H0, H_end = hist_a[0][3], hist_a[-1][3]
    f_a = dispatch_fractions(top_k_route(x @ W_r_a, k)[0], E)
    print(f"      max possible entropy = log(E) = {H_max:.4f}")
    print(f"      final utilization: {np.round(f_a, 3)}")
    starved_a = int((f_a < 0.5 / E).sum())
    print(f"      {starved_a} of {E} experts end below half their uniform share of {1/E:.3f}")
    # Anchor the entropy scale before reading anything into it. Three values
    # known in closed form: uniform -> log E, one expert -> 0, and the dyadic
    # histogram (1/2, 1/4, 1/8, 1/8, 0, 0, 0, 0), whose entropy is
    # (1/2)ln2 + (1/4)ln4 + 2*(1/8)ln8 = (1/2 + 1/2 + 3/4) ln 2 = 1.75 ln 2.
    f_dyadic = np.array([0.5, 0.25, 0.125, 0.125] + [0.0] * (E - 4))
    print(f"      anchors: H(uniform) = {routing_entropy(np.full(E, 1.0 / E)):.6f} = log {E}, "
          f"H(one expert) = {routing_entropy(np.eye(E)[0]):.6f}, "
          f"H(1/2,1/4,1/8,1/8) = {routing_entropy(f_dyadic):.6f} = 1.75 ln 2")
    check("uniform utilization has entropy exactly log E",
          routing_entropy(np.full(E, 1.0 / E)), H_max, tol=1e-12)
    check("a one-expert histogram has entropy exactly 0",
          routing_entropy(np.eye(E)[0]), 0.0, tol=1e-12)
    check("a hand-computed dyadic histogram has entropy 1.75 * ln 2",
          routing_entropy(f_dyadic), 1.75 * np.log(2.0), tol=1e-12)
    check("so exp(H) really is the effective expert count: E at uniform, 1 at collapse",
          [float(np.exp(routing_entropy(np.full(E, 1.0 / E)))),
           float(np.exp(routing_entropy(np.eye(E)[0])))], [float(E), 1.0], tol=1e-12)
    check("and every measured entropy respects the log E ceiling",
          max(H for _, _, _, H, _ in hist_a) <= H_max + 1e-12)
    check("routing entropy falls during training", H_end < H0)
    check("the most popular expert grows past its uniform share",
          hist_a[-1][4] > 1.0 / E)
    check("effective experts in use drops below E", float(np.exp(H_end)) < E)
    check("and some experts are starved outright", starved_a > 0)

    print("\n--- adding the f_i * P_i load-balancing loss ---")
    alpha = 0.05
    W_r_b, W_e_b, hist_b, _ = train_moe(x, y_target, W_r0, W_e0, k, steps, lr, aux_alpha=alpha)
    for s, t, a, H, fm in hist_b:
        print(f"      {s:6d} {t:12.6f} {a:8.4f} {H:9.4f} {np.exp(H):8.3f} {fm:9.3f}")
    f_b = dispatch_fractions(top_k_route(x @ W_r_b, k)[0], E)
    H_bal = hist_b[-1][3]
    print(f"      final utilization: {np.round(f_b, 3)}")
    print(f"      entropy {H_end:.4f} -> {H_bal:.4f} of a possible {H_max:.4f}; "
          f"effective experts {np.exp(H_end):.2f} -> {np.exp(H_bal):.2f} of {E}; "
          f"starved experts {starved_a} -> {int((f_b < 0.5 / E).sum())}")
    check("the aux loss recovers utilization entropy", H_bal > H_end)
    check("and un-starves every expert", int((f_b < 0.5 / E).sum()), 0)
    # Balance is not charity: with all E experts learning, the task loss is
    # LOWER, not higher — the collapsed run wasted most of its parameters.
    print(f"      task loss after {steps} steps: collapsed {hist_a[-1][1]:.3e}, "
          f"balanced {hist_b[-1][1]:.3e}")
    check("balancing did not cost task loss here — it improved it",
          hist_b[-1][1] < hist_a[-1][1])
    # "The aux loss has a floor of 1.0" is folklore, and false. What IS true —
    # and both halves are pinned on the actual implementation here: a
    # round-robin dispatch has f exactly uniform, and then
    # E * sum_i (1/E) * P_i = sum_i P_i = 1 for ANY router probabilities
    # whatsoever — a flat plateau, not a minimum. And the balanced run itself
    # undercuts 1 on its way in, because a mildly non-uniform f that is
    # anti-aligned with P gives E * f.P < 1.
    idx_uniform = np.arange(N * k).reshape(N, k) % E     # exactly N*k/E slots each
    check("uniform f gives aux exactly 1 under the trained router's P",
          load_balance_loss(x @ W_r_b, idx_uniform, E)[0], 1.0, tol=1e-12)
    check("...and under a freshly random P too — a plateau, not a floor",
          load_balance_loss(rng.standard_normal((N, E)), idx_uniform, E)[0],
          1.0, tol=1e-12)
    aux_b_min = min(a for _, _, a, _, _ in hist_b)
    print(f"      balanced run's recorded aux bottoms out at {aux_b_min:.6f} — "
          f"strictly below the alleged floor of 1.0")
    check("the balanced run's aux dips strictly below 1.0", aux_b_min < 1.0)
    check("balanced run ends near 1, collapsed run well above it",
          hist_b[-1][2] < hist_a[-1][2])

    print("\n--- capacity factor and token dropping ---")
    print(f"      {'cf':>6} {'capacity':>9} {'collapsed drop':>15} {'balanced drop':>14}")
    idx_a = top_k_route(x @ W_r_a, k)[0]
    idx_b = top_k_route(x @ W_r_b, k)[0]
    drops_a, drops_b = [], []
    for cf in (1.0, 1.25, 1.5, 2.0, 4.0):
        C = expert_capacity(N, k, E, cf)
        da = 1.0 - capacity_mask(idx_a, E, C).mean()
        db = 1.0 - capacity_mask(idx_b, E, C).mean()
        drops_a.append(da); drops_b.append(db)
        print(f"      {cf:6.2f} {C:9d} {da*100:14.1f}% {db*100:13.1f}%")
    C1 = expert_capacity(N, k, E, 1.0)
    check("measured drop rate matches the closed form sum_i max(0, n_i - C)",
          drops_a[0], predicted_drop_rate(f_a, N, k, C1), tol=1e-12)
    check("...for the balanced router too",
          drops_b[0], predicted_drop_rate(f_b, N, k, C1), tol=1e-12)
    check("drop rate falls monotonically as capacity grows",
          all(drops_a[i] >= drops_a[i + 1] for i in range(len(drops_a) - 1)))
    # The computed floor: a perfectly uniform assignment (idx_uniform, the
    # round-robin dispatch from the balance section) drops exactly nothing at
    # cf = 1.0. Anything above zero is the histogram's excess mass, not the
    # capacity mechanism being lossy in itself.
    check("a perfectly uniform assignment drops nothing at cf=1.0",
          1.0 - capacity_mask(idx_uniform, E, C1).mean(), 0.0, tol=0.0)
    # The mechanism, sharply: drops vanish exactly when C >= max_i n_i, i.e. at
    # capacity factor cf* = E * max_i f_i. Balance is what lowers cf*.
    for name, f_, idx_ in (("collapsed", f_a, idx_a), ("balanced", f_b, idx_b)):
        cf_star = E * float(f_.max())
        below = 1.0 - capacity_mask(idx_, E, expert_capacity(N, k, E, cf_star * (1 - 1e-3))).mean()
        above = 1.0 - capacity_mask(idx_, E, expert_capacity(N, k, E, cf_star * (1 + 1e-6))).mean()
        print(f"      {name}: critical cf* = E*max_i f_i = {cf_star:.3f}  "
              f"(drop just below {below*100:.2f}%, just above {above*100:.2f}%)")
        check(f"{name}: tokens are still dropped just below cf* ", below > 0.0)
        check(f"{name}: and exactly zero are dropped at cf*", above, 0.0, tol=0.0)
    # What a dropped token actually experiences: with both of its k slots full,
    # the MoE layer returns the zero vector for it and it survives only on the
    # residual stream.
    y_cap, _, _, kept_a = moe_forward(x, W_r_a, W_e_a, k, capacity=C1)
    all_dropped = (~kept_a).all(axis=1)
    print(f"      at cf=1.0 the collapsed router loses {int(all_dropped.sum())}/{N} tokens "
          f"entirely ({all_dropped.mean()*100:.1f}%) — the layer returns zeros for them")
    check("a token whose every slot overflowed gets exactly the zero vector",
          float(np.abs(y_cap[all_dropped]).max()), 0.0, tol=0.0)
    check("tokens that kept a slot did not", float(np.abs(y_cap[~all_dropped]).max()) > 0.0)
    print(f"      balancing cuts the cf=1.0 drop rate {drops_a[0]/max(drops_b[0],1e-12):.1f}x, "
          f"{drops_a[0]*100:.1f}% -> {drops_b[0]*100:.1f}%")
    check("balancing lowers the critical capacity factor",
          E * float(f_b.max()) < E * float(f_a.max()))
    check("and with it the drop rate at every capacity factor tested",
          all(db <= da for da, db in zip(drops_a, drops_b)))

    print("\n--- the FLOP argument ---")
    d_model, d_ff = 1024, 4096
    b = moe_flop_budget(d_model, d_ff, E, k)
    print(f"      config: d_model={d_model}, d_ff={d_ff}, E={E}, k={k}")
    print(f"      params: one FFN {b['params_one_ffn']/1e6:.1f}M  ->  MoE layer "
          f"{b['params_moe']/1e6:.1f}M  ({b['params_moe']/b['params_one_ffn']:.0f}x)")
    print(f"      FLOPs/token: all {E} experts {b['flops_all_experts']/1e6:.1f}M  ->  "
          f"top-{k} MoE {b['flops_moe']/1e6:.1f}M "
          f"({100*b['flops_moe']/b['flops_all_experts']:.2f}%)")
    print(f"      router overhead {b['flops_router']/1e3:.1f}k FLOPs/token "
          f"({100*b['flops_router']/b['flops_all_experts']:.4f}%)")
    # These pin the arithmetic itself, recomputed here from first principles
    # rather than from the dict's own entries (params_moe == E*params_one_ffn
    # and (k*pe + r)/(E*pe) == k/E + r/(E*pe) hold for ANY pe and r, so checks
    # of that shape cannot catch a wrong formula). One (p x q) matmul is p*q
    # multiply-accumulates = 2*p*q FLOPs; an FFN is the pair d_model x d_ff
    # then d_ff x d_model; the router is a single d_model x E.
    check("one FFN is two d_model x d_ff matrices",
          b["params_one_ffn"], 2 * d_model * d_ff)
    check("the MoE layer holds E of them", b["params_moe"], E * 2 * d_model * d_ff)
    check("running all E experts costs 4*d_model*d_ff FLOPs each",
          b["flops_all_experts"], 4 * d_model * d_ff * E)
    check("the router is one d_model x E matmul: 2*d_model*E FLOPs",
          b["flops_router"], 2 * d_model * E)
    check("top-k runs k experts plus the router",
          b["flops_moe"], 4 * d_model * d_ff * k + 2 * d_model * E)
    # 33570816 / 134217728 — both dyadic, so the quotient is exact in float:
    # 2^-2 + 2^-13 = 0.2501220703125. That IS the headline 25.01%.
    check("MoE spends exactly 25.0122% of the dense FLOPs",
          b["flops_moe"] / b["flops_all_experts"], 0.2501220703125, tol=0.0)

    if HAVE_TORCH:
        print("\n--- torch cross-check of the hand-written gradients ---")
        tx = torch.tensor(x)
        ty = torch.tensor(y_target)
        tWr = torch.tensor(W_r0, requires_grad=True)
        tWe = torch.tensor(W_e0, requires_grad=True)
        tlog = tx @ tWr
        tv, ti = torch.topk(tlog, k, dim=-1)
        tg = torch.softmax(tv, dim=-1)
        teo = torch.einsum("nd,nkdo->nko", tx, tWe[ti])
        tyh = torch.einsum("nk,nko->no", tg, teo)
        tloss = ((tyh - ty) ** 2).sum(dim=1).mean()
        # the aux term, with f detached exactly as the formula prescribes
        tf = torch.tensor(dispatch_fractions(ti.numpy(), E))
        tP = torch.softmax(tlog, dim=-1).mean(dim=0)
        taux = E * (tf * tP).sum()
        (tloss + alpha * taux).backward()
        task, aux, dW_r, dW_e, _ = moe_grads(x, y_target, W_r0, W_e0, k, aux_alpha=alpha)
        tloss_v = float(tloss.detach())
        print(f"      loss: numpy {task:.8f}  torch {tloss_v:.8f}")
        check("torch agrees on the loss", task, tloss_v, tol=1e-12)
        check("torch agrees on dW_r (router, task + aux)", dW_r, tWr.grad.numpy(), tol=1e-12)
        check("torch agrees on dW_e (experts)", dW_e, tWe.grad.numpy(), tol=1e-12)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) softmax over ALL experts, THEN slice the top-k — the one-character
    #     reordering that silently rescales every token's output.
    def top_k_route_wrong(logits, k_):
        p = softmax(logits, axis=-1)
        idx_ = np.argsort(-logits, axis=-1, kind="stable")[:, :k_]
        return idx_, np.take_along_axis(p, idx_, axis=-1)   # NOT renormalized

    idx_w, g_w = top_k_route_wrong(x @ W_r_a, k)
    idx_r, g_r = top_k_route(x @ W_r_a, k)
    s = g_w.sum(axis=1)
    print(f"      correct gates sum to {top_k_route(x @ W_r_a, k)[1].sum(axis=1).mean():.6f}")
    print(f"      sliced  gates sum to {s.mean():.4f} on average "
          f"(range {s.min():.4f} .. {s.max():.4f})")
    check("the correct route always sums to 1", g_r.sum(axis=1), np.ones(N), tol=1e-12)
    check("the sliced route never does", bool(np.all(s < 1.0)))
    # The damage is exactly a per-token rescale: y_wrong = s_n * y_correct.
    eo = np.einsum("nd,nkdo->nko", x, W_e_a[idx_r])
    y_ok = np.einsum("nk,nko->no", g_r, eo)
    y_bad = np.einsum("nk,nko->no", g_w, eo)
    check("the bug is exactly a per-token rescale by the missing mass",
          y_bad, s[:, None] * y_ok, tol=1e-12)
    rel = float(np.abs(y_bad - y_ok).sum() / np.abs(y_ok).sum())
    mass_weighted = float((( 1 - s) * np.abs(y_ok).sum(axis=1)).sum() / np.abs(y_ok).sum())
    print(f"      -> output shrinks: relative L1 error {rel*100:.1f}%, "
          f"mean gate mass {s.mean():.4f}")
    check("the measured output error IS the missing softmax mass, nothing else",
          rel, mass_weighted, tol=1e-12)

    # (b) the load-balancing loss computed on the ONE-HOT dispatch. It prints a
    #     plausible number every step and has ZERO gradient — the failure mode
    #     that survives code review because the log looks right.
    def aux_correct_of(Wr):
        i2, _ = top_k_route(x @ Wr, k)
        return load_balance_loss(x @ Wr, i2, E)[0]

    def aux_wrong_of(Wr):
        i2, _ = top_k_route(x @ Wr, k)
        return load_balance_loss_onehot(x @ Wr, i2, E)[0]

    print(f"      {'ckpt':>5} {'aux(f,P)':>9} {'aux(f,f)':>9} "
          f"{'|d aux(f,P)| fd':>16} {'|d aux(f,f)| fd':>16} {'(eps)':>8}")
    W_probe, W_e_probe = W_r0.copy(), W_e0.copy()
    gmax_wrong_fine, gmax_wrong_coarse, fd_vs_analytic = [], [], []
    logged_wrong = []
    for cp in range(3):
        for eps in (1e-6, 1e-8):
            gc = numeric_grad_Wr(aux_correct_of, W_probe, eps)
            gw = numeric_grad_Wr(aux_wrong_of, W_probe, eps)
            gmax_wrong_fine.append(float(np.abs(gw).max()))
            print(f"      {cp*200:5d} {aux_correct_of(W_probe):9.4f} "
                  f"{aux_wrong_of(W_probe):9.4f} {np.abs(gc).max():16.3e} "
                  f"{np.abs(gw).max():16.3e} {eps:8.0e}")
        gmax_wrong_coarse.append(float(np.abs(numeric_grad_Wr(aux_wrong_of, W_probe, 1e-4)).max()))
        logged_wrong.append(aux_wrong_of(W_probe))
        # the analytic aux gradient, isolated by differencing the two alphas
        _, _, dWr1, _, _ = moe_grads(x, y_target, W_probe, W_e_probe, k, aux_alpha=1.0)
        _, _, dWr0, _, _ = moe_grads(x, y_target, W_probe, W_e_probe, k, aux_alpha=0.0)
        fd_vs_analytic.append(float(np.abs((dWr1 - dWr0)
                                           - numeric_grad_Wr(aux_correct_of, W_probe, 1e-6)).max()))
        W_probe, W_e_probe, _, _ = train_moe(x, y_target, W_probe, W_e_probe, k,
                                          200, lr, aux_alpha=0.0, record_every=1000)
    print(f"      analytic d aux(f,P) vs finite differences: max diff "
          f"{max(fd_vs_analytic):.2e}  (so the f*P loss really is being differentiated)")
    check("the f*P aux gradient matches finite differences", max(fd_vs_analytic) < 1e-8)
    check("the one-hot aux gradient is EXACTLY zero, every entry, every checkpoint",
          all(g == 0.0 for g in gmax_wrong_fine))
    # It is not "small": it is a staircase. A coarse difference straddles the
    # top-k flips and reports cliff heights instead — the classic reason someone
    # "verifies" this loss with a sloppy eps and believes it works.
    print(f"      at eps=1e-4 the same finite difference reports up to "
          f"{max(gmax_wrong_coarse):.2f} — pure top-k flips, gone by eps=1e-6")
    check("coarse finite differences see staircase cliffs, not a slope",
          min(gmax_wrong_coarse) > 0.0 and max(gmax_wrong_fine) == 0.0)
    if HAVE_TORCH:
        # What this looks like in a real framework: the one-hot dispatch is an
        # integer scatter, so the aux tensor leaves the autograd graph entirely.
        mask = torch.zeros_like(tlog).scatter_(1, ti, 1.0)
        tf_hard = mask.sum(dim=0) / (N * k)
        aux_onehot = E * (tf_hard * tf_hard).sum()
        print(f"      torch: aux(f,P).requires_grad = {bool(taux.requires_grad)}, "
              f"aux(f,f).requires_grad = {bool(aux_onehot.requires_grad)}")
        check("torch: the one-hot aux is detached from the graph entirely",
              bool(aux_onehot.requires_grad) is False)
        check("torch: the f*P aux is not", bool(taux.requires_grad) is True)
    # The consequence: the loss is logged, it even gets steadily WORSE, and the
    # router never feels it. Training with it is bit-identical to no aux at all.
    # (Unlike E*f.P, the one-hot E*f.f really IS bounded below by 1: by
    # Cauchy-Schwarz, sum f_i^2 >= (sum f_i)^2 / E = 1/E, equality iff f is
    # uniform. A genuine floor it cannot descend to, since nothing pulls it.)
    print(f"      logged aux(f,f) at steps 0/200/400: "
          f"{logged_wrong[0]:.4f} -> {logged_wrong[1]:.4f} -> {logged_wrong[2]:.4f} "
          f"(up from its uniform floor of 1.0, and nothing ever pushes back)")
    check("the logged one-hot balance loss drifts away from its uniform floor",
          logged_wrong[-1] > logged_wrong[0])
    W_r_w, _, hist_w, gmax_w = train_moe(x, y_target, W_r0, W_e0, k, steps, lr,
                                 aux_alpha=alpha, wrong_aux=True, record_every=1000)
    print(f"      measured d aux(f,f)/d W_r at the starting router: "
          f"max|entry| = {gmax_w:.3e} — that array is what the run applies")
    check("the applied one-hot aux gradient was measured to be exactly zero",
          gmax_w, 0.0, tol=0.0)
    print(f"      final entropy: one-hot aux {hist_w[-1][3]:.6f}, "
          f"no aux {hist_a[-1][3]:.6f}, f*P aux {hist_b[-1][3]:.6f}")
    check("applying that measured gradient is bit-identical to no aux at all",
          W_r_w, W_r_a, tol=0.0)
    check("...so the balance it 'enforces' is exactly no balance",
          hist_w[-1][3], hist_a[-1][3], tol=0.0)
    check("while the f*P version genuinely moves it", hist_b[-1][3] > hist_a[-1][3])

    summary()
