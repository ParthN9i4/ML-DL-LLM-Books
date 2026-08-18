"""
ex23 — LoRA: low-rank adaptation, exactly.  (Book: Chapter 26)

Fine-tuning updates a weight matrix W by some ΔW. LoRA's bet is that the ΔW a
task actually needs is close to low-rank, so instead of storing and optimizing
a full d x d update it parameterizes

    ΔW = (alpha / r) * B A ,     A: r x d_in (Gaussian init),  B: d_out x r (ZERO init)

and trains only A and B while W stays frozen. Two properties make this more
than a compression trick, and this exercise is built to make both undeniable.

The first is that B = 0 is not a detail, it is the whole safety argument. At
step zero ΔW is the zero matrix, so the adapted network is not "close to" the
base network, it IS the base network — bit for bit, in float64 and float32
alike. Fine-tuning therefore starts from exactly the pretrained function, with
no initialization shock to recover from. Initialize BOTH factors randomly and
that guarantee evaporates immediately; the break-it section measures how far.

The second is that the adapter is a linear correction, so once training ends
you can fold it in: W_merged = W + (alpha/r) B A reproduces the adapted forward
pass to round-off. Inference cost after merging is exactly the cost of the base
model — this is the property that separates LoRA from adapter layers, which add
depth you can never remove.

What the rank buys is a different question, and it is an empirical one. The
best any rank-r matrix can do against a target ΔW is fixed by Eckart-Young:
the error floor is the Frobenius tail of the singular values, sqrt(sum_{i>r}
sigma_i^2). Fit LoRA against a target whose spectrum you control and the fit
tracks that floor. Two unit-norm targets, the same rank 8: the genuinely
rank-4 one is reproduced to 7.8e-15, the one with sigma_i ~ 1/i is stuck at
0.250 — and doubling to rank 32 only takes it to 0.097. Rank is not a knob you
tune blind; it is a claim about a spectrum.

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
# The adapter itself
# ---------------------------------------------------------------------------

def lora_init(d_in, d_out, r, rng):
    """A ~ N(0, 1/d_in), B = 0 — the asymmetry IS the design.

    A must be random: two zero factors would sit at a saddle with zero gradient
    in both, and nothing would ever move. B must be zero: that is what makes
    the initial ΔW exactly zero rather than merely small.
    """
    # === YOUR CODE HERE ===
    A = rng.standard_normal((r, d_in)) / np.sqrt(d_in)
    B = np.zeros((d_out, r))
    return A, B


def lora_delta(A, B, alpha):
    """The implied weight update ΔW = (alpha/r) B A, with r read off A."""
    r = A.shape[0]
    # === YOUR CODE HERE ===
    return (alpha / r) * (B @ A)


def lora_forward(X, W, A, B, alpha):
    """Adapted forward pass, base and adapter paths kept SEPARATE.

    X is (n, d_in), W is (d_out, d_in), output is (n, d_out). The adapter path
    goes through the r-dimensional bottleneck and never materializes a d x d
    matrix — that is what makes the extra cost O(n r d) instead of O(n d^2).
    """
    r = A.shape[0]
    # === YOUR CODE HERE ===
    base = X @ W.T
    adapt = (X @ A.T) @ B.T
    return base + (alpha / r) * adapt


def merge_lora(W, A, B, alpha):
    """W + ΔW: one dense matrix, no adapter left at inference time."""
    # === YOUR CODE HERE ===
    return W + lora_delta(A, B, alpha)


# ---------------------------------------------------------------------------
# Fitting a target update, and the floor it cannot beat
# ---------------------------------------------------------------------------

def matrix_with_spectrum(sigmas, rng):
    """A square matrix with exactly the prescribed singular values, unit ||.||_F.

    Random orthogonal U, V from QR; the spectrum is the experimental variable.
    """
    n = len(sigmas)
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    # === YOUR CODE HERE ===
    D = (U * sigmas) @ V.T
    return D / np.linalg.norm(D)


def eckart_young_bound(D, r):
    """min over rank-r matrices of ||D - M||_F = sqrt(sum_{i>r} sigma_i^2).

    Not an estimate: the truncated SVD attains it, and nothing beats it. Any
    LoRA fit of rank r is bounded below by this number, whatever the optimizer.
    """
    sv = np.linalg.svd(D, compute_uv=False)
    # === YOUR CODE HERE ===
    return float(np.sqrt(np.sum(sv[r:] ** 2)))


def lora_fit_grads(D, A, B, alpha):
    """Gradients of 0.5 * ||(alpha/r) B A - D||_F^2 wrt A and B.

    Note what happens at B = 0: the residual is -D, so grad_B = -(alpha/r) D A^T
    is nonzero while grad_A = (alpha/r) B^T R is EXACTLY zero. The first step of
    LoRA training moves B alone; A only starts moving once B has left zero.
    """
    s = alpha / A.shape[0]
    R = s * (B @ A) - D
    # === YOUR CODE HERE ===
    gA = s * (B.T @ R)
    gB = s * (R @ A.T)
    return gA, gB


def fit_lora_als(D, r, alpha, rng, iters=30):
    """Fit ΔW ≈ (alpha/r) B A by alternating least squares.

    Each half-step is a linear least-squares problem with a closed-form
    solution, so this converges to the global optimum — i.e. to the truncated
    SVD — in a handful of sweeps, independent of the scale alpha/r. Using it
    here keeps the rank experiment about CAPACITY; the gradient-descent version
    below is where the scale starts to matter.

    Starts from the LoRA initialization, so the first half-step is the B step:
    exactly the asymmetry that lora_fit_grads describes.
    """
    s = alpha / r
    A, B = lora_init(D.shape[1], D.shape[0], r, rng)
    # === YOUR CODE HERE ===
    for _ in range(iters):
        B = (D @ np.linalg.pinv(A)) / s          # least squares for B given A
        A = np.linalg.pinv(s * B) @ D            # least squares for A given B
    return A, B


def fit_lora_gd(D, r, alpha, rng, lr, steps, scale=None):
    """Plain gradient descent on the same objective, with an OVERRIDABLE scale.

    `scale` defaults to the correct alpha/r; passing alpha itself is the classic
    bug the break-it section measures. Returns (A, B, final error), with the
    error set to inf if the iteration blew up.
    """
    s = (alpha / r) if scale is None else scale
    A, B = lora_init(D.shape[1], D.shape[0], r, rng)
    # === YOUR CODE HERE ===
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps):
            R = s * (B @ A) - D
            gA = s * (B.T @ R)
            gB = s * (R @ A.T)
            A = A - lr * gA
            B = B - lr * gB
            if not np.isfinite(A).all() or not np.isfinite(B).all():
                return A, B, float("inf")
        err = float(np.linalg.norm(s * (B @ A) - D))
    return A, B, err


# ---------------------------------------------------------------------------
# Honest accounting
# ---------------------------------------------------------------------------

def param_counts(d, r, bytes_per_scalar=4, adam_states=2):
    """Training-memory accounting for one d x d layer, in scalars and bytes.

    Full fine-tuning holds, per weight: the weight, its gradient, and
    `adam_states` optimizer moments (Adam keeps m and v). LoRA holds the frozen
    weight once — no gradient, no moments — plus that same 1 + 1 + adam_states
    stack for each of the 2dr adapter scalars.
    """
    # === YOUR CODE HERE ===
    full_train = d * d
    lora_train = 2 * d * r
    per_trained = 1 + 1 + adam_states                # weight + grad + moments
    full_total = full_train * per_trained
    lora_total = d * d + lora_train * per_trained    # frozen W held once
    return {
        "full_trainable": full_train,
        "lora_trainable": lora_train,
        "full_optimizer_state": adam_states * full_train,
        "lora_optimizer_state": adam_states * lora_train,
        "full_bytes": full_total * bytes_per_scalar,
        "lora_bytes": lora_total * bytes_per_scalar,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d_in = d_out = 64
    n = 32
    alpha = 16.0
    W = rng.standard_normal((d_out, d_in)) / np.sqrt(d_in)
    X = rng.standard_normal((n, d_in))

    print("\n--- at initialization the adapted model IS the base model ---")
    y_base = X @ W.T
    for r in (1, 4, 16):
        A, B = lora_init(d_in, d_out, r, rng)
        y_lora = lora_forward(X, W, A, B, alpha)
        n_diff = int(np.count_nonzero(y_lora - y_base))
        check(f"r={r}: adapted output is bit-exactly the base output", n_diff, 0)
        check(f"r={r}: ΔW is exactly the zero matrix",
              int(np.count_nonzero(lora_delta(A, B, alpha))), 0)
    print(f"      max|adapted - base| = {float(np.abs(y_lora - y_base).max()):.1e} "
          f"(identically zero, not merely small)")
    # float32 too: a tolerance-based check would pass on a merely-small ΔW, so
    # the point is that the difference is not small, it is absent.
    A, B = lora_init(d_in, d_out, 8, rng)
    X32, W32 = X.astype(np.float32), W.astype(np.float32)
    y32 = lora_forward(X32, W32, A.astype(np.float32), B.astype(np.float32), alpha)
    check("float32: same, zero differing entries",
          int(np.count_nonzero(y32 - X32 @ W32.T)), 0)
    # And the reason, stated as an assertion about the gradients: at B = 0 only
    # B moves. A random-A / zero-B init is the unique choice that gives both an
    # exactly-zero start AND a nonzero first step.
    D_dummy = rng.standard_normal((d_out, d_in))
    gA0, gB0 = lora_fit_grads(D_dummy, A, B, alpha)
    print(f"      at init: |grad_A| = {np.abs(gA0).max():.1e}, "
          f"|grad_B| = {np.abs(gB0).max():.3f}")
    check("at init grad_A is exactly zero", int(np.count_nonzero(gA0)), 0)
    check("at init grad_B is not", float(np.abs(gB0).max()) > 0.0)

    print("\n--- merging: zero inference latency, to round-off ---")
    # Train something so B is genuinely nonzero, then compare the two-path
    # forward against the single merged matmul.
    D_target = matrix_with_spectrum(
        np.concatenate([np.array([4.0, 3.0, 2.0, 1.0]), np.zeros(d_in - 4)]), rng)
    A_t, B_t = fit_lora_als(D_target, r=8, alpha=alpha, rng=rng)
    y_adapt = lora_forward(X, W, A_t, B_t, alpha)
    W_merged = merge_lora(W, A_t, B_t, alpha)
    y_merged = X @ W_merged.T
    err_merge = float(np.abs(y_adapt - y_merged).max())
    # Two floors to compare against, neither of them invented. The textbook
    # backward-error bound for a length-k dot product is k * eps * sum|x_i w_i|;
    # and empirically, the SAME base product summed in reversed index order
    # already disagrees with itself at that scale.
    eps = float(np.finfo(np.float64).eps)
    round_off_bound = float(d_in * eps *
                            (np.abs(X) @ (np.abs(W) + np.abs(W_merged - W)).T).max())
    reversal_floor = float(np.abs(X @ W.T - X[:, ::-1] @ W.T[::-1, :]).max())
    print(f"      max|adapted - merged| = {err_merge:.2e}  (max|y| = {np.abs(y_merged).max():.2f})")
    print(f"      dot-product round-off bound d*eps*max|X||W| = {round_off_bound:.2e}")
    print(f"      same product, reversed summation order, disagrees by {reversal_floor:.2e}")
    check("merged dense weight reproduces the adapted forward pass",
          y_merged, y_adapt, tol=1e-12)
    check("and the disagreement is below the dot-product round-off bound",
          err_merge < round_off_bound)
    check("ΔW itself is rank <= r", int(np.linalg.matrix_rank(lora_delta(A_t, B_t, alpha))) <= 8)

    print("\n--- parameter and optimizer-state accounting (d=4096, r=16) ---")
    d_real, r_real = 4096, 16
    c = param_counts(d_real, r_real)
    print(f"      trainable:        full {c['full_trainable']:,}   "
          f"LoRA {c['lora_trainable']:,}   ratio {c['full_trainable']/c['lora_trainable']:.0f}x")
    print(f"      Adam m,v scalars: full {c['full_optimizer_state']:,}   "
          f"LoRA {c['lora_optimizer_state']:,}")
    print(f"      total fp32 training memory for the layer: "
          f"full {c['full_bytes']/2**20:.1f} MB   LoRA {c['lora_bytes']/2**20:.1f} MB   "
          f"({c['full_bytes']/c['lora_bytes']:.2f}x)")
    check("trainable-parameter ratio is exactly d/(2r)",
          c["full_trainable"] / c["lora_trainable"], d_real / (2 * r_real), tol=1e-12)
    check("optimizer state shrinks by the same factor",
          c["full_optimizer_state"] / c["lora_optimizer_state"], d_real / (2 * r_real), tol=1e-12)
    # The larger practical win: with Adam, EVERY trainable weight drags 3 extra
    # copies (grad + m + v). LoRA pays that toll on 2dr scalars, not d^2 — but
    # the frozen weight still has to be stored, which caps the total saving.
    check("total-memory ratio is 4d^2 / (d^2 + 8dr), i.e. capped by the frozen weight",
          c["full_bytes"] / c["lora_bytes"],
          4 * d_real**2 / (d_real**2 + 8 * d_real * r_real), tol=1e-12)
    param_ratio = c["full_trainable"] / c["lora_trainable"]
    mem_ratio = c["full_bytes"] / c["lora_bytes"]
    print(f"      so the {param_ratio:.0f}x parameter saving is only a {mem_ratio:.2f}x memory "
          f"saving: {param_ratio/mem_ratio:.1f}x of it is eaten by the frozen weight")
    check("the parameter saving overstates the memory saving by more than 30x",
          param_ratio / mem_ratio > 30.0)
    # And the crossover: LoRA stops saving anything at all at r = d/2.
    check("LoRA saves nothing once r >= d/2",
          param_counts(d_real, d_real // 2)["lora_trainable"] >= d_real**2)

    print("\n--- rank is a claim about a spectrum (Eckart-Young) ---")
    sig_low = np.concatenate([np.array([4.0, 3.0, 2.0, 1.0]), np.zeros(d_in - 4)])
    sig_slow = np.arange(1, d_in + 1) ** -1.0
    targets = {"exactly rank 4": matrix_with_spectrum(sig_low, rng),
               "sigma_i ~ 1/i": matrix_with_spectrum(sig_slow, rng)}
    ranks = (1, 2, 4, 8, 16, 32)
    fits, bounds = {}, {}
    print(f"      {'target':16}{'':7}" + " ".join(f"{('r=%d' % r):>11}" for r in ranks))
    for name, D in targets.items():
        fits[name], bounds[name] = {}, {}
        for r in ranks:
            A_r, B_r = fit_lora_als(D, r, alpha, np.random.default_rng(1))
            fits[name][r] = float(np.linalg.norm(lora_delta(A_r, B_r, alpha) - D))
            bounds[name][r] = eckart_young_bound(D, r)
        print(f"      {name:16}  fit  " + " ".join(f"{fits[name][r]:11.3e}" for r in ranks))
        print(f"      {'':16}  bound" + " ".join(f"{bounds[name][r]:11.3e}" for r in ranks))
        print()
    # THE theorem: no rank-r matrix beats the singular-value tail. LoRA is a
    # rank-r matrix, so every fit above must sit on or above its bound.
    check("every LoRA fit respects the Eckart-Young lower bound",
          all(fits[k][r] >= bounds[k][r] - 1e-12 for k in targets for r in ranks))
    # And ALS actually attains it, so the numbers below are about capacity and
    # not about a lazy optimizer.
    worst = max(fits[k][r] / bounds[k][r] for k in targets for r in ranks
                if bounds[k][r] > 1e-12)
    print(f"      worst fit/bound ratio over all (target, rank) with a nonzero bound: {worst:.6f}")
    check("ALS attains the bound to within 0.5%", worst < 1.005)
    # The contrast that is the whole exercise: at the SAME rank 8, one target is
    # reproduced to machine precision and the other is not reproduced at all.
    lo, sl = fits["exactly rank 4"][8], fits["sigma_i ~ 1/i"][8]
    print(f"      at r=8: low-rank target error {lo:.2e}, slow-spectrum target error {sl:.3f} "
          f"(both targets have ||ΔW||_F = 1)")
    check("rank 8 is exact for the low-rank target and useless for the other",
          lo < 1e-10 * sl)
    # Doubling the rank buys a factor on one target and almost nothing on the other.
    gain_low = bounds["exactly rank 4"][2] / max(bounds["exactly rank 4"][4], 1e-300)
    gain_slow = bounds["sigma_i ~ 1/i"][16] / bounds["sigma_i ~ 1/i"][32]
    print(f"      going r=2 -> 4 on the low-rank target cuts the floor by {gain_low:.1e}x;")
    print(f"      going r=16 -> 32 on the slow-spectrum target cuts it by {gain_slow:.2f}x")
    check("the slow spectrum's floor decays only gently with rank", gain_slow < 2.0)
    check("the low-rank target's floor collapses to zero at the true rank", gain_low > 1e12)

    if HAVE_TORCH:
        print("\n--- torch cross-check ---")
        D = targets["sigma_i ~ 1/i"]
        A_c, B_c = fit_lora_als(D, 8, alpha, np.random.default_rng(2), iters=3)
        At = torch.tensor(A_c, requires_grad=True)
        Bt = torch.tensor(B_c, requires_grad=True)
        loss = 0.5 * ((alpha / 8) * (Bt @ At) - torch.tensor(D)).pow(2).sum()
        loss.backward()
        gA, gB = lora_fit_grads(D, A_c, B_c, alpha)
        check("hand-derived grad_A matches torch autograd", gA, At.grad.numpy(), tol=1e-12)
        check("hand-derived grad_B matches torch autograd", gB, Bt.grad.numpy(), tol=1e-12)
        # And the merged-equals-adapted property in torch's own linear layer.
        lin = torch.nn.Linear(d_in, d_out, bias=False).double()
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(W))
        Xt = torch.tensor(X)
        y_two_path = lin(Xt) + (alpha / 8) * (Xt @ torch.tensor(A_t).T) @ torch.tensor(B_t).T
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(W_merged))
        check("torch nn.Linear: merged weight == two-path adapter",
              lin(Xt).detach().numpy(), y_two_path.detach().numpy(), tol=1e-12)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) Initialize BOTH factors at random. Nothing errors, nothing is NaN, and
    #     the loss curve will look perfectly normal — but the model you started
    #     fine-tuning is no longer the model you pretrained.
    r_b = 8
    A_bad = rng.standard_normal((r_b, d_in)) / np.sqrt(d_in)
    B_bad = rng.standard_normal((d_out, r_b)) / np.sqrt(r_b)   # the "symmetric" choice
    dW_bad = lora_delta(A_bad, B_bad, alpha)
    # Entries of B A sum r products of independent N(0,1/r) and N(0,1/d_in)
    # terms, so std(ΔW) = (alpha/r) * sqrt(r * (1/r) * (1/d_in)) = (alpha/r)/sqrt(d_in).
    predicted_std = (alpha / r_b) / np.sqrt(d_in)
    print(f"      std(ΔW) measured {dW_bad.std():.4f} vs predicted {predicted_std:.4f}; "
          f"std(W) = {W.std():.4f}")
    check("the spurious ΔW has exactly the magnitude the init variance predicts",
          float(dW_bad.std()), predicted_std, tol=0.1 * predicted_std)
    y_bad = lora_forward(X, W, A_bad, B_bad, alpha)
    rel = float(np.linalg.norm(y_bad - y_base) / np.linalg.norm(y_base))
    rel_good = float(np.linalg.norm(lora_forward(X, W, *lora_init(d_in, d_out, r_b, rng), alpha)
                                    - y_base) / np.linalg.norm(y_base))
    print(f"      relative output perturbation at step 0: random-B {rel:.2f}, correct init {rel_good:.1f}")
    check("random B perturbs the base model's output on the order of the signal itself",
          rel > 0.5)
    check("the correct init perturbs it by exactly nothing", rel_good == 0.0)
    check("and ||ΔW||_F is comparable to ||W||_F, so this is not a small nudge",
          float(np.linalg.norm(dW_bad) / np.linalg.norm(W)) > 0.5)

    # (b) Scale by alpha instead of alpha/r. The scale multiplies the adapter
    #     OUTPUT linearly and the per-step change in ΔW quadratically (one factor
    #     from the gradient, one from the forward), so at fixed lr the effective
    #     learning rate on ΔW moves by r^2. A rank sweep then silently varies the
    #     step size across its own arms.
    A0, B0 = lora_init(d_in, d_out, r_b, rng)
    B1 = np.full_like(B0, 0.01)                        # a common, nonzero B
    out_right = np.linalg.norm(lora_forward(X, W, A0, B1, alpha) - y_base)
    out_wrong = np.linalg.norm(X @ W.T + r_b * (alpha / r_b) * ((X @ A0.T) @ B1.T) - y_base)
    check("at fixed adapters the wrong scale multiplies the adapter output by exactly r",
          out_wrong / out_right, float(r_b), tol=1e-9)
    D_sw = targets["sigma_i ~ 1/i"]
    A_r1, B_r1, _ = fit_lora_gd(D_sw, r_b, alpha, np.random.default_rng(3), lr=0.05, steps=1)
    A_w1, B_w1, _ = fit_lora_gd(D_sw, r_b, alpha, np.random.default_rng(3), lr=0.05, steps=1,
                                scale=alpha)
    step_right = float(np.linalg.norm((alpha / r_b) * (B_r1 @ A_r1)))
    step_wrong = float(np.linalg.norm(alpha * (B_w1 @ A_w1)))
    print(f"      after ONE step, ||ΔW||_F: correct scale {step_right:.4e}, "
          f"wrong scale {step_wrong:.4e}, ratio {step_wrong/step_right:.1f} (r^2 = {r_b**2})")
    check("one step of the mis-scaled run moves ΔW exactly r^2 times as far",
          step_wrong / step_right, float(r_b**2), tol=1e-6 * r_b**2)

    #     Now the confound in the form a practitioner would actually meet it:
    #     one learning rate, one step budget, a sweep over rank.
    lr_sweep, steps_sweep = 0.01, 3000
    sweep_ranks = (2, 4, 8, 16, 32)
    err_right, err_wrong = {}, {}
    for r_s in sweep_ranks:
        err_right[r_s] = fit_lora_gd(D_sw, r_s, alpha, np.random.default_rng(4),
                                     lr=lr_sweep, steps=steps_sweep)[2]
        err_wrong[r_s] = fit_lora_gd(D_sw, r_s, alpha, np.random.default_rng(4),
                                     lr=lr_sweep, steps=steps_sweep, scale=alpha)[2]
    print(f"      lr={lr_sweep}, {steps_sweep} steps    " +
          " ".join(f"{('r=%d' % r):>10}" for r in sweep_ranks))
    print("      scale alpha/r (right) " + " ".join(f"{err_right[r]:10.4f}" for r in sweep_ranks))
    print("      scale alpha   (wrong) " + " ".join(f"{err_wrong[r]:10.4f}" for r in sweep_ranks))
    print("      Eckart-Young floor    " +
          " ".join(f"{eckart_young_bound(D_sw, r):10.4f}" for r in sweep_ranks))
    check("with the right scale every arm of the sweep converges",
          all(np.isfinite(err_right[r]) for r in sweep_ranks))
    check("with the right scale the error decreases with rank, as the floor does",
          all(err_right[sweep_ranks[i + 1]] < err_right[sweep_ranks[i]]
              for i in range(len(sweep_ranks) - 1)))
    check("with the wrong scale, at the SAME lr, the largest-rank arm diverges",
          not np.isfinite(err_wrong[32]))
    check("and it is the high-rank arms that blow up, not the low-rank ones",
          np.isfinite(err_wrong[2]) and np.isfinite(err_wrong[4]))
    # The conclusion "rank 16 and up is unstable" would be drawn from the second
    # row — and it is false: capacity is monotone in r, only the step size moved.
    check("yet capacity is monotone in rank: the floor never rises",
          all(eckart_young_bound(D_sw, sweep_ranks[i + 1]) <
              eckart_young_bound(D_sw, sweep_ranks[i])
              for i in range(len(sweep_ranks) - 1)))
    # The correct scale is not a cure either, only a much gentler dose of the
    # same disease: its effective step also carries (alpha/r)^2, so at a fixed
    # budget the high-rank arms are the ones left furthest from their floor.
    gap = {r: err_right[r] / eckart_young_bound(D_sw, r) for r in sweep_ranks}
    print("      right-scale error / floor " +
          " ".join(f"{gap[r]:10.3f}" for r in sweep_ranks))
    check("even the correct scale leaves higher ranks relatively less converged",
          all(gap[sweep_ranks[i + 1]] > gap[sweep_ranks[i]]
              for i in range(len(sweep_ranks) - 1)))

    summary()
