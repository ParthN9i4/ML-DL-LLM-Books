"""
ex24 — Preference optimization you can check in closed form.  (Book: Chapter 27)

The KL-regularized RLHF objective

    max_pi  E_pi[ r(a) ] - beta * KL(pi || pi_ref)

has an EXACT solution:  pi*(a) proportional to pi_ref(a) * exp(r(a) / beta).

On a bandit — a handful of discrete actions, no context — everything is
computable in closed form, which makes it the one setting where the claims
of Chapter 27 can be tested rather than believed:

  * DPO's derivation inverts pi* for r and regresses on preference pairs. So
    with FULL preference coverage (every pair, labelled with exact
    Bradley-Terry probabilities), minimizing the DPO loss must recover pi*
    exactly. Measured below: total-variation distance 1.9e-11.

  * With PARTIAL coverage (finitely many sampled pairs), the DPO solution
    drifts away from pi* — measured, not asserted. This gap, not the loss
    function, is where real DPO-vs-PPO differences come from.

  * GRPO replaces the critic with a group-relative advantage: sample G
    responses, subtract the group mean. Its gradient is checked against the
    exact policy gradient.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def rlhf_optimum(pi_ref, r, beta):
    """The closed-form KL-regularized optimum: pi* ∝ pi_ref * exp(r / beta).

    Computed in LOG space. The naive form pi_ref * exp(r/beta) overflows the
    moment beta is small (exp(1.3/0.001) is inf) — the first draft of this very
    function did exactly that and returned NaN. The lesson of ex02 applies to
    everything shaped like a softmax, including this one.
    """
    # === YOUR CODE HERE ===
    logw = np.log(pi_ref) + r / beta
    logw = logw - np.max(logw)
    w = np.exp(logw)
    return w / w.sum()


def bt_preference_prob(r, i, j):
    """Bradley-Terry: P(i preferred over j) = sigmoid(r_i - r_j)."""
    # === YOUR CODE HERE ===
    return 1.0 / (1.0 + np.exp(-(r[i] - r[j])))


def dpo_loss_and_grad(theta, pi_ref, pairs, weights, beta):
    """DPO on a bandit. Policy pi_theta = softmax(theta).

    Per pair (w, l):  loss = -log sigmoid( beta * (m_w - m_l) )
    where m_a = log pi_theta(a) - log pi_ref(a) is the implicit reward margin.

    Returns (weighted mean loss, gradient wrt theta). The gradient is derived,
    not autodiff'd — deriving it is the exercise.
    """
    K = len(theta)
    logp = theta - np.max(theta)
    logp = logp - np.log(np.exp(logp).sum())
    pi = np.exp(logp)
    m = logp - np.log(pi_ref)
    # === YOUR CODE HERE ===
    loss = 0.0
    grad = np.zeros(K)
    wsum = weights.sum()
    for (a_w, a_l), wt in zip(pairs, weights):
        z = beta * (m[a_w] - m[a_l])
        s = 1.0 / (1.0 + np.exp(z))          # 1 - sigmoid(z)
        loss += wt * np.log1p(np.exp(-z))
        # d m_a / d theta_k = delta_{ak} - pi_k  =>
        # d z / d theta = beta * (e_w - e_l)  (the -pi terms cancel)
        g = np.zeros(K)
        g[a_w] += beta
        g[a_l] -= beta
        grad += wt * (-s) * g
    return loss / wsum, grad / wsum


def fit_dpo(pi_ref, pairs, weights, beta, lr=0.5, steps=4000):
    """Plain gradient descent on the DPO loss."""
    theta = np.zeros(len(pi_ref))
    # === YOUR CODE HERE ===
    for _ in range(steps):
        _, g = dpo_loss_and_grad(theta, pi_ref, pairs, weights, beta)
        theta = theta - lr * g
    return softmax(theta)


def grpo_advantages(rewards):
    """Group-relative advantages: (r_i - mean) / std. No critic anywhere."""
    r = np.asarray(rewards, dtype=np.float64)
    # === YOUR CODE HERE ===
    return (r - r.mean()) / max(r.std(), 1e-8)


def exact_policy_gradient(theta, r):
    """The exact REINFORCE gradient with the optimal (mean-reward) baseline:
    grad_k = pi_k * (r_k - E_pi[r])."""
    pi = softmax(theta)
    # === YOUR CODE HERE ===
    return pi * (r - pi @ r)


def grpo_gradient_estimate(theta, r, G, rng):
    """Monte-Carlo GRPO gradient: sample a group of G actions from pi_theta,
    compute group-relative advantages, accumulate grad log pi weighted by them."""
    pi = softmax(theta)
    K = len(theta)
    # === YOUR CODE HERE ===
    acts = rng.choice(K, size=G, p=pi)
    adv = grpo_advantages(r[acts])
    grad = np.zeros(K)
    for a, A in zip(acts, adv):
        g = -pi.copy()
        g[a] += 1.0                          # grad log pi(a)
        grad += A * g
    return grad / G


def total_variation(p, q):
    return 0.5 * float(np.abs(p - q).sum())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K, beta = 6, 0.5
    pi_ref = softmax(rng.standard_normal(K))
    r = rng.standard_normal(K)

    print("\n--- the closed form itself ---")
    pi_star = rlhf_optimum(pi_ref, r, beta)
    check("pi* is a distribution", float(pi_star.sum()), 1.0, tol=1e-12)
    # beta -> infinity: KL dominates, pi* -> pi_ref.
    check("beta -> inf recovers the reference policy",
          rlhf_optimum(pi_ref, r, 1e6), pi_ref, tol=1e-5)
    # beta -> 0: reward dominates, pi* -> argmax(r) (among pi_ref's support).
    hard = rlhf_optimum(pi_ref, r, 1e-3)
    check("beta -> 0 collapses onto the best action", float(hard[np.argmax(r)]), 1.0, tol=1e-6)

    print("\n--- DPO gradient is correct (finite differences) ---")
    pairs_all = [(i, j) for i in range(K) for j in range(K) if i != j]
    w_all = np.array([bt_preference_prob(r, i, j) for i, j in pairs_all])
    theta0 = rng.standard_normal(K) * 0.3
    _, g_analytic = dpo_loss_and_grad(theta0, pi_ref, pairs_all, w_all, beta)
    eps = 1e-6
    g_num = np.zeros(K)
    for k in range(K):
        tp, tm = theta0.copy(), theta0.copy()
        tp[k] += eps; tm[k] -= eps
        lp, _ = dpo_loss_and_grad(tp, pi_ref, pairs_all, w_all, beta)
        lm, _ = dpo_loss_and_grad(tm, pi_ref, pairs_all, w_all, beta)
        g_num[k] = (lp - lm) / (2 * eps)
    check("hand-derived DPO gradient matches central differences", g_analytic, g_num, tol=1e-8)

    print("\n--- THE theorem: full coverage recovers the RLHF optimum ---")
    # Full coverage: every ordered pair, weighted by its exact BT probability.
    # DPO's implicit reward must then equal r up to a constant, so pi_dpo = pi*.
    # NOTE the labels must come from BT applied to r at the SAME beta scaling
    # DPO uses for its margin: DPO's margin beta*(m_w - m_l) regresses toward
    # the BT logit r_w - r_l.
    pi_dpo = fit_dpo(pi_ref, pairs_all, w_all, beta)
    tv_full = total_variation(pi_dpo, pi_star)
    print(f"      TV(pi_DPO, pi*) with full coverage: {tv_full:.2e}")
    check("full-coverage DPO recovers pi* (TV < 1e-4)", tv_full < 1e-4)

    print("\n--- and partial coverage does not ---")
    print(f"      {'pairs':>8} {'TV to pi*':>12}")
    tvs = {}
    for n_pairs in (10, 100, 10000):
        idx = rng.choice(len(pairs_all), size=n_pairs, replace=True, p=w_all / w_all.sum())
        # Sampled labels: each drawn pair counts once (empirical preferences).
        sampled = [pairs_all[i] for i in idx]
        pi_hat = fit_dpo(pi_ref, sampled, np.ones(n_pairs), beta)
        tvs[n_pairs] = total_variation(pi_hat, pi_star)
        print(f"      {n_pairs:8d} {tvs[n_pairs]:12.4f}")
    check("coverage matters: 10 pairs is far from pi*", tvs[10] > 5 * tvs[10000])
    check("more pairs converge toward pi*", tvs[10000] < 0.1)

    print("\n--- GRPO ---")
    g_rewards = rng.standard_normal(8)
    adv = grpo_advantages(g_rewards)
    check("group advantages have zero mean", float(adv.mean()), 0.0, tol=1e-12)
    check("group advantages have unit std", float(adv.std()), 1.0, tol=1e-9)

    # The estimator points the right way: average many GRPO gradient estimates
    # and compare the direction against the exact policy gradient.
    theta_g = rng.standard_normal(K) * 0.5
    exact = exact_policy_gradient(theta_g, r)
    est = np.mean([grpo_gradient_estimate(theta_g, r, G=16, rng=rng) for _ in range(4000)], axis=0)
    cos = float(exact @ est / (np.linalg.norm(exact) * np.linalg.norm(est)))
    print(f"      cosine(exact policy gradient, mean GRPO estimate over 4000 groups) = {cos:.4f}")
    check("GRPO estimator aligns with the exact policy gradient (cos > 0.98)", cos > 0.98)

    # And it actually optimizes: run GRPO ascent, expected reward must rise
    # toward the max.
    theta_run = np.zeros(K)
    for _ in range(300):
        theta_run += 0.3 * grpo_gradient_estimate(theta_run, r, G=32, rng=rng)
    final_er = float(softmax(theta_run) @ r)
    print(f"      expected reward: uniform {float(np.mean(r)):.3f} -> GRPO {final_er:.3f} "
          f"(max {r.max():.3f})")
    check("GRPO ascent reaches near-optimal expected reward", final_er > r.max() - 0.15)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) Drop the reference model from the DPO margin (use log pi alone).
    #     The regression target silently changes and the optimum is no longer
    #     pi* — it ignores pi_ref entirely. Measurable on a skewed reference.
    def fit_dpo_no_ref(pairs, weights, beta_, lr=0.5, steps=4000):
        theta = np.zeros(K)
        for _ in range(steps):
            logp = theta - np.max(theta)
            logp = logp - np.log(np.exp(logp).sum())
            grad = np.zeros(K)
            for (a_w, a_l), wt in zip(pairs, weights):
                z = beta_ * (logp[a_w] - logp[a_l])
                s = 1.0 / (1.0 + np.exp(z))
                g = np.zeros(K)
                g[a_w] += beta_
                g[a_l] -= beta_
                grad += wt * (-s) * g
            theta = theta - lr * grad / weights.sum()
        return softmax(theta)

    pi_noref = fit_dpo_no_ref(pairs_all, w_all, beta)
    tv_noref = total_variation(pi_noref, pi_star)
    print(f"      TV to pi*: with reference {tv_full:.2e}, without reference {tv_noref:.3f}")
    check("dropping the reference model moves the optimum materially",
          tv_noref > 100 * max(tv_full, 1e-6))

    # (b) Advantage normalization with a degenerate group: all-equal rewards
    #     give std 0, and without the epsilon guard the advantages are NaN and
    #     the update silently poisons the parameters.
    equal = grpo_advantages(np.full(8, 2.5))
    check("degenerate group (all rewards equal) yields zero advantages, not NaN",
          bool(np.all(np.isfinite(equal)) and np.allclose(equal, 0.0)))
    with np.errstate(invalid="ignore"):
        naive = (np.full(8, 2.5) - 2.5) / np.full(8, 2.5).std()
    print(f"      without the guard: 0/0 -> {naive[0]}")
    check("the unguarded version really does produce NaN", bool(np.isnan(naive[0])))

    summary()
