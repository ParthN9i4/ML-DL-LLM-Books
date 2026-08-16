"""Artifact 27.1 -- DPO/IPO/SimPO losses, the GRPO advantage, and the bandit
where DPO's equivalence to RLHF can be checked against a closed form.

Setup: a K-armed bandit with known reward r(a) and reference policy pi_ref.
The KL-regularized RLHF optimum is available in closed form:
    pi*(a) = pi_ref(a) * exp(r(a)/beta) / Z.
We (1) gradient-check every loss against central finite differences,
(2) show that minimizing the *population* DPO loss over ALL preference pairs
recovers pi* to numerical precision, (3) show that with finite sampled
preference data the DPO solution measurably diverges from pi*, and
(4) verify GRPO group-relative advantages are zero-mean within each group.
Pure NumPy core; torch appears only as an autograd cross-check.
"""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(0)

# ----------------------------- utilities ------------------------------------
def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def softplus(x):                       # log(1+e^x), stable
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# ------------------------- losses (analytic grads) --------------------------
# Policy is pi_theta = softmax(theta) over K actions. For pairwise losses the
# log-partition cancels in log pi(w) - log pi(l), so Delta uses raw logits.
def dpo_loss_grad(theta, ref_logp, w, l, wt, beta):
    """L = sum_i wt_i * -log sigmoid(beta*Delta_i);  Delta = (th_w-th_l)-(ref_w-ref_l)."""
    d = (theta[w] - theta[l]) - (ref_logp[w] - ref_logp[l])
    z = beta * d
    loss = float(np.sum(wt * softplus(-z)))
    coef = -beta * wt * sigmoid(-z)            # dL/dtheta_w per pair
    g = np.zeros_like(theta)
    np.add.at(g, w, coef); np.add.at(g, l, -coef)
    return loss, g

def ipo_loss_grad(theta, ref_logp, w, l, wt, beta):
    """IPO (Azar et al.): L = sum wt_i (Delta_i - 1/(2 beta))^2."""
    d = (theta[w] - theta[l]) - (ref_logp[w] - ref_logp[l])
    e = d - 1.0 / (2.0 * beta)
    loss = float(np.sum(wt * e * e))
    coef = 2.0 * wt * e
    g = np.zeros_like(theta)
    np.add.at(g, w, coef); np.add.at(g, l, -coef)
    return loss, g

def simpo_loss_grad(theta, w, l, wt, beta, gamma):
    """SimPO (Meng et al.): reference-free, length-normalized margin.
    Bandit 'responses' have length 1, so L = sum wt_i -log sigmoid(beta*(th_w-th_l)-gamma)."""
    z = beta * (theta[w] - theta[l]) - gamma
    loss = float(np.sum(wt * softplus(-z)))
    coef = -beta * wt * sigmoid(-z)
    g = np.zeros_like(theta)
    np.add.at(g, w, coef); np.add.at(g, l, -coef)
    return loss, g

def grpo_advantages(rewards):
    """Group-relative advantage: standardize rewards within each group (row)."""
    mu = rewards.mean(axis=1, keepdims=True)
    sd = rewards.std(axis=1, keepdims=True)
    return (rewards - mu) / (sd + 1e-8)

def grpo_loss_grad(theta, actions, old_pi, adv, eps=0.2):
    """Clipped surrogate, single-token episodes: L = -mean_i min(rho_i A_i, clip(rho_i)A_i),
    rho_i = pi_theta(a_i)/pi_old(a_i). Analytic grad uses d pi_a/d theta = pi_a (e_a - pi)."""
    pi = softmax(theta)
    rho = pi[actions] / old_pi[actions]
    unc, cl = rho * adv, np.clip(rho, 1 - eps, 1 + eps) * adv
    loss = float(-np.mean(np.minimum(unc, cl)))
    g = np.zeros_like(theta)
    for i, a in enumerate(actions):                 # only unclipped terms carry gradient
        if unc[i] <= cl[i]:
            drho = rho[i] * ((np.arange(len(theta)) == a) - pi)
            g += -adv[i] * drho / len(actions)
    return loss, g

def fd_check(f, theta, h=1e-5):
    """Max abs error between analytic grad and central finite differences."""
    _, g = f(theta)
    num = np.zeros_like(theta)
    for k in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h; tm[k] -= h
        num[k] = (f(tp)[0] - f(tm)[0]) / (2 * h)
    return np.abs(g - num).max()

# ------------------------------- experiment ---------------------------------
if __name__ == "__main__":
    K, beta = 6, 0.3
    r = np.array([1.0, 0.5, 0.0, -0.5, -1.0, 0.25])          # known reward
    ref_logits = rng.normal(0, 1, K)
    pi_ref = softmax(ref_logits)
    ref_logp = np.log(pi_ref)
    pi_star = softmax(ref_logp + r / beta)                    # closed-form RLHF optimum

    # 1) gradient checks at a generic point
    th0 = rng.normal(0, 1, K)
    ii, jj = np.triu_indices(K, 1)                            # all 15 unordered pairs
    p_win = sigmoid(r[ii] - r[jj])                            # Bradley-Terry soft labels
    w_all = np.concatenate([ii, jj]); l_all = np.concatenate([jj, ii])
    wt_all = np.concatenate([p_win, 1 - p_win]) / len(ii)
    print(f"[grad] DPO   max|analytic-FD| = {fd_check(lambda t: dpo_loss_grad(t, ref_logp, w_all, l_all, wt_all, beta), th0):.2e}")
    print(f"[grad] IPO   max|analytic-FD| = {fd_check(lambda t: ipo_loss_grad(t, ref_logp, w_all, l_all, wt_all, beta), th0):.2e}")
    print(f"[grad] SimPO max|analytic-FD| = {fd_check(lambda t: simpo_loss_grad(t, w_all, l_all, wt_all, 2.0, 0.5), th0):.2e}")
    G, n_g = 8, 16
    rew = rng.normal(0, 1, (n_g, G))
    adv = grpo_advantages(rew)
    print(f"[GRPO] max |group mean of advantages| = {np.abs(adv.mean(axis=1)).max():.2e}")
    assert np.abs(adv.mean(axis=1)).max() < 1e-9, "GRPO advantages must be zero-mean per group"
    acts = rng.integers(0, K, 32)
    old_pi = softmax(rng.normal(0, 1, K))
    adv1 = grpo_advantages(rng.normal(0, 1, (2, 16))).ravel()
    err = fd_check(lambda t: grpo_loss_grad(t, acts, old_pi, adv1), th0)
    print(f"[grad] GRPO clipped surrogate max|analytic-FD| = {err:.2e}")
    for e in [err]:
        assert e < 1e-6
    assert fd_check(lambda t: dpo_loss_grad(t, ref_logp, w_all, l_all, wt_all, beta), th0) < 1e-6

    try:
        import torch
        tt = torch.tensor(th0, requires_grad=True)
        d = (tt[w_all] - tt[l_all]) - torch.tensor(ref_logp[w_all] - ref_logp[l_all])
        Lt = (torch.tensor(wt_all) * torch.nn.functional.softplus(-beta * d)).sum()
        Lt.backward()
        print(f"[torch] DPO grad max diff vs autograd = {np.abs(tt.grad.numpy() - dpo_loss_grad(th0, ref_logp, w_all, l_all, wt_all, beta)[1]).max():.2e}")
    except ImportError:
        print("[skipped: torch not installed]")

    # 2) full preference coverage: population DPO loss over all pairs w/ exact
    #    Bradley-Terry labels. Its minimizer must be the closed-form pi*.
    res = minimize(lambda t: dpo_loss_grad(t, ref_logp, w_all, l_all, wt_all, beta),
                   np.zeros(K), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-14})
    pi_hat = softmax(res.x)
    tv_full = 0.5 * np.abs(pi_hat - pi_star).sum()
    print(f"\npi*      = {np.array2string(pi_star, precision=4)}")
    print(f"pi_DPO   = {np.array2string(pi_hat, precision=4)}  (full coverage)")
    print(f"[full coverage] TV(pi_DPO, pi*) = {tv_full:.2e}")
    assert tv_full < 1e-4, "DPO under full coverage must recover the RLHF optimum"

    # 3) finite coverage: sample N comparisons (uniform pair, BT winner), fit
    #    the empirical DPO loss, measure divergence from pi*.
    print("\n  N pairs   TV(pi_DPO, pi*)")
    tvs = {}
    for N in [30000, 3000, 300, 30]:
        pk = rng.integers(0, len(ii), N)                      # which unordered pair
        first_wins = rng.random(N) < p_win[pk]
        w = np.where(first_wins, ii[pk], jj[pk])
        l = np.where(first_wins, jj[pk], ii[pk])
        res = minimize(lambda t: dpo_loss_grad(t, ref_logp, w, l, np.full(N, 1.0 / N), beta),
                       np.zeros(K), jac=True, method="L-BFGS-B", options={"maxiter": 500})
        tvs[N] = 0.5 * np.abs(softmax(res.x) - pi_star).sum()
        print(f"  {N:7d}   {tvs[N]:.4f}")
    assert tvs[30] > 10 * tvs[30000], "divergence must grow as coverage shrinks"
    assert tvs[30000] < 0.05
    print("\nAll assertions passed: DPO = RLHF optimum under full coverage; "
          "measurably not otherwise.")
