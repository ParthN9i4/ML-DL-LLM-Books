"""Artifact 35.1 -- DDPM and conditional flow matching from scratch on an
analytic 2D Gaussian-mixture target.

Everything is plain NumPy: the MLPs, the backprop, the Adam updates, the
samplers. The target is a 3-component isotropic Gaussian mixture, chosen
because BOTH the exact score of every noised marginal AND the exact
flow-matching velocity field are available in closed form, so the learned
networks can be checked against ground truth -- not against a vibe.

Self-checks (hard assertions):
  (a) learned score (from eps-prediction) matches the analytic score on a
      density-weighted grid at two noise levels, rel. error < 0.15; same
      for the learned FM velocity at t=0.5.
  (b) samples from both methods match the target's mean, covariance
      (Frobenius rel. error < 0.15) and mode weights (abs error < 0.06).
  (c) FM crosses a fixed energy-distance threshold in >=2x fewer function
      evaluations than ancestral DDPM, and beats DDPM head-to-head at 2
      and 4 NFE. Full NFE table printed.
  (d) NumPy backprop gradients match torch autograd to < 1e-5 (cross-check).
"""
import os, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # tiny matmuls: BLAS threads only spin
import numpy as np

rng = np.random.default_rng(0)
t0 = time.time()

# ---------------- target: 3-mode isotropic Gaussian mixture ----------------
W  = np.array([0.5, 0.3, 0.2])                       # mode weights
MU = np.array([[-2.5, 0.0], [2.5, 1.5], [1.0, -2.5]])  # mode means
S2 = np.array([0.45, 0.55, 0.35]) ** 2               # per-mode variances (iso)

def sample_target(n, rg):
    k = rg.choice(3, size=n, p=W)
    return MU[k] + np.sqrt(S2[k])[:, None] * rg.standard_normal((n, 2))

def resp(x, m, v):        # posterior mode responsibilities under N(m_k, v_k I)
    lg = -np.sum((x[:, None, :] - m[None]) ** 2, -1) / (2 * v) - np.log(v)
    lg = lg + np.log(W); lg -= lg.max(1, keepdims=True)
    r = np.exp(lg); return r / r.sum(1, keepdims=True)

def mixture_score(x, m, v):   # exact grad log p for mixture N(m_k, v_k I)
    r = resp(x, m, v)
    return np.einsum('nk,nkd->nd', r / v, m[None] - x[:, None, :])

def mixture_logdens(x, m, v):  # log p_t up to const, for grid weighting
    lg = -np.sum((x[:, None, :] - m[None]) ** 2, -1) / (2 * v) - np.log(v)
    return np.log(np.exp(lg + np.log(W)).sum(1))

def fm_marginal_velocity(x, t):
    """Exact E[x1 - x0 | x_t] for path x_t=(1-t)x0 + t x1, x0~N(0,I), x1~mix."""
    c = t * t * S2 + (1 - t) ** 2                    # per-mode marginal var
    r = resp(x, t * MU, c)
    Ex1 = MU[None] + (t * S2 / c)[:, None] * (x[:, None, :] - t * MU[None])
    Ex0 = (x[:, None, :] - t * Ex1) / (1 - t)                       # E[x0|x,k]
    return np.einsum('nk,nkd->nd', r, Ex1 - Ex0)

# ---------------- tiny MLP + manual backprop + Adam ----------------
def timefeat(t):              # Fourier features of scalar time in [0,1]
    c = 2 * np.pi * t[:, None] * np.array([1., 2., 4., 8.])
    return np.concatenate([t[:, None], np.sin(c), np.cos(c)], 1)   # 9 feats

class MLP:                    # 11 -> 64 -> 64 -> 2, tanh hidden, linear out
    def __init__(self, rg, sizes=(11, 64, 64, 2)):
        self.Ws = [rg.standard_normal((a, b)) * np.sqrt(1.0 / a)
                   for a, b in zip(sizes[:-1], sizes[1:])]
        self.bs = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.Ws + self.bs]
        self.v = [np.zeros_like(p) for p in self.Ws + self.bs]
        self.step = 0

    def forward(self, X):
        self.hs = [X]
        for i, (Wt, b) in enumerate(zip(self.Ws, self.bs)):
            X = X @ Wt + b
            if i < len(self.Ws) - 1: X = np.tanh(X)
            self.hs.append(X)
        return X

    def train_batch(self, X, Y, lr):        # MSE loss; returns loss value
        out = self.forward(X); n = len(X)
        g = 2 * (out - Y) / n                       # dL/dout
        gW, gb = [None] * len(self.Ws), [None] * len(self.bs)
        for i in reversed(range(len(self.Ws))):
            gW[i] = self.hs[i].T @ g; gb[i] = g.sum(0)
            if i: g = (g @ self.Ws[i].T) * (1 - self.hs[i] ** 2)
        self.step += 1; b1, b2 = 0.9, 0.999          # Adam
        for j, (p, gr) in enumerate(zip(self.Ws + self.bs, gW + gb)):
            self.m[j] = b1 * self.m[j] + (1 - b1) * gr
            self.v[j] = b2 * self.v[j] + (1 - b2) * gr ** 2
            mh = self.m[j] / (1 - b1 ** self.step)
            vh = self.v[j] / (1 - b2 ** self.step)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        return np.mean((out - Y) ** 2)

# ---------------- DDPM: schedule, training, ancestral/DDIM sampler ---------
T = 1000
beta = np.linspace(1e-4, 0.02, T)
abar = np.concatenate([[1.0], np.cumprod(1 - beta)])   # abar[k], k=0..T

net_eps, net_v = MLP(rng), MLP(rng)
ITERS = 4000
lr_at = lambda it: 3e-3 if it < 2000 else (1e-3 if it < 3200 else 3e-4)
def main():
    for it in range(ITERS):                                 # DDPM training
        x0 = sample_target(512, rng)
        k = rng.integers(1, T + 1, 512)
        eps = rng.standard_normal((512, 2))
        a = abar[k][:, None]
        xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * eps         # closed-form marginal
        loss_d = net_eps.train_batch(np.concatenate([xt, timefeat(k / T)], 1),
                                     eps, lr_at(it))
    for it in range(ITERS):                                 # flow-matching training
        x1 = sample_target(512, rng)
        x0 = rng.standard_normal((512, 2))
        t = rng.uniform(1e-3, 1 - 1e-3, 512)[:, None]
        xt = (1 - t) * x0 + t * x1                          # linear (rectified) path
        loss_f = net_v.train_batch(np.concatenate([xt, timefeat(t[:, 0])], 1),
                                   x1 - x0, lr_at(it))
    print(f"final losses  DDPM eps-MSE {loss_d:.4f} | FM velocity-MSE {loss_f:.4f}")

    def ddim_sample(n, nfe, eta, rg):       # eta=1: ancestral DDPM; eta=0: PF-ODE
        x = rg.standard_normal((n, 2))
        seq = np.linspace(T, 0, nfe + 1).astype(int)
        for k, kp in zip(seq[:-1], seq[1:]):
            a, ap = abar[k], abar[kp]
            e = net_eps.forward(np.concatenate([x, timefeat(np.full(n, k / T))], 1))
            x0h = (x - np.sqrt(1 - a) * e) / np.sqrt(a)
            s2 = (eta ** 2) * (1 - ap) / (1 - a) * (1 - a / ap) if kp else 0.0
            x = (np.sqrt(ap) * x0h + np.sqrt(max(1 - ap - s2, 0.0)) * e
                 + np.sqrt(s2) * rg.standard_normal((n, 2)))
        return x

    def fm_sample(n, nfe, rg):              # forward-Euler on dx/dt = v_theta(x,t)
        x, dt = rg.standard_normal((n, 2)), 1.0 / nfe
        for i in range(nfe):
            t = np.full(n, i * dt + dt / 2)      # midpoint time label, still 1 NFE
            x += dt * net_v.forward(np.concatenate([x, timefeat(t)], 1))
        return x

    # ---------------- (a) learned score / velocity vs analytic -----------------
    g1 = np.linspace(-4, 4, 41)
    grid = np.stack(np.meshgrid(g1, g1), -1).reshape(-1, 2)

    def wrel(err_vec, ref_vec, logw):       # density-weighted relative L2 error
        w = np.exp(logw - logw.max())
        num = np.sum(w * np.sum(err_vec ** 2, 1))
        den = np.sum(w * np.sum(ref_vec ** 2, 1))
        return np.sqrt(num / den)

    print("\n(a) learned vs analytic fields (density-weighted rel. L2 on grid)")
    for k in (int(0.25 * T), int(0.55 * T)):
        a = abar[k]
        m_t, v_t = np.sqrt(a) * MU, a * S2 + (1 - a)
        s_true = mixture_score(grid, m_t, v_t)
        e = net_eps.forward(np.concatenate([grid, timefeat(np.full(len(grid), k / T))], 1))
        s_hat = -e / np.sqrt(1 - a)                     # score from eps-prediction
        r = wrel(s_hat - s_true, s_true, mixture_logdens(grid, m_t, v_t))
        print(f"  DDPM score, abar={a:.3f}: rel err {r:.4f}")
        assert r < 0.15, f"score mismatch {r}"
    tv = 0.5
    v_true = fm_marginal_velocity(grid, tv)
    v_hat = net_v.forward(np.concatenate([grid, timefeat(np.full(len(grid), tv))], 1))
    lw = mixture_logdens(grid, tv * MU, tv * tv * S2 + (1 - tv) ** 2)
    rv = wrel(v_hat - v_true, v_true, lw)
    print(f"  FM velocity, t=0.5   : rel err {rv:.4f}")
    assert rv < 0.15, f"velocity mismatch {rv}"

    # ---------------- (b) sample statistics vs target --------------------------
    mean_true = W @ MU
    cov_true = (np.einsum('k,kd,ke->de', W, MU, MU)
                + (W @ S2) * np.eye(2) - np.outer(mean_true, mean_true))
    def check_stats(x, name):
        dm = np.abs(x.mean(0) - mean_true).max()
        C = np.cov(x.T)
        dc = np.linalg.norm(C - cov_true) / np.linalg.norm(cov_true)
        hit = np.argmin(np.sum((x[:, None, :] - MU[None]) ** 2, -1), 1)
        dw = np.abs(np.bincount(hit, minlength=3) / len(x) - W).max()
        print(f"  {name}: |dmean| {dm:.3f}  cov rel-F {dc:.3f}  |dweights| {dw:.3f}")
        assert dm < 0.15 and dc < 0.15 and dw < 0.06, name
    print("\n(b) sample moments vs target (4096 samples each)")
    check_stats(ddim_sample(4096, 200, 1.0, rng), "DDPM 200-step ancestral")
    check_stats(fm_sample(4096, 32, rng), "FM   32-step Euler     ")

    # ---------------- (c) energy distance vs NFE -------------------------------
    def energy_dist(x, y):                  # sqrt(2E|X-Y| - E|X-X'| - E|Y-Y'|)
        d = lambda a, b: np.sqrt(((a[:, None, :] - b[None]) ** 2).sum(-1) + 1e-12)
        return np.sqrt(max(2 * d(x, y).mean() - d(x, x).mean() - d(y, y).mean(), 0))

    ref = sample_target(2048, rng)
    floor = energy_dist(sample_target(2048, rng), ref)     # sampling-noise floor
    print(f"\n(c) energy distance to 2048 target samples (floor ~{floor:.4f})")
    print("  NFE | DDPM ancestral | DDPM PF-ODE(DDIM) | flow matching")
    ed, NFES = {}, (1, 2, 4, 8, 16, 50, 200)
    for nfe in NFES:
        ed[('d', nfe)] = energy_dist(ddim_sample(2048, nfe, 1.0, rng), ref)
        ed[('o', nfe)] = energy_dist(ddim_sample(2048, nfe, 0.0, rng), ref)
        ed[('f', nfe)] = energy_dist(fm_sample(2048, nfe, rng), ref)
        print(f"  {nfe:3d} |     {ed[('d', nfe)]:.4f}     |      {ed[('o', nfe)]:.4f}"
              f"       |    {ed[('f', nfe)]:.4f}")
    thr = 0.2                       # quality threshold ~ a few times the floor
    first = lambda tag: next(n for n in NFES if ed[(tag, n)] < thr)
    nf, nd = first('f'), first('d')
    print(f"  => energy distance < {thr}: FM at {nf} NFE, ancestral DDPM at {nd}")
    assert ed[('f', 2)] < ed[('d', 2)] and ed[('f', 4)] < ed[('d', 4)], "low-NFE"
    assert 2 * nf <= nd, "FM should need materially (>=2x) fewer NFE"

    # ---------------- (d) cross-check NumPy backprop against torch -------------
    try:
        import torch
        X = rng.standard_normal((16, 11)); Y = rng.standard_normal((16, 2))
        net = MLP(np.random.default_rng(7))
        tW = [torch.tensor(w, requires_grad=True) for w in net.Ws]
        tb = [torch.tensor(b, requires_grad=True) for b in net.bs]
        h = torch.tensor(X)
        for i in range(3):
            h = h @ tW[i] + tb[i]
            if i < 2: h = torch.tanh(h)
        (((h - torch.tensor(Y)) ** 2).sum() / 16).backward()  # match manual scaling
        out = net.forward(X); g = 2 * (out - Y) / 16     # replicate manual backward
        gW = [None] * 3
        for i in reversed(range(3)):
            gW[i] = net.hs[i].T @ g
            if i: g = (g @ net.Ws[i].T) * (1 - net.hs[i] ** 2)
        r = max(np.abs(gW[i] - tW[i].grad.numpy()).max() for i in range(3))
        print(f"\n(d) max |NumPy grad - torch grad| = {r:.2e}"); assert r < 1e-5
    except ImportError:
        print("\n(d) [skipped: torch not installed]")

    print(f"\nall assertions passed  ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
