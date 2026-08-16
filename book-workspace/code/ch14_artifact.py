"""Artifact 14.1 - Regularization ablation harness (pure NumPy, ~18 s CPU).
(1) dropout as a geometric-mean ensemble: exact for linear-softmax (2^d masks
enumerated), approximate for a ReLU net; (2) label smoothing: optimal logit gap
= log((K-1)(1-a)/a); (3) five regularizers, fixed budget, small vs large data;
(4) grokking. torch appears only as a gradient cross-check.
"""
import time
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def onehot(y, K):
    Q = np.zeros((len(y), K)); Q[np.arange(len(y)), y] = 1.0; return Q

def smooth(Q, a):
    """q_y = 1-a, q_j = a/(K-1): the convention whose optimal logit gap is
    exactly log((K-1)(1-a)/a). (PyTorch instead uses q_j = a/K.)"""
    K = Q.shape[1]
    return Q * (1.0 - a) + (1.0 - Q) * (a / (K - 1.0))

class MLP:
    """Manual forward/backward. Dropout is the classic non-inverted form: multiply
    by a Bernoulli mask while training, multiply activations by p_keep at test
    time -- i.e. weight scaling, the thing Section 14.3 is about."""
    def __init__(self, sizes, rng, scale=1.0, act="relu"):
        self.W = [rng.normal(0, 1, (a, b)) * np.sqrt(2.0 / a) * scale
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.act = act

    def _phi(self, x):
        return np.maximum(x, 0.0) if self.act == "relu" else x ** 2

    def _dphi(self, x):
        return (x > 0).astype(x.dtype) if self.act == "relu" else 2.0 * x

    def forward(self, X, p_keep=1.0, rng=None, masks=None):
        """masks -> use these exact masks; rng -> sample them (train);
        neither -> deterministic weight-scaled inference."""
        h, cache, L = X, [X], len(self.W)
        for i in range(L):
            pre = h @ self.W[i] + self.b[i]
            if i == L - 1:
                h = pre
            else:
                h = self._phi(pre)
                if masks is not None:
                    h = h * masks[i]
                elif rng is not None and p_keep < 1.0:
                    h = h * (rng.random(h.shape) < p_keep)
                elif p_keep < 1.0:
                    h = h * p_keep
            cache.append((pre, h))
        return h, cache

    def backward(self, cache, G):
        """G = dL/d(logits); returns (grads_W, grads_b)."""
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = (cache[0] if i == 0 else cache[i][1]).T @ G
            gb[i] = G.sum(0)
            if i > 0:
                G = (G @ self.W[i].T) * self._dphi(cache[i][0])
        return gW, gb

class Adam:
    """Adam with *decoupled* weight decay (AdamW): decay is not fed to the moments."""
    def __init__(self, params, lr, wd=0.0, b1=0.9, b2=0.98, eps=1e-8):
        self.p, self.lr, self.wd = params, lr, wd
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = [np.zeros_like(q) for q in params]
        self.v = [np.zeros_like(q) for q in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh, vh = self.m[i] / (1 - self.b1 ** self.t), self.v[i] / (1 - self.b2 ** self.t)
            self.p[i] -= self.lr * (mh / (np.sqrt(vh) + self.eps) + self.wd * self.p[i])

def exp1_linear_exact(d=12, K=4, p_keep=0.5):
    """Linear-softmax: normalized geometric mean over ALL 2^d input-dropout masks
    equals softmax of the weight-scaled input. Exact, not approximate."""
    rng = np.random.default_rng(1)
    W, x = rng.normal(0, 1, (d, K)), rng.normal(0, 1, d)
    bits = ((np.arange(2 ** d)[:, None] >> np.arange(d)) & 1).astype(float)
    k = bits.sum(1)
    w = p_keep ** k * (1 - p_keep) ** (d - k)                 # probability of each mask
    geo = np.exp(w @ np.log(softmax((bits * x) @ W)))         # geometric mean
    geo /= geo.sum()
    err = np.abs(geo - softmax((p_keep * x) @ W)).max()
    print(f"  [1a] linear-softmax, {2**d} masks enumerated exactly")
    print(f"       max |geometric-mean ensemble - weight-scaled| = {err:.3e}")
    assert err < 1e-12, err

def exp2_relu_mc(n=64, d=16, hid=64, K=4, p_keep=0.5, n_masks=4000, tol=0.02):
    """Nonlinear case: MC dropout over many masks vs one weight-scaled pass."""
    rng = np.random.default_rng(2)
    net = MLP([d, hid, K], rng, scale=0.7)
    X = rng.normal(0, 1, (n, d))
    Pref = softmax(net.forward(X, p_keep=p_keep)[0])          # weight-scaled
    lin, log = np.zeros((n, K)), np.zeros((n, K))
    for _ in range(n_masks):
        P = softmax(net.forward(X, masks=[(rng.random((n, hid)) < p_keep) * 1.0])[0])
        lin += P; log += np.log(P)
    geo = np.exp(log / n_masks); geo /= geo.sum(1, keepdims=True)
    e_geo, e_ari = np.abs(geo - Pref).mean(), np.abs(lin / n_masks - Pref).mean()
    print(f"  [1b] 1-hidden-layer ReLU net, {n_masks} MC masks, p_keep={p_keep}")
    print(f"       mean |geometric MC  - weight-scaled| = {e_geo:.4f}  (tol {tol})")
    print(f"       mean |arithmetic MC - weight-scaled| = {e_ari:.4f}")
    assert e_geo < tol, e_geo

def exp3_label_smoothing(n=128, K=8, steps=4000, lr=0.05):
    """Square invertible design matrix -> the model can hit any target
    distribution, so the converged logits show the unconstrained optimum."""
    rng = np.random.default_rng(3)
    X, y = rng.normal(0, 1, (n, n)), rng.integers(0, K, n)
    Q1 = onehot(y, K)
    print("  [2] label smoothing: measured vs analytic logit gap log((K-1)(1-a)/a), K=8")
    for a in (0.05, 0.1, 0.2):
        Q, W = smooth(Q1, a), np.zeros((n, K))
        opt = Adam([W], lr=lr)
        for _ in range(steps):
            opt.step([X.T @ ((softmax(X @ W) - Q) / n)])
        Z = X @ W
        zy = Z[np.arange(n), y]
        gap = float((zy - (Z.sum(1) - zy) / (K - 1)).mean())
        pred = float(np.log((K - 1) * (1 - a) / a))
        print(f"      a={a:.2f}: measured {gap:.4f}   analytic {pred:.4f}   "
              f"|diff| {abs(gap - pred):.2e}")
        assert abs(gap - pred) < 2e-2, (a, gap, pred)

def make_task(n_tr, n_te, d=20, K=4, noise=0.10, seed=7):
    """Labels from a fixed random 2-layer teacher plus 10% symmetric label noise."""
    rng = np.random.default_rng(seed)
    T1, T2 = rng.normal(0, 1, (d, 32)), rng.normal(0, 1, (32, K))
    def gen(n, r):
        X = r.normal(0, 1, (n, d))
        y = (np.maximum(X @ T1, 0) @ T2).argmax(1)
        flip = r.random(n) < noise
        y[flip] = r.integers(0, K, flip.sum())
        return X, y
    return gen(n_tr, rng), gen(n_te, np.random.default_rng(seed + 100))

def train(mode, Xtr, ytr, Xte, yte, K=4, steps=2000, B=64, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    net = MLP([Xtr.shape[1], 64, 64, K], rng)
    opt = Adam(net.W + net.b, lr=lr, wd=3e-2 if mode == "weight decay" else 0.0)
    p_keep = 0.7 if mode == "dropout" else 1.0
    alpha = 0.1 if mode == "label smoothing" else 0.0
    for _ in range(steps):
        idx = rng.integers(0, len(ytr), B)
        xb, Q = Xtr[idx], onehot(ytr[idx], K)
        if mode == "mixup":                       # vicinal risk: mix inputs AND targets
            lam, j = rng.beta(0.4, 0.4, (B, 1)), rng.permutation(B)
            xb, Q = lam * xb + (1 - lam) * xb[j], lam * Q + (1 - lam) * Q[j]
        if alpha:
            Q = smooth(Q, alpha)
        z, cache = net.forward(xb, p_keep=p_keep, rng=rng)
        gW, gb = net.backward(cache, (softmax(z) - Q) / B)
        opt.step(gW + gb)
    acc = lambda X, y: float((net.forward(X, p_keep=p_keep)[0].argmax(1) == y).mean())
    return acc(Xtr, ytr), acc(Xte, yte)

def exp4_ablation(seeds=(0, 1, 2)):
    modes = ["none", "dropout", "weight decay", "label smoothing", "mixup"]
    res = {}
    for n_tr, key in ((250, "small"), (25000, "large")):
        (Xtr, ytr), (Xte, yte) = make_task(n_tr, 4000)
        for m in modes:
            res[(key, m)] = np.mean(
                [train(m, Xtr, ytr, Xte, yte, seed=s) for s in seeds], axis=0)
    print("  [3] ablation: same net, same 2000-step budget, 3 seeds averaged")
    print(f"      {'regularizer':<16}{'n=250 train':>12}{'n=250 test':>12}{'n=25000 test':>14}")
    for m in modes:
        print(f"      {m:<16}{res[('small', m)][0]:>12.3f}"
              f"{res[('small', m)][1]:>12.3f}{res[('large', m)][1]:>14.3f}")
    sp = lambda k: (max(res[(k, m)][1] for m in modes) - min(res[(k, m)][1] for m in modes))
    print(f"      test-accuracy spread across regularizers: "
          f"{sp('small'):.3f} (n=250) vs {sp('large'):.3f} (n=25000)")
    assert sp("large") < sp("small"), (sp("small"), sp("large"))

def exp5_grokking(p=23, frac=0.5, hid=128, init=4.0, lr=1e-2, wd=0.7,
                  steps=6000, every=25, seed=0):
    """(a+b) mod p, one-hot inputs, quadratic activation, large init, heavy
    decoupled weight decay. Memorization is fast; generalization is late."""
    rng = np.random.default_rng(seed)
    a, b = np.repeat(np.arange(p), p), np.tile(np.arange(p), p)
    y = (a + b) % p
    X = np.zeros((p * p, 2 * p))
    X[np.arange(p * p), a] = 1.0
    X[np.arange(p * p), p + b] = 1.0
    perm, ntr = rng.permutation(p * p), int(frac * p * p)
    tr, te = perm[:ntr], perm[ntr:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    net = MLP([2 * p, hid, p], rng, scale=init, act="quad")
    opt = Adam(net.W, lr=lr, wd=wd)
    Q = onehot(ytr, p)
    s_mem = s_gen = te_at_mem = None
    for t in range(1, steps + 1):
        z, cache = net.forward(Xtr)
        opt.step(net.backward(cache, (softmax(z) - Q) / ntr)[0])
        if t % every:
            continue
        atr = float((z.argmax(1) == ytr).mean())
        ate = float((net.forward(Xte)[0].argmax(1) == yte).mean())
        if s_mem is None and atr >= 1.0:
            s_mem, te_at_mem = t, ate
        if s_gen is None and ate >= 0.95:
            s_gen = t; break
    print(f"  [4] grokking on (a+b) mod {p}, {ntr}/{p*p} pairs seen, wd={wd}")
    print(f"      train acc 100% at step {s_mem} (test acc there: {te_at_mem:.3f}, "
          f"chance {1/p:.3f})")
    print(f"      test  acc  95% at step {s_gen}   delay factor {s_gen/s_mem:.1f}x")
    assert s_gen >= 10 * s_mem, (s_mem, s_gen)

def crosscheck():
    """Only cross-check: our smoothed-CE logit gradient against torch autograd."""
    if torch is None:
        print("  [x] [skipped: torch not installed]"); return
    rng = np.random.default_rng(11)
    Z, y, K = rng.normal(0, 1, (8, 5)), rng.integers(0, 5, 8), 5
    Q = smooth(onehot(y, K), 0.1)
    Zt = torch.tensor(Z, requires_grad=True)
    (-(torch.tensor(Q) * torch.log_softmax(Zt, dim=1)).sum() / 8).backward()
    err = np.abs((softmax(Z) - Q) / 8 - Zt.grad.numpy()).max()
    print(f"  [x] torch cross-check of smoothed-CE logit gradient: max abs diff {err:.3e}")
    assert err < 1e-10, err

if __name__ == "__main__":
    t0 = time.time()
    print("Artifact 14.1 - regularization levers, measured\n")
    exp1_linear_exact(); exp2_relu_mc(); print()
    exp3_label_smoothing(); print()
    exp4_ablation(); print()
    exp5_grokking(); print()
    crosscheck()
    print(f"\nall assertions passed in {time.time() - t0:.1f} s")
