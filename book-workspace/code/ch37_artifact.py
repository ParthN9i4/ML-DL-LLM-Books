"""Artifact 37.1 -- Superposition, sparse autoencoders, and a causal steering test.

Three experiments, pure NumPy (torch appears only as an autograd cross-check):
  (1) The toy-superposition model of Elhage et al. (2022): an autoencoder with
      m=20 features and d=5 dimensions. With SPARSE features it packs >2d
      features at low loss (superposition); with DENSE features it keeps ~d.
  (2) A TopK sparse autoencoder (Gao et al., 2024) trained on activations
      built from a KNOWN planted dictionary; we assert recovery of >=80% of
      planted features at cosine similarity > 0.9.
  (3) A causal steering test: clamping a recovered latent moves the
      reconstruction along the planted direction; a control feature does not.
Also prints the L0-vs-reconstruction-error frontier (the sparsity-fidelity
trade-off of Section 37.3.1). Runs CPU-only in ~20 s.
"""
import numpy as np

rng = np.random.default_rng(0)

# ----------------------------------------------------------- tiny Adam in NumPy
def adam(params, grads, state, lr, t, b1=0.9, b2=0.999, eps=1e-8):
    for i, (p, g) in enumerate(zip(params, grads)):
        m, v = state[i]
        m[:] = b1 * m + (1 - b1) * g
        v[:] = b2 * v + (1 - b2) * g * g
        p -= lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)

# ============ (1) TOY MODEL OF SUPERPOSITION: x_hat = ReLU(W^T W x + b) =======
def train_toy(p_active, d=5, m=20, steps=6000, batch=512, lr=3e-3, seed=1):
    r = np.random.default_rng(seed)
    W = r.normal(0, 0.3, (d, m)); b = np.zeros(m)
    I = 0.9 ** np.arange(m)                       # importance weights
    st = [(np.zeros_like(W), np.zeros_like(W)), (np.zeros_like(b), np.zeros_like(b))]
    for t in range(1, steps + 1):
        X = (r.random((batch, m)) < p_active) * r.random((batch, m))
        H = X @ W.T                               # (B,d)  compress
        Z = H @ W + b                             # (B,m)  reconstruct
        Xh = np.maximum(Z, 0)
        dXh = 2 * I * (Xh - X) / batch            # grad of mean_B sum_i I_i e_i^2
        dZ = dXh * (Z > 0)
        dW = H.T @ dZ + (dZ @ W.T).T @ X          # two paths: Z=HW and H=XW^T
        adam([W, b], [dW, dZ.sum(0)], st, lr, t)
    # importance-weighted eval loss, relative to the zero-predictor
    X = (r.random((8192, m)) < p_active) * r.random((8192, m))
    Xh = np.maximum(X @ W.T @ W + b, 0)
    rel = (I * (Xh - X)**2).sum() / (I * X**2).sum()
    return W, rel

# ================== (2) TopK SPARSE AUTOENCODER, manual backprop ==============
def gen_acts(r, B, F, k_true=3, exclude=None):
    """Activations a = sum_i z_i f_i with exactly k_true planted features on."""
    n = F.shape[0]
    scores = r.random((B, n))
    if exclude is not None:
        scores[:, exclude] = -1.0                 # never activate this feature
    idx = np.argpartition(scores, -k_true, 1)[:, -k_true:]
    Zt = np.zeros((B, n))
    np.put_along_axis(Zt, idx, r.uniform(0.5, 1.5, (B, k_true)), 1)
    return Zt @ F

def sae_forward(A, We, be, Wd, bd, k):
    Ac = A - bd
    P = Ac @ We.T + be                            # (B,m) pre-codes
    idx = np.argpartition(P, -k, 1)[:, -k:]
    mask = np.zeros_like(P); np.put_along_axis(mask, idx, 1.0, 1)
    Z = P * mask                                  # TopK latents
    return Ac, P, mask, Z, Z @ Wd + bd

def sae_grads(A, We, be, Wd, bd, k):
    B = A.shape[0]
    Ac, P, mask, Z, Ah = sae_forward(A, We, be, Wd, bd, k)
    R = Ah - A
    dAh = 2 * R / B                               # loss = mean_B ||a_hat - a||^2
    dWd, dbd = Z.T @ dAh, dAh.sum(0)
    dP = (dAh @ Wd.T) * mask                      # TopK: grad only through kept
    dWe, dbe = dP.T @ Ac, dP.sum(0)
    dbd -= (dP @ We).sum(0)                       # bd also enters via Ac = A-bd
    return [dWe, dbe, dWd, dbd], (R**2).sum(1).mean()

def train_sae(F, d, m, k, steps=2500, batch=256, lr=2e-3, seed=2):
    r = np.random.default_rng(seed)
    Wd = r.normal(size=(m, d)); Wd /= np.linalg.norm(Wd, axis=1, keepdims=True)
    We, be, bd = Wd.copy(), np.zeros(m), np.zeros(d)
    ps = [We, be, Wd, bd]
    st = [(np.zeros_like(p), np.zeros_like(p)) for p in ps]
    for t in range(1, steps + 1):
        g, _ = sae_grads(gen_acts(r, batch, F), We, be, Wd, bd, k)
        adam(ps, g, st, lr, t)
        Wd /= np.linalg.norm(Wd, axis=1, keepdims=True)   # unit decoder rows
    A = gen_acts(r, 4096, F)
    *_, Ah = sae_forward(A, We, be, Wd, bd, k)
    nmse = ((Ah - A)**2).sum() / ((A - A.mean(0))**2).sum()
    return (We, be, Wd, bd), nmse

if __name__ == "__main__":
    # ---- (1) superposition: sparse packs > d features, dense keeps ~ d ------
    d, m = 5, 20
    Ws, rel_s = train_toy(p_active=0.05)          # sparse regime
    Wd_, rel_d = train_toy(p_active=1.00)         # dense regime
    nf = lambda W: int((np.linalg.norm(W, axis=0) > 0.5).sum())
    ns, nd = nf(Ws), nf(Wd_)
    # a model keeping d features and reconstructing 0 for the rest pays at
    # least the importance mass of the m-d dropped features (E[x_i^2] equal):
    I = 0.9 ** np.arange(m)
    bound_d_feats = I[d:].sum() / I.sum()
    print(f"[toy] sparse p=0.05: {ns}/{m} features in {d} dims, rel-loss {rel_s:.4f}")
    print(f"[toy]   (floor for d kept features, zero on the rest: {bound_d_feats:.4f})")
    print(f"[toy] dense  p=1.00: {nd}/{m} features in {d} dims, rel-loss {rel_d:.4f}")
    assert ns >= 2 * d, f"superposition failed: only {ns} features packed"
    assert nd <= d + 1, f"dense model kept {nd} > d+1 features"
    assert rel_s < 0.6 * bound_d_feats, "superposition loss not below d-feature bound"

    # ---- (2) SAE recovery of planted features + L0/error frontier -----------
    D, n_true, m_sae = 64, 32, 64
    F = rng.normal(size=(n_true, D)); F /= np.linalg.norm(F, axis=1, keepdims=True)
    print("\n[frontier]  L0(k) | norm. recon MSE | planted features recovered")
    models = {}
    for k in (1, 2, 3, 6, 12):
        params, nmse = train_sae(F, D, m_sae, k)
        C = F @ params[2].T                       # cosines: planted x decoder
        rec = int((C.max(1) > 0.9).sum())
        models[k] = (params, C)
        print(f"            {k:5d} | {nmse:15.2e} | {rec}/{n_true}")
    (We, be, Wd, bd), C = models[3]               # k = true sparsity
    frac = (C.max(1) > 0.9).mean()
    print(f"[sae] k=3 recovery: {frac:.0%} of planted features at cos > 0.9")
    assert frac >= 0.8, f"recovered only {frac:.0%} of planted features"

    # ---- (3) causal steering: clamp matched latent vs control ---------------
    tgt = int(C.max(1).argmax())                  # best-recovered planted feat
    lat = int(C[tgt].argmax())                    # its matching SAE latent
    ctrl = (tgt + 1) % n_true                     # a different planted feature
    A = gen_acts(rng, 512, F, exclude=tgt)        # target feature never active
    *_, Zb, Ab = sae_forward(A, We, be, Wd, bd, 3)
    Zc = Zb.copy(); Zc[:, lat] = 8.0              # clamp the latent on
    Delta = (Zc @ Wd + bd) - Ab                   # steering displacement
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    c_t = cos(Delta.mean(0), F[tgt]); c_c = cos(Delta.mean(0), F[ctrl])
    print(f"[steer] cos(delta, planted target) = {c_t:+.4f}")
    print(f"[steer] cos(delta, control feat)   = {c_c:+.4f}")
    assert c_t > 0.9, "steering did not move along the planted direction"
    assert abs(c_c) < 0.35, "control direction moved too much"

    # ---- torch cross-check of the manual TopK-SAE gradient ------------------
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None:
        print("[skipped: torch not installed]")
    else:
        r = np.random.default_rng(7); A0 = gen_acts(r, 64, F)
        g_np, _ = sae_grads(A0, We, be, Wd, bd, 3)
        tp = [torch.tensor(p, dtype=torch.float64, requires_grad=True)
              for p in (We, be, Wd, bd)]
        tA = torch.tensor(A0, dtype=torch.float64)
        Ac = tA - tp[3]; P = Ac @ tp[0].T + tp[1]
        v, i = torch.topk(P, 3, dim=1)
        Z = torch.zeros_like(P).scatter(1, i, v)
        ((Z @ tp[2] + tp[3] - tA)**2).sum(1).mean().backward()
        diff = max(float((t.grad - torch.tensor(g)).abs().max())
                   for t, g in zip(tp, g_np))
        print(f"[check] max |manual grad - torch autograd| = {diff:.2e}")
        assert diff < 1e-9, "manual gradient disagrees with autograd"
    print("\nAll assertions passed.")
