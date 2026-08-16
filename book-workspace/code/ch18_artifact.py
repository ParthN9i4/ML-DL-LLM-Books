"""
Artifact 18.1 -- Multi-head attention and a pre-norm transformer block, from scratch.
Verifies: (1) logit variance is d_k unscaled and ~1 after dividing by sqrt(d_k);
(2) NumPy MHA forward == torch.nn.MultiheadAttention with transplanted weights;
(3) analytic backward (incl. softmax Jacobian) == central finite differences;
(4) incremental KV-cached decoding == full recomputation;
(5) FLOP and KV-cache tables locating where quadratic attention overtakes the FFN.
Pure NumPy core; torch is a guarded cross-check only.
"""
import numpy as np
try:
    import torch
    import torch.nn as nn
except ImportError:                      # the artifact must run without torch
    torch = None

# --------------------------------------------------------------------- primitives

def softmax(z, axis=-1):
    """Max-subtracted softmax: the shift is exact (softmax is shift-invariant) and
    is the only thing keeping exp() out of overflow. See Chapter 5."""
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def softmax_backward(dA, A, axis=-1):
    """VJP for p = softmax(s): ds = p * (dp - <dp, p>). This is dp contracted with
    J = diag(p) - p p^T, done without ever forming the T x T Jacobian."""
    return A * (dA - np.sum(dA * A, axis=axis, keepdims=True))

def gelu(x):
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))

def gelu_grad(x):
    c = np.sqrt(2.0 / np.pi)
    t = np.tanh(c * (x + 0.044715 * x ** 3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x ** 2)

def layernorm(x, g, b, eps=1e-5):
    inv = 1.0 / np.sqrt(x.var(-1, keepdims=True) + eps)
    xhat = (x - x.mean(-1, keepdims=True)) * inv
    return g * xhat + b, (xhat, inv, g)

def layernorm_backward(dy, cache):
    xhat, inv, g = cache
    d = xhat.shape[-1]
    dg = (dy * xhat).reshape(-1, d).sum(0)
    db = dy.reshape(-1, d).sum(0)
    dxhat = dy * g
    dx = inv * (dxhat - dxhat.mean(-1, keepdims=True)
                - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db

# --------------------------------------------------------------------- attention

class MHA:
    """Weights act as y = W x on column vectors, so row-of-tokens code is X @ W.T."""

    def __init__(self, d, H, rng):
        self.d, self.H, self.dh = d, H, d // H
        s = d ** -0.5
        self.p = dict(Wq=rng.normal(0, s, (d, d)), Wk=rng.normal(0, s, (d, d)),
                      Wv=rng.normal(0, s, (d, d)), Wo=rng.normal(0, s, (d, d)),
                      bq=rng.normal(0, .1, d), bk=rng.normal(0, .1, d),
                      bv=rng.normal(0, .1, d), bo=rng.normal(0, .1, d))

    def _split(self, Z):                 # (T,d) -> (H,T,dh)
        return Z.reshape(Z.shape[0], self.H, self.dh).transpose(1, 0, 2)

    def _merge(self, Z):                 # (H,T,dh) -> (T,d)
        return Z.transpose(1, 0, 2).reshape(-1, self.d)

    def forward(self, X, causal=True, cache=None, offset=0):
        """cache=None: full forward. cache={'K':...,'V':...}: append this step's K,V
        and attend over everything stored. offset = absolute index of X[0]."""
        p = self.p
        Q = self._split(X @ p["Wq"].T + p["bq"])
        K = self._split(X @ p["Wk"].T + p["bk"])
        V = self._split(X @ p["Wv"].T + p["bv"])
        if cache is not None:
            if cache["K"] is not None:
                K = np.concatenate([cache["K"], K], axis=1)
                V = np.concatenate([cache["V"], V], axis=1)
            cache["K"], cache["V"] = K, V
        S = (Q @ K.transpose(0, 2, 1)) / np.sqrt(self.dh)        # (H,Tq,Tk)
        if causal:
            i = np.arange(S.shape[1])[:, None] + offset
            j = np.arange(S.shape[2])[None, :]
            # MASK THEN SOFTMAX: additive -inf on logits, never a multiply on the
            # probabilities. Zeroing after softmax leaves the denominator wrong.
            S = np.where(j <= i, S, -np.inf)
        A = softmax(S)
        Ctx = self._merge(A @ V)
        self.cache = (X, Q, K, V, A, Ctx)
        return Ctx @ p["Wo"].T + p["bo"]

    def backward(self, dO):
        X, Q, K, V, A, Ctx = self.cache
        p, g = self.p, {}
        g["Wo"], g["bo"] = dO.T @ Ctx, dO.sum(0)
        dCtx = self._split(dO @ p["Wo"])
        dA = dCtx @ V.transpose(0, 2, 1)
        dV = A.transpose(0, 2, 1) @ dCtx
        dS = softmax_backward(dA, A) / np.sqrt(self.dh)
        dQ, dK = dS @ K, dS.transpose(0, 2, 1) @ Q
        dX = np.zeros_like(X)
        for nm, dZ in (("q", self._merge(dQ)), ("k", self._merge(dK)),
                       ("v", self._merge(dV))):
            g["W" + nm], g["b" + nm] = dZ.T @ X, dZ.sum(0)
            dX = dX + dZ @ p["W" + nm]
        return dX, g

class Block:
    """Pre-norm: h = x + Attn(LN(x)); y = h + FFN(LN(h)). The residual stream is never
    normalized in place -- each sublayer reads a normalized copy of it and writes back
    an unnormalized increment."""

    def __init__(self, d, H, dff, rng):
        self.attn = MHA(d, H, rng)
        self.p = dict(g1=np.ones(d), b1=np.zeros(d), g2=np.ones(d), b2=np.zeros(d),
                      W1=rng.normal(0, d ** -0.5, (dff, d)), c1=np.zeros(dff),
                      W2=rng.normal(0, dff ** -0.5, (d, dff)), c2=np.zeros(d))

    def forward(self, X, cache=None, offset=0):
        p = self.p
        n1, c1 = layernorm(X, p["g1"], p["b1"])
        h = X + self.attn.forward(n1, cache=cache, offset=offset)
        n2, c2 = layernorm(h, p["g2"], p["b2"])
        u = n2 @ p["W1"].T + p["c1"]
        a = gelu(u)
        self.cache = (c1, c2, n2, u, a)
        return h + a @ p["W2"].T + p["c2"]

    def backward(self, dY):
        p, (c1, c2, n2, u, a) = self.p, self.cache
        g = {"W2": dY.T @ a, "c2": dY.sum(0)}
        du = (dY @ p["W2"]) * gelu_grad(u)
        g["W1"], g["c1"] = du.T @ n2, du.sum(0)
        dh, g["g2"], g["b2"] = layernorm_backward(du @ p["W1"], c2)
        dh = dh + dY                                   # residual around the FFN
        dn1, ga = self.attn.backward(dh)
        dX, g["g1"], g["b1"] = layernorm_backward(dn1, c1)
        return dX + dh, {**g, **{"attn." + k: v for k, v in ga.items()}}

    def params(self):
        return {**self.p, **{"attn." + k: v for k, v in self.attn.p.items()}}

# --------------------------------------------------------------------- checks

def check_scaling(rng, dk=64, n=200_000):
    q, k = rng.normal(size=(n, dk)), rng.normal(size=(n, dk))
    raw = (q * k).sum(1)
    print(f"  Var(q.k)          = {raw.var():8.3f}   (predicted d_k = {dk})")
    print(f"  Var(q.k/sqrt(dk)) = {(raw / np.sqrt(dk)).var():8.3f}   (predicted 1.000)")
    lo = softmax(raw[:8][None, :])[0].max()
    hi = softmax(raw[:8][None, :] / np.sqrt(dk))[0].max()
    print(f"  max prob over 8 keys: unscaled {lo:.4f}  scaled {hi:.4f}")

def check_torch(blk, X, d, H):
    if torch is None:
        print("  [skipped: torch not installed]")
        return
    m = nn.MultiheadAttention(d, H, batch_first=True, dtype=torch.float64)
    p, T = blk.attn.p, X.shape[0]
    with torch.no_grad():
        m.in_proj_weight.copy_(torch.tensor(np.concatenate([p["Wq"], p["Wk"], p["Wv"]])))
        m.in_proj_bias.copy_(torch.tensor(np.concatenate([p["bq"], p["bk"], p["bv"]])))
        m.out_proj.weight.copy_(torch.tensor(p["Wo"]))
        m.out_proj.bias.copy_(torch.tensor(p["bo"]))
        mask = torch.triu(torch.full((T, T), float("-inf"), dtype=torch.float64), 1)
        t = torch.tensor(X)[None]
        ref, _ = m(t, t, t, attn_mask=mask, need_weights=False)
    err = np.abs(blk.attn.forward(X) - ref[0].numpy()).max()
    print(f"  max|numpy - torch.nn.MultiheadAttention| = {err:.3e}")
    assert err < 1e-9, err

def check_backward(blk, X, G, rng, k=4, eps=1e-6):
    Y = blk.forward(X)
    dX, grads = blk.backward(G)
    def loss():
        return np.sum(blk.forward(X) * G)
    worst = 0.0
    targets = [(P, grads[nm]) for nm, P in blk.params().items()] + [(X, dX)]
    for P, gm in targets:
        for _ in range(k):
            ix = tuple(rng.integers(0, s) for s in P.shape)
            old = P[ix]
            P[ix] = old + eps; lp = loss()
            P[ix] = old - eps; lm = loss()
            P[ix] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - gm[ix]) / max(1.0, abs(num)))
    print(f"  worst relative grad error, {k} coords per tensor = {worst:.3e}")
    assert worst < 1e-6, worst
    return Y

def check_kv_cache(blk, X, Y_full):
    """The assertion that catches real cache bugs: token-at-a-time decoding must
    reproduce the full-sequence forward exactly, offset in the causal mask included."""
    cache, rows = {"K": None, "V": None}, []
    for t in range(X.shape[0]):
        rows.append(blk.forward(X[t:t + 1], cache=cache, offset=t)[0])
    err = np.abs(np.stack(rows) - Y_full).max()
    print(f"  max|cached decode - full recompute| = {err:.3e}")
    assert err < 1e-6, err

def flop_table(d=768, H=12, dff=None):
    dff = dff or 4 * d
    print(f"  d_model={d} heads={H} d_head={d//H} d_ff={dff}  (one block, fwd, 2 FLOP/MAC)")
    print(f"  {'n':>6} {'QKVO 8nd^2':>14} {'FFN 4n d dff':>15} {'attn 4n^2 d':>15} {'ratio':>7}")
    for n in (128, 512, 1024, 2048, 3072, 4096, 8192, 16384):
        qkvo, ffn, att = 8 * n * d * d, 4 * n * d * dff, 4 * n * n * d
        print(f"  {n:>6} {qkvo:>14,} {ffn:>15,} {att:>15,} {att/ffn:>7.3f}")
    print(f"  attn == FFN at n = d_ff = {dff} = {dff//d}*d_model")
    print(f"  attn == all linear terms at n = 2d + d_ff = {2*d + dff}")

def kv_cache_table(d=4096, H=32, L=32, n=8192, bytes_per=2):
    dh = d // H
    print(f"  d={d} H={H} d_head={dh} L={L} n={n}  fp16 KV cache, one sequence")
    for name, Hkv in (("MHA (H_kv=32)", 32), ("GQA (H_kv=8)", 8), ("GQA (H_kv=4)", 4),
                      ("MQA (H_kv=1)", 1)):
        b = 2 * L * n * Hkv * dh * bytes_per
        print(f"  {name:<16} {b/2**30:8.3f} GiB   reduction {H/Hkv:5.1f}x")

def main():
    rng = np.random.default_rng(0)
    d, H, dff, T = 32, 4, 64, 12
    blk = Block(d, H, dff, rng)
    X, G = rng.normal(size=(T, d)), rng.normal(size=(T, d))
    print("[1] scaling factor and logit variance")
    check_scaling(rng)
    print("[2] forward vs torch.nn.MultiheadAttention (transplanted weights)")
    check_torch(blk, X, d, H)
    print("[3] backward vs central finite differences")
    Y = check_backward(blk, X, G, rng)
    print("[4] KV-cached incremental decode vs full recompute")
    check_kv_cache(blk, X, Y)
    print("[5] FLOP accounting")
    flop_table()
    print("[6] KV-cache economics")
    kv_cache_table()
    print("all checks passed")

if __name__ == "__main__":
    main()
