"""
Artifact 17.1 -- An LSTM cell with full backpropagation through time, from scratch.

[1] BPTT checked against torch.nn.LSTM with the same weights transplanted in
    (torch packs the gates as [i, f, g, o]; a wrong permutation is also shown).
[2] The constant-error carousel: with the h-path cut, dc_T/dc_0 = diag(prod_t f_t).
[3] || dh_T / dh_0 ||_2 versus T -- Gelfand's rho for a linear RNN, then a tanh
    RNN and an LSTM driven by real inputs.
[4] The parallelism deficit: T sequential GEMMs vs one time-fused GEMM.

Pure NumPy for everything that matters; torch is a cross-check only.
"""
import time
import numpy as np

try:
    import torch
except ImportError:
    torch = None

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
# Gate block order is torch's: rows [0:H)=i, [H:2H)=f, [2H:3H)=g, [3H:4H)=o,
# so weights transplant without a permutation.
class LSTM:
    def __init__(self, d_in, H, forget_bias=0.0, scale=1.0, rng=None):
        rng = rng or np.random.default_rng(0)
        self.d_in, self.H = d_in, H
        s = scale / np.sqrt(H)
        self.Wx = rng.normal(0.0, s, (4 * H, d_in))   # input-to-hidden
        self.Wh = rng.normal(0.0, s, (4 * H, H))      # hidden-to-hidden
        self.bx = np.zeros(4 * H)                     # torch keeps two biases
        self.bh = np.zeros(4 * H)
        self.bx[H:2 * H] = forget_bias                # push the forget gate open

    def forward(self, X, h0, c0):
        """X: (T,B,d_in). Returns h stack (T,B,H) and a cache for BPTT."""
        T, B, H = X.shape[0], X.shape[1], self.H
        h, c = h0.copy(), c0.copy()
        Hs, cache = np.empty((T, B, H)), []
        for t in range(T):
            z = X[t] @ self.Wx.T + h @ self.Wh.T + self.bx + self.bh
            i, f = sigmoid(z[:, 0:H]), sigmoid(z[:, H:2 * H])
            g, o = np.tanh(z[:, 2 * H:3 * H]), sigmoid(z[:, 3 * H:4 * H])
            c_new = f * c + i * g                 # the additive cell update
            tc = np.tanh(c_new)
            cache.append((X[t], h, c, i, f, g, o, tc))
            h, c = o * tc, c_new
            Hs[t] = h
        return Hs, (h, c), cache

    def backward(self, cache, dHs, dh_T=None, dc_T=None, cut_h_path=False):
        """Reverse-mode BPTT.

        dHs : (T,B,H) upstream gradient injected at every step (may be zeros).
        dh_T, dc_T : extra gradient injected at the final state.
        cut_h_path : if True, do not route gate gradients back into h_{t-1}.
                     This isolates the cell-state highway (demo [2]).
        """
        T, H, B = len(cache), self.H, cache[0][1].shape[0]
        gWx, gWh = np.zeros_like(self.Wx), np.zeros_like(self.Wh)
        gb, gX = np.zeros(4 * H), np.zeros((T, B, self.d_in))
        dh = np.zeros((B, H)) if dh_T is None else dh_T.copy()
        dc = np.zeros((B, H)) if dc_T is None else dc_T.copy()
        for t in reversed(range(T)):
            x, h_prev, c_prev, i, f, g, o, tc = cache[t]
            dh = dh + dHs[t]
            do = dh * tc
            dc = dc + dh * o * (1.0 - tc * tc)      # through tanh(c_t)
            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc = dc * f                              # <-- the carousel: multiply by f
            # through the gate nonlinearities
            dz = np.concatenate([di * i * (1 - i),
                                 df * f * (1 - f),
                                 dg * (1 - g * g),
                                 do * o * (1 - o)], axis=1)
            gWx += dz.T @ x
            gWh += dz.T @ h_prev
            gb += dz.sum(axis=0)
            gX[t] = dz @ self.Wx
            dh = np.zeros((B, H)) if cut_h_path else dz @ self.Wh
        # torch carries two identical bias vectors, so each receives gb
        return dict(Wx=gWx, Wh=gWh, bx=gb, bh=gb.copy(), X=gX, h0=dh, c0=dc)

class VanillaRNN:
    """h_t = tanh(Wx x_t + Wh h_{t-1} + b), with Wh rescaled to a chosen rho."""

    def __init__(self, d_in, H, rho, rng=None):
        rng = rng or np.random.default_rng(1)
        W = rng.normal(0.0, 1.0 / np.sqrt(H), (H, H))
        self.Wh = W * (rho / max(abs(np.linalg.eigvals(W))))
        self.Wx = rng.normal(0.0, 1.0 / np.sqrt(H), (H, d_in))
        self.H = H

    def jac_norm(self, X, h0):
        """|| d h_T / d h_0 ||_2, built exactly as prod_t diag(1-h_t^2) Wh."""
        h = h0.copy()
        J = np.eye(self.H)
        sat = 0.0
        for t in range(X.shape[0]):
            h = np.tanh(X[t] @ self.Wx.T + h @ self.Wh.T)
            J = (np.diag(1.0 - h[0] ** 2) @ self.Wh) @ J
            sat = max(sat, np.abs(h).max())
        return np.linalg.norm(J, 2), sat

def lstm_jac_norm(net, X, h0, c0):
    """|| d h_T / d h_0 ||_2 by running BPTT once per unit basis vector."""
    _, _, cache = net.forward(X, h0, c0)
    T, B, H = X.shape[0], h0.shape[0], net.H
    Z, J = np.zeros((T, B, H)), np.empty((H, H))
    for k in range(H):
        seed = np.zeros((B, H)); seed[0, k] = 1.0
        J[k] = net.backward(cache, Z, dh_T=seed)["h0"][0]  # row k of the Jacobian
    return np.linalg.norm(J, 2)

# ----------------------------------------------------------------------------
def demo_torch_check(T=12, B=4, d_in=7, H=5):
    print("[1] BPTT vs torch.nn.LSTM, weights transplanted, float64")
    rng = np.random.default_rng(0)
    net = LSTM(d_in, H, forget_bias=0.7, scale=1.0, rng=rng)
    X = rng.normal(size=(T, B, d_in))
    h0, c0 = rng.normal(size=(B, H)), rng.normal(size=(B, H))
    R = rng.normal(size=(T, B, H))                     # loss = sum(R * h)
    Hs, _, cache = net.forward(X, h0, c0)
    gn = net.backward(cache, R)
    print(f"    loss (numpy)              = {float((R * Hs).sum()):.10f}")
    if torch is None:
        print("    [skipped: torch not installed]"); return
    tl = torch.nn.LSTM(d_in, H, num_layers=1, batch_first=False, dtype=torch.float64)
    with torch.no_grad():
        tl.weight_ih_l0.copy_(torch.from_numpy(net.Wx))
        tl.weight_hh_l0.copy_(torch.from_numpy(net.Wh))
        tl.bias_ih_l0.copy_(torch.from_numpy(net.bx))
        tl.bias_hh_l0.copy_(torch.from_numpy(net.bh))
    tX = torch.tensor(X, requires_grad=True)
    th0, tc0 = (torch.tensor(v[None], requires_grad=True) for v in (h0, c0))
    out, _ = tl(tX, (th0, tc0))
    loss = (torch.tensor(R) * out).sum()
    loss.backward()
    print(f"    loss (torch)              = {loss.item():.10f}   "
          f"|dloss| = {abs(loss.item() - float((R * Hs).sum())):.3e}")
    pairs = [("h stack", Hs, out), ("dWx", gn["Wx"], tl.weight_ih_l0.grad),
             ("dWh", gn["Wh"], tl.weight_hh_l0.grad), ("dbx", gn["bx"], tl.bias_ih_l0.grad),
             ("dX", gn["X"], tX.grad), ("dh0", gn["h0"], th0.grad[0]),
             ("dc0", gn["c0"], tc0.grad[0])]
    worst = 0.0
    for name, a, b in pairs:
        b = b.detach().numpy()
        r = np.abs(a - b).max() / max(1e-12, np.abs(b).max())
        worst = max(worst, r)
        print(f"    max rel err {name:<8s}      = {r:.3e}")
    assert worst < 1e-9, worst
    # ---- the classic bug: torch's block order is (i,f,g,o), not (i,g,f,o) ----
    bad = LSTM(d_in, H, rng=np.random.default_rng(0))
    perm = np.r_[0:H, 2 * H:3 * H, H:2 * H, 3 * H:4 * H]   # swap f and g blocks
    bad.Wx, bad.Wh = net.Wx[perm], net.Wh[perm]
    bad.bx, bad.bh = net.bx[perm], net.bh[perm]
    Hb, _, _ = bad.forward(X, h0, c0)
    print(f"    same weights, gates read (i,g,f,o): rel err "
          f"{np.abs(Hb - Hs).max() / np.abs(Hs).max():.3e}  <-- silent, not a crash")
    assert np.abs(Hb - Hs).max() / np.abs(Hs).max() > 0.1

def demo_carousel(T=40, B=3, H=6):
    print("\n[2] Cell-state highway: d c_T / d c_0 with the h-path cut")
    rng = np.random.default_rng(2)
    net = LSTM(4, H, forget_bias=1.5, scale=1.0, rng=rng)
    X = rng.normal(size=(T, B, 4))
    h0, c0 = rng.normal(size=(B, H)), rng.normal(size=(B, H))
    _, _, cache = net.forward(X, h0, c0)
    dc0 = net.backward(cache, np.zeros((T, B, H)), dc_T=np.ones((B, H)),
                       cut_h_path=True)["c0"]
    prod_f = np.ones((B, H))
    for t in range(T):
        prod_f *= cache[t][4]
    err = np.abs(dc0 - prod_f).max()
    print(f"    T = {T}, mean forget gate = {np.mean([c[4].mean() for c in cache]):.6f}")
    print(f"    prod_t f_t  in [{prod_f.min():.3e}, {prod_f.max():.3e}]")
    print(f"    max |dc0 - prod_t f_t|    = {err:.3e}")
    assert err < 1e-12, err

def demo_spectral(H=32, Ts=(16, 64, 256)):
    """Linear RNN: J = Wh^T exactly, so Gelfand's formula ||Wh^T||^(1/T) -> rho."""
    print("\n[3a] Linear RNN: the spectral radius is the growth rate (Gelfand)")
    rng = np.random.default_rng(4)
    W = rng.normal(0.0, 1.0 / np.sqrt(H), (H, H))
    print(f"    {'rho':>5} {'T':>5} {'||Wh^T||_2':>13} {'rho^T':>13} {'||Wh^T||^(1/T)':>16}")
    for rho in (0.5, 0.9, 1.2):
        Wh = W * (rho / max(abs(np.linalg.eigvals(W))))
        for T in Ts:
            n = np.linalg.norm(np.linalg.matrix_power(Wh, T), 2)
            print(f"    {rho:5.1f} {T:5d} {n:13.4e} {rho ** T:13.4e} {n ** (1.0 / T):16.5f}")
        assert abs(n ** (1.0 / T) - rho) < 0.01, (rho, n ** (1.0 / T))

def demo_decay(H=32, d_in=8, Ts=(1, 2, 4, 8, 16, 32, 64, 128, 256)):
    print("\n[3b] || d h_T / d h_0 ||_2 versus sequence length, driven by real inputs")
    X = np.random.default_rng(3).normal(size=(max(Ts), 1, d_in))
    h0, c0 = np.zeros((1, H)), np.zeros((1, H))
    rnns = {r: VanillaRNN(d_in, H, r, rng=np.random.default_rng(4)) for r in (0.5, 0.9, 1.2)}
    lstm = LSTM(d_in, H, forget_bias=4.0, scale=0.4, rng=np.random.default_rng(5))
    print(f"    {'T':>5} | {'tanh rho=0.5':>13} {'tanh rho=0.9':>13} {'tanh rho=1.2':>13}"
          f" | {'LSTM':>11} | {'LSTM / rho=0.9':>15}")
    print("    " + "-" * 84)
    for T in Ts:
        v = {rho: r.jac_norm(X[:T], h0)[0] for rho, r in rnns.items()}
        vl = lstm_jac_norm(lstm, X[:T], h0, c0)
        print(f"    {T:5d} | {v[0.5]:13.3e} {v[0.9]:13.3e} {v[1.2]:13.3e} | "
              f"{vl:11.3e} | {vl / v[0.9]:15.3e}")
    sat = rnns[1.2].jac_norm(X, h0)[1]      # why rho > 1 does not explode here
    print(f"    tanh saturation kills even rho=1.2: max |h_t| reached {sat:.4f}, "
          f"so diag(1-h^2) shrinks the product")
    assert v[0.5] < 1e-90 and v[0.9] < 1e-40                 # decayed to nothing
    assert v[0.5] <= 0.5 ** max(Ts) and v[0.9] <= 0.9 ** max(Ts)   # |tanh'| <= 1
    assert vl > 1e-2 and vl / v[0.9] > 1e30                  # the LSTM does not

def demo_parallelism(T=512, d=256, Bs=(1, 8, 32, 64), reps=5, dt=np.float32):
    print("\n[4] Parallelism deficit: T sequential GEMMs vs one time-fused GEMM")
    rng = np.random.default_rng(6)
    Wx = (rng.normal(size=(d, 4 * d)) / np.sqrt(d)).astype(dt)
    Wh = (rng.normal(size=(d, 4 * d)) / np.sqrt(d)).astype(dt)
    print(f"    T={T}, d={d}, fp32. Sequential depth {T} vs 1; only B is parallel.")
    print(f"    {'B':>5} {'seq ms':>9} {'seq GF/s':>10} {'fused ms':>10} "
          f"{'fused GF/s':>11} {'gap':>7}")
    for B in Bs:
        X = rng.normal(size=(T, B, d)).astype(dt)
        flops = 2.0 * T * B * d * (4 * d)
        seq, par = np.inf, np.inf
        for _ in range(reps):
            h = np.zeros((B, d), dt); t0 = time.perf_counter()
            for t in range(T):                      # the loop that cannot be unrolled
                h = np.tanh((X[t] @ Wx + h @ Wh)[:, :d])
            seq = min(seq, time.perf_counter() - t0)
        Xf = X.reshape(T * B, d)
        for _ in range(reps):
            t0 = time.perf_counter()
            _ = Xf @ Wx                             # all T steps at once
            par = min(par, time.perf_counter() - t0)
        sr, pr = 2 * flops / seq / 1e9, flops / par / 1e9   # seq does 2 GEMMs/step
        print(f"    {B:5d} {seq * 1e3:9.1f} {sr:10.1f} {par * 1e3:10.2f} "
              f"{pr:11.1f} {pr / sr:6.1f}x")
        assert pr > sr, (B, sr, pr)

if __name__ == "__main__":
    t0 = time.perf_counter()
    demo_torch_check()
    demo_carousel()
    demo_spectral()
    demo_decay()
    demo_parallelism()
    print(f"\nall assertions passed in {time.perf_counter() - t0:.1f} s")
