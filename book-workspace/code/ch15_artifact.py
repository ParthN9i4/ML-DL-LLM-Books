import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

DT = np.float32
D_IN, D_OUT = 8, 1
BASE_WIDTH = 64
WIDTHS = [64, 128, 256, 512, 1024]
LOG2_LRS = [-13, -12, -11, -10, -9, -8, -7, -6]
N_TRAIN, BATCH, STEPS, SEEDS = 2048, 256, 36, 2


def init_params(n, param, rng, dtype=DT):
    """Return ([W1,b1,W2,b2,W3], per-tensor Adam LR multipliers)."""
    m = n / BASE_WIDTH                                   # width multiplier
    if param == "sp":
        var = (1.0 / D_IN, 1.0 / n, 1.0 / n)             # LeCun everywhere
        lr_mult = [1.0, 1.0, 1.0, 1.0, 1.0]              # one global LR
    elif param == "mup":
        var = (1.0 / D_IN, 1.0 / n, 1.0 / (n * m))       # readout var ~ 1/fan_in^2
        lr_mult = [1.0, 1.0, 1.0 / m, 1.0, 1.0 / m]      # hidden & readout / m
    else:
        raise ValueError(param)
    p = [rng.normal(0, np.sqrt(var[0]), (D_IN, n)).astype(dtype),
         np.zeros(n, dtype),
         rng.normal(0, np.sqrt(var[1]), (n, n)).astype(dtype),
         np.zeros(n, dtype),
         rng.normal(0, np.sqrt(var[2]), (n, D_OUT)).astype(dtype)]
    return p, lr_mult


def forward(p, X):
    """Post-ReLU activations h1, h2 and the network output f."""
    W1, b1, W2, b2, W3 = p
    h1 = np.maximum(X @ W1 + b1, 0)
    h2 = np.maximum(h1 @ W2 + b2, 0)
    return h1, h2, h2 @ W3


def loss_and_grads(p, X, Y):
    """Mean-squared error and its exact gradient by backprop."""
    W1, b1, W2, b2, W3 = p
    h1, h2, f = forward(p, X)
    r = f - Y
    loss = float(np.mean(r ** 2))
    df = (2.0 / (X.shape[0] * D_OUT)) * r
    gW3 = h2.T @ df
    dz2 = (df @ W3.T) * (h2 > 0)
    gW2, gb2 = h1.T @ dz2, dz2.sum(0)
    dz1 = (dz2 @ W2.T) * (h1 > 0)
    gW1, gb1 = X.T @ dz1, dz1.sum(0)
    return loss, [gW1, gb1, gW2, gb2, gW3]


def adam_train(p, lr_mult, eta, X, Y, steps):
    """Adam with bias correction and per-tensor LR multipliers; in-place."""
    M = [np.zeros_like(q) for q in p]
    V = [np.zeros_like(q) for q in p]
    b1, b2, eps = DT(0.9), DT(0.999), DT(1e-8)
    for t in range(1, steps + 1):
        s = ((t - 1) * BATCH) % (X.shape[0] - BATCH + 1)   # fixed cyclic order
        _, g = loss_and_grads(p, X[s:s + BATCH], Y[s:s + BATCH])
        c1, c2 = DT(1 / (1 - 0.9 ** t)), DT(1 / (1 - 0.999 ** t))
        for i in range(5):
            M[i] *= b1
            M[i] += (1 - b1) * g[i]
            V[i] *= b2
            V[i] += (1 - b2) * g[i] * g[i]
            p[i] -= DT(eta * lr_mult[i]) * (M[i] * c1) / (np.sqrt(V[i] * c2) + eps)
    return float(np.mean((forward(p, X)[2] - Y) ** 2))     # full training loss


def make_data():
    """Fixed random-teacher task y = tanh(xA)b; identical for every run."""
    r = np.random.default_rng(1234)
    X = r.normal(0, 1, (N_TRAIN, D_IN)).astype(DT)
    A = (r.normal(0, 1, (D_IN, 16)) / np.sqrt(D_IN)).astype(DT)
    b = (r.normal(0, 1, (16, D_OUT)) / 4.0).astype(DT)
    return X, (np.tanh(X @ A) @ b).astype(DT)


def gradient_check():
    """Central finite differences vs backprop, in float64."""
    rng = np.random.default_rng(0)
    p, _ = init_params(32, "mup", rng, dtype=np.float64)
    X = rng.normal(0, 1, (16, D_IN))
    Y = np.tanh(X[:, :1])
    _, g = loss_and_grads(p, X, Y)
    worst = 0.0
    for idx in range(5):
        flat, gflat = p[idx].ravel(), g[idx].ravel()
        for k in rng.integers(0, flat.size, size=6):
            h, old = 1e-6, flat[k]
            flat[k] = old + h
            lp, _ = loss_and_grads(p, X, Y)
            flat[k] = old - h
            lm, _ = loss_and_grads(p, X, Y)
            flat[k] = old
            num = (lp - lm) / (2 * h)
            worst = max(worst, abs(num - gflat[k]) / (abs(num) + abs(gflat[k]) + 1e-15))
    return worst


def torch_crosscheck():
    """One Adam step: our update rule vs torch.optim.Adam."""
    if torch is None:
        return None
    g = torch.tensor([3.0, -0.5])
    w = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.01)
    w.grad = g.clone()
    opt.step()
    m, v = 0.1 * g.numpy(), 0.001 * g.numpy() ** 2
    ours = -0.01 * (m / 0.1) / (np.sqrt(v / 0.001) + 1e-8)
    return float(np.max(np.abs(ours - w.detach().numpy())))


def sweep(param, X, Y):
    table, argmins = {}, {}
    for n in WIDTHS:
        row = []
        for e in LOG2_LRS:
            tot = 0.0
            for sd in range(SEEDS):
                p, lm = init_params(n, param, np.random.default_rng(sd))
                L = adam_train(p, lm, 2.0 ** e, X, Y, STEPS)
                tot += L if np.isfinite(L) else 1e6
            row.append(tot / SEEDS)
        table[n] = row
        argmins[n] = LOG2_LRS[int(np.argmin(row))]
    return table, argmins


def coord_check(param, X, Y, eta=2.0 ** -7, steps=5):
    """Average coordinate size (RMS) of each activation after a few steps."""
    out = {}
    for n in WIDTHS:
        p, lm = init_params(n, param, np.random.default_rng(7))
        adam_train(p, lm, eta, X, Y, steps)
        h1, h2, f = forward(p, X[:BATCH])
        out[n] = tuple(float(np.sqrt(np.mean(a ** 2))) for a in (h1, h2, f))
    return out


def print_sweep(name, table, argmins):
    print(f"\n  {name}: mean training MSE, {SEEDS} seeds x {STEPS} Adam steps")
    print("  width |" + "".join(f"   2^{e:<3d}" for e in LOG2_LRS) + "  | argmin")
    for n in WIDTHS:
        best = int(np.argmin(table[n]))
        cells = "".join(f"{v:7.4f}" + ("*" if i == best else " ")
                        for i, v in enumerate(table[n]))
        print(f"  {n:5d} |{cells} | 2^{argmins[n]}")


def print_coord(name, c):
    print(f"\n  {name} coordinate check: activation RMS after 5 steps, eta = 2^-7")
    print("  width |      h1      h2   f(out)")
    for n in WIDTHS:
        print(f"  {n:5d} | {c[n][0]:7.4f} {c[n][1]:7.4f} {c[n][2]:8.4f}")


if __name__ == "__main__":
    res = gradient_check()
    print(f"\n  backprop vs central differences, worst rel. error : {res:.3e}")
    assert res < 1e-7, res
    tc = torch_crosscheck()
    print("  our Adam step vs torch.optim.Adam, max abs diff    : "
          + (f"{tc:.3e}" if tc is not None else "[skipped: torch not installed]"))
    if tc is not None:
        assert tc < 1e-8, tc

    X, Y = make_data()
    sp_t, sp_a = sweep("sp", X, Y)
    mu_t, mu_a = sweep("mup", X, Y)
    print_sweep("STANDARD PARAMETERIZATION (SP)", sp_t, sp_a)
    print_sweep("MAXIMAL UPDATE PARAMETERIZATION (muP)", mu_t, mu_a)

    mu_spread = max(mu_a.values()) - min(mu_a.values())
    sp_spread = max(sp_a.values()) - min(sp_a.values())
    print(f"\n  argmin log2(eta) spread over widths {WIDTHS[0]}-{WIDTHS[-1]}:"
          f"  muP = {mu_spread} octaves, SP = {sp_spread} octaves")
    assert mu_spread == 0, f"muP argmin LR moved by 2^{mu_spread}"
    assert sp_spread >= 2, f"SP argmin LR moved only 2^{sp_spread} (< 4x)"

    c_sp, c_mu = coord_check("sp", X, Y), coord_check("mup", X, Y)
    print_coord("SP", c_sp)
    print_coord("muP", c_mu)
    ratio = lambda c: [max(c[n][k] for n in WIDTHS) / min(c[n][k] for n in WIDTHS)
                       for k in range(3)]
    r_mu, r_sp = ratio(c_mu), ratio(c_sp)
    print(f"\n  max/min RMS across widths   muP : h1={r_mu[0]:.2f}  h2={r_mu[1]:.2f}"
          f"  f={r_mu[2]:.2f}")
    print(f"                               SP : h1={r_sp[0]:.2f}  h2={r_sp[1]:.2f}"
          f"  f={r_sp[2]:.2f}")
    assert max(r_mu) < 3.0, r_mu
    assert max(r_sp) > 4.0, r_sp
    print("\n  all assertions passed.")
