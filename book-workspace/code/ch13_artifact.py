"""
Artifact 13.1 -- Optimizers from scratch: SGD, momentum, Adam, AdamW, Lion, Muon.

Core is pure NumPy. torch is used ONLY as a cross-check and is guarded.
Every optimizer is driven by the SAME analytic gradient so the comparison
isolates the update rule (no autograd noise, no data-order noise).

Problem: multiclass softmax regression, one 2D weight matrix W (16 x 12),
so that Muon -- which requires a 2D parameter -- applies unchanged.
"""

import math
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

RNG = np.random.default_rng(0)
D_IN, D_OUT, N = 16, 12, 256

# == Problem: softmax regression. loss(W) and grad(W) in closed form.
X = RNG.normal(size=(N, D_IN)) * np.linspace(0.2, 4.0, D_IN)   # ill-conditioned
Y = RNG.integers(0, D_OUT, size=N)
ONEHOT = np.eye(D_OUT)[Y]
W0 = RNG.normal(size=(D_IN, D_OUT)) * 0.3

def loss_and_grad(W):
    z = X @ W                                   # (N, D_OUT)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(p[np.arange(N), Y] + 1e-30))
    g = X.T @ (p - ONEHOT) / N                  # (D_IN, D_OUT)
    return loss, g

class SGD:
    """torch.optim.SGD. buf <- mu*buf + g (buf starts AT g, not 0)."""
    slots = 1  # momentum buffer (0 if momentum == 0)

    def __init__(self, lr, momentum=0.0, nesterov=False, weight_decay=0.0):
        self.lr, self.mu, self.nesterov, self.wd = lr, momentum, nesterov, weight_decay
        self.buf = None

    def step(self, W, g):
        if self.wd:
            g = g + self.wd * W                 # L2, folded into the gradient
        if self.mu:
            self.buf = g.copy() if self.buf is None else self.mu * self.buf + g
            d = g + self.mu * self.buf if self.nesterov else self.buf
        else:
            d = g
        return W - self.lr * d

class Adam:
    """torch.optim.Adam (decoupled=False => L2) and AdamW (decoupled=True)."""
    slots = 2

    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0,
                 decoupled=False):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, weight_decay
        self.decoupled = decoupled
        self.m = self.v = None
        self.t = 0

    def step(self, W, g):
        if self.m is None:
            self.m, self.v = np.zeros_like(W), np.zeros_like(W)
        if self.wd and self.decoupled:
            W = W * (1 - self.lr * self.wd)     # AdamW: shrink W, leave moments alone
        elif self.wd:
            g = g + self.wd * W                 # Adam+L2: the coupling AdamW removes
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        denom = np.sqrt(self.v) / math.sqrt(bc2) + self.eps   # torch's eps placement
        return W - (self.lr / bc1) * self.m / denom

def AdamW(lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01):
    """torch.optim.AdamW: identical to Adam except the decay path."""
    return Adam(lr, b1, b2, eps, weight_decay, decoupled=True)

class Lion:
    """Chen et al. 2023. One buffer; update is exactly +-1 elementwise."""
    slots = 1

    def __init__(self, lr, b1=0.9, b2=0.99, weight_decay=0.0):
        self.lr, self.b1, self.b2, self.wd = lr, b1, b2, weight_decay
        self.m = None

    def step(self, W, g):
        if self.m is None:
            self.m = np.zeros_like(W)
        c = self.b1 * self.m + (1 - self.b1) * g     # interpolated, NOT the buffer
        self.m = self.b2 * self.m + (1 - self.b2) * g
        return W - self.lr * (np.sign(c) + self.wd * W)

def newton_schulz(G, steps, coeffs, eps=1e-7, record=None):
    """Orthogonalize G (2D) by polynomial iteration on X <- a X + b (XX^T) X + c (XX^T)^2 X.

    Normalizing by ||G||_F puts every singular value in (0, 1]; the polynomial
    p(s) = a s + b s^3 + c s^5 pushes each toward 1 without ever forming an SVD.
    Transposing tall matrices keeps the Gram matrix the smaller of the two.
    """
    a, b, c = coeffs
    Xm = G.astype(np.float64)
    transposed = Xm.shape[0] > Xm.shape[1]
    if transposed:
        Xm = Xm.T
    Xm = Xm / max(np.linalg.norm(Xm), eps)
    for i in range(steps):
        A = Xm @ Xm.T
        B = b * A + c * (A @ A)
        Xm = a * Xm + B @ Xm
        if record is not None:
            k = Xm.shape[0]
            resid = np.linalg.norm(Xm @ Xm.T - np.eye(k)) / math.sqrt(k)
            record.append((i + 1, resid))
    return Xm.T if transposed else Xm

class Muon:
    """torch.optim.Muon (2D params only). Momentum -> orthogonalize -> decoupled decay."""
    slots = 1

    def __init__(self, lr, momentum=0.95, nesterov=True, weight_decay=0.1,
                 ns_steps=5, coeffs=(3.4445, -4.7750, 2.0315), adjust="original"):
        self.lr, self.mu, self.nesterov, self.wd = lr, momentum, nesterov, weight_decay
        self.ns_steps, self.coeffs, self.adjust = ns_steps, coeffs, adjust
        self.buf = None

    def _adjusted_lr(self, shape):
        A, B = shape[:2]
        if self.adjust == "original":
            return self.lr * math.sqrt(max(1.0, A / B))
        if self.adjust == "match_rms_adamw":
            return self.lr * 0.2 * math.sqrt(max(A, B))
        return self.lr

    def step(self, W, g):
        if self.buf is None:
            self.buf = np.zeros_like(W)
        self.buf = self.buf + (1 - self.mu) * (g - self.buf)       # torch lerp_
        upd = g + self.mu * (self.buf - g) if self.nesterov else self.buf
        O = newton_schulz(upd, self.ns_steps, self.coeffs)
        W = W * (1 - self.lr * self.wd)                            # decoupled decay
        return W - self._adjusted_lr(W.shape) * O

# == Harness
def run_numpy(opt, steps):
    W = W0.copy()
    traj = []
    for _ in range(steps):
        _, g = loss_and_grad(W)
        W = opt.step(W, g)
        traj.append(W.copy())
    return np.array(traj)

def run_torch(make_opt, steps):
    W = torch.tensor(W0.copy(), dtype=torch.float64, requires_grad=False)
    W = torch.nn.Parameter(W)
    opt = make_opt([W])
    traj = []
    for _ in range(steps):
        _, g = loss_and_grad(W.detach().numpy())
        W.grad = torch.tensor(g, dtype=torch.float64)
        opt.step()
        traj.append(W.detach().numpy().copy())
    return np.array(traj)

def compare(name, np_opt, torch_factory, steps=200, tol=1e-6):
    tr = run_numpy(np_opt, steps)
    if torch is None or torch_factory is None:
        print(f"  {name:<22s} [skipped: no torch reference]  final |W|_F = {np.linalg.norm(tr[-1]):.6f}")
        return
    tt = run_torch(torch_factory, steps)
    dev = np.abs(tr - tt).max()
    print(f"  {name:<22s} max|numpy-torch| over {steps} steps = {dev:.3e}   (tol {tol:.0e})  "
          f"{'PASS' if dev < tol else 'FAIL'}")
    assert dev < tol, f"{name} diverged from torch: {dev}"

if __name__ == "__main__":
    print("=" * 78)
    print("1. STEP-FOR-STEP PARITY WITH torch.optim (float64, identical gradients)")
    print("=" * 78)
    T = torch is not None
    print(f"  torch available: {T}" + (f" (version {torch.__version__})" if T else ""))
    O = torch.optim if T else None
    for name, mine, ref in [
        ("SGD (plain)", SGD(0.5), lambda p: O.SGD(p, lr=0.5)),
        ("SGD + momentum 0.9", SGD(0.5, momentum=0.9), lambda p: O.SGD(p, lr=0.5, momentum=0.9)),
        ("SGD + Nesterov", SGD(0.5, 0.9, True), lambda p: O.SGD(p, lr=0.5, momentum=0.9, nesterov=True)),
        ("Adam", Adam(1e-2), lambda p: O.Adam(p, lr=1e-2)),
        ("Adam (L2 wd=0.1)", Adam(1e-2, weight_decay=0.1), lambda p: O.Adam(p, lr=1e-2, weight_decay=0.1)),
        ("AdamW (wd=0.1)", AdamW(1e-2, weight_decay=0.1), lambda p: O.AdamW(p, lr=1e-2, weight_decay=0.1)),
        ("Adam beta2=0.95", Adam(1e-2, b2=0.95), lambda p: O.Adam(p, lr=1e-2, betas=(0.9, 0.95))),
    ]:
        compare(name, mine, ref if T else None)

    has_lion = T and hasattr(torch.optim, "Lion")
    compare("Lion", Lion(1e-3, weight_decay=0.1),
            (lambda p: torch.optim.Lion(p, lr=1e-3, weight_decay=0.1)) if has_lion else None)
    if not has_lion:
        print("    (torch.optim.Lion absent in this build -- NumPy Lion checked below instead)")
        lt = run_numpy(Lion(1e-3, weight_decay=0.1), 50)
        d = np.abs(lt[1] - lt[0]) / 1e-3
        # Lion's raw update magnitude is sign(.) +- decay, so |dW|/lr is ~1 everywhere.
        print(f"    Lion |dW|/lr in [{d.min():.4f}, {d.max():.4f}]  (sign step => ~1.0)")
        assert 0.9 < d.min() and d.max() < 1.1

    has_muon = T and hasattr(torch.optim, "Muon")
    print(f"  torch.optim.Muon present: {has_muon}")
    if has_muon:
        # torch runs the NS iteration in bfloat16 by design; parity is bf16-limited.
        tr = run_numpy(Muon(2e-2, weight_decay=0.1), 100)
        tt = run_torch(lambda p: torch.optim.Muon(p, lr=2e-2, weight_decay=0.1), 100)
        dev = np.abs(tr - tt).max()
        rel = dev / np.abs(tt[-1]).max()
        print(f"  {'Muon vs torch.optim.Muon':<22s} max abs dev over 100 steps = {dev:.3e}"
              f"  rel = {rel:.3e}  (torch casts NS to bfloat16; 1e-6 is unreachable)")
        assert rel < 5e-2, rel

    print()
    print("=" * 78)
    print("2. NEWTON-SCHULZ ORTHOGONALITY RESIDUAL  ||XX^T - I||_F / sqrt(k)")
    print("=" * 78)
    G = RNG.normal(size=(D_IN, D_OUT))
    for label, coeffs, steps in [
        ("convergent cubic (1.5,-0.5,0)", (1.5, -0.5, 0.0), 12),
        ("Muon quintic (3.4445,-4.775,2.0315)", (3.4445, -4.7750, 2.0315), 8),
    ]:
        rec = []
        newton_schulz(G, steps, coeffs, record=rec)
        print(f"  {label}")
        print("    " + "  ".join(f"it{i}={r:.2e}" for i, r in rec))
    rec = []
    newton_schulz(G, 30, (1.5, -0.5, 0.0), record=rec)
    print(f"  cubic residual after 30 iters = {rec[-1][1]:.3e}  (drives to zero)")
    assert rec[-1][1] < 1e-9, rec[-1][1]

    print()
    print("=" * 78)
    print("3. Adam+L2 vs AdamW REACH DIFFERENT WEIGHT NORMS (same lr, same wd, same data)")
    print("=" * 78)
    S = 600
    w_l2 = run_numpy(Adam(1e-2, weight_decay=0.1), S)[-1]
    w_aw = run_numpy(AdamW(1e-2, weight_decay=0.1), S)[-1]
    n_l2, n_aw = np.linalg.norm(w_l2), np.linalg.norm(w_aw)
    print(f"  Adam + L2 (wd=0.1) : |W|_F = {n_l2:.6f}   loss = {loss_and_grad(w_l2)[0]:.6f}")
    print(f"  AdamW     (wd=0.1) : |W|_F = {n_aw:.6f}   loss = {loss_and_grad(w_aw)[0]:.6f}")
    print(f"  ratio |W|_L2 / |W|_AdamW = {n_l2 / n_aw:.4f}")
    assert abs(n_l2 - n_aw) / max(n_l2, n_aw) > 0.05, "norms must differ measurably"

    print()
    print("=" * 78)
    print("4. OPTIMIZER STATE MEMORY (multiplier x parameter bytes; fp32 master weights)")
    print("=" * 78)
    Nparam = 1_000_000_000
    print(f"  {'method':<18s} {'x params':>8s}  {'bytes @ 1B params':>18s}  note")
    for name, mult, note in [("SGD", 0, "none"),
                             ("SGD + momentum", 1, "momentum buffer"),
                             ("Adam / AdamW", 2, "first + second moment"),
                             ("Lion", 1, "single momentum buffer"),
                             ("Muon (2D params)", 1, "momentum buffer only"),
                             ("Shampoo", 2, "L,R preconditioners + inverse roots"),
                             ("SOAP", 3, "Shampoo eigenbasis + Adam moments in it")]:
        print(f"  {name:<18s} {mult:>8d}  {mult * 4 * Nparam / 1e9:>15.1f} GB  {note}")
    print("  (Shampoo/SOAP multipliers are order-of-magnitude and shape-dependent:")
    print("   for an m x n matrix Shampoo's L,R cost m^2+n^2, i.e. (m^2+n^2)/(mn) x params.")
    print("   Blocking into b x b blocks pins that ratio at 2 but does NOT shrink the")
    print("   state; what it cuts is the eigendecomposition, O(m^3+n^3) -> O(mnb).)")
    print()
    print("all assertions passed")
