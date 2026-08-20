"""
ex10 — An MLP and a training loop, end to end.  (Book: Chapters 10 and 11)

Everything before this file was a piece. This one assembles them: a reverse-mode
engine, a two-hidden-layer tanh MLP, cross-entropy from logits, and full-batch
gradient descent — then trains the thing on two interleaved spiral arms, where a
linear model is not merely weak but provably capped. A scan over every direction
and every threshold puts the best halfplane at 71.25% on this data; the MLP
reaches 99.75% from the same loop, and logistic regression trained by that same
loop lands at 70.00%, just under the ceiling where it belongs.

The insight the file is built to make undeniable is that a training loop is
THREE independent claims, and they fail in three different ways.

The gradient is right. That is checked against central differences on every one
of the 1185 parameters (worst disagreement 1.06e-11) and against torch autograd
on an identically initialized net (6.7e-17, a fraction of one ulp).

The step is right. Forgetting `zero_grad()` does not make the gradient wrong —
.grad becomes exactly the SUM of the per-step gradients, verified here
bit-for-bit. What it destroys is the dynamics. Near a minimum with a = eta*L,
correct GD contracts the top mode by (1-a) each step, while the accumulating
version obeys e_{t+1} = (2-a) e_t - e_{t-1}, whose two roots multiply to 1. For
every 0 < a < 4 they are a complex pair of modulus EXACTLY 1 — measured over 160
steps as 1.000000, to within 3e-5. The bug does not merely slow training: the error
rotates forever and never decays, at ANY learning rate, however small.

The step size is admissible. For a QUADRATIC — equivalently, inside the local
quadratic model at a fixed minimum, which is exactly what the last section
measures — GD converges iff eta < 2/L with L the top Hessian eigenvalue, and
there is no grey zone. The file builds the full Hessian of a 17-parameter net at
a genuine minimum (L = 0.5262, so 2/L = 3.8008) and watches eta*L = 1.9999
contract while 2.0001 grows, each at the predicted rate |1 - eta*L| to four
digits. That "iff" stops being a theorem the moment the loss stops being
quadratic: on a real net L is a function of where you are, so eta*L moves as you
train. The MLP here trains at lr = 1.0 to 99.75% while its own eta*L wanders
from 1.75 at step 50 up past 3.86 at step 110 and back to 2.10 at step 150 —
above its own stated threshold for most of that stretch, still converging. That
is the edge of stability, and it is exactly the grey zone the quadratic
statement forbids.

The third claim is why the first is not enough. Along that same lr = 1.0 run our
trajectory and torch's — identical to 3.7e-16 after fifty steps, and 6.7e-17
after one — sit ~5.7e-3 apart by step 150: eta*L crosses 2 near step 60, and
from step 80 to 120 the accumulated rounding grows about 2.09x per step, inside
the |1 - eta*L| band of [1.05, 2.87] measured over exactly those steps. (The
step-150 figure depends on BLAS rounding; the mechanism does not.) Correct
gradients do not buy a reproducible trajectory.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402

try:
    import torch
    HAVE_TORCH = True
except ImportError:                                     # pragma: no cover
    HAVE_TORCH = False


# ---------------------------------------------------------------------------
# A compact reverse-mode engine. Self-contained on purpose: every file in this
# ladder stands alone, so nothing here is imported from ex06/ex07.
# ---------------------------------------------------------------------------

def unbroadcast(grad, shape):
    """Undo numpy broadcasting: sum away prepended axes, sum-and-keep stretched ones."""
    # === YOUR CODE HERE ===
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, size in enumerate(shape):
        if size == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A node in the computation graph: data, an accumulated gradient, and a
    closure that pushes that gradient to its inputs."""

    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            # === YOUR CODE HERE ===
            self.grad = self.grad + unbroadcast(out.grad, self.data.shape)
            other.grad = other.grad + unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            # === YOUR CODE HERE ===
            self.grad = self.grad + unbroadcast(other.data * out.grad, self.data.shape)
            other.grad = other.grad + unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            # Z = A B  =>  Abar = Zbar B^T,  Bbar = A^T Zbar.
            # === YOUR CODE HERE ===
            self.grad = self.grad + out.grad @ other.data.T
            other.grad = other.grad + self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), "sum")

        def _backward():
            # The adjoint of a sum is a broadcast.
            # === YOUR CODE HERE ===
            self.grad = self.grad + np.broadcast_to(out.grad, self.data.shape).copy()

        out._backward = _backward
        return out

    def tanh(self):
        """Smooth activation, chosen deliberately: the loss is then twice
        differentiable everywhere, which is what makes the Hessian (and the
        2/L stability threshold measured at the end of this file) exist."""
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            # === YOUR CODE HERE ===
            self.grad = self.grad + (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def softplus(self):
        """log(1 + exp(z)), evaluated in the shifted form that never overflows:
            max(z,0) + log1p(exp(-|z|)).
        Its derivative is the logistic sigmoid — which is why the whole binary
        cross-entropy gradient collapses to (sigma(z) - y)."""
        z = self.data
        out = Tensor(np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))), (self,), "softplus")

        def _backward():
            # === YOUR CODE HERE ===
            sig = np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                           np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))
            self.grad = self.grad + sig * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Iterative topological sort, then one reverse sweep."""
        topo, visited, stack = [], set(), [(self, False)]
        while stack:
            v, expanded = stack.pop()
            if expanded:
                topo.append(v)
                continue
            if id(v) in visited:
                continue
            visited.add(id(v))
            stack.append((v, True))
            for child in v._prev:
                if id(child) not in visited:
                    stack.append((child, False))
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def __neg__(self): return self * -1.0
    def __sub__(self, o): return self + (-(o if isinstance(o, Tensor) else Tensor(o)))
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def __repr__(self): return f"Tensor(shape={self.data.shape})"


# ---------------------------------------------------------------------------
# The model, the loss, and the training loop.
# ---------------------------------------------------------------------------

class MLP:
    """dims = [2, 32, 32, 1] means two hidden tanh layers and a linear head.

    Initialization is LeCun/Xavier-style, sd = 1/sqrt(fan_in), which is the
    right scale for tanh: it keeps the pre-activations inside the linear part
    of the nonlinearity at step 0 (ex08 is the exercise about why).
    """

    def __init__(self, dims, seed=0):
        rng = np.random.default_rng(seed)
        self.dims = list(dims)
        self.W, self.b = [], []
        # === YOUR CODE HERE ===
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            self.W.append(Tensor(rng.standard_normal((fan_in, fan_out)) / np.sqrt(fan_in)))
            self.b.append(Tensor(np.zeros((1, fan_out))))

    @property
    def params(self):
        """Every parameter tensor, in a fixed order: W0, b0, W1, b1, ..."""
        return [t for pair in zip(self.W, self.b) for t in pair]

    def names(self):
        return [n for i in range(len(self.W)) for n in (f"W{i}{self.W[i].shape}",
                                                       f"b{i}{self.b[i].shape}")]

    def __call__(self, x):
        """Forward pass through the autodiff graph. Returns logits, not probabilities."""
        # === YOUR CODE HERE ===
        h = x
        for i in range(len(self.W) - 1):
            h = (h @ self.W[i] + self.b[i]).tanh()
        return h @ self.W[-1] + self.b[-1]

    def forward_numpy(self, X):
        """The same function written in plain numpy, with no graph. Used to
        cross-check the engine's forward pass and to drive finite differences,
        so the gradient check does not lean on the code it is checking."""
        # === YOUR CODE HERE ===
        h = X
        for i in range(len(self.W) - 1):
            h = np.tanh(h @ self.W[i].data + self.b[i].data)
        return h @ self.W[-1].data + self.b[-1].data


def bce_with_logits(logits, y):
    """Mean binary cross-entropy from logits, built out of engine ops.

        BCE(z, y) = log(1 + e^z) - y z = softplus(z) - y z

    so it needs exactly one nonlinear primitive, and gradients flow through the
    same graph that produced z.
    """
    n = float(logits.data.size)
    # === YOUR CODE HERE ===
    return (logits.softplus() - Tensor(y) * logits).sum() * (1.0 / n)


def loss_numpy(model, X, y, lam=0.0):
    """Graph-free loss, for finite differences. Must agree with the graph one."""
    z = model.forward_numpy(X)
    bce = float(np.mean(np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))) - y * z))
    l2 = 0.5 * lam * float(sum(np.sum(p.data ** 2) for p in model.params))
    return bce + l2


def train(model, X, y, lr, steps, lam=0.0, zero_grad=True, record_every=0):
    """Full-batch gradient descent. Returns (loss history, final accuracy).

    `zero_grad=False` reproduces the single most common bug in hand-written
    training loops: gradients accumulate across steps instead of being reset.
    """
    Xt = Tensor(X)
    hist = []
    # === YOUR CODE HERE ===
    for step in range(steps):
        if zero_grad:
            for p in model.params:
                p.zero_grad()
        loss = bce_with_logits(model(Xt), y)
        if lam:
            reg = (model.params[0] * model.params[0]).sum()
            for p in model.params[1:]:
                reg = reg + (p * p).sum()
            loss = loss + reg * (0.5 * lam)
        loss.backward()
        for p in model.params:
            p.data = p.data - lr * p.grad
        if record_every and (step % record_every == 0 or step == steps - 1):
            hist.append((step, float(loss.data)))
    acc = accuracy(model, X, y)
    return hist, acc


def accuracy(model, X, y):
    """Fraction correct with the decision boundary at logit 0."""
    # === YOUR CODE HERE ===
    return float(((model.forward_numpy(X) > 0).astype(float) == y).mean())


# ---------------------------------------------------------------------------
# The data, and the exact ceiling on what a linear model can do with it.
# ---------------------------------------------------------------------------

def make_spiral(n_per=200, turns=2.0, noise=0.06, seed=0):
    """Two interleaved Archimedean spiral arms, 180 degrees out of phase.

    Every straight line through this cloud cuts both arms many times, so no
    halfplane can separate the classes — and unlike "two moons" the ceiling is
    close to chance rather than merely low.
    """
    rng = np.random.default_rng(seed)
    # === YOUR CODE HERE ===
    arms = []
    for phase in (0.0, np.pi):
        t = np.sqrt(rng.uniform(0.05, 1.0, n_per)) * turns * 2 * np.pi
        r = t / (turns * 2 * np.pi)
        arms.append(np.column_stack([r * np.cos(t + phase), r * np.sin(t + phase)]))
    X = np.vstack(arms) + noise * rng.standard_normal((2 * n_per, 2))
    y = np.concatenate([np.zeros(n_per), np.ones(n_per)]).reshape(-1, 1)
    return X, y


def best_linear_accuracy(X, y, n_angles=3600):
    """The best accuracy ANY halfplane classifier can reach on this 2-D data.

    Not a trained model — a scan. For each direction w on a grid of angles,
    project, sort, and take the best of all n+1 thresholds in closed form:

        correct(k) = #{i >= k : y=1} + #{i < k : y=0} = n1 + k - 2*cumsum(y)[k]

    and the complement n - correct(k) covers the flipped orientation. So the
    only approximation is the angular grid, and the printed refinement check
    shows the answer has stopped moving well before the grid we use.
    """
    yy = y.ravel()
    n, n1 = len(yy), yy.sum()
    ang = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    # === YOUR CODE HERE ===
    W = np.column_stack([np.cos(ang), np.sin(ang)])          # (A, 2)
    P = X @ W.T                                              # (n, A)
    ys = np.take_along_axis(np.broadcast_to(yy[:, None], P.shape),
                            np.argsort(P, axis=0), axis=0)   # labels in projection order
    S = np.vstack([np.zeros((1, n_angles)), np.cumsum(ys, axis=0)])   # (n+1, A)
    correct = n1 + np.arange(n + 1)[:, None] - 2.0 * S
    return float(np.maximum(correct, n - correct).max()) / n


# ---------------------------------------------------------------------------
# Finite differences, flat parameter vectors, and the Hessian.
# ---------------------------------------------------------------------------

def numeric_grads(f, params, eps=1e-5):
    """Central differences over every entry of every parameter tensor."""
    grads = []
    for t in params:
        g = np.zeros_like(t.data)
        flat, gflat = t.data.reshape(-1), g.reshape(-1)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps; up = f()
            flat[i] = orig - eps; dn = f()
            flat[i] = orig
            gflat[i] = (up - dn) / (2 * eps)
        grads.append(g)
    return grads


def get_flat(model):
    return np.concatenate([p.data.ravel() for p in model.params])


def set_flat(model, v):
    i = 0
    for p in model.params:
        k = p.data.size
        p.data = v[i:i + k].reshape(p.data.shape).copy()
        i += k


def grad_flat(model, X, y, lam):
    """Analytic gradient of (BCE + L2) as one flat vector, via the engine."""
    for p in model.params:
        p.zero_grad()
    loss = bce_with_logits(model(Tensor(X)), y)
    reg = (model.params[0] * model.params[0]).sum()
    for p in model.params[1:]:
        reg = reg + (p * p).sum()
    (loss + reg * (0.5 * lam)).backward()
    return float(loss.data) + 0.5 * lam * float(get_flat(model) @ get_flat(model)), \
        np.concatenate([p.grad.ravel() for p in model.params])


def top_curvature(model, X, y, lam=0.0, iters=40, eps=1e-6, seed=0):
    """Largest-magnitude Hessian eigenvalue by power iteration on Hessian-vector
    products, which are just central differences of the analytic gradient:

        H v ~ (grad(w + eps v) - grad(w - eps v)) / (2 eps)

    Never forms H, so it works on the 1185-parameter net where the full matrix
    would be a 1185 x 1185 finite-difference job.
    """
    v0 = get_flat(model)
    rng_ = np.random.default_rng(seed)
    v = rng_.standard_normal(v0.size)
    v /= np.linalg.norm(v)
    lam_max = 0.0
    # === YOUR CODE HERE ===
    for _ in range(iters):
        set_flat(model, v0 + eps * v); _, gp = grad_flat(model, X, y, lam)
        set_flat(model, v0 - eps * v); _, gm = grad_flat(model, X, y, lam)
        Hv = (gp - gm) / (2 * eps)
        lam_max = float(v @ Hv)
        v = Hv / np.linalg.norm(Hv)
    set_flat(model, v0)
    return lam_max


def hessian(model, X, y, lam, eps=1e-5):
    """Full Hessian by central differences OF THE ANALYTIC GRADIENT.

    Differencing the gradient (not the loss) costs 2P backward passes instead
    of P^2 forward passes and loses only one factor of eps in accuracy.
    """
    v0 = get_flat(model)
    P = v0.size
    H = np.zeros((P, P))
    # === YOUR CODE HERE ===
    for i in range(P):
        v = v0.copy(); v[i] += eps; set_flat(model, v)
        _, gp = grad_flat(model, X, y, lam)
        v = v0.copy(); v[i] -= eps; set_flat(model, v)
        _, gm = grad_flat(model, X, y, lam)
        H[:, i] = (gp - gm) / (2 * eps)
    set_flat(model, v0)
    return 0.5 * (H + H.T)


if __name__ == "__main__":
    t_start = time.perf_counter()
    rng = np.random.default_rng(0)

    X, y = make_spiral(n_per=200, turns=2.0, noise=0.06, seed=0)
    print(f"\n      data: two spiral arms, {X.shape[0]} points, 2 turns each")

    # =======================================================================
    print("\n--- the engine: every parameter tensor against finite differences ---")
    # =======================================================================
    net = MLP([2, 32, 32, 1], seed=1)
    Xs, ys = X[:16], y[:16]

    # First: the graph forward and the graph-free forward must be the same function.
    z_graph = net(Tensor(Xs)).data
    z_numpy = net.forward_numpy(Xs)
    check("graph forward == graph-free forward", z_graph, z_numpy, tol=1e-14)

    for p in net.params:
        p.zero_grad()
    bce_with_logits(net(Tensor(Xs)), ys).backward()
    fd = numeric_grads(lambda: loss_numpy(net, Xs, ys), net.params, eps=1e-5)
    worst = 0.0
    for name, p, g in zip(net.names(), net.params, fd):
        worst = max(worst, float(np.abs(p.grad - g).max()))
        check(f"d loss / d {name}", p.grad, g, tol=1e-9)
    print(f"      worst disagreement over all {sum(p.data.size for p in net.params)} "
          f"parameters: {worst:.2e}")

    # =======================================================================
    print("\n--- what a linear model can do here, at best ---")
    # =======================================================================
    ceiling = {A: best_linear_accuracy(X, y, A) for A in (900, 3600, 14400)}
    for A, v in ceiling.items():
        print(f"      best halfplane over {A:6d} directions: {v*100:.2f}%")
    best_lin = ceiling[14400]
    check("the halfplane scan has converged in the angular grid",
          ceiling[900] == ceiling[3600] == ceiling[14400])

    lin = MLP([2, 1], seed=2)
    _, lin_acc = train(lin, X, y, lr=1.0, steps=3000)
    print(f"      logistic regression trained by the same loop: {lin_acc*100:.2f}%")
    # A trained linear model cannot beat the optimum over all linear models.
    check("the trained linear model does not exceed the scanned optimum",
          lin_acc <= best_lin + 1e-12)
    check("and it gets within 2 points of it", best_lin - lin_acc < 0.02)

    # Control: the SAME model and the SAME loop on data that IS separable. This
    # is the computed floor — it proves the loop is not what is failing above.
    blob = rng.standard_normal((400, 2)) * 0.6
    ysep = (blob[:, [0]] + blob[:, [1]] > 0).astype(float)
    blob = blob + 2.0 * ysep * np.array([[1.0, 1.0]])
    lin2 = MLP([2, 1], seed=3)
    _, sep_acc = train(lin2, blob, ysep, lr=1.0, steps=3000)
    print(f"      the same linear model on separable blobs:    {sep_acc*100:.2f}%")
    check("the linear trainer itself works (separable control)", sep_acc == 1.0)

    # =======================================================================
    print("\n--- the MLP training loop ---")
    # =======================================================================
    net = MLP([2, 32, 32, 1], seed=1)
    t0 = time.perf_counter()
    hist, mlp_acc = train(net, X, y, lr=1.0, steps=5000, record_every=1000)
    t_train = time.perf_counter() - t0
    for step, loss in hist:
        print(f"      step {step:5d}   loss {loss:.4f}")
    print(f"      final accuracy {mlp_acc*100:.2f}%   ({t_train:.1f}s for 5000 full-batch steps)")
    check("the loss fell by more than an order of magnitude", hist[-1][1] < hist[0][1] / 10)
    check("the MLP separates the spiral", mlp_acc > 0.99)
    print(f"      MLP {mlp_acc*100:.2f}%  vs  best-possible linear {best_lin*100:.2f}%")
    check("the MLP beats every linear model by a wide margin", mlp_acc - best_lin > 0.20)

    # =======================================================================
    print("\n--- the same net, the same step, in torch ---")
    # =======================================================================
    if HAVE_TORCH:
        ref = MLP([2, 32, 32, 1], seed=1)
        tnet = torch.nn.Sequential(
            torch.nn.Linear(2, 32).double(), torch.nn.Tanh(),
            torch.nn.Linear(32, 32).double(), torch.nn.Tanh(),
            torch.nn.Linear(32, 1).double(),
        )
        layers = [m for m in tnet if isinstance(m, torch.nn.Linear)]
        with torch.no_grad():
            for m, W, b in zip(layers, ref.W, ref.b):
                m.weight.copy_(torch.tensor(W.data.T))    # torch stores (out, in)
                m.bias.copy_(torch.tensor(b.data.ravel()))
        Xt_, yt_ = torch.tensor(X), torch.tensor(y)
        lossfn = torch.nn.BCEWithLogitsLoss(reduction="mean")

        l_t = lossfn(tnet(Xt_), yt_)
        tnet.zero_grad()
        l_t.backward()
        for p in ref.params:
            p.zero_grad()
        l_o = bce_with_logits(ref(Tensor(X)), y)
        l_o.backward()

        check("loss value matches torch", float(l_o.data), float(l_t.detach()), tol=1e-15)
        gmax = 0.0
        for name, m, gW, gb in zip(("W0/b0", "W1/b1", "W2/b2"), layers, ref.W, ref.b):
            dW = float(np.abs(gW.grad - m.weight.grad.numpy().T).max())
            db = float(np.abs(gb.grad.ravel() - m.bias.grad.numpy()).max())
            gmax = max(gmax, dW, db)
            check(f"gradient of {name} matches torch", max(dW, db), 0.0, tol=1e-15)
        print(f"      worst gradient disagreement on one step: {gmax:.2e}  "
              f"(machine epsilon is {np.finfo(np.float64).eps:.1e})")

        # Now step both, in lockstep, and watch the two trajectories.
        LR = 1.0

        def param_gap():
            return max(max(float(np.abs(W.data - m.weight.detach().numpy().T).max()),
                           float(np.abs(b.data.ravel() - m.bias.detach().numpy()).max()))
                       for m, W, b in zip(layers, ref.W, ref.b))

        # eta*L is NOT a constant of the run, so measure it ALONG the way rather
        # than once at the end. GROW is the window over which the gap actually
        # grows geometrically; the endpoints are read off the gap table below.
        MARKS = (1, 50, 60, 80, 90, 100, 110, 120, 150)
        GROW = (80, 120)
        gap, curv = {}, {}
        for k in range(1, 151):
            tnet.zero_grad()
            lossfn(tnet(Xt_), yt_).backward()
            with torch.no_grad():
                for m in layers:
                    m.weight -= LR * m.weight.grad
                    m.bias -= LR * m.bias.grad
            train(ref, X, y, lr=LR, steps=1)
            if k in MARKS:
                gap[k] = param_gap()
                curv[k] = top_curvature(ref, X, y)          # restores the params
        for k in MARKS:
            print(f"      after {k:3d} identical steps: gap {gap[k]:.3e}   "
                  f"eta*L {LR * curv[k]:.4f}")
        check("one SGD step reproduces torch to machine precision", gap[1] < 1e-15)
        check("and 50 steps still do", gap[50] < 1e-14)

        # The gap explodes by ~13 orders between step 50 and step 150. That is
        # NOT a wrong gradient — a wrong gradient shows up as drift from step 1.
        # It is the loop amplifying its own rounding, and the mechanism is
        # measurable: the top curvature times the learning rate crosses 2, so
        # the top mode expands by |1 - eta*L| per step. But eta*L is not fixed:
        # it is 1.75 at step 50, crosses 2 near step 60, peaks near 3.87 at step
        # 110, then drops back near 2.05. So there is no single rate to quote —
        # the honest claim is that the OBSERVED per-step growth over the window
        # where the gap is actually growing lies inside the |1 - eta*L| range
        # measured over that same window. That band is a derived bound, not a
        # guessed threshold: both ends come from top_curvature above.
        lo, hi = GROW
        rate = (gap[hi] / gap[lo]) ** (1.0 / (hi - lo))
        band = [abs(1 - LR * curv[k]) for k in MARKS if lo <= k <= hi]
        print(f"      eta*L crosses 2 between step 50 ({LR*curv[50]:.3f}) and step 60 "
              f"({LR*curv[60]:.3f}), and is still {LR*curv[150]:.3f} at step 150")
        print(f"      the gap is still at rounding level at step {lo} ({gap[lo]:.3e}) "
              f"even though eta*L passed 2 twenty steps earlier: 1e-16 takes a while "
              f"to become visible")
        print(f"      -> from step {lo} to {hi} the gap grows {rate:.3f}x per step, "
              f"inside the measured |1 - eta*L| band [{min(band):.3f}, {max(band):.3f}]")
        print(f"      -> total growth between step 50 and 150: {gap[150]/gap[50]:.1e}x")
        check("no drift for 50 steps (so the gradients agree exactly)", gap[50] / gap[1] < 100)
        check("the gap is still pure rounding at step 80", gap[80] < 1e-14)
        check("then rounding is amplified geometrically", gap[150] / gap[50] > 1e9)
        check("because this run sits past the eta*L = 2 stability edge",
              LR * curv[150] > 2.0)
        # ... but it was NOT past the edge at step 50: eta*L moves along the run,
        # which is the whole difference between a real loss and a quadratic.
        check("and eta*L crossed that edge mid-run, not at the start",
              LR * curv[50] < 2.0 < LR * curv[60])
        check("the observed amplification matches the measured |1 - eta*L| band",
              min(band) <= rate <= max(band))

        # And the sting in the tail. This is the SAME net, the SAME lr = 1.0 that
        # reached 99.75% earlier: eta*L sits above 2 for most of steps 60-150 and
        # the run trains anyway. "GD converges iff eta < 2/L" is a theorem about a
        # FIXED quadratic; on a loss whose curvature moves it is the edge of
        # stability instead, and there very much is a grey zone.
        print(f"      note: eta*L > 2 from step 60 on, yet this same net at this same "
              f"lr reached {mlp_acc*100:.2f}% above — 'converges iff eta < 2/L' is a")
        print(f"      statement about a FIXED quadratic, not about a loss whose "
              f"curvature moves. Section (b) below measures it where it IS exact.")
        check("edge of stability: eta*L is past 2 and the same run trains anyway",
              LR * curv[110] > 2.0 and mlp_acc > 0.99)
    else:                                                # pragma: no cover
        print("  ....  [skipped: torch not installed] torch cross-check")

    # =======================================================================
    # BREAK IT
    # =======================================================================
    print("\n--- break it ---")

    # A small net on a small slice, so the FULL Hessian is affordable and the
    # loss has a genuine isolated minimum. The L2 term is what makes the
    # minimizer finite: with pure cross-entropy on separable data the weights
    # run off to infinity and there is no minimum to sit at.
    LAM = 1e-2
    Xh, yh = make_spiral(n_per=50, turns=2.0, noise=0.06, seed=7)
    small = MLP([2, 4, 1], seed=4)
    train(small, Xh, yh, lr=0.5, steps=4000, lam=LAM)
    for _ in range(8):                                   # Newton polish to the minimum
        _, g = grad_flat(small, Xh, yh, LAM)
        H = hessian(small, Xh, yh, LAM)
        set_flat(small, get_flat(small) - np.linalg.solve(H, g))
        if np.linalg.norm(grad_flat(small, Xh, yh, LAM)[1]) < 1e-13:
            break
    loss_star, g_star = grad_flat(small, Xh, yh, LAM)
    H = hessian(small, Xh, yh, LAM)
    evals, evecs = np.linalg.eigh(H)
    L, v_top = float(evals[-1]), evecs[:, -1]
    w_star = get_flat(small)
    P = w_star.size
    DELTA = 1e-4                                         # the perturbation we study
    newton_step = np.linalg.solve(H, g_star)
    newton_residual = float(np.linalg.norm(newton_step))
    # How far our w* sits from the TRUE minimizer, measured along v_top. Every
    # error trajectory below decays to this value rather than to zero, so it is
    # the floor any "it contracted" claim has to stay clear of. It is computed,
    # not assumed.
    FLOOR = abs(float(newton_step @ v_top))
    print(f"      minimized a {P}-parameter net: loss {loss_star:.6f}, "
          f"||grad|| {np.linalg.norm(g_star):.1e}")
    print(f"      remaining Newton step {newton_residual:.1e}, i.e. w* is pinned "
          f"{DELTA/newton_residual:.1e}x tighter than the {DELTA:.0e} perturbation below")
    print(f"      that residual is {FLOOR:.2e} along v_top — the floor every "
          f"contracting run below decays to")
    print(f"      Hessian eigenvalues in [{evals[0]:.6f}, {evals[-1]:.6f}], "
          f"condition number {evals[-1]/evals[0]:.0f}")
    print(f"      -> L = {L:.4f} and the stability threshold 2/L = {2/L:.4f}")
    # The claim that matters is not "the gradient is small" but "the minimizer
    # is located far more precisely than the perturbation we are about to make".
    check("w* is located far inside the perturbation we study",
          newton_residual < 1e-3 * DELTA)
    check("and it is a minimum, not a saddle (Hessian positive definite)",
          float(evals[0]) > 0.0)

    def gd_along_top(eta, steps, delta=DELTA, zero=True):
        """GD from w* + delta*v_top; returns the error's component along v_top.

        Starting along the top eigenvector means the quadratic model predicts a
        one-dimensional recursion, so every number below has a closed form to
        be compared against.
        """
        set_flat(small, w_star + delta * v_top)
        acc_g = np.zeros(P)
        e = []
        for _ in range(steps):
            _, g = grad_flat(small, Xh, yh, LAM)
            acc_g = g if zero else acc_g + g
            set_flat(small, get_flat(small) - eta * acc_g)
            e.append(float((get_flat(small) - w_star) @ v_top))
        return np.array(e)

    # -- (a) forgetting to zero the gradients --------------------------------
    print("\n      (a) forgetting to zero the gradients between steps")

    # First the exact mechanism: the un-zeroed .grad is the SUM of the true
    # per-step gradients. Nothing approximate about that — check it bit-for-bit.
    fresh = MLP([2, 8, 1], seed=5)
    running = np.zeros(get_flat(fresh).size)
    ok_sum = True
    for p in fresh.params:
        p.zero_grad()
    for t in range(5):
        bce_with_logits(fresh(Tensor(Xh)), yh).backward()          # NO zero_grad
        accumulated = np.concatenate([p.grad.ravel() for p in fresh.params])
        probe = MLP([2, 8, 1], seed=5)                              # this step's gradient alone
        set_flat(probe, get_flat(fresh))
        _, g_true = grad_flat(probe, Xh, yh, 0.0)
        running = running + g_true
        ok_sum = ok_sum and np.abs(accumulated - running).max() <= 1e-15
        set_flat(fresh, get_flat(fresh) - 0.05 * accumulated)
    print(f"      un-zeroed .grad == running sum of the true gradients: {ok_sum}")
    check("un-zeroed .grad is exactly the sum of the per-step gradients", ok_sum)

    # And what that sum does to the dynamics. Near the minimum, with a = eta*L,
    # correct GD contracts the top mode by (1 - a) every step. Accumulating turns
    # the same iteration into a SECOND-ORDER recurrence:
    #
    #     e_{t+1} = e_t - eta * sum_{s<=t} L e_s   =>   e_{t+1} = (2 - a) e_t - e_{t-1}
    #
    # whose characteristic roots multiply to 1. For 0 < a < 4 they are a complex
    # conjugate pair of modulus EXACTLY 1: the error rotates forever and never
    # decays, no matter how small the learning rate is. Past a = 4 the roots turn
    # real and one exceeds 1, and it diverges geometrically. Either way it never
    # converges — which is the real statement, and it is stronger than "the loss
    # blows up".
    def accum_root(a):
        """Larger root modulus of r^2 - (2-a) r + 1 = 0."""
        return float(np.abs(np.roots([1.0, -(2.0 - a), 1.0])).max())

    # Correct GD along v_top has the closed form e_K = (1 - a)^K e_0 in the
    # quadratic model, so that is what we compare against — not a threshold.
    # The horizon K is DERIVED rather than fixed, because the closed form is
    # only honest inside a window:
    #
    #   * upper end: |e| must stay inside the bowl. CAP = 1e-3 is 1600x smaller
    #     than ||w*|| = 1.61, so the Hessian is still the one we measured. Run
    #     an expanding case deeper and it saturates the tanh, the quadratic model
    #     stops applying, AND the number becomes chaotic: at a = 3.9 a 200-step
    #     run multiplies any last-bit difference in w* by 2.9^200 ~ 1e92, which
    #     is how this table used to produce a different answer on every run.
    #   * lower end: |e| must stay a factor 1e3 above FLOOR (4.7e-12, computed
    #     above), or we are just measuring how well we located w*.
    #
    # The accumulating column can safely use all 200 steps precisely because its
    # modulus is 1 — it neither grows out of the bowl nor decays into the floor.
    CAP = 1e-3

    def horizon(a, delta):
        """Largest K with the predicted |e_K| = delta*|1-a|^K inside [1e3*FLOOR, CAP]."""
        r = abs(1.0 - a)
        if r == 0.0:
            return 1                # 1 - a = 0: one step is exact, nothing to grow
        if r == 1.0:
            return 200              # neutral mode, no window to derive
        end = CAP if r > 1.0 else 1e3 * FLOOR
        return int(np.clip(np.floor(np.log(end / delta) / np.log(r)), 1, 200))

    print(f"      {'eta*L':>7} {'K':>4} {'GD |e_K|/|e_0|':>16} {'pred (1-a)^K':>14} "
          f"{'accum |e| ratio':>16} {'pred |r|':>9} {'meas rate':>10}")
    for a in (0.25, 1.0, 1.9, 3.0, 3.9):
        # An expanding run has to start small enough to survive K steps inside
        # the bowl; a contracting one starts at the perturbation we advertised.
        delta = DELTA if a < 2 else 1e-9
        K = horizon(a, delta)
        with np.errstate(over="ignore", invalid="ignore"):
            e_gd = gd_along_top(a / L, K, delta=delta, zero=True)
            e_ac = gd_along_top(a / L, 200, zero=False)
        gd_ratio = e_gd[-1] / delta
        pred = (1.0 - a) ** K
        env_early = float(np.abs(e_ac[10:30]).max())
        env_late = float(np.abs(e_ac[170:190]).max())
        rate = (env_late / env_early) ** (1 / 160)
        print(f"      {a:7.2f} {K:4d} {gd_ratio:16.3e} {pred:14.3e} "
              f"{env_late/env_early:16.4f} {accum_root(a):9.4f} {rate:10.6f}")
        if a == 1.0:
            # 1 - a = 0: the quadratic model says GD lands ON the minimum in a
            # single step along this mode. What is left is the cubic term of the
            # loss, so the residual must be O(delta^2), not O(delta) -- and
            # delta^2 = 1e-8 is the bound, derived, no threshold invented.
            check("eta*L=1.0: correct GD lands on the minimum in ONE step",
                  abs(e_gd[0]) < delta ** 2)
            r_big = abs(gd_along_top(a / L, 1, delta=3 * delta)[0])
            check("eta*L=1.0: and the leftover residual is O(delta^2), not O(delta)",
                  r_big / abs(e_gd[0]), 9.0, tol=1.0)
        else:
            check(f"eta*L={a}: correct GD follows e_K = (1 - eta*L)^K e_0",
                  gd_ratio, pred, tol=0.02 * abs(pred))
        # Correct GD obeys its own threshold: contracts below eta*L = 2, not
        # above. Both bounds below are the closed form, not invented magnitudes.
        if a < 2:
            check(f"eta*L={a}: correct GD contracts the error", abs(gd_ratio) < 1e-3)
        else:
            check(f"eta*L={a}: correct GD diverges (past 2/L)", abs(gd_ratio) > 1e3)
        # Accumulating: modulus exactly 1 at every one of these step sizes.
        check(f"eta*L={a}: accumulating rotates instead of contracting (|r| = 1)",
              rate, accum_root(a), tol=1e-3)

    # Past a = 4 the same recurrence goes real and diverges at a rate the
    # quadratic model predicts to four digits.
    for a in (4.5, 6.0):
        e_ac = np.abs(gd_along_top(a / L, 14, delta=1e-9, zero=False))
        rate = float(np.exp(np.polyfit(np.arange(3, 13), np.log(e_ac[3:13]), 1)[0]))
        print(f"      {a:7.2f} {'':>18} {'':>17} {accum_root(a):9.4f} {rate:10.6f}")
        check(f"eta*L={a}: accumulating diverges at the predicted root modulus",
              rate, accum_root(a), tol=1e-3 * accum_root(a))

    # On the real net the nonlinearity finishes the job: the loop that trains
    # fine ends up above where it started, with weights three orders larger.
    net_ok = MLP([2, 32, 32, 1], seed=1)
    hist_ok, _ = train(net_ok, X, y, lr=1.0, steps=500, record_every=499)
    net_bad = MLP([2, 32, 32, 1], seed=1)
    with np.errstate(over="ignore", invalid="ignore"):
        hist_bad, _ = train(net_bad, X, y, lr=1.0, steps=500,
                            zero_grad=False, record_every=499)
    wn_ok = float(np.sqrt(sum(np.sum(p.data ** 2) for p in net_ok.params)))
    wn_bad = float(np.sqrt(sum(np.sum(p.data ** 2) for p in net_bad.params)))
    print(f"      real net, 500 steps:  zeroing -> loss {hist_ok[-1][1]:.4f}, "
          f"||w|| {wn_ok:.1f}")
    print(f"                        accumulating -> loss {hist_bad[-1][1]:.4f}, "
          f"||w|| {wn_bad:.1f}")
    check("with zeroing the loss goes down", hist_ok[-1][1] < hist_ok[0][1])
    check("without zeroing it ends above where it started",
          not np.isfinite(hist_bad[-1][1]) or hist_bad[-1][1] > 10 * hist_bad[0][1])
    check("and the parameters travel orders of magnitude further",
          wn_bad > 100 * wn_ok)

    # -- (b) a learning rate past 2/L ----------------------------------------
    print("\n      (b) a learning rate past the 2/L stability threshold")
    print(f"      L = {L:.4f} measured from the Hessian above, so 2/L = {2/L:.4f}")
    print(f"      {'eta':>10} {'eta*L':>8} {'pred 1-eta*L':>13} {'measured':>10} "
          f"{'|e_60|/|e_0|':>13} {'sign flips':>11}")
    for a in (0.5, 1.9, 1.9999, 2.0001, 2.1, 2.5):
        e = gd_along_top(a / L, 60)
        r = e[9] / e[8]
        flips = int(np.sum(np.sign(e[:-1]) != np.sign(e[1:])))
        print(f"      {a/L:10.4f} {a:8.4f} {1-a:13.4f} {r:10.4f} "
              f"{abs(e[-1])/DELTA:13.3e} {flips:11d}")
        check(f"eta*L={a}: measured per-step factor is exactly 1 - eta*L",
              r, 1 - a, tol=2e-3)
        if a < 2:
            check(f"eta*L={a}: converges", abs(e[-1]) < DELTA)
        else:
            check(f"eta*L={a}: diverges", abs(e[-1]) > DELTA)
        if a > 1:
            check(f"eta*L={a}: and oscillates (error flips sign every single step)",
                  flips == 59)

    print(f"\n      total runtime {time.perf_counter() - t_start:.1f}s")
    summary()
