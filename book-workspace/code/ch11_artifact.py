"""Artifact 11.1 -- a tape-based reverse-mode autodiff engine, pure NumPy.
Verified (a) against central finite differences on random computation graphs,
(b) against torch.autograd on a 3-layer MLP, (c) by training that MLP on
synthetic two-moons with nothing but this engine.  torch is optional.
"""
import time
import numpy as np
try:
    import torch
except ImportError:
    torch = None

# Every Tensor takes a monotonically increasing id at construction, so sorting
# by DESCENDING id is a reverse topological order: a node can only be built
# after all of its parents exist.  That ordering is the whole of "the tape".
_COUNTER = [0]

def _unbroadcast(g, shape):
    """Reduce gradient `g` back to `shape`, undoing NumPy broadcasting.
    Broadcasting is a linear map that COPIES entries; its transpose SUMS them.
    Omitting this is the classic autodiff bug: the forward pass is fine and the
    backward pass silently returns a gradient of the wrong shape."""
    while g.ndim > len(shape):            # axes prepended by broadcasting
        g = g.sum(axis=0)
    for i, s in enumerate(shape):         # axes stretched from length 1
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)

class Tensor:
    """Tape node.  `_bw(g)` maps the output cotangent to a tuple of parent
    cotangents: it applies the local transposed Jacobian, i.e. one VJP."""
    __slots__ = ("data", "grad", "_parents", "_bw", "_op", "_id")

    def __init__(self, data, parents=(), bw=None, op="leaf"):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad, self._parents, self._bw, self._op = None, parents, bw, op
        _COUNTER[0] += 1
        self._id = _COUNTER[0]

    @property
    def shape(self):
        return self.data.shape

    def _w(self, x):
        return x if isinstance(x, Tensor) else Tensor(x)

    # ---- primitives: forward value, parents, and the local VJP ----
    def __add__(self, other):
        o = self._w(other)
        sa, sb = self.shape, o.shape
        return Tensor(self.data + o.data, (self, o),
                      lambda g: (_unbroadcast(g, sa), _unbroadcast(g, sb)), "+")

    def __mul__(self, other):
        o = self._w(other)
        sa, sb, a, b = self.shape, o.shape, self.data, o.data
        return Tensor(a * b, (self, o),
                      lambda g: (_unbroadcast(g * b, sa),
                                 _unbroadcast(g * a, sb)), "*")

    def __matmul__(self, other):
        o = self._w(other)
        a, b = self.data, o.data
        return Tensor(a @ b, (self, o), lambda g: (g @ b.T, a.T @ g), "@")

    def relu(self):
        m = (self.data > 0.0).astype(np.float64)   # subgradient 0 at exactly 0
        return Tensor(self.data * m, (self,), lambda g: (g * m,), "relu")

    def exp(self):
        e = np.exp(self.data)
        return Tensor(e, (self,), lambda g: (g * e,), "exp")

    def log(self):
        a = self.data
        return Tensor(np.log(a), (self,), lambda g: (g / a,), "log")

    def sum(self, axis=None, keepdims=False):
        shp, ax, kd = self.shape, axis, keepdims
        def bw(g):                        # transpose of "sum" is "broadcast"
            g = np.asarray(g)
            if ax is not None and not kd:
                g = np.expand_dims(g, ax)
            return (np.broadcast_to(g, shp).copy(),)
        return Tensor(self.data.sum(axis=ax, keepdims=kd), (self,), bw, "sum")

    __radd__, __rmul__ = __add__, __mul__
    def __neg__(self):
        return self * -1.0
    def __sub__(self, other):
        return self + (-self._w(other))
    def __truediv__(self, c):
        return self * (1.0 / float(c))

    def backward(self):
        """One reverse sweep: seed the output cotangent with 1, then apply each
        node's VJP once, in reverse creation order, accumulating at every leaf."""
        assert self.data.size == 1, "backward() must start from a scalar"
        seen, nodes, stack = set(), [], [self]
        while stack:                             # reachable sub-tape only
            n = stack.pop()
            if id(n) in seen:
                continue
            seen.add(id(n)); nodes.append(n); stack.extend(n._parents)
        nodes.sort(key=lambda t: -t._id)
        for n in nodes:
            n.grad = np.zeros_like(n.data)
        self.grad = np.ones_like(self.data)
        for n in nodes:
            if n._bw is None:
                continue
            for p, gp in zip(n._parents, n._bw(n.grad)):
                p.grad = p.grad + gp             # fan-out => sum of paths

def log_softmax(z):
    """Stable log-softmax over axis 1, built only from the primitives above.
    The max shift is a CONSTANT: log-softmax is shift-invariant, so no true
    derivative flows through the max and detaching it is exact, not a hack."""
    zs = z + Tensor(-z.data.max(axis=1, keepdims=True))
    return zs - zs.exp().sum(axis=1, keepdims=True).log()

# ---- (a) random computation graphs vs central finite differences ----
SHAPES = [(3, 4), (4,), (1, 4), (3, 1), ()]      # every shape broadcasts to (3,4)

def random_graph(seed, leaves):
    """Random DAG over `leaves`; `seed` fixes the STRUCTURE so the identical
    function is rebuilt for every finite-difference probe."""
    rng = np.random.default_rng(seed)
    pool, mat = list(leaves[:-1]), leaves[-1]    # mat: the (4,4) matmul partner
    for _ in range(12):
        k = rng.integers(0, 6)
        a, b = pool[rng.integers(0, len(pool))], pool[rng.integers(0, len(pool))]
        if k == 0:   pool.append(a + b)
        elif k == 1: pool.append(a * b)
        elif k == 2: pool.append((a + b).relu())
        elif k == 3: pool.append((a * 0.25).exp())
        elif k == 4: pool.append((a * a + 1.0).log())      # argument > 0 always
        else:        pool.append((a + Tensor(np.zeros((3, 4)))) @ mat)
    # summing the leaves too guarantees each is reachable from the output
    return sum(p.sum() for p in pool) + mat.sum()

def fd_check(n_graphs=60, probes=3, eps=1e-5):
    """Each probed partial against (f(x+eps)-f(x-eps))/2eps, error O(eps^2)."""
    worst, n = 0.0, 0
    for trial in range(n_graphs):
        rng = np.random.default_rng(trial)
        data = [np.array(rng.normal(size=s) * 0.5) for s in SHAPES]
        data.append(rng.normal(size=(4, 4)) * 0.3)
        def f(vals):                     # same graph, fresh leaves
            return random_graph(1000 + trial, [Tensor(v.copy()) for v in vals])
        leaves = [Tensor(v.copy()) for v in data]
        random_graph(1000 + trial, leaves).backward()      # ONE reverse sweep
        for li in range(len(data)):
            for _ in range(probes):
                idx = tuple(rng.integers(0, d) for d in data[li].shape)
                base = float(data[li][idx])
                data[li][idx] = base + eps; fp = float(f(data).data)
                data[li][idx] = base - eps; fm = float(f(data).data)
                data[li][idx] = base
                num, ana = (fp - fm) / (2 * eps), float(leaves[li].grad[idx])
                worst = max(worst, abs(num - ana) / max(1.0, abs(num)))
                n += 1
    return worst, n

# ---- the 3-layer MLP used by checks (b) and (c) ----
def init_mlp(rng, d_in=2, h=32, d_out=2):
    ps = []
    for (a, b) in [(d_in, h), (h, h), (h, d_out)]:
        ps += [Tensor(rng.normal(size=(a, b)) * np.sqrt(2.0 / a)),
               Tensor(np.zeros((1, b)))]         # bias relies on broadcasting
    return ps

def mlp_loss(ps, X, Y):
    W1, b1, W2, b2, W3, b3 = ps
    h1 = (X @ W1 + b1).relu()                    # (B,h) + (1,h) broadcasts
    h2 = (h1 @ W2 + b2).relu()
    return -(Y * log_softmax(h2 @ W3 + b3)).sum() / X.shape[0]

def two_moons(n, rng, noise=0.18):
    t = rng.uniform(0.0, np.pi, size=n // 2)
    X = np.concatenate([np.stack([np.cos(t), np.sin(t)], 1),
                        np.stack([1.0 - np.cos(t), 0.5 - np.sin(t)], 1)])
    X = X + rng.normal(size=(n, 2)) * noise
    return X, np.concatenate([np.zeros(n // 2, int), np.ones(n // 2, int)])

if __name__ == "__main__":
    t0 = time.time()
    print("(a) random graphs vs central finite differences")
    worst, nprobe = fd_check()
    print(f"    60 graphs, {nprobe} probed partials: max rel error = {worst:.3e}")
    assert worst < 1e-6, worst

    print("(b) 3-layer MLP vs torch.autograd")
    X, y = two_moons(600, np.random.default_rng(1))
    Y = np.eye(2)[y]
    ps = init_mlp(np.random.default_rng(0))
    loss = mlp_loss(ps, Tensor(X), Tensor(Y))
    loss.backward()
    if torch is None:
        print("    [skipped: torch not installed]")
    else:
        tp = [torch.tensor(p.data, requires_grad=True) for p in ps]
        W1, b1, W2, b2, W3, b3 = tp
        h1 = torch.relu(torch.tensor(X) @ W1 + b1)
        h2 = torch.relu(h1 @ W2 + b2)
        tl = torch.nn.functional.cross_entropy(h2 @ W3 + b3, torch.tensor(y))
        tl.backward()
        dl = abs(float(tl.detach()) - float(loss.data))
        errs = [float(np.abs(p.grad - q.grad.numpy()).max()) for p, q in zip(ps, tp)]
        print(f"    loss ours={float(loss.data):.10f} torch={float(tl.detach()):.10f}"
              f"  |diff|={dl:.2e}")
        print("    max |grad diff|: " + "  ".join(
            f"{k}={e:.2e}" for k, e in zip("W1 b1 W2 b2 W3 b3".split(), errs)))
        assert dl < 1e-9 and max(errs) < 1e-6, (dl, errs)

    print("(c) training two-moons with this engine only")
    ps = init_mlp(np.random.default_rng(2))
    vel = [np.zeros_like(p.data) for p in ps]
    tX, tY, acc = Tensor(X), Tensor(Y), 0.0
    for ep in range(1, 601):                     # full-batch SGD + momentum
        loss = mlp_loss(ps, tX, tY)
        loss.backward()
        for p, v in zip(ps, vel):
            v *= 0.9
            v -= 0.5 * p.grad
            p.data += v
        if ep % 200 == 0:
            W1, b1, W2, b2, W3, b3 = ps
            h1 = np.maximum(X @ W1.data + b1.data, 0)
            h2 = np.maximum(h1 @ W2.data + b2.data, 0)
            acc = ((h2 @ W3.data + b3.data).argmax(1) == y).mean()
            print(f"    epoch {ep:4d}  loss {float(loss.data):.6f}"
                  f"  train acc {acc*100:.2f}%")
    assert acc > 0.99, acc
    print(f"done in {time.time()-t0:.1f}s")
