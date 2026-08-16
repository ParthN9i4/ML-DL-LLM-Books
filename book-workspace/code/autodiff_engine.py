"""autodiff_engine.py -- the reverse-mode AD engine from Chapter 11, standalone.

Later chapters import this:

    from autodiff_engine import Tensor, log_softmax

Pure NumPy, float64, tape-based.  Run this file directly to self-test:
    python autodiff_engine.py
"""
import numpy as np

__all__ = ["Tensor", "log_softmax", "numerical_grad"]

# Every Tensor takes a monotonically increasing id at construction, so sorting
# by DESCENDING id is a reverse topological order: a node can only be built
# after all of its parents exist.  That ordering is the whole of "the tape".
_COUNTER = [0]


def _unbroadcast(g, shape):
    """Reduce gradient `g` back to `shape`, undoing NumPy broadcasting.
    Broadcasting is a linear map that COPIES entries; its transpose SUMS them."""
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

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op})"

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

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    __radd__, __rmul__ = __add__, __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._w(other))

    def __rsub__(self, other):
        return self._w(other) + (-self)

    def __truediv__(self, c):
        return self * (1.0 / float(c))

    def zero_grad(self):
        self.grad = None

    def backward(self):
        """One reverse sweep: seed the output cotangent with 1, then apply each
        node's VJP once, in reverse creation order, accumulating at every leaf."""
        assert self.data.size == 1, "backward() must start from a scalar"
        seen, nodes, stack = set(), [], [self]
        while stack:                             # reachable sub-tape only
            n = stack.pop()
            if id(n) in seen:
                continue
            seen.add(id(n))
            nodes.append(n)
            stack.extend(n._parents)
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


def numerical_grad(f, x, eps=1e-5):
    """Central-difference gradient of a scalar-valued f: ndarray -> float."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        b = x[i]
        x[i] = b + eps
        fp = f(x)
        x[i] = b - eps
        fm = f(x)
        x[i] = b
        g[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Xd = rng.normal(size=(5, 3))
    Wd = rng.normal(size=(3, 4)) * 0.5
    bd = rng.normal(size=(1, 4)) * 0.5          # broadcast over the batch
    Yd = np.eye(4)[rng.integers(0, 4, size=5)]

    def loss_of(Wt, bt):
        z = (Tensor(Xd) @ Wt + bt).relu() + 0.1     # bias broadcasts over batch
        return -(Tensor(Yd) * log_softmax(z.log() + z)).sum() / 5.0

    W, b = Tensor(Wd), Tensor(bd)
    loss_of(W, b).backward()
    gW = numerical_grad(lambda w: float(loss_of(Tensor(w), Tensor(bd)).data),
                        Wd.copy())
    gb = numerical_grad(lambda bb: float(loss_of(Tensor(Wd), Tensor(bb)).data),
                        bd.copy())
    eW = np.abs(W.grad - gW).max()
    eb = np.abs(b.grad - gb).max()
    assert b.grad.shape == (1, 4), b.grad.shape       # broadcast reduction
    assert eW < 1e-7 and eb < 1e-7, (eW, eb)
    print(f"finite-difference check: max |dL/dW| err = {eW:.3e}, "
          f"max |dL/db| err = {eb:.3e}, db shape {b.grad.shape}")

    # a broadcast gradient must be the SUM over the broadcast axis
    v = Tensor(np.arange(6.0).reshape(2, 3))
    c = Tensor(np.array([[10.0, 20.0, 30.0]]))
    (v * c).sum().backward()
    assert np.allclose(c.grad, v.data.sum(axis=0, keepdims=True))
    print(f"broadcast reduction: dc = {c.grad.ravel()} == column sums "
          f"{v.data.sum(axis=0)}")
    print("autodiff_engine self-test OK")
