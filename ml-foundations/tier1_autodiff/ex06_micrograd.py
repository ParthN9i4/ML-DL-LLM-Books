"""
ex06 — Scalar reverse-mode automatic differentiation.  (Book: Chapter 11)

Backpropagation is not a neural-network algorithm. It is reverse-mode AD, and
the fact that matters is asymptotic:

    forward mode : one pass per INPUT   -> cost O(n) passes for n parameters
    reverse mode : one pass per OUTPUT  -> cost O(1) passes, any n

A loss is one scalar output. So reverse mode computes the gradient of a
400-billion-parameter model for roughly the price of two forward passes, and
that single asymptotic fact is why deep learning is possible at all. Forward
mode would need 400 billion passes.

The mechanism, in three lines:

  1. Every operation records itself on a tape as it executes (the forward pass).
  2. Seed the output's adjoint to 1.
  3. Walk the tape backwards, and at each node push the adjoint to its inputs
     via the local derivative (the chain rule, applied in reverse order).

The only subtle part is that a node used TWICE accumulates two contributions.
Assigning instead of accumulating is the single most common bug in a
hand-written engine, and the "break it" section below shows exactly what it
costs you.

To learn: delete the body of every `_backward` closure and reimplement.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


class Value:
    """A scalar node in a dynamically built computation graph."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # -- arithmetic ---------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1.  ACCUMULATE (+=), never assign:
            # if a node feeds two consumers both must contribute.
            # === YOUR CODE HERE ===
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(ab)/da = b, d(ab)/db = a
            # === YOUR CODE HERE ===
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only numeric powers"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            # === YOUR CODE HERE ===
            self.grad += k * (self.data ** (k - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            # d/dx exp(x) = exp(x), which is exactly out.data — reuse it rather
            # than recomputing. Every engine does this.
            # === YOUR CODE HERE ===
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            # === YOUR CODE HERE ===
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            # Non-differentiable at exactly 0. Every framework picks a
            # subgradient; the near-universal convention is 0.
            # === YOUR CODE HERE ===
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # === YOUR CODE HERE ===
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    # -- the reverse sweep --------------------------------------------------

    def backward(self):
        """Topologically sort the graph, seed d(self)/d(self) = 1, sweep backwards.

        The sort is ITERATIVE, not recursive. A recursive post-order DFS is the
        obvious way to write this and it blows Python's ~1000-frame stack on any
        genuinely deep graph — a 1000-step chain is nothing, a real network has
        far more. PyTorch traverses its graph iteratively for the same reason.
        """
        topo, visited = [], set()
        # Explicit stack of (node, children_expanded?) — the second flag is what
        # turns a recursive post-order into an iterative one.
        stack = [(self, False)]
        # === YOUR CODE HERE ===
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

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # -- conveniences -------------------------------------------------------

    def __neg__(self): return self * -1
    def __radd__(self, o): return self + o
    def __sub__(self, o): return self + (-o)
    def __rsub__(self, o): return (-self) + o
    def __rmul__(self, o): return self * o
    def __truediv__(self, o): return self * (o ** -1 if isinstance(o, Value) else 1.0 / o)
    def __rtruediv__(self, o): return (self ** -1) * o
    def __repr__(self): return f"Value({self.data:.6g}, grad={self.grad:.6g})"


# ---------------------------------------------------------------------------
# Reference: central finite differences
# ---------------------------------------------------------------------------

def numeric_grad(f, xs, eps=1e-6):
    """d f / d x_i by central differences. O(eps^2) truncation, O(eps^-1) roundoff."""
    g = []
    for i in range(len(xs)):
        up = list(xs); up[i] += eps
        dn = list(xs); dn[i] -= eps
        g.append((f(up) - f(dn)) / (2 * eps))
    return g


def autodiff_grad(f, xs):
    """d f / d x_i by one reverse sweep — regardless of how many inputs there are."""
    nodes = [Value(x) for x in xs]
    out = f(nodes)
    out.backward()
    return [n.grad for n in nodes]


if __name__ == "__main__":
    import random
    random.seed(0)

    print("\n--- a hand-checkable case ---")
    # f(a,b) = a*b + a  at a=3, b=4  ->  f=15, df/da = b+1 = 5, df/db = a = 3.
    a, b = Value(3.0), Value(4.0)
    f = a * b + a
    f.backward()
    check("f(3,4) = 15", f.data, 15.0, tol=1e-12)
    check("df/da = b + 1 = 5", a.grad, 5.0, tol=1e-12)
    check("df/db = a = 3", b.grad, 3.0, tol=1e-12)

    print("\n--- a node used twice accumulates ---")
    # f = x*x with ONE node reused. df/dx = 2x = 6 only if both paths contribute.
    x = Value(3.0)
    y = x * x
    y.backward()
    check("df/dx for f = x*x is 2x, not x", x.grad, 6.0, tol=1e-12)

    # Deeper reuse: f = (x+x)*(x+x) = 4x^2, df/dx = 8x = 24.
    x2 = Value(3.0)
    s = x2 + x2
    (s * s).backward()
    check("df/dx for f = (x+x)^2 is 8x", x2.grad, 24.0, tol=1e-12)

    print("\n--- every operation against finite differences ---")
    cases = {
        "polynomial":  (lambda v: v[0] ** 3 * v[1] + v[1] ** 2, [1.7, -0.9]),
        "exp/log":     (lambda v: (v[0].exp() + v[1] * v[0]).log(), [0.6, 1.1]),
        "tanh chain":  (lambda v: (v[0] * v[1]).tanh() * v[0].tanh(), [0.8, -1.3]),
        "relu (pos)":  (lambda v: (v[0] * v[1]).relu() * v[1], [1.2, 0.7]),
        "division":    (lambda v: v[0] / (v[1] * v[1] + 1.0), [2.1, 0.4]),
        "deep chain":  (lambda v: ((((v[0] * v[1]).tanh() + v[0]) * v[1]).tanh() + v[1]).tanh(),
                        [0.5, -0.8]),
    }
    for name, (fn, pt) in cases.items():
        ad = autodiff_grad(fn, pt)
        fd = numeric_grad(lambda xs: fn([Value(x) for x in xs]).data, pt)
        check(f"{name}: autodiff == finite differences", ad, fd, tol=1e-6)

    print("\n--- the asymptotic claim ---")
    # ONE reverse sweep produces all n partials. Finite differences needs 2n
    # forward evaluations. Count them and watch the gap widen with n.
    print(f"      {'n inputs':>10} {'fwd evals (FD)':>16} {'fwd evals (AD)':>16}")
    for n in (10, 100, 1000):
        calls = {"n": 0}

        def big(v):
            calls["n"] += 1
            acc = v[0]
            for t in v[1:]:
                acc = acc * 1.0001 + t.tanh()
            return acc

        pt = [random.uniform(-1, 1) for _ in range(n)]
        calls["n"] = 0
        g_ad = autodiff_grad(big, pt)
        ad_calls = calls["n"]
        print(f"      {n:10d} {2*n:16d} {ad_calls:16d}")
        check(f"n={n}: reverse mode needs exactly one forward evaluation", ad_calls, 1)
        check(f"n={n}: it still produces all {n} partials", len(g_ad), n)

    # Spot-check correctness at n=10 against finite differences.
    def big10(v):
        acc = v[0]
        for t in v[1:]:
            acc = acc * 1.0001 + t.tanh()
        return acc

    pt10 = [random.uniform(-1, 1) for _ in range(10)]
    check("the n=10 gradient is actually correct",
          autodiff_grad(big10, pt10),
          numeric_grad(lambda xs: big10([Value(x) for x in xs]).data, pt10),
          tol=1e-6)

    print("\n--- training a tiny network with it ---")
    # Fit y = xor-ish decision with a 2-4-1 tanh network, by hand, using only
    # this engine. If the gradients were wrong the loss would not fall.
    def init(nin, nout):
        return ([[Value(random.uniform(-1, 1)) for _ in range(nin)] for _ in range(nout)],
                [Value(0.0) for _ in range(nout)])

    W1, b1 = init(2, 4)
    W2, b2 = init(4, 1)
    params = [w for row in W1 for w in row] + b1 + [w for row in W2 for w in row] + b2

    data = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)]

    def forward(xs):
        h = [sum((W1[j][i] * xs[i] for i in range(2)), b1[j]).tanh() for j in range(4)]
        return sum((W2[0][j] * h[j] for j in range(4)), b2[0]).tanh()

    losses = []
    for step in range(600):
        loss = Value(0.0)
        for xs, t in data:
            pred = forward([Value(v) for v in xs])
            loss = loss + (pred - (t * 2 - 1)) ** 2
        for p in params:
            p.grad = 0.0
        loss.backward()
        for p in params:
            p.data -= 0.05 * p.grad
        losses.append(loss.data)

    print(f"      loss: {losses[0]:.4f} -> {losses[-1]:.6f} over 600 steps")
    preds = [forward([Value(v) for v in xs]).data for xs, _ in data]
    print(f"      predictions: {' '.join(f'{p:+.3f}' for p in preds)}   (targets: -1 +1 +1 -1)")
    check("XOR loss falls by orders of magnitude", losses[-1] < losses[0] / 100)
    check("all four XOR points are classified correctly",
          all((p > 0) == (t > 0.5) for p, (_, t) in zip(preds, data)))

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) THE bug: assigning the adjoint instead of accumulating it. With a node
    #     used twice, one path silently overwrites the other and you get half
    #     the gradient — a model that trains, slowly, and wrongly.
    class BrokenValue(Value):
        def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data * other.data, (self, other), "*")

            def _backward():
                self.grad = other.data * out.grad      # = instead of +=
                other.grad = self.data * out.grad
            out._backward = _backward
            return out

    xb = BrokenValue(3.0)
    (xb * xb).backward()
    print(f"      f = x*x at x=3: correct df/dx = 6, broken engine gives {xb.grad}")
    check("the assigning engine gets it wrong", xb.grad != 6.0)
    check("specifically, it loses one of the two paths", xb.grad, 3.0, tol=1e-12)

    # (b) Finite differences cannot be made arbitrarily accurate. Truncation
    #     falls as eps^2 while roundoff grows as eps^-1, so the total error is
    #     minimized near eps = u^(1/3) ~ 6e-6 and gets WORSE on both sides.
    def g(xs): return math.sin(xs[0]) * math.exp(xs[1])
    pt = [0.7, 0.3]
    exact = [math.cos(0.7) * math.exp(0.3), math.sin(0.7) * math.exp(0.3)]
    print(f"      {'eps':>10} {'max |FD - exact|':>20}")
    errs = {}
    for e in (1e-1, 1e-3, 1e-6, 1e-10, 1e-13):
        fd = numeric_grad(g, pt, eps=e)
        errs[e] = max(abs(fd[i] - exact[i]) for i in range(2))
        print(f"      {e:10.0e} {errs[e]:20.3e}")
    check("large eps is inaccurate (truncation)", errs[1e-1] > errs[1e-6])
    check("tiny eps is inaccurate too (roundoff)", errs[1e-13] > errs[1e-6])
    # Autodiff has no such tradeoff: it is exact to machine precision.
    ad = autodiff_grad(lambda v: v[0].tanh() * v[1].exp(), pt)
    exact_ad = [(1 - math.tanh(0.7) ** 2) * math.exp(0.3), math.tanh(0.7) * math.exp(0.3)]
    check("autodiff has no step-size tradeoff at all", ad, exact_ad, tol=1e-15)

    summary()
