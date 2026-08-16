"""
ex07 — Tensor reverse-mode autodiff, and the broadcast bug.  (Book: Chapter 11)

ex06 built a scalar engine. Scaling it to tensors adds exactly one genuinely
hard idea, and it is the one everybody gets wrong:

    IF THE FORWARD PASS BROADCAST, THE BACKWARD PASS MUST SUM.

A bias of shape (1, d) added to activations of shape (B, d) is silently
replicated across the batch. Each of those B copies receives its own gradient,
and the bias's true gradient is their SUM. Skip the reduction and one of two
things happens:

  * the shapes disagree and you get a loud error (the lucky case), or
  * the shapes happen to be compatible and numpy broadcasts the update too,
    so the code runs, the loss even falls, and the gradient is wrong by a
    factor of B. That is the unlucky case, and it is why this exercise exists.

The rule generalizes. In reverse-mode AD, the adjoint of an operation that
REPLICATES data is a SUM, and the adjoint of a SUM is a replication. They are
transposes of each other.

To learn: delete each `_backward` body and reimplement. `unbroadcast` is the
function to get right; everything else follows from it.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


def unbroadcast(grad, shape):
    """Reduce `grad` back to `shape`, undoing whatever numpy broadcasting did.

    Two things happened during the forward broadcast and both must be undone:
      1. Leading axes were PREPENDED  -> sum over them and drop them.
      2. Size-1 axes were STRETCHED   -> sum over them but KEEP the axis.

    This is the whole of the tensor chain rule that scalars do not teach you.
    """
    # === YOUR CODE HERE ===
    # 1. drop the extra leading axes
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2. collapse the stretched size-1 axes, keeping the dimension
    for i, size in enumerate(shape):
        if size == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """An n-dimensional node in a computation graph."""

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
            # Derive it once by indices and you never have to remember it.
            # === YOUR CODE HERE ===
            self.grad = self.grad + out.grad @ other.data.T
            other.grad = other.grad + self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            # The adjoint of a sum is a broadcast — the exact transpose of the
            # add case above. Restore the reduced axes before broadcasting.
            # === YOUR CODE HERE ===
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad = self.grad + np.broadcast_to(g, self.data.shape).copy()

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward():
            # === YOUR CODE HERE ===
            self.grad = self.grad + (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            # === YOUR CODE HERE ===
            self.grad = self.grad + out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            # === YOUR CODE HERE ===
            self.grad = self.grad + out.grad / self.data

        out._backward = _backward
        return out

    def backward(self):
        """Iterative topological sort, then the reverse sweep (see ex06)."""
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


def numeric_grad(f, tensors, eps=1e-6):
    """Central differences over every entry of every input tensor."""
    grads = []
    for t in tensors:
        g = np.zeros_like(t.data)
        flat = t.data.reshape(-1)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps; up = f()
            flat[i] = orig - eps; dn = f()
            flat[i] = orig
            g.reshape(-1)[i] = (up - dn) / (2 * eps)
        grads.append(g)
    return grads


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("\n--- unbroadcast in isolation ---")
    # (B,d) gradient reduced to a (1,d) bias: sum over the batch, keep the axis.
    g = np.ones((4, 3))
    check("(4,3) -> (1,3) sums the batch", unbroadcast(g, (1, 3)), np.full((1, 3), 4.0), tol=0)
    # (B,d) reduced to a bare (d,) bias: sum over the prepended axis and drop it.
    check("(4,3) -> (3,) sums and drops", unbroadcast(g, (3,)), np.full((3,), 4.0), tol=0)
    # A scalar collects everything.
    check("(4,3) -> () sums all", unbroadcast(g, ()), np.array(12.0), tol=0)
    # No broadcasting happened: nothing to undo.
    check("(4,3) -> (4,3) is the identity", unbroadcast(g, (4, 3)), g, tol=0)
    # Both mechanisms at once.
    check("(2,4,3) -> (1,3) handles both", unbroadcast(np.ones((2, 4, 3)), (1, 3)),
          np.full((1, 3), 8.0), tol=0)

    print("\n--- gradients against finite differences ---")
    A = Tensor(rng.standard_normal((4, 5)))
    B = Tensor(rng.standard_normal((5, 3)))
    bias = Tensor(rng.standard_normal((1, 3)))       # broadcast over the batch
    scale = Tensor(rng.standard_normal(()))          # scalar, broadcast over everything

    def loss_fn():
        return float((((A.data @ B.data) + bias.data) * scale.data).clip(min=0).sum())

    for t in (A, B, bias, scale):
        t.zero_grad()
    out = (((A @ B) + bias) * scale).relu().sum()
    out.backward()
    ref = numeric_grad(loss_fn, [A, B, bias, scale])
    for name, t, r in zip(("A", "B", "bias(1,3)", "scale(scalar)"), (A, B, bias, scale), ref):
        check(f"d loss / d {name}", t.grad, r, tol=1e-6)

    print("\n--- against torch.autograd ---")
    try:
        import torch
        an = rng.standard_normal((6, 4))
        bn = rng.standard_normal((4, 3))
        cn = rng.standard_normal((1, 3))

        at, bt, ct = (torch.tensor(x, requires_grad=True) for x in (an, bn, cn))
        ((at @ bt + ct).relu() * (at @ bt + ct).relu()).sum().backward()

        A2, B2, C2 = Tensor(an), Tensor(bn), Tensor(cn)
        h = (A2 @ B2 + C2).relu()
        (h * h).sum().backward()

        check("matmul gradient matches torch", A2.grad, at.grad.numpy(), tol=1e-9)
        check("weight gradient matches torch", B2.grad, bt.grad.numpy(), tol=1e-9)
        check("broadcast bias gradient matches torch", C2.grad, ct.grad.numpy(), tol=1e-9)
    except ImportError:
        print("  ....  [skipped: torch not installed] torch.autograd cross-check")

    print("\n--- train an MLP with it ---")
    # Two moons, 2-16-1, trained end to end on this engine alone.
    n = 300
    theta = rng.uniform(0, np.pi, n)
    Xa = np.column_stack([np.cos(theta), np.sin(theta)]) + 0.12 * rng.standard_normal((n, 2))
    Xb = np.column_stack([1 - np.cos(theta), 0.5 - np.sin(theta)]) + 0.12 * rng.standard_normal((n, 2))
    Xd = np.vstack([Xa, Xb])
    yd = np.concatenate([np.zeros(n), np.ones(n)]).reshape(-1, 1)

    W1 = Tensor(rng.standard_normal((2, 16)) * np.sqrt(2 / 2))    # He init
    b1 = Tensor(np.zeros((1, 16)))
    W2 = Tensor(rng.standard_normal((16, 1)) * np.sqrt(2 / 16))
    b2 = Tensor(np.zeros((1, 1)))
    params = [W1, b1, W2, b2]

    Xt = Tensor(Xd)
    first = last = None
    for step in range(1500):
        for p in params:
            p.zero_grad()
        logits = (Xt @ W1 + b1).relu() @ W2 + b2
        # Binary cross-entropy as log(1 + exp(z)) - y*z, built through the graph
        # so gradients flow. NOTE this is the NAIVE softplus: it computes exp(z)
        # directly and overflows for z beyond ~709 in float64. It is fine here
        # because the logits stay small, but a real engine needs the shifted form
        # max(z,0) + log1p(exp(-|z|)) — which would require adding abs and
        # maximum to this engine. ex02 shows what the unshifted version costs.
        loss = ((Tensor(1.0) + logits.exp()).log() - Tensor(yd) * logits).sum() * (1.0 / len(yd))
        loss.backward()
        for p in params:
            p.data -= 0.5 * p.grad
        if step == 0:
            first = float(loss.data)
        last = float(loss.data)

    logits_final = (Xd @ W1.data + b1.data).clip(min=0) @ W2.data + b2.data
    acc = float(((logits_final > 0).astype(float) == yd).mean())
    print(f"      loss {first:.4f} -> {last:.4f},  train accuracy {acc*100:.1f}%")
    check("loss decreased substantially", last < first / 3)
    check("two moons is separated (>99% accuracy)", acc > 0.99)

    # -----------------------------------------------------------------------
    # BREAK IT — the reason this exercise exists
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    class NoReduceTensor(Tensor):
        """An engine that forgets to sum over broadcast axes."""
        def __add__(self, other):
            other = other if isinstance(other, Tensor) else Tensor(other)
            out = Tensor(self.data + other.data, (self, other), "+")

            def _backward():
                self.grad = self.grad + out.grad          # no unbroadcast
                other.grad = other.grad + out.grad        # no unbroadcast
            out._backward = _backward
            return out

    Bsz = 8
    x_ok = Tensor(np.ones((Bsz, 3)))
    bias_ok = Tensor(np.zeros((1, 3)))
    (x_ok + bias_ok).sum().backward()
    print(f"      correct bias gradient           : {bias_ok.grad.ravel()}  (= batch size {Bsz})")

    x_bad = NoReduceTensor(np.ones((Bsz, 3)))
    bias_bad = Tensor(np.zeros((1, 3)))
    try:
        (x_bad + bias_bad).sum().backward()
        bad = bias_bad.grad
        print(f"      un-reduced bias gradient shape  : {bad.shape}  <- should be (1, 3)")
        wrong = bad.shape != (1, 3)
    except ValueError as e:
        print(f"      un-reduced engine raised: {e}")
        wrong = True
    check("skipping unbroadcast corrupts the bias gradient", wrong)

    # And the dangerous variant: with a (d,) bias the shapes stay compatible,
    # numpy broadcasts the update, nothing raises, and every bias step is B
    # times too large. The model still trains. It is just wrong.
    bias_1d = Tensor(np.zeros(3))
    x2 = NoReduceTensor(np.ones((Bsz, 3)))
    (x2 + bias_1d).sum().backward()
    print(f"      1-D bias: gradient has shape {bias_1d.grad.shape}, "
          f"correct is (3,) with value {Bsz}")
    silently_wrong = bias_1d.grad.shape == (Bsz, 3)
    print("      -> no exception raised; a (8,3) 'gradient' for a (3,) parameter"
          if silently_wrong else "      -> raised, which is the lucky case")
    check("the 1-D case fails silently rather than loudly", silently_wrong)

    summary()
