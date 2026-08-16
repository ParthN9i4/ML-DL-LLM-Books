"""Artifact 22.1 -- A top-k mixture-of-experts layer from scratch.

Implements: top-k gating, batched dispatch/combine with capacity dropping,
and BOTH load-balancing schemes -- the Switch/GShard auxiliary loss and the
DeepSeek-style auxiliary-loss-free bias adjustment. Verifies:
  (a) dispatch -> expert -> combine == a dense per-token loop, to ~1e-6;
  (b) coefficient of variation (CV) of expert load falls under both schemes;
  (c) active/total parameter counts match hand-derived formulas.
Pure NumPy. Fixed seed. Runs in seconds on CPU.
"""
import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

class MoELayer:
    """E two-layer ReLU experts of width h, a linear router, top-k gating."""
    def __init__(self, d, h, E, k, rng, logit_offset=None):
        self.d, self.h, self.E, self.k = d, h, E, k
        self.W1 = rng.normal(0, d ** -0.5, (E, d, h))   # expert up-proj
        self.W2 = rng.normal(0, h ** -0.5, (E, h, d))   # expert down-proj
        self.Wg = rng.normal(0, d ** -0.5, (d, E))      # router
        # bg: a fixed router offset used to CREATE imbalance for the demo.
        self.bg = np.zeros(E) if logit_offset is None else logit_offset.copy()
        self.b  = np.zeros(E)  # aux-loss-free balancing bias (routing only)

    def route(self, X, use_bias=False):
        """Returns softmax probs, top-k expert ids (T,k), renormalized gates."""
        logits = X @ self.Wg + self.bg                 # (T, E)
        probs = softmax(logits)
        sel = logits + self.b if use_bias else logits  # bias steers SELECTION
        topk = np.argpartition(-sel, self.k - 1, axis=1)[:, :self.k]
        # order the k winners by descending selection score (deterministic)
        order = np.argsort(-np.take_along_axis(sel, topk, 1), axis=1)
        topk = np.take_along_axis(topk, order, 1)
        g = np.take_along_axis(probs, topk, 1)         # gate from RAW probs
        gates = g / g.sum(axis=1, keepdims=True)       # renormalize over top-k
        return logits, probs, topk, gates

    def forward_dispatch(self, X, capacity_factor=1.25, use_bias=False):
        """Batched dispatch/combine path with capacity dropping."""
        T, (d, h, E, k) = X.shape[0], (self.d, self.h, self.E, self.k)
        _, _, topk, gates = self.route(X, use_bias)
        C = int(np.ceil(capacity_factor * T * k / E))  # per-expert capacity
        # Flatten (token, slot) assignments; priority = token order.
        tok = np.repeat(np.arange(T), k)
        exp = topk.reshape(-1)
        gat = gates.reshape(-1)
        # Position of each assignment within its expert's queue.
        pos = np.zeros(T * k, dtype=int)
        for e in range(E):
            m = exp == e
            pos[m] = np.arange(m.sum())
        keep = pos < C                                  # overflow is DROPPED
        # Scatter kept tokens into an (E, C, d) buffer, run experts batched.
        buf = np.zeros((E, C, d))
        buf[exp[keep], pos[keep]] = X[tok[keep]]
        out_buf = np.maximum(buf @ self.W1, 0.0) @ self.W2   # (E, C, d)
        # Combine: each kept assignment adds gate * expert output.
        Y = np.zeros_like(X)
        np.add.at(Y, tok[keep], gat[keep, None] * out_buf[exp[keep], pos[keep]])
        return Y, topk, gates, keep.reshape(T, k)

    def forward_dense(self, X, topk, gates, keep):
        """Reference: naive loop over each token's selected experts."""
        Y = np.zeros_like(X)
        for t in range(X.shape[0]):
            for j in range(self.k):
                if keep[t, j]:
                    e = topk[t, j]
                    hdn = np.maximum(X[t] @ self.W1[e], 0.0)
                    Y[t] += gates[t, j] * (hdn @ self.W2[e])
        return Y

def cv_of_load(topk, E):
    load = np.bincount(topk.reshape(-1), minlength=E).astype(float)
    return load.std() / load.mean(), load

def demo():
    rng = np.random.default_rng(0)
    d, h, E, k, T = 32, 64, 8, 2, 512
    # A skewed router offset makes experts 0-1 initially dominate.
    offset = np.array([3.0, 2.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
    X = rng.normal(size=(T, d))

    # ---- (c) parameter accounting -------------------------------------
    m = MoELayer(d, h, E, k, rng, logit_offset=offset)
    total = m.W1.size + m.W2.size + m.Wg.size            # E*2*d*h + d*E
    active = k * 2 * d * h + d * E                       # k experts + router
    assert total == E * 2 * d * h + d * E, "total-param formula mismatch"
    print(f"[params] total={total}  active/token={active}  "
          f"ratio={active/total:.4f} (formula: (k*2dh+dE)/(E*2dh+dE))")

    # ---- (a) dispatch/combine == dense loop ---------------------------
    Y_fast, topk, gates, keep = m.forward_dispatch(X, capacity_factor=1.25)
    Y_ref = m.forward_dense(X, topk, gates, keep)
    err = np.abs(Y_fast - Y_ref).max()
    print(f"[equivalence] max |dispatch/combine - dense loop| = {err:.3e}")
    assert err < 1e-6, "dispatch path diverges from dense reference"
    dropped = (~keep).sum()
    print(f"[capacity] CF=1.25: capacity={int(np.ceil(1.25*T*k/E))}/expert, "
          f"dropped {dropped}/{T*k} assignments ({100*dropped/(T*k):.1f}%)")
    _, _, tk2, g2 = m.route(X)
    C1 = int(np.ceil(1.0 * T * k / E))
    _, _, _, keep1 = m.forward_dispatch(X, capacity_factor=1.0)
    print(f"[capacity] CF=1.00: capacity={C1}/expert, dropped "
          f"{(~keep1).sum()}/{T*k} ({100*(~keep1).sum()/(T*k):.1f}%)")

    # ---- (b) two balancing schemes, same skewed start -----------------
    # alpha here is larger than in real training (~1e-2) because the aux
    # loss is the ONLY loss in this isolated demo, so its scale is free.
    steps, alpha, lr, gamma = 300, 1.0, 2.0, 0.05
    # Scheme 1: auxiliary loss  L = alpha * E * sum_e f_e * Pbar_e,
    # SGD on router (Wg, bg) using dL/dlogits with f treated as constant.
    m1 = MoELayer(d, h, E, k, np.random.default_rng(0), logit_offset=offset)
    # Scheme 2: loss-free bias update  b_e += gamma * sign(mean - load_e).
    m2 = MoELayer(d, h, E, k, np.random.default_rng(0), logit_offset=offset)
    print(f"[balance] CV trajectories over {steps} steps "
          f"(fresh batch of {T} tokens each step):")
    print("  step |  aux-loss CV | loss-free CV")
    for s in range(steps + 1):
        Xb = np.random.default_rng(100 + s).normal(size=(T, d))
        # aux-loss scheme
        logits, probs, tk, _ = m1.route(Xb)
        cv1, load1 = cv_of_load(tk, E)
        f = load1 / load1.sum()                    # fraction routed to e
        dP = alpha * E * f / T                     # dL/dP_te (per token)
        dlog = probs * (dP - (probs * dP).sum(1, keepdims=True))
        m1.Wg -= lr * Xb.T @ dlog
        m1.bg -= lr * dlog.sum(0)
        # loss-free scheme: selection uses logits + b
        _, _, tk2, _ = m2.route(Xb, use_bias=True)
        cv2, load2 = cv_of_load(tk2, E)
        m2.b += gamma * np.sign(load2.mean() - load2)
        if s == 0:
            cv1_0, cv2_0 = cv1, cv2
        if s % 50 == 0:
            print(f"  {s:4d} |    {cv1:8.4f}  |   {cv2:8.4f}")
    assert cv1 < 0.25 * cv1_0, "aux loss failed to balance"
    assert cv2 < 0.25 * cv2_0, "loss-free bias failed to balance"
    print(f"[balance] aux-loss CV {cv1_0:.3f} -> {cv1:.3f}; "
          f"loss-free CV {cv2_0:.3f} -> {cv2:.3f}  (both < 25% of start)")
    print(f"[balance] final loss-free biases b = "
          f"{np.array2string(m2.b, precision=2)}")

    # Cross-check softmax + top-k gate against torch, if present.
    try:
        import torch
        lt = torch.from_numpy(X @ m.Wg + m.bg)
        pt = torch.softmax(lt, dim=-1).numpy()
        _, probs_np, _, _ = m.route(X)
        terr = np.abs(pt - probs_np).max()
        print(f"[torch cross-check] max softmax diff = {terr:.3e}")
        assert terr < 1e-12
    except ImportError:
        print("[skipped: torch not installed]")
    print("All assertions passed.")

if __name__ == "__main__":
    demo()
