"""Artifact 26.1 -- LoRA from scratch + a normal-float 4-bit quantizer.

Verifies four claims about LoRA (Hu et al., 2021) and one about NF4
(Dettmers et al., QLoRA, 2023), all in pure NumPy (scipy only for the
Gaussian quantile function; torch only as an optional gradient cross-check):

  (a) at init (B = 0) the wrapped layer is BIT-identical to the base layer;
  (b) merging W + (alpha/r) B A reproduces the two-path output to ~1e-12;
  (c) at full rank r = min(d_in, d_out), LoRA reaches the same loss as full
      fine-tuning on a convex regression problem (same global optimum);
  (d) Adam state memory scales as 4*r*d (two moments for A and B) versus
      2*d_in*d_out for full fine-tuning -- verified by counting floats.

The NF4 codebook is rebuilt from the QLoRA recipe (quantiles of N(0,1) with
offset 0.5*(1/32 + 1/30)) and checked against the bitsandbytes constants,
then compared with uniform int4 on blockwise-absmax round-trip error.
"""
import numpy as np
from scipy.stats import norm

# ----------------------------------------------------------------------
# LoRA wrapper around a frozen linear layer  y = W x + b
# ----------------------------------------------------------------------
class LoRALinear:
    def __init__(self, W, b, r, alpha, rng):
        self.W, self.b = W, b            # frozen base weights
        self.r, self.s = r, alpha / r    # LoRA scale s = alpha / r
        d_out, d_in = W.shape
        # Kaiming-style A, zero B: the update s*B@A starts exactly at 0.
        self.A = rng.normal(0.0, 1.0 / np.sqrt(d_in), (r, d_in))
        self.B = np.zeros((d_out, r))

    def forward(self, x):                # two-path: base + low-rank update
        return (x @ self.W.T + self.b) + self.s * (x @ self.A.T) @ self.B.T

    def merged(self):                    # fold the update into the base
        return self.W + self.s * (self.B @ self.A)

    def grads(self, x, dY):              # dY = dL/d(output), shape (m, d_out)
        dW = dY.T @ x                    # what full FT would use
        return self.s * (dW @ self.A.T), self.s * (self.B.T @ dW)  # dB, dA


class Adam:
    """Minimal Adam; keeps two moment arrays per parameter (state counting)."""
    def __init__(self, shapes, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0

    def state_floats(self):
        return sum(a.size for a in self.m) + sum(a.size for a in self.v)

    def step(self, params, grads):
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ----------------------------------------------------------------------
# NF4: 16 levels at the quantiles of N(0,1), with an exact zero
# ----------------------------------------------------------------------
def nf4_codebook():
    """QLoRA appendix recipe: asymmetric quantile grid so that 0 is a level.
    8 positive levels, 7 negative levels, plus exact 0 -> 16 = 2^4 levels."""
    offset = 0.5 * (1 / 32 + 1 / 30)                    # tail probability
    pos = norm.ppf(np.linspace(1 - offset, 0.5, 9)[:-1])  # 8 positive quantiles
    neg = norm.ppf(np.linspace(offset, 0.5, 8)[:-1])       # 7 negative quantiles
    q = np.sort(np.concatenate([neg, [0.0], pos]))
    return q / np.abs(q).max()                          # normalize to [-1, 1]


def quantize_blockwise(w, code, block=64):
    """Absmax blockwise quantization: per-block scale, nearest codebook level."""
    wb = w.reshape(-1, block)
    scale = np.abs(wb).max(axis=1, keepdims=True)       # one fp32 per block
    idx = np.abs(wb[:, :, None] / scale[:, :, None] - code).argmin(axis=2)
    return (code[idx] * scale).reshape(-1)              # dequantized round trip


# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d_in, d_out, m = 10, 8, 200
    W0 = rng.normal(0, 0.3, (d_out, d_in)); b0 = rng.normal(0, 0.1, d_out)
    X = rng.normal(0, 1, (m, d_in))

    # --- (a) bit-identity at initialization --------------------------------
    lora = LoRALinear(W0, b0, r=4, alpha=8, rng=rng)
    base_out, lora_out = X @ W0.T + b0, lora.forward(X)
    assert base_out.tobytes() == lora_out.tobytes(), "init not bit-identical"
    print("(a) B=0 output bit-identical to base layer: PASS")

    # --- (b) merged == unmerged -------------------------------------------
    lora.B = rng.normal(0, 0.5, lora.B.shape)           # make the update nonzero
    err = np.abs(X @ lora.merged().T + b0 - lora.forward(X)).max()
    assert err < 1e-12, err
    print(f"(b) merge residual |merged - two-path|_max = {err:.2e}  (< 1e-12)")

    # --- (c) full-rank LoRA reaches the full fine-tuning loss --------------
    Wstar = rng.normal(0, 0.5, (d_out, d_in))           # target weights
    Y = X @ Wstar.T + b0 + rng.normal(0, 0.01, (m, d_out))
    loss = lambda P: 0.5 * np.mean((X @ P.T + b0 - Y) ** 2)

    Wf = W0.copy()                                      # full fine-tuning
    opt_f = Adam([Wf.shape])
    for t in range(8000):
        opt_f.lr = 0.02 if t < 4000 else 0.002          # decay to kill Adam noise
        dY = (X @ Wf.T + b0 - Y) / (m * d_out)
        opt_f.step([Wf], [dY.T @ X])

    r_full = min(d_in, d_out)                           # r = 8 = full rank
    lr8 = LoRALinear(W0, b0, r=r_full, alpha=r_full, rng=rng)  # s = 1
    opt_l = Adam([lr8.B.shape, lr8.A.shape])
    for t in range(8000):
        opt_l.lr = 0.02 if t < 4000 else 0.002
        dY = (lr8.forward(X) - Y) / (m * d_out)
        dB, dA = lr8.grads(X, dY)
        opt_l.step([lr8.B, lr8.A], [dB, dA])

    Lf, Ll = loss(Wf), loss(lr8.merged())
    rel = abs(Ll - Lf) / Lf
    assert rel < 1e-3, (Lf, Ll)
    print(f"(c) full FT loss {Lf:.8f} vs full-rank LoRA loss {Ll:.8f}"
          f"  (rel diff {rel:.2e} < 1e-3)")

    # --- (d) optimizer-state memory: 4rd vs 2*d_in*d_out -------------------
    nf, nl = opt_f.state_floats(), opt_l.state_floats()
    assert nf == 2 * d_in * d_out and nl == 2 * r_full * (d_in + d_out)
    d, r = 4096, 16                                     # a realistic layer
    print(f"(d) Adam state floats: full={nf} (=2*d_in*d_out), "
          f"LoRA={nl} (=2r(d_in+d_out))")
    print(f"    at d={d}, r={r}: full={2*d*d:,} vs LoRA={4*r*d:,} floats "
          f"-> {2*d*d/(4*r*d):.0f}x smaller")

    # --- optional torch cross-check of the LoRA gradients ------------------
    try:
        import torch
        tX = torch.tensor(X); tA = torch.tensor(lr8.A, requires_grad=True)
        tB = torch.tensor(lr8.B, requires_grad=True)
        out = tX @ torch.tensor(W0).T + torch.tensor(b0) \
            + lr8.s * (tX @ tA.T) @ tB.T
        (0.5 * ((out - torch.tensor(Y)) ** 2).mean()).backward()
        dY = (lr8.forward(X) - Y) / (m * d_out)
        dB, dA = lr8.grads(X, dY)
        ga = np.abs(tA.grad.numpy() - dA).max()
        gb = np.abs(tB.grad.numpy() - dB).max()
        assert max(ga, gb) < 1e-10
        print(f"    torch autograd cross-check: max grad diff {max(ga, gb):.2e}")
    except ImportError:
        print("    [skipped: torch not installed]")

    # --- NF4 vs uniform int4 round-trip error ------------------------------
    code = nf4_codebook()
    # spot-check against the bitsandbytes NF4 constants
    ref = {1: -0.6961928009986877, 7: 0.0, 14: 0.7229568362236023}
    for i, v in ref.items():
        assert abs(code[i] - v) < 1e-6, (i, code[i], v)
    print("NF4 codebook matches bitsandbytes constants to <1e-6: PASS")

    w = rng.normal(0, 1, 1 << 16)                       # 65,536 Gaussian weights
    int4 = np.linspace(-1, 1, 16)                       # uniform 16-level grid
    rms = lambda cb: np.sqrt(np.mean((quantize_blockwise(w, cb) - w) ** 2))
    e_nf4, e_int4 = rms(code), rms(int4)
    print(f"round-trip RMSE on N(0,1) weights (block=64): "
          f"NF4={e_nf4:.5f}  int4={e_int4:.5f}  "
          f"(int4/NF4 = {e_int4/e_nf4:.2f}x)")
    assert e_nf4 < e_int4, "NF4 should beat uniform int4 on Gaussian weights"
    # memory arithmetic incl. double quantization of the per-block scales
    print(f"bits/param: fp32 scale 4+32/64 = {4+32/64:.3f}; "
          f"double-quantized 4+8/64+32/(64*256) = {4+8/64+32/(64*256):.3f}")
