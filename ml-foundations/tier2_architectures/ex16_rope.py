"""
ex16 — Rotary position embedding (RoPE), two ways.  (Book: Chapter 19)

RoPE encodes position by ROTATING each (even, odd) coordinate pair of a query
or key by an angle proportional to its position:

    pair j of x at position m  ->  rotate by  m * theta_j,
    theta_j = base^(-2j/d),  base = 10000 (classically; much larger for
    long-context models — Llama 3 uses 500000).

Why this won over additive position embeddings, provably and checkably:

    <RoPE(q, m), RoPE(k, n)> depends on (m - n) ONLY.

A rotation is orthogonal, and composing a rotation by m*theta with the inverse
of one by n*theta leaves a rotation by (m-n)*theta inside the inner product.
Position becomes RELATIVE displacement, exactly, with no learned table and no
sequence-length cap baked into the parameters.

Two implementations must agree:
  * complex form: view pairs as complex numbers, multiply by e^{i m theta_j}
  * real form:    explicit 2x2 rotation on interleaved pairs

Context extension is then a statement about wavelengths. Dimension j has
wavelength 2*pi/theta_j; dimensions whose wavelength exceeds the trained
context have never completed a full cycle and are the ones extension methods
(position interpolation, NTK-aware, YaRN) treat specially.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


def rope_freqs(d, base=10000.0):
    """theta_j = base^(-2j/d) for j = 0..d/2-1."""
    # === YOUR CODE HERE ===
    return base ** (-2.0 * np.arange(d // 2) / d)


def rope_complex(x, m, base=10000.0):
    """RoPE via complex multiplication.

    x: (..., d) with d even. Views (x[2j], x[2j+1]) as x[2j] + i*x[2j+1],
    multiplies by exp(i * m * theta_j), returns to real interleaved layout.
    `m` may be a scalar or an array broadcastable over the leading axes.
    """
    d = x.shape[-1]
    # === YOUR CODE HERE ===
    z = x[..., 0::2] + 1j * x[..., 1::2]                  # (..., d/2)
    angles = np.multiply.outer(np.asarray(m, dtype=np.float64), rope_freqs(d, base))
    z = z * np.exp(1j * angles)
    out = np.empty_like(x, dtype=np.float64)
    out[..., 0::2] = z.real
    out[..., 1::2] = z.imag
    return out


def rope_real(x, m, base=10000.0):
    """RoPE via explicit 2x2 rotations — no complex arithmetic.

    [x_even', x_odd'] = [cos*x_even - sin*x_odd,  sin*x_even + cos*x_odd]
    """
    d = x.shape[-1]
    # === YOUR CODE HERE ===
    angles = np.multiply.outer(np.asarray(m, dtype=np.float64), rope_freqs(d, base))
    c, s = np.cos(angles), np.sin(angles)
    xe, xo = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x, dtype=np.float64)
    out[..., 0::2] = c * xe - s * xo
    out[..., 1::2] = s * xe + c * xo
    return out


def wavelengths(d, base=10000.0):
    """Per-pair wavelength in positions: lambda_j = 2*pi / theta_j."""
    # === YOUR CODE HERE ===
    return 2.0 * np.pi / rope_freqs(d, base)


def position_interpolation(m, factor):
    """PI (Chen et al.): compress ALL positions uniformly, m -> m / factor.

    Every dimension slows down equally — including the fast ones that encode
    local order, which is why PI degrades short-range precision.
    """
    # === YOUR CODE HERE ===
    return np.asarray(m, dtype=np.float64) / factor


def ntk_aware_base(d, base, factor):
    """NTK-aware scaling: raise the base so the LOWEST frequency stretches by
    `factor` while the highest is nearly untouched.

    base' = base * factor^(d/(d-2)). High-frequency dims (j=0) keep theta ~ 1;
    the last pair's wavelength grows by ~factor.
    """
    # === YOUR CODE HERE ===
    return base * factor ** (d / (d - 2.0))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = 64

    print("\n--- two implementations agree ---")
    x = rng.standard_normal((8, d))
    ms = np.arange(8, dtype=float)
    check("complex form == real 2x2 form", rope_complex(x, ms), rope_real(x, ms), tol=1e-12)
    check("position 0 is the identity", rope_complex(x, 0.0), x, tol=1e-15)

    print("\n--- rotations preserve norms ---")
    check("||RoPE(x, m)|| == ||x|| (orthogonality)",
          np.linalg.norm(rope_complex(x, 1234.0), axis=-1),
          np.linalg.norm(x, axis=-1), tol=1e-10)

    print("\n--- THE property: inner products depend on m - n only ---")
    q = rng.standard_normal(d)
    k = rng.standard_normal(d)
    # Same displacement, wildly different absolute positions:
    pairs = [(3, 1), (103, 101), (5003, 5001), (100002, 100000)]
    vals = [float(rope_complex(q, m) @ rope_complex(k, n)) for m, n in pairs]
    print(f"      <q_m, k_n> at displacement 2, positions {[p[0] for p in pairs]}:")
    print(f"      {[f'{v:.10f}' for v in vals]}")
    check("identical at every absolute position", vals, [vals[0]] * len(vals), tol=1e-8)
    # And different displacements give different values (position is not erased).
    v_disp5 = float(rope_complex(q, 6.0) @ rope_complex(k, 1.0))
    check("different displacement, different inner product", abs(vals[0] - v_disp5) > 1e-6)
    # Batch version over random position pairs with matched differences.
    ok = True
    for _ in range(200):
        delta = rng.integers(0, 5000)
        a, b = (x0 := rng.integers(0, 100000)), x0 + delta
        c, e = (x1 := rng.integers(0, 100000)), x1 + delta
        va = float(rope_complex(q, float(b)) @ rope_complex(k, float(a)))
        vb = float(rope_complex(q, float(e)) @ rope_complex(k, float(c)))
        ok &= abs(va - vb) < 1e-7 * max(1.0, abs(va))
    check("relative-position property holds over 200 random matched pairs", ok)

    print("\n--- wavelength table ---")
    base = 10000.0
    lam = wavelengths(d, base)
    trained_ctx = 4096
    n_long = int(np.sum(lam > trained_ctx))
    print(f"      d={d}, base={base:.0f}: wavelengths span {lam[0]:.2f} to {lam[-1]:,.0f} positions")
    print(f"      {n_long} of {d//2} pairs have wavelength > trained context {trained_ctx}")
    check("fastest pair has wavelength 2*pi", float(lam[0]), 2 * np.pi, tol=1e-12)
    check("wavelengths increase monotonically", bool(np.all(np.diff(lam) > 0)))
    check("some dimensions never complete a cycle in-context", n_long >= 5)

    print("\n--- context extension ---")
    factor = 8.0
    # PI: after compression, position L*factor lands where L used to.
    m_ext = position_interpolation(np.array([trained_ctx * factor]), factor)
    check("PI maps the extended boundary onto the trained boundary",
          float(m_ext[0]), float(trained_ctx), tol=1e-9)
    # NTK-aware: last pair slows by ~factor, first pair barely moves.
    base2 = ntk_aware_base(d, base, factor)
    f_old, f_new = rope_freqs(d, base), rope_freqs(d, base2)
    slow_last = f_old[-1] / f_new[-1]
    slow_first = f_old[0] / f_new[0]
    print(f"      NTK-aware base {base:.0f} -> {base2:,.0f}: "
          f"last pair slows {slow_last:.2f}x, first pair {slow_first:.4f}x")
    check("NTK-aware stretches the slowest dimension by ~factor",
          slow_last, factor, tol=0.35 * factor)
    check("...while leaving the fastest nearly untouched", slow_first < 1.15)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) Additive learned/sinusoidal embeddings do NOT have the relative
    #     property: <q + p_m, k + p_n> carries absolute-position cross terms.
    def sinusoidal(m, d_, base_=10000.0):
        pos = np.zeros(d_)
        f = base_ ** (-2.0 * np.arange(d_ // 2) / d_)
        pos[0::2] = np.sin(m * f)
        pos[1::2] = np.cos(m * f)
        return pos

    add_vals = [float((q + sinusoidal(m, d)) @ (k + sinusoidal(n, d))) for m, n in pairs]
    spread = max(add_vals) - min(add_vals)
    print(f"      additive embeddings at fixed displacement: spread = {spread:.4f} "
          f"(RoPE's was < 1e-8)")
    check("additive embeddings leak absolute position", spread > 1e-3)

    # (b) The pairing convention bug: rotating (x[j], x[j + d/2]) pairs (the
    #     half-split layout used by HF/Llama) while your table assumes
    #     interleaved (x[2j], x[2j+1]) gives DIFFERENT vectors. Both are valid
    #     conventions internally — mixing them across a codebase is the bug.
    def rope_halfsplit(x_, m, base_=10000.0):
        d_ = x_.shape[-1]
        f = base_ ** (-2.0 * np.arange(d_ // 2) / d_)
        ang = m * f
        c, s = np.cos(ang), np.sin(ang)
        x1, x2 = x_[..., : d_ // 2], x_[..., d_ // 2:]
        return np.concatenate([c * x1 - s * x2, s * x1 + c * x2], axis=-1)

    a_int = rope_complex(q, 7.0)
    a_half = rope_halfsplit(q, 7.0)
    print(f"      interleaved vs half-split at m=7: max diff {np.abs(a_int - a_half).max():.3f}")
    check("the two pairing conventions produce different vectors",
          float(np.abs(a_int - a_half).max()) > 0.1)
    # Consistently used, EITHER convention has the relative property...
    h_vals = [float(rope_halfsplit(q, m) @ rope_halfsplit(k, n)) for m, n in pairs]
    check("half-split, used consistently, is also position-relative",
          h_vals, [h_vals[0]] * len(pairs), tol=1e-8)
    # ...but mixing them (queries interleaved, keys half-split) destroys it.
    mix = [float(rope_complex(q, m) @ rope_halfsplit(k, n)) for m, n in pairs]
    check("MIXING conventions destroys the relative property",
          max(mix) - min(mix) > 1e-3)

    summary()
