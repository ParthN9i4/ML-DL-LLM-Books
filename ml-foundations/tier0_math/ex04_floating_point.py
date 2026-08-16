"""
ex04 — Floating point, precision, and stability.  (Book: Chapter 5)

Low precision is not compression. It is a change to the arithmetic your
theorems assume, and most "mysterious" training instabilities are traceable to
a specific unit in the last place.

The one table worth memorizing:

    format   exp bits  mantissa  max finite   eps (2^-p)   why it exists
    fp32        8        23       3.4e38      1.19e-07     the default
    fp16        5        10       65504       9.77e-04     more precision, tiny range
    bf16        8         7       3.4e38      7.81e-03     fp32's RANGE, less precision
    fp8 e4m3    4         3       448         6.25e-02     inference, and now training

bf16 replaced fp16 for training because of the EXPONENT, not the mantissa. It
has the same dynamic range as fp32, so gradients do not flush to zero and loss
scaling becomes unnecessary. It has *worse* precision than fp16 — that turned
out to matter less than range.

Summation error growth, which decides why optimizer states stay in fp32:

    naive     O(n)      error grows linearly in the number of terms
    pairwise  O(log n)  what numpy's .sum() actually does
    Kahan     O(1)      compensated, error independent of n

To learn: replace each function body with `pass` and reimplement.
"""

import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Anatomy of a float
# ---------------------------------------------------------------------------

def float_bits(x):
    """The IEEE-754 binary32 bit pattern of x, as a 32-character string."""
    # === YOUR CODE HERE ===
    return format(struct.unpack("<I", struct.pack("<f", np.float32(x)))[0], "032b")


def decompose(x):
    """Split a float32 into (sign, biased_exponent, mantissa_bits).

    Layout is 1 sign bit, 8 exponent bits (bias 127), 23 mantissa bits.
    """
    b = float_bits(x)
    # === YOUR CODE HERE ===
    return int(b[0]), int(b[1:9], 2), int(b[9:], 2)


def machine_epsilon(dtype):
    """Smallest eps with 1 + eps != 1 in `dtype`, found by halving.

    Do NOT read np.finfo — that is the reference we check against.
    """
    one = dtype(1.0)
    eps = dtype(1.0)
    # === YOUR CODE HERE ===
    while dtype(one + eps / dtype(2.0)) != one:
        eps = dtype(eps / dtype(2.0))
    return eps


# ---------------------------------------------------------------------------
# 2. Low-precision simulation by masking
# ---------------------------------------------------------------------------

def to_bf16(x):
    """Round-to-nearest-even truncation of float32 to bfloat16, via integer masking.

    bf16 IS fp32 with the low 16 mantissa bits removed, which is exactly why the
    conversion is so cheap in hardware.
    """
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    # === YOUR CODE HERE ===
    # round-to-nearest-even on the bit being dropped
    rounding_bias = ((u >> 16) & 1) + np.uint32(0x7FFF)
    u = (u + rounding_bias) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


def kahan_sum(xs, dtype=np.float32):
    """Compensated summation: carry the lost low-order bits in a running term."""
    total = dtype(0.0)
    c = dtype(0.0)
    # === YOUR CODE HERE ===
    for x in xs:
        y = dtype(dtype(x) - c)
        t = dtype(total + y)
        c = dtype(dtype(t - total) - y)
        total = t
    return total


def naive_sum(xs, dtype=np.float32):
    """Left-to-right accumulation — the one that loses O(n) precision."""
    total = dtype(0.0)
    # === YOUR CODE HERE ===
    for x in xs:
        total = dtype(total + dtype(x))
    return total


# ---------------------------------------------------------------------------
# 3. Stable softmax
# ---------------------------------------------------------------------------

def overflow_threshold(dtype):
    """Largest z with exp(z) finite in `dtype`: log(max_finite)."""
    # === YOUR CODE HERE ===
    return float(np.log(np.finfo(dtype).max))


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("\n--- anatomy ---")
    # 1.0 = sign 0, exponent 127 (bias), mantissa 0. Check by hand.
    check("bits of 1.0f", float_bits(1.0), "00111111100000000000000000000000")
    check("1.0 decomposes to (0, 127, 0)", decompose(1.0), (0, 127, 0))
    check("-2.0 decomposes to (1, 128, 0)", decompose(-2.0), (1, 128, 0))
    # 0.1 is not representable: this is the whole reason 0.1+0.2 != 0.3.
    check("0.1 has a nonzero mantissa (it is not exact)", decompose(0.1)[2] != 0)

    print("\n--- machine epsilon ---")
    for dt, name in ((np.float32, "float32"), (np.float64, "float64"), (np.float16, "float16")):
        got, ref = machine_epsilon(dt), np.finfo(dt).eps
        print(f"      {name:8s} eps = {float(got):.6e}   (np.finfo: {float(ref):.6e})")
        check(f"{name} epsilon matches np.finfo", float(got), float(ref), tol=float(ref) * 1e-6)

    print("\n--- bf16 has fp32's range, fp16 does not ---")
    # THE argument for bf16. A small gradient survives bf16 and dies in fp16.
    tiny = np.float32(1e-8)
    print(f"      1e-8 in fp16 -> {np.float16(tiny)}   (flushed to zero)")
    print(f"      1e-8 in bf16 -> {float(to_bf16(tiny)):.3e}  (survives)")
    check("fp16 flushes 1e-8 to zero", float(np.float16(tiny)) == 0.0)
    check("bf16 preserves 1e-8", float(to_bf16(tiny)) > 0.0)

    big = np.float32(1e30)
    print(f"      1e30 in fp16 -> {np.float16(big)}    (overflows)")
    print(f"      1e30 in bf16 -> {float(to_bf16(big)):.3e}  (fine)")
    check("fp16 overflows at 1e30", not np.isfinite(np.float16(big)))
    check("bf16 does not", np.isfinite(to_bf16(big)))

    # ...and the price: bf16 is much coarser than fp16 where both are in range.
    vals = rng.standard_normal(20000).astype(np.float32)
    err_bf = float(np.max(np.abs(to_bf16(vals) - vals) / np.abs(vals)))
    err_f16 = float(np.max(np.abs(np.float16(vals).astype(np.float32) - vals) / np.abs(vals)))
    print(f"      max relative error: bf16 {err_bf:.2e}, fp16 {err_f16:.2e}")
    check("bf16 relative error is within its 2^-8 bound", err_bf < 2 ** -7)
    check("bf16 is coarser than fp16 (that is the trade)", err_bf > err_f16)

    try:
        import torch
        t = torch.tensor(vals[:512], dtype=torch.float32)
        ref_bf = t.to(torch.bfloat16).to(torch.float32).numpy()
        check("bf16 simulator matches torch.bfloat16 exactly", to_bf16(vals[:512]), ref_bf, tol=0.0)
    except ImportError:
        print("  ....  [skipped: torch not installed] bf16 cross-check against torch")

    print("\n--- summation error growth ---")
    # A hard case: one large value followed by many tiny ones. The addend must
    # be BELOW the ULP of the running total, or nothing goes wrong. At 1e7 the
    # float32 ULP is exactly 1.0 (1e7 sits between 2^23 and 2^24, where the
    # spacing is 2^-23 * 2^23 = 1), so adding 1.0 repeatedly accumulates
    # perfectly. Use 0.1, which is well under half an ULP and rounds away.
    n = 200000
    xs = np.concatenate([[np.float32(1e7)], np.full(n, np.float32(0.1))])
    exact = 1e7 + n * 0.1                 # computed in float64, the truth
    print(f"      ULP of 1e7 in float32 is {np.spacing(np.float32(1e7)):.3f};"
          f" each addend is 0.1, i.e. well below half of it")

    naive = float(naive_sum(xs))
    pair = float(np.asarray(xs, dtype=np.float32).sum())   # numpy uses pairwise
    kahan = float(kahan_sum(xs))
    print(f"      exact    {exact:.1f}")
    print(f"      naive    {naive:.1f}   (error {abs(naive-exact):.1f})")
    print(f"      pairwise {pair:.1f}   (error {abs(pair-exact):.1f})")
    print(f"      Kahan    {kahan:.1f}   (error {abs(kahan-exact):.1f})")

    check("naive float32 summation loses essentially everything",
          abs(naive - exact) > 0.5 * n * 0.1)
    check("pairwise is far better than naive", abs(pair - exact) < abs(naive - exact) / 10)
    check("Kahan is essentially exact", abs(kahan - exact) < 1.0)
    check("Kahan beats naive by orders of magnitude",
          abs(naive - exact) > 1000 * max(abs(kahan - exact), 1e-6))

    print("\n--- softmax overflow, exactly where predicted ---")
    for dt, name in ((np.float32, "float32"), (np.float16, "float16")):
        thr = overflow_threshold(dt)
        with np.errstate(over="ignore"):
            just_under = np.exp(dt(thr * 0.999))
            just_over = np.exp(dt(thr * 1.001))
        print(f"      {name}: exp overflows above z = {thr:.2f}")
        check(f"{name}: exp finite just below the threshold", np.isfinite(just_under))
        check(f"{name}: exp overflows just above it", not np.isfinite(just_over))

    check("fp32 threshold is ~88.7", overflow_threshold(np.float32), 88.7, tol=0.1)
    check("fp16 threshold is ~11.09", overflow_threshold(np.float16), 11.09, tol=0.1)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) Catastrophic cancellation: subtracting nearly equal numbers destroys
    #     every significant digit. This is why variance is computed with the
    #     two-pass or Welford algorithm, never as E[x^2] - E[x]^2.
    # At 1e8 the float32 ULP is 8, so adding 1 does not change the value at all
    # and the subtraction returns 0 instead of 1 — every digit is gone.
    x = np.float32(1e8) + np.float32(1.0)
    y = np.float32(1e8)
    print(f"      (1e8 + 1) - 1e8 in float32 = {float(x - y)}  (want 1.0; ULP here is "
          f"{np.spacing(np.float32(1e8)):.0f})")
    check("adding 1 to 1e8 in float32 is a no-op", float(x - y) == 0.0)
    data = (rng.standard_normal(10000).astype(np.float32) + np.float32(1e4))
    naive_var = float(np.mean(data.astype(np.float32) ** 2) - np.mean(data) ** 2)
    good_var = float(np.var(data.astype(np.float64)))
    print(f"      variance via E[x^2]-E[x]^2 (fp32): {naive_var:.4f}")
    print(f"      variance via the two-pass form   : {good_var:.4f}")
    check("the E[x^2]-E[x]^2 shortcut is badly wrong here",
          abs(naive_var - good_var) > 0.1 * good_var)

    # (b) Addition is not associative in floating point, so reduction ORDER
    #     changes the answer. This is why multi-GPU runs are not bit-reproducible
    #     unless the reduction order is pinned.
    v = rng.standard_normal(4096).astype(np.float32) * np.float32(1e3)
    fwd = float(naive_sum(v))
    bwd = float(naive_sum(v[::-1]))
    print(f"      sum forward {fwd:.6f}, reversed {bwd:.6f}, differ by {abs(fwd-bwd):.3e}")
    check("float addition is not associative", fwd != bwd)

    summary()
