"""Chapter 5 artifact -- bit-exact low precision, error growth, and softmax overflow.

(A) round_to_format(): fp32 -> {fp16, bf16, fp8-E4M3} with correct round-to-nearest-even,
    subnormals and overflow, from integer masking of the fp32 bit pattern alone; checked
    bit-for-bit against torch on 400k values.  (B) summation: naive vs Kahan vs pairwise
    in real fp32 against an exact (fsum) reference, plus accumulator stagnation.
(C) catastrophic cancellation in the one-pass variance formula.  (D) the exact largest
    logit whose exp does not overflow, predicted then measured.  Pure NumPy; torch is an
    optional oracle only.
"""
import numpy as np
from math import fsum, log, ldexp
try:
    import torch
except ImportError:                              # the artifact must run without torch
    torch = None

# fp16/bf16 are IEEE-754: the all-ones exponent field is reserved, so e_max = 2^E-2-bias.
# OCP fp8 E4M3 ("E4M3FN") has no infinities and spends only S.1111.111 on NaN, buying one
# extra exponent value (e_max = 8) and a largest finite value of 448; casts saturate.
FORMATS = {"fp16": dict(E=5, M=10, sat=False),
           "bf16": dict(E=8, M=7, sat=False),
           "fp8_e4m3": dict(E=4, M=3, sat=True)}

def fmt_info(name):
    """Bias, exponent range, largest finite value, epsilon -- all derived, none hardcoded."""
    f = FORMATS[name]
    E, M, sat = f["E"], f["M"], f["sat"]
    bias = 2 ** (E - 1) - 1
    e_max = (2 ** E - 1 - bias) if sat else (2 ** E - 2 - bias)
    top_sig = (2 ** (M + 1) - 2) if sat else (2 ** (M + 1) - 1)   # E4M3 loses S.1111.111
    return dict(E=E, M=M, sat=sat, bias=bias, e_min=1 - bias, e_max=e_max,
                max_finite=ldexp(top_sig, e_max - M), min_normal=ldexp(1.0, 1 - bias),
                min_subnormal=ldexp(1.0, 1 - bias - M),
                eps=ldexp(1.0, -M), u=ldexp(1.0, -M - 1))

def round_to_format(x, name):
    """Round float32 values to `name` (returned as float32), bit-exactly.

    A float32 with biased exponent field ef and mantissa m equals sig * 2^(e-23) with
    sig = m + 2^23, e = ef - 127 for normals and sig = m, e = -126 for subnormals: one
    uniform form.  The target grid near that value has spacing 2^g, g = clip(e,e_min,e_max)
    - M.  So round the integer sig to a multiple of 2^(g-(e-23)) with half-to-even.
    """
    i = fmt_info(name)
    M, e_min, e_max = i["M"], i["e_min"], i["e_max"]
    x = np.asarray(x, dtype=np.float32)
    u32 = x.view(np.uint32).astype(np.int64)
    sign, ef, man = (u32 >> 31) & 1, (u32 >> 23) & 0xFF, u32 & 0x7FFFFF
    sig = np.where(ef == 0, man, man + (1 << 23))          # explicit significand
    e = np.where(ef == 0, -126, ef - 127)                  # value = sig * 2^(e-23)
    g = np.clip(e, e_min, e_max) - M                       # target grid exponent
    sh = np.clip(23 + g - e, 0, 32)                        # bits to discard
    q, rem = sig >> sh, sig & ((np.int64(1) << sh) - 1)
    half = np.where(sh > 0, np.int64(1) << np.maximum(sh - 1, 0), np.int64(0))
    up = (rem > half) | ((rem == half) & (sh > 0) & ((q & 1) == 1))   # half-to-even
    q = q + up.astype(np.int64)
    out = np.ldexp(q.astype(np.float64), g)                # exact: q < 2^(M+2)
    big = i["max_finite"] if i["sat"] else np.inf
    out = np.where((e > e_max) | (out > i["max_finite"]), big, out)
    out = np.where((ef == 255) & (man == 0), big, out)                # +-inf
    out = np.where((ef == 255) & (man != 0), np.nan, out)             # NaN
    return np.where(sign == 1, -out, out).astype(np.float32)

def encode(x, name):
    """Encode one scalar into the format's integer code word (display only)."""
    i = fmt_info(name)
    M, bias = i["M"], i["bias"]
    v = float(round_to_format(np.float32(x), name))
    s, a = (1 if v < 0 else 0), abs(v)
    if a < i["min_normal"]:                        # zero or subnormal: exponent field 0
        ef, mf = 0, int(round(a / ldexp(1.0, i["e_min"] - M)))
    else:
        e = int(np.floor(np.log2(a)))
        ef, mf = e + bias, int(round((a / ldexp(1.0, e) - 1.0) * 2 ** M))
    return (s << (i["E"] + M)) | (ef << M) | mf

def part_a():
    print("(A) format anatomy and bit-exact rounding")
    print(f"    {'format':9s} {'E':>2s} {'M':>2s} {'bias':>5s} {'e_min':>6s} {'e_max':>6s}"
          f" {'max finite':>12s} {'min normal':>11s} {'min subn.':>11s} {'eps=2^-M':>10s}")
    for n in FORMATS:
        i = fmt_info(n)
        print(f"    {n:9s} {i['E']:2d} {i['M']:2d} {i['bias']:5d} {i['e_min']:6d} "
              f"{i['e_max']:6d} {i['max_finite']:12.6g} {i['min_normal']:11.5g} "
              f"{i['min_subnormal']:11.5g} {i['eps']:10.4g}")
    for val, n, w in ((3.125, "fp16", 16), (1.375, "fp8_e4m3", 8)):
        print(f"    hand check: {val} in {n} -> {float(round_to_format(np.float32(val), n))}"
              f"  code = {format(encode(val, n), '0%db' % w)}")
    # adversarial values: ties, overflow boundaries, subnormal boundaries, specials
    special = np.array([0.0, -0.0, 1.0, 1.0 + 2 ** -11, 65504.0, 65519.0, 65520.0, 65536.0,
                        448.0, 464.0, 464.5, 2 ** -24, 2 ** -25, 2 ** -26, 2 ** -9,
                        2 ** -10, 0.1, 1e-40, 3.4e38, np.inf, -np.inf, np.nan], np.float32)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2 ** 32, size=300_000, dtype=np.uint64).astype(np.uint32)
    logu = (rng.uniform(-45, 40, size=100_000) * np.log(10)).astype(np.float32)
    with np.errstate(over="ignore"):             # deliberately generate overflowing values
        mags = (np.exp(logu) * rng.choice([-1.0, 1.0], 100_000)).astype(np.float32)
    x = np.concatenate([special, bits.view(np.float32), mags])
    if torch is None:
        print("    [skipped: torch not installed] -- no oracle for the cross-check")
        return
    tmap = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8_e4m3": torch.float8_e4m3fn}
    xt = torch.from_numpy(x)
    for n, dt in tmap.items():
        mine = round_to_format(x, n)
        ref = xt.to(dt).to(torch.float32).numpy()
        same = (mine == ref) | (np.isnan(mine) & np.isnan(ref))
        nbad = int((~same).sum())
        print(f"    {n:9s} vs torch on {x.size:,} values: mismatches = {nbad}"
              f"   (fraction finite after the cast: {np.isfinite(ref).mean():.4f})")
        assert nbad == 0, (n, x[~same][:5], mine[~same][:5], ref[~same][:5])
    print("    all three formats reproduce torch bit-for-bit")

def kahan_sum32(v):
    """Compensated (Kahan) summation, genuinely in float32."""
    s = c = np.float32(0.0)
    for xi in v:
        y = xi - c
        t = s + y
        c = (t - s) - y              # the part of y that did not fit into t
        s = t
    return float(s)

def pairwise_sum32(v):
    """Explicit binary-tree summation in float32 (length must be a power of two)."""
    v = v.copy()
    while v.size > 1:
        v = (v[0::2] + v[1::2]).astype(np.float32)
    return float(v[0])

def part_b():
    print("\n(B) summation: naive (left-to-right) vs Kahan vs pairwise, all in fp32")
    u = 2.0 ** -24
    print(f"    fp32 unit roundoff u = 2^-24 = {u:.4e}; bounds: naive (n-1)u, "
          f"pairwise log2(n)u, Kahan 2u")
    rng = np.random.default_rng(0)
    for label, gen in (("x_i = 0.1 (systematic drift)", lambda n: np.full(n, 0.1, np.float32)),
                       ("x_i ~ U(0,1) (random walk)", lambda n: rng.random(n, dtype=np.float32))):
        print(f"    {label}\n      {'n':>9s} {'naive rel.err':>14s} {'/(n-1)u':>9s} "
              f"{'pairwise':>12s} {'/log2(n)u':>10s} {'Kahan':>12s} {'/2u':>7s}")
        for k in (10, 14, 17, 20):
            n = 2 ** k
            v = gen(n)
            exact = fsum(float(t) for t in v)
            naive = float(np.cumsum(v, dtype=np.float32)[-1])   # cumsum IS sequential
            pw, kh = pairwise_sum32(v), kahan_sum32(v)
            en, ep, ek = (abs(naive - exact) / exact, abs(pw - exact) / exact,
                          abs(kh - exact) / exact)
            print(f"      {n:9d} {en:14.4e} {en/((n-1)*u):9.4f} {ep:12.4e} "
                  f"{ep/(k*u):10.4f} {ek:12.4e} {ek/(2*u):7.4f}")
            assert en <= (n - 1) * u and ep <= k * u + 1e-12 and ek <= 2 * u + 1e-12
    n = 2 ** 20
    v = np.full(n, 0.1, np.float32)
    print(f"    2^20 copies of 0.1: naive fp32 = {float(np.cumsum(v, dtype=np.float32)[-1]):.4f}"
          f", exact = {fsum(float(t) for t in v):.4f}, Kahan = {kahan_sum32(v):.4f}")
    # stagnation: once the addend drops below half an ulp of the running total, the sum
    # stops moving.  Adding 1.0 repeatedly, that happens at 2^(M+1) exactly (tie-to-even).
    print("    accumulating 1.0 repeatedly in a low-precision accumulator (8192 adds):")
    for name in ("fp16", "bf16"):
        s, stalled = np.float32(0.0), None
        for k in range(8192):
            t = round_to_format(np.float32(float(s) + 1.0), name)
            if float(t) == float(s) and stalled is None:
                stalled = (k, float(s))
            s = np.float32(t)
        pred = ldexp(1.0, fmt_info(name)["M"] + 1)
        print(f"      {name}: final sum = {float(s):.0f}, first stalled after {stalled[0]}"
              f" adds at {stalled[1]:.0f}; predicted 2^(M+1) = {pred:.0f}")
        assert float(s) == pred and stalled[1] == pred

def part_c():
    print("\n(C) catastrophic cancellation: one-pass variance E[x^2]-E[x]^2 in fp32")
    print("    (every sum below is pairwise, so summation error is NOT the culprit)")
    rng = np.random.default_rng(0)
    n = 2 ** 17
    for mu in (0.0, 1e2, 1e3, 1e4):
        x = (rng.standard_normal(n) + mu).astype(np.float32)
        exact = float(np.var(x.astype(np.float64)))              # truth for THIS fp32 data
        m1 = pairwise_sum32(x) / n
        m2 = pairwise_sum32((x * x).astype(np.float32)) / n
        one_pass = m2 - m1 * m1
        d = (x - np.float32(m1)).astype(np.float32)
        two_pass = pairwise_sum32((d * d).astype(np.float32)) / n
        cond = (abs(m2) + m1 * m1) / abs(one_pass)               # condition of the subtraction
        print(f"    mean={mu:6.0e}  exact={exact:.6f}  one-pass={one_pass:11.6f} "
              f"(rel.err {abs(one_pass-exact)/exact:8.2e})  two-pass={two_pass:.6f} "
              f"(rel.err {abs(two_pass-exact)/exact:8.2e})  cond={cond:.3e}")
    assert abs(one_pass - exact) / exact > 1.0 and abs(two_pass - exact) / exact < 1e-3
    print("    the two-pass form loses nothing; the one-pass form loses roughly two "
          "digits per decade of mean/sigma")

def decode(code, name):
    """Integer code word -> real value (finite codes only)."""
    i = fmt_info(name)
    E, M = i["E"], i["M"]
    s, ef, mf = (code >> (E + M)) & 1, (code >> M) & (2 ** E - 1), code & (2 ** M - 1)
    v = ldexp(mf, i["e_min"] - M) if ef == 0 else ldexp(2 ** M + mf, ef - i["bias"] - M)
    return -v if s else v

def part_d():
    print("\n(D) softmax overflow: predicted vs measured largest safe logit")
    # RNE sends a value to infinity once it reaches the midpoint between the largest
    # finite number and 2^(e_max+1), i.e. 2^(e_max+1) * (1 - 2^-(M+2)).
    ovf32 = ldexp(1.0, 128) * (1.0 - 2.0 ** -25)
    z32 = log(ovf32)
    pred = np.float32(z32)
    while float(pred) >= z32:                       # step down onto the fp32 grid
        pred = np.nextafter(pred, np.float32(0.0))
    meas = pred
    with np.errstate(over="ignore"):
        while np.isfinite(np.exp(np.nextafter(meas, np.float32(1e9)))):
            meas = np.nextafter(meas, np.float32(1e9))
    print(f"    fp32: first value that rounds to inf = {ovf32:.10e}, ln = {z32:.10f}\n"
          f"          predicted last safe logit {float(pred):.9f}   measured "
          f"{float(meas):.9f}   exp = {float(np.exp(meas)):.6e}")
    assert float(pred) == float(meas), (pred, meas)
    # fp16: enumerate the entire finite non-negative fp16 grid, 0x0000..0x7BFF
    i16 = fmt_info("fp16")
    ovf16 = ldexp(1.0, i16["e_max"] + 1) * (1.0 - 2.0 ** (-(i16["M"] + 2)))
    z16 = log(ovf16)
    vals = np.array([decode(c, "fp16") for c in range(0x7C00)], dtype=np.float64)
    with np.errstate(over="ignore"):                # exp evaluated exactly, rounded once
        ex = round_to_format(np.exp(vals).astype(np.float32), "fp16")
    pred16, meas16 = vals[vals < z16].max(), vals[np.isfinite(ex)].max()
    print(f"    fp16: first value that rounds to inf = {ovf16:.1f}, ln = {z16:.10f}\n"
          f"          predicted last safe logit {pred16:.9f}   measured {meas16:.9f}"
          f"   exp = {float(round_to_format(np.float32(np.exp(meas16)), 'fp16')):.1f}")
    assert pred16 == meas16, (pred16, meas16)
    # the softmax itself, on logits that straddle the fp32 threshold
    rng = np.random.default_rng(0)
    z = (rng.standard_normal(8).astype(np.float32) * 2.0 + 90.0).astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(z) / np.exp(z).sum()
    m = z.max()
    e = np.exp(z - m)
    stable = e / e.sum()
    lse = float(m + np.log(e.sum()))
    lse_exact = float(np.log(np.exp(z.astype(np.float64)).sum()))   # fp64 has the range
    print(f"    8 logits ~ N(90, 2^2) in fp32: max = {float(m):.4f} (above "
          f"{float(pred):.4f}), naive softmax = {naive[:3]}\n"
          f"          stable softmax = {stable[:3]} sums to {float(stable.sum()):.8f}\n"
          f"          logsumexp shifted = {lse:.8f}, fp64 reference = {lse_exact:.8f}, "
          f"abs diff = {abs(lse - lse_exact):.3e}")
    assert not np.isfinite(naive).all() and np.isfinite(stable).all()
    assert abs(lse - lse_exact) < 1e-4
    z16v = round_to_format(np.float32(12.0), "fp16")
    e16 = round_to_format(np.exp(np.float32(z16v)), "fp16")
    print(f"    fp16: one logit of {float(z16v):.1f} already gives exp = {float(e16)} "
          f"(fp16 max is {i16['max_finite']:.0f}); shifted, it is exp(0) = 1")
    print("    predicted overflow thresholds matched the measured ones exactly, "
          "in both formats")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
    print("\nall assertions passed")
