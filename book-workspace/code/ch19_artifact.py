"""
Artifact 19.1 -- Rotary position embedding two independent ways, plus PI / NTK-aware /
YaRN reduced to one question: which frequency bands get their angular velocity divided
by the scale factor s?  Core is pure NumPy float64; torch is an optional cross-check.

Frequencies: theta_i = base ** (-2i/d_h), i = 0..d_h/2-1.  theta_0 = 1 (wavelength
2*pi tokens, fastest); theta_{d_h/2-1} ~ 1/base (slowest).  Wavelength lam_i = 2*pi/theta_i.
"Interleaved" pairing (RoFormer): coord 2i with 2i+1.  "Half-split" (GPT-NeoX/Llama):
coord j with j+d_h/2.  Same operator up to a fixed permutation -- checked below.
"""
import numpy as np
try:
    import torch
except ImportError:
    torch = None


# --- frequencies -------------------------------------------------------------------
def rope_freqs(d_h, base=10000.0):
    """theta_i = base**(-2i/d_h), radians per token, shape (d_h//2,)."""
    return base ** (-2.0 * np.arange(d_h // 2, dtype=np.float64) / d_h)


def wavelengths(theta):
    """lam_i = 2*pi/theta_i -- tokens per full rotation of coordinate pair i."""
    return 2.0 * np.pi / theta


# --- RoPE, implementation A: complex multiply --------------------------------------
def rope_complex(x, pos, theta):
    """z_i = x[2i] + j*x[2i+1]; multiply by exp(j*m*theta_i). x:(...,d_h) pos:(...)"""
    z = (x[..., 0::2] + 1j * x[..., 1::2]) * np.exp(1j * (pos[..., None] * theta))
    out = np.empty_like(x)
    out[..., 0::2], out[..., 1::2] = z.real, z.imag
    return out


# --- RoPE, implementation B: real 2x2 blocks ---------------------------------------
def rope_real_pairs(x, pos, theta):
    """Explicit [[c,-s],[s,c]] on each interleaved pair. No complex arithmetic."""
    ang = pos[..., None] * theta
    c, s = np.cos(ang), np.sin(ang)
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2], out[..., 1::2] = a * c - b * s, a * s + b * c
    return out


def rope_half_split(x, pos, theta):
    """GPT-NeoX / Llama "rotate_half" layout: coordinate j pairs with j + d_h/2."""
    h = x.shape[-1] // 2
    ang = pos[..., None] * theta
    c, s = np.cos(ang), np.sin(ang)
    a, b = x[..., :h], x[..., h:]
    return np.concatenate([a * c - b * s, a * s + b * c], axis=-1)


def perm(x):
    """P: interleaved coord 2i -> j=i, coord 2i+1 -> j=i+d_h/2."""
    return np.concatenate([x[..., 0::2], x[..., 1::2]], axis=-1)


# --- context extension: each returns a modified frequency vector -------------------
def freqs_pi(theta, s):
    """Position interpolation (Chen et al. 2023): m -> m/s, every theta divided by s."""
    return theta / s


def freqs_ntk(d_h, base, s):
    """NTK-aware: base' = base * s**(d_h/(d_h-2)). Fastest pair fixed, slowest /s."""
    return rope_freqs(d_h, base * s ** (d_h / (d_h - 2.0)))


def yarn_ramp(r, alpha=1.0, beta=32.0):
    """gamma(r) = 0 for r<alpha, 1 for r>beta, linear between. r = L/lambda."""
    return np.clip((r - alpha) / (beta - alpha), 0.0, 1.0)


def freqs_yarn(theta, s, L, alpha=1.0, beta=32.0):
    """
    YaRN "NTK-by-parts": theta'_i = (1-g_i)*theta_i/s + g_i*theta_i, g_i = ramp(L/lam_i).
    r < alpha (pair never completes a rotation in training) -> fully interpolated.
    r > beta  (pair completes >= beta rotations)            -> left exactly alone.
    """
    g = yarn_ramp(L / wavelengths(theta), alpha, beta)
    return (1.0 - g) * (theta / s) + g * theta


def yarn_logit_scale(s):
    """YaRN attention temperature: logits multiplied by sqrt(1/t) = 0.1*ln(s) + 1."""
    return 0.1 * np.log(s) + 1.0


# --- checks -------------------------------------------------------------------------
def check_implementations(rng, d_h=128, base=10000.0):
    theta = rope_freqs(d_h, base)
    x = rng.standard_normal((512, d_h))
    pos = rng.integers(0, 200000, size=512).astype(np.float64)
    a, b = rope_complex(x, pos, theta), rope_real_pairs(x, pos, theta)
    print(f"  complex vs real-pair rotation, max |diff| = {np.max(np.abs(a - b)):.3e}")
    assert np.max(np.abs(a - b)) < 1e-7
    dn = np.max(np.abs(np.linalg.norm(a, axis=-1) - np.linalg.norm(x, axis=-1)))
    print(f"  ||R_m q|| - ||q||, max |diff|             = {dn:.3e}")
    assert dn < 1e-9
    c = rope_half_split(perm(x), pos, theta)
    print(f"  half-split(Px) vs P(interleaved x)        = {np.max(np.abs(c-perm(a))):.3e}")
    assert np.max(np.abs(c - perm(a))) < 1e-7
    return theta


def check_relative(rng, theta, d_h=128, n_trials=4000):
    """<R_m q, R_n k> must depend on m-n only: fix (q,k,delta), vary absolute (m,n)."""
    offsets = np.array([0, 1, 7, 113, 4096, 65536, 1_000_000], dtype=np.float64)
    worst = 0.0
    for _ in range(n_trials):
        q, k = rng.standard_normal(d_h), rng.standard_normal(d_h)
        delta = float(rng.integers(-8192, 8192))
        m = offsets + max(0.0, -delta)
        qm = rope_complex(np.broadcast_to(q, (m.size, d_h)), m, theta)
        kn = rope_complex(np.broadcast_to(k, (m.size, d_h)), m - delta, theta)
        v = np.einsum("ij,ij->i", qm, kn)
        worst = max(worst, float(v.max() - v.min()))
    print(f"  {n_trials} trials x {offsets.size} absolute positions (up to 1e6)")
    print(f"  max spread of <R_m q, R_n k> at fixed m-n = {worst:.3e}")
    assert worst < 1e-8, worst
    # The same test in float32: m*theta_i is formed before sin/cos, so a large absolute
    # position eats the low bits of the phase and the invariance degrades.
    for dt, tag in [(np.float64, "float64"), (np.float32, "float32")]:
        th = theta.astype(dt)
        q, k = (rng.standard_normal(d_h).astype(dt) for _ in range(2))
        m = np.array([0.0, 4096.0, 131072.0, 1048576.0], dtype=dt)
        qm = rope_real_pairs(np.broadcast_to(q, (4, d_h)).copy(), m, th)
        kn = rope_real_pairs(np.broadcast_to(k, (4, d_h)).copy(), m - dt(1000.0), th)
        v = np.einsum("ij,ij->i", qm, kn).astype(np.float64)
        print(f"  {tag} spread at m-n=1000, m in (0,4k,128k,1M) = {v.max()-v.min():.3e}")


def check_closed_form(rng, theta, d_h=128):
    """<R_m q,R_n k> = sum_i [A_i cos((m-n)th_i) - B_i sin((m-n)th_i)], A=qk pair dot,
    B = q_{2i+1}k_{2i} - q_{2i}k_{2i+1}. Only m-n appears on the right."""
    q, k = rng.standard_normal(d_h), rng.standard_normal(d_h)
    m, n = 9001.0, 137.0
    lhs = float(rope_complex(q, np.array(m), theta) @ rope_complex(k, np.array(n), theta))
    A = q[0::2] * k[0::2] + q[1::2] * k[1::2]
    B = q[1::2] * k[0::2] - q[0::2] * k[1::2]
    rhs = float(np.sum(A * np.cos((m - n) * theta) - B * np.sin((m - n) * theta)))
    print(f"  numeric {lhs:+.10f}  closed form {rhs:+.10f}  |diff| {abs(lhs-rhs):.3e}")
    assert abs(lhs - rhs) / abs(lhs) < 1e-12


def check_torch(rng, theta, d_h=128):
    if torch is None:
        print("  [skipped: torch not installed]")
        return
    x = rng.standard_normal((64, d_h))
    pos = rng.integers(0, 50000, size=64).astype(np.float64)
    ang = torch.from_numpy(pos)[:, None] * torch.from_numpy(theta)
    z = torch.view_as_complex(torch.from_numpy(x).reshape(64, d_h // 2, 2).contiguous())
    out = torch.view_as_real(z * torch.polar(torch.ones_like(ang), ang)).reshape(64, d_h)
    res = np.max(np.abs(out.numpy() - rope_complex(x, pos, theta)))
    print(f"  torch.view_as_complex path vs NumPy, max |diff| = {res:.3e}")
    assert res < 1e-9


def wavelength_table(d_h=128, base=10000.0, L=4096.0, s=8.0):
    theta = rope_freqs(d_h, base)
    lam, r = wavelengths(theta), L / wavelengths(theta)
    pi_, ntk, yarn, g = (freqs_pi(theta, s), freqs_ntk(d_h, base, s),
                         freqs_yarn(theta, s, L), yarn_ramp(L / wavelengths(theta)))
    assert abs(ntk[0] / theta[0] - 1.0) < 1e-12          # fastest pair untouched
    assert abs(ntk[-1] / theta[-1] - 1.0 / s) < 1e-12    # slowest pair == PI exactly
    assert np.all(yarn <= theta * (1 + 1e-12)) and np.all(yarn >= theta / s - 1e-18)
    print(f"  d_h={d_h} base={base:g} trained L={L:g} s={s:g} -> target {L*s:g} tokens; "
          f"YaRN alpha=1 beta=32, logit scale {yarn_logit_scale(s):.4f}")
    print("                                    theta'_i / theta_i")
    hdr = "  pair   lambda_i(tok)     r=L/lam      PI     NTK    YaRN   gamma"
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for i in [0, 4, 8, 12, 16, 20, 21, 24, 28, 32, 36, 40, 44, 45, 46, 48, 52, 56, 60, 63]:
        print(f"  {i:4d} {lam[i]:15.2f} {r[i]:11.4f}  {pi_[i]/theta[i]:6.4f}  "
              f"{ntk[i]/theta[i]:6.4f}  {yarn[i]/theta[i]:6.4f}  {g[i]:5.3f}")
    hi, lo = int(np.sum(g >= 1 - 1e-12)), int(np.sum(g <= 1e-12))
    print(f"\n  PI       : all {d_h//2} pairs scaled by 1/s -- including the {hi} "
          f"short-wavelength pairs that never needed it")
    print(f"  NTK-aware: smooth sweep, pair 0 ratio {ntk[0]/theta[0]:.4f} -> pair "
          f"{d_h//2-1} ratio {ntk[-1]/theta[-1]:.4f} (= 1/s exactly)")
    print(f"  YaRN     : {hi} pairs untouched (r>32), {d_h//2-hi-lo} ramped, "
          f"{lo} fully interpolated (r<1)")
    print(f"  YaRN boundaries: last untouched pair i={hi-1} (lam={lam[hi-1]:.1f}); first "
          f"fully-interpolated pair i={d_h//2-lo} (lam={lam[d_h//2-lo]:.1f})")
    print(f"  {lo} of {d_h//2} pairs have lam > L, so positions in [{L:g},{L*s:g}) turn "
          f"them through angles never seen in training")


def base_table(d_h=128, L=8192.0):
    print(f"  d_h={d_h}; 'pairs with lam>L' at L={L:g}")
    print("  base           lam_max (tokens)   pairs with lam>L")
    for b in [1e4, 5e5, 1e6, 1e7]:
        lam = wavelengths(rope_freqs(d_h, b))
        print(f"  {b:>10.0f}   {lam[-1]:15.1f}   {int(np.sum(lam > L)):>10d} / {d_h//2}")


def sink_demo(rng, n=1024, d_h=64, window=128, sink_logit=6.0, trials=64):
    """
    Why the first KV pair must survive eviction. Token 0 carries a large content-
    independent logit and absorbs surplus softmax mass. Weights on the window are
    e_j/Z; dropping tokens shrinks Z and inflates every survivor.
    """
    theta = rope_freqs(d_h, 10000.0)
    pos = np.arange(n, dtype=np.float64)
    w = np.arange(n - window, n)
    ee, ek, eg, sm, ze, zk, mf, me = ([] for _ in range(8))
    for _ in range(trials):
        q = rng.standard_normal(d_h) / np.sqrt(d_h)
        k = rng.standard_normal((n, d_h)) / np.sqrt(d_h)
        kr = rope_real_pairs(k, pos, theta)
        qr = rope_real_pairs(q[None, :], np.array([float(n - 1)]), theta)[0]
        logits = kr @ qr * np.sqrt(d_h)
        logits[0] += sink_logit                       # the learned sink
        e = np.exp(logits - logits.max())
        Z, a_full = e.sum(), e / e.sum()
        Ze = e[w].sum()                               # window only
        Zk = e[w].sum() + e[0]                        # sink + window (StreamingLLM)
        Zg = e[w].sum() + np.exp(sink_logit - logits.max())   # gpt-oss: scalar, no KV
        ee.append(np.abs(e[w] / Ze - a_full[w]).sum())
        ek.append(np.abs(e[w] / Zk - a_full[w]).sum())
        eg.append(np.abs(e[w] / Zg - a_full[w]).sum())
        sm.append(a_full[0]); ze.append(Ze / Z); zk.append(Zk / Z)
        mf.append(a_full[w].max()); me.append((e[w] / Ze).max())
    print(f"  n={n}, window={window}, sink logit bonus +{sink_logit:g}, {trials} heads")
    print(f"  mean attention mass on token 0 (full context) : {np.mean(sm):.4f}")
    print(f"  denominator recovered, window only            : {np.mean(ze):.4f} of Z")
    print(f"  denominator recovered, sink + window          : {np.mean(zk):.4f} of Z")
    print(f"  L1 error on window weights, sink EVICTED      : {np.mean(ee):.4f}")
    print(f"  L1 error on window weights, sink KEPT         : {np.mean(ek):.4f}")
    print(f"  L1 error, learned scalar sink (no KV stored)  : {np.mean(eg):.4f}")
    print(f"  largest window weight: true {np.mean(mf):.4f} -> evicted "
          f"{np.mean(me):.4f} ({np.mean(me)/np.mean(mf):.2f}x inflated)")
    assert np.mean(ek) < np.mean(ee) and np.mean(eg) < np.mean(ee)


def head(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    head("[1] Two independent RoPE implementations agree")
    theta = check_implementations(rng)
    head("[2] <R_m q, R_n k> depends on m-n alone")
    check_relative(rng, theta)
    head("[3] Closed form of the rotated inner product")
    check_closed_form(rng, theta)
    head("[4] torch cross-check")
    check_torch(rng, theta)
    head("[5] Per-dimension wavelengths: who gets interpolated")
    wavelength_table()
    head("[6] Raising the base instead of interpolating")
    base_table()
    head("[7] Attention sink under a sliding window")
    sink_demo(rng)
    print("\nall assertions passed")
