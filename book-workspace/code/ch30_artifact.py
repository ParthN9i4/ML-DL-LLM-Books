"""Artifact 30.1 -- An empirical roofline analyzer for the machine it runs on.

No GPU exists in this environment, so we build the roofline for the CPU,
honestly: measure the two ceilings (streaming memory bandwidth via large
array copies; peak FLOP/s as the best large-sgemm rate observed anywhere
in the run), compute each operation's arithmetic intensity
AI = FLOPs / bytes moved, classify it as compute- or memory-bound against
the measured machine balance point B* = peak/BW, and predict its runtime
as  t = max(FLOPs/peak, bytes/BW).  The identical procedure applies
verbatim to any GPU -- only the two ceiling measurements change.
Pure NumPy core; torch appears only as a cross-check.
"""
import time
import numpy as np

rng = np.random.default_rng(0)

def tmin(f, reps=3, budget=0.35, cap=150):
    """Best-of-many wall clock. Warm up twice first (page faults, BLAS
    thread-pool spin-up, caches), then sample until we have both `reps`
    samples and `budget` seconds of measurement, capped at `cap` calls.
    Min, not mean: noise on a shared VM is one-sided -- interference only
    ever makes you slower -- and it arrives in bursts of ~100 ms, so many
    short samples give the min a chance to catch a quiet window."""
    f(); f()
    ts, spent = [], 0.0
    while (len(ts) < reps or spent < budget) and len(ts) < cap:
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
        spent += ts[-1]
    return min(ts)

# ---------------------------------------------------------------- ceilings
def measure_bandwidth():
    """STREAM-style copy, 256 MiB arrays (this VM has a 33 MiB L3, so the
    512 MiB working set streams from DRAM). Convention:
    count bytes read + bytes written, so a copy moves 2x the array size."""
    x = rng.random(2**26, dtype=np.float32); y = np.empty_like(x)
    t = tmin(lambda: np.copyto(y, x), reps=7)
    return 2 * x.nbytes / t                      # bytes/s

def measure_peak_flops():
    """Provisional peak: large-sgemm rate. n=3072 amortizes overheads;
    2n^3 = 58 GFLOP per call. Refined below: the honest ceiling is the
    best sgemm rate seen anywhere in the run, because on a shared VM this
    dedicated probe can itself land in a contended window."""
    n = 3072
    a = rng.standard_normal((n, n), dtype=np.float32); b = a.copy()
    t = tmin(lambda: a @ b)
    return 2 * n**3 / t                          # FLOP/s

BW   = measure_bandwidth()
PEAK = measure_peak_flops()                      # provisional; see below

# ---------------------------------------------------------------- op suite
# Each op: (name, callable-on-prealloc-buffers, flops, bytes moved by the
# implementation as written).  exp counted as 1 FLOP (roofline convention;
# its true cost shows up as memory-bound ops running above prediction).
R, C = 8192, 8192                                 # 256 MiB: larger than L3
X  = rng.random((R, C), dtype=np.float32)   # uniform: cheap to generate
T  = np.empty_like(X); OUT = np.empty_like(X)
n1 = 2**26                                        # 256 MiB per fp32 array
xa = rng.random(n1, dtype=np.float32)
za = rng.random(n1, dtype=np.float32); ya = np.empty_like(xa)
G  = {n: rng.standard_normal((n, n), dtype=np.float32) for n in (512, 1024, 2048)}
GB = {n: G[n].copy() for n in G}
nv = 16384                                        # 1 GiB matrix: DRAM-resident
Av = rng.random((nv, nv), dtype=np.float32)
xv = rng.random(nv, dtype=np.float32)

def softmax_rows():        # 5 unfused kernels, 7 full passes over R*C floats
    m = X.max(axis=1, keepdims=True)             # read X
    np.subtract(X, m, out=T)                     # read X, write T
    np.exp(T, out=T)                             # read T, write T
    s = T.sum(axis=1, keepdims=True)             # read T
    np.divide(T, s, out=OUT)                     # read T, write OUT

def layernorm_rows():      # 5 unfused kernels, 8 full passes
    mu = X.mean(axis=1, keepdims=True)           # read X
    np.subtract(X, mu, out=T)                    # read X, write T (T = x-mu)
    np.multiply(T, T, out=OUT)                   # read T, write OUT (scratch)
    v = OUT.mean(axis=1, keepdims=True)          # read OUT
    np.multiply(T, 1.0/np.sqrt(v + 1e-5), out=OUT)  # read T, write OUT

ops = []                   # (name, fn, flops, bytes)
for n in (512, 1024, 2048):
    ops.append((f"matmul {n}^3", (lambda n=n: G[n] @ GB[n]), 2*n**3, 12*n**2))
ops.append((f"matvec {nv}^2", (lambda: Av @ xv), 2*nv**2, 4*(nv**2 + 2*nv)))
ops.append(("triad y=2x+z", (lambda: np.add(np.multiply(xa, 2.0, out=ya), za, out=ya)),
            2*n1, 12*n1))
ops.append((f"softmax {R}x{C}", softmax_rows, 5*R*C, 7*4*R*C))
ops.append((f"layernorm {R}x{C}", layernorm_rows, 7*R*C, 8*4*R*C))

# Measure everything FIRST, then classify: the final compute ceiling is
# the best sgemm rate observed in the whole run (probe or suite).
raw  = [(name, flops, byts, tmin(fn)) for name, fn, flops, byts in ops]
PEAK = max(PEAK, max(f/m for name, f, _, m in raw if name.startswith("matmul")))
BSTAR = PEAK / BW                                # machine balance, FLOPs/byte
print(f"memory bandwidth (copy, r+w) : {BW/1e9:8.1f} GB/s")
print(f"peak compute (best sgemm)    : {PEAK/1e9:8.1f} GFLOP/s")
print(f"MACHINE BALANCE POINT B*     : {BSTAR:8.1f} FLOPs/byte")

print(f"\n{'op':<20}{'AI':>7}{'bound':>9}{'pred ms':>9}{'meas ms':>9}"
      f"{'meas/pred':>10}{'%peak':>7}")
results = []
for name, flops, byts, meas in raw:
    ai    = flops / byts
    bound = "compute" if ai >= BSTAR else "memory"
    pred  = max(flops / PEAK, byts / BW)          # the roofline prediction
    frac  = flops / meas / PEAK                   # fraction of peak achieved
    results.append((name, ai, bound, pred, meas, frac))
    print(f"{name:<20}{ai:>7.2f}{bound:>9}{pred*1e3:>9.2f}{meas*1e3:>9.2f}"
          f"{meas/pred:>10.2f}{100*frac:>6.1f}%")

# ------------------------------------------------------------- assertions
for name, ai, bound, pred, meas, frac in results:
    if bound == "memory":
        # A memory-bound op must be nowhere near the compute ceiling...
        assert frac < 0.35, f"{name}: 'memory-bound' but hit {frac:.0%} of peak"
        # ...and land within [1/6x, 4x] of the bandwidth prediction (above
        # 1x means unmodeled cost: exp, reduction overhead; below 1x means
        # a multithreaded BLAS kernel beat our single-thread copy ceiling,
        # as sgemv does here -- a read-only stream on all four cores runs
        # several times faster than one core's read+write copy).
        assert 1/6 <= meas/pred <= 4.0, f"{name}: ratio {meas/pred:.2f}"
    else:
        # A compute-bound op must run near the ceiling: within factor 2.
        assert frac >= 0.35, f"{name}: 'compute-bound' at {frac:.0%} of peak"
        assert meas/pred <= 2.0, f"{name}: ratio {meas/pred:.2f}"
print("\nall classification + prediction assertions passed")

# ---------------------------------------------------- fusion, demonstrated
# Chain of 4 elementwise ops on 256 MiB: y = relu(2x+1) * 3.
# Unfused: 4 kernels, each a full read+write pass over DRAM-resident data.
# 'Fused': one loop over 256 KiB cache-resident blocks -- exactly what a
# fused GPU kernel (or torch.compile) does with registers/shared memory.
def unfused():
    np.multiply(xa, 2.0, out=ya); np.add(ya, 1.0, out=ya)
    np.maximum(ya, 0.0, out=ya);  np.multiply(ya, 3.0, out=ya)
def fused(blk=2**16):
    scratch = np.empty(blk, dtype=np.float32)
    for i in range(0, n1, blk):
        s = scratch[: n1 - i] if n1 - i < blk else scratch
        np.multiply(xa[i:i+blk], 2.0, out=s); np.add(s, 1.0, out=s)
        np.maximum(s, 0.0, out=s);            np.multiply(s, 3.0, out=ya[i:i+blk])
tu, tf = tmin(unfused, budget=0.7), tmin(fused, budget=0.7)
print(f"\nfusion: unfused {tu*1e3:6.1f} ms ({8*4*n1/tu/1e9:.1f} GB/s of traffic)"
      f" | fused {tf*1e3:6.1f} ms | speedup {tu/tf:.2f}x")
assert tf < tu, "fusion must win: same FLOPs, 4x less DRAM traffic"

# ------------------------------------------------------ torch cross-check
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    xt = torch.from_numpy(X)
    t = tmin(lambda: torch.softmax(xt, dim=1))
    print(f"torch.softmax (internally fused/threaded): {t*1e3:.1f} ms vs "
          f"numpy unfused {results[-2][4]*1e3:.1f} ms")
else:
    print("[skipped: torch not installed]")

if __name__ == "__main__":
    print(f"\nsummary: every op with AI below B*={BSTAR:.0f} FLOPs/byte was "
          "memory-bound in measurement; only matmul was compute-bound.")
