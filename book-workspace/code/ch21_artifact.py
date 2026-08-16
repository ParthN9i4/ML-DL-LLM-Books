import time
import numpy as np

try:
    import torch
except ImportError:
    torch = None

RNG = np.random.default_rng(0)

# --- (A) Selective scan: h_t = abar_t (*) h_{t-1} + bbar_t ------------------
def make_selective(T, P, N, rng):
    A = -np.exp(rng.normal(0.0, 0.5, size=(P, N)))          # A < 0: stable poles
    x = rng.normal(size=(T, P))
    # Delta = softplus(.) > 0 is the selectivity: per-token, per-channel step.
    Delta = np.log1p(np.exp(rng.normal(-0.5, 0.5, size=(T, P))))
    B = rng.normal(size=(T, N)) / np.sqrt(N)
    C = rng.normal(size=(T, N)) / np.sqrt(N)
    D = rng.normal(size=(P,))
    abar = np.exp(Delta[:, :, None] * A[None, :, :])        # (T,P,N)
    bbar = Delta[:, :, None] * B[:, None, :] * x[:, :, None]
    return abar, bbar, C, D, x

def scan_sequential(abar, bbar, C, D, x):
    """The obvious O(T) loop. Depth T: step t waits on step t-1."""
    T, P, N = abar.shape
    h = np.zeros((P, N)); y = np.empty((T, P))
    for t in range(T):
        h = abar[t] * h + bbar[t]
        y[t] = h @ C[t]
    return y + D * x

def blelloch_scan(a, b):
    """Work-efficient inclusive scan of the affine maps f_t(h) = a_t*h + b_t.
    (a_L,b_L) o (a_R,b_R) = (a_R a_L, a_R b_L + b_R), identity (1, 0).
    2T-2 total applications, ceil(log2 T) levels each way."""
    n = a.shape[0]
    m = 1 << (n - 1).bit_length()                # pad to a power of two
    tail = (m - n,) + a.shape[1:]
    A  = np.concatenate([a, np.ones(tail, a.dtype)])
    Bv = np.concatenate([b, np.zeros(tail, b.dtype)])
    a0, b0 = A.copy(), Bv.copy()
    s = 2                                        # ---- up-sweep ----
    while s <= m:
        h = s >> 1
        aL, bL = A[h - 1::s], Bv[h - 1::s]
        aR, bR = A[s - 1::s], Bv[s - 1::s]
        Bv[s - 1::s] = aR * bL + bR
        A[s - 1::s]  = aR * aL
        s <<= 1
    A[m - 1] = 1.0; Bv[m - 1] = 0.0              # root <- identity
    s = m                                        # ---- down-sweep ----
    while s >= 2:
        h = s >> 1
        ta, tb = A[h - 1::s].copy(), Bv[h - 1::s].copy()   # left aggregate
        A[h - 1::s]  = A[s - 1::s]                         # left <- prefix
        Bv[h - 1::s] = Bv[s - 1::s]
        Bv[s - 1::s] = ta * Bv[s - 1::s] + tb              # right <- prefix o left
        A[s - 1::s]  = ta * A[s - 1::s]
        s >>= 1
    return (a0 * Bv + b0)[:n]                    # exclusive -> inclusive

def scan_parallel(abar, bbar, C, D, x):
    h = blelloch_scan(abar, bbar)
    return np.einsum('tpn,tn->tp', h, C) + D * x

# --- (B) S4D: time-INVARIANT, so the whole map is one kernel ---------------
def s4d_kernel(T, N, dt):
    """S4D-Lin poles A_n = -1/2 + i*pi*n, ZOH-discretized."""
    A = -0.5 + 1j * np.pi * np.arange(N)
    Bc = np.ones(N, dtype=complex)
    Cc = (RNG.normal(size=N) + 1j * RNG.normal(size=N)) / np.sqrt(N)
    Abar = np.exp(dt * A)                        # ZOH state matrix
    Bbar = (Abar - 1.0) / A * Bc                 # ZOH input matrix
    powers = Abar[None, :] ** np.arange(T)[:, None]
    K = 2.0 * np.real(powers @ (Cc * Bbar))      # conj pairs -> real kernel
    return K, Abar, Bbar, Cc

def conv_fft(u, K):
    L = 1 << (2 * len(u) - 1).bit_length()        # zero-pad so it cannot wrap
    return np.fft.irfft(np.fft.rfft(u, L) * np.fft.rfft(K, L), L)[:len(u)]

def conv_naive(u, K):
    T = len(u)
    idx = np.arange(T)[:, None] - np.arange(T)[None, :]
    M = np.where(idx >= 0, K[np.abs(idx)], 0.0)  # lower-triangular Toeplitz
    return M @ u

def lti_recurrence(u, Abar, Bbar, Cc):
    h = np.zeros_like(Abar); y = np.empty(len(u))
    for t, ut in enumerate(u):
        h = Abar * h + Bbar * ut
        y[t] = 2.0 * np.real(Cc @ h)
    return y

# --- (C) SSD: linear recurrence vs masked matmul ---------------------------
def make_ssd(T, P, N, rng):
    log_a = -np.exp(rng.normal(-1.0, 0.4, size=T))          # log a_t < 0
    B = rng.normal(size=(T, N)) / np.sqrt(N)
    C = rng.normal(size=(T, N)) / np.sqrt(N)
    X = rng.normal(size=(T, P))
    return log_a, B, C, X

def ssd_linear(log_a, B, C, X):
    T, N = B.shape; P = X.shape[1]
    a = np.exp(log_a); h = np.zeros((N, P)); Y = np.empty((T, P))
    for t in range(T):
        h = a[t] * h + np.outer(B[t], X[t])      # rank-1 state write
        Y[t] = C[t] @ h
    return Y

def semiseparable_mask(log_a):
    g = np.cumsum(log_a)
    E = np.tril(g[:, None] - g[None, :])         # mask first...
    return np.tril(np.exp(E))                    # ...then exponentiate

def ssd_quadratic(log_a, B, C, X):
    """Y = (L o C B^T) X. Mask the exponent BEFORE exponentiating: above the
    diagonal it is large and positive, exp() overflows to inf, and inf*0 = nan."""
    return (semiseparable_mask(log_a) * (C @ B.T)) @ X

def timed(fn, *args):
    t0 = time.perf_counter(); out = fn(*args)
    return out, time.perf_counter() - t0

if __name__ == "__main__":
    T, P, N = 4096, 64, 16
    print(f"selective SSM: T={T} P={P} N={N}  (state = P*N = {P*N} scalars/layer)")
    abar, bbar, Cs, Ds, xs = make_selective(T, P, N, RNG)
    y_seq, t_seq = timed(scan_sequential, abar, bbar, Cs, Ds, xs)
    y_par, t_par = timed(scan_parallel,   abar, bbar, Cs, Ds, xs)
    r = np.max(np.abs(y_seq - y_par)) / np.max(np.abs(y_seq))
    print(f"  sequential recurrence   {t_seq*1e3:8.1f} ms   depth O(T)")
    print(f"  Blelloch parallel scan  {t_par*1e3:8.1f} ms   depth O(log T) "
          f"= {int(np.ceil(np.log2(T)))} levels")
    print(f"  max rel. difference     {r:.3e}")
    assert r < 1e-6, r

    Tc, Nc, dt = 4096, 64, 0.01
    K, Abar, Bbar, Cc = s4d_kernel(Tc, Nc, dt)
    u = RNG.normal(size=Tc)
    y_fft, t_fft = timed(conv_fft, u, K)
    y_nai, t_nai = timed(conv_naive, u, K)
    y_rec, t_rec = timed(lti_recurrence, u, Abar, Bbar, Cc)
    r_fft = np.max(np.abs(y_fft - y_nai)) / np.max(np.abs(y_nai))
    r_rec = np.max(np.abs(y_rec - y_nai)) / np.max(np.abs(y_nai))
    print(f"\nS4D convolution: T={Tc} N={Nc} dt={dt}")
    print(f"  FFT convolution         {t_fft*1e3:8.1f} ms   O(T log T)")
    print(f"  naive Toeplitz matmul   {t_nai*1e3:8.1f} ms   O(T^2)")
    print(f"  step-by-step recurrence {t_rec*1e3:8.1f} ms   O(T N)")
    print(f"  rel. err FFT vs naive       {r_fft:.3e}")
    print(f"  rel. err recurrence vs conv {r_rec:.3e}")
    assert r_fft < 1e-6 and r_rec < 1e-6, (r_fft, r_rec)

    Ts, Ps, Ns = 2048, 64, 64
    log_a, Bq, Cq, Xq = make_ssd(Ts, Ps, Ns, RNG)
    Y_lin,  t_lin  = timed(ssd_linear,    log_a, Bq, Cq, Xq)
    Y_quad, t_quad = timed(ssd_quadratic, log_a, Bq, Cq, Xq)
    r_ssd = np.max(np.abs(Y_lin - Y_quad)) / np.max(np.abs(Y_lin))
    print(f"\nSSD duality: T={Ts} P={Ps} N={Ns}")
    print(f"  linear recurrent form   {t_lin*1e3:8.1f} ms   O(T N P) time, O(N P) state")
    print(f"  quadratic masked matmul {t_quad*1e3:8.1f} ms   O(T^2 N) time, O(T^2) memory")
    print(f"  MAX REL DIFFERENCE      {r_ssd:.3e}   <-- the duality")
    assert r_ssd < 1e-5, r_ssd

    # Every strictly-lower submatrix of L has rank one: L[t,s] = e^{g_t} e^{-g_s}.
    Lmask = semiseparable_mask(log_a)
    sv = np.linalg.svd(Lmask[1024:1044, 1000:1020], compute_uv=False)
    print(f"  off-diagonal 20x20 block singular values: "
          f"s1={sv[0]:.3e}  s2/s1={sv[1]/sv[0]:.2e}  (rank 1)")
    assert sv[1] / sv[0] < 1e-12

    if torch is None:
        print("\n[skipped: torch not installed]")
    else:
        with torch.no_grad():
            ut, Kt = torch.from_numpy(u), torch.from_numpy(K)
            Lp = 1 << (2 * Tc - 1).bit_length()
            yt = torch.fft.irfft(torch.fft.rfft(ut, Lp) * torch.fft.rfft(Kt, Lp),
                                 Lp)[:Tc].numpy()
            Mt = torch.from_numpy(Lmask * (Cq @ Bq.T)) @ torch.from_numpy(Xq)
        print(f"\n[torch] FFT conv rel. err   "
              f"{np.max(np.abs(yt - y_nai))/np.max(np.abs(y_nai)):.3e}")
        print(f"[torch] SSD matmul rel. err "
              f"{np.max(np.abs(Mt.numpy() - Y_lin))/np.max(np.abs(Y_lin)):.3e}")
    print("\nall assertions passed")
