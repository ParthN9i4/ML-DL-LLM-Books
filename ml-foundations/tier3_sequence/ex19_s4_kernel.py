"""
ex19 — The S4 kernel: HiPPO, the convolution view, and FFT.  (Book: Chapter 21)

A structured state space layer is, per channel, the most boring object in signal
processing: a linear time-invariant system.

    x_t = A_bar x_{t-1} + B_bar u_t          (state, N-dimensional)
    y_t = C x_t                              (scalar output)

Unroll it and the whole layer collapses into a single causal convolution,

    y = K * u        with       K = (C B_bar, C A_bar B_bar, C A_bar^2 B_bar, ...)

so the same weights can be run two completely different ways: as an O(1)-memory
recurrence at inference time, or as one FFT-based convolution over the entire
sequence at training time, in O(T log T) with no sequential dependency at all.
That dual identity is the entire architectural argument for S4, and this file
makes it a numerical fact: recurrence, direct convolution, and FFT convolution
agree to within 3e-15 here, and the FFT path is only correct because it is
zero-padded to at least 2T so the wraparound of the circular convolution falls
off the end — drop the padding and 65% of the output's peak amplitude is garbage.

The second half is about where A comes from. A random stable A gives you a state
that forgets in some arbitrary way. HiPPO-LegS gives you the matrix for which the
state IS the coefficient vector of the input history projected onto Legendre
polynomials under an exponentially decaying measure — the optimal N-number
summary of everything you have seen. That is not a slogan here: the file builds
the Legendre memory functionals in closed form and measures how much of them the
state can linearly reconstruct, for HiPPO versus random stable matrices of the
same size and the same slowest decay rate. HiPPO recovers them to 1.0% relative
error; the best of five random draws manages 40%. The residual 1.0% is not a
modelling gap either — refine dt and it shrinks like the quadrature error it is.

Discretization is the joint between the two halves. The bilinear (Tustin) map
sends each continuous eigenvalue lambda to (1 + dt*lambda/2)/(1 - dt*lambda/2),
a Mobius transform carrying the open left half-plane exactly onto the open unit
disc — which is why "Re lambda < 0" is not a stylistic preference but the literal
condition for the discrete system not to explode.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402

try:
    import scipy.signal as scipy_signal
    HAVE_SCIPY = True
except ImportError:                                    # pragma: no cover
    HAVE_SCIPY = False

try:
    import torch
    HAVE_TORCH = True
except ImportError:                                    # pragma: no cover
    HAVE_TORCH = False


# ---------------------------------------------------------------------------
# The continuous system
# ---------------------------------------------------------------------------

def hippo_legs(N):
    """The HiPPO-LegS state matrix, from its closed form.

        A[n, k] = -sqrt(2n+1) sqrt(2k+1)   for n > k
                = -(n + 1)                 for n == k
                =  0                       for n < k

    Lower triangular, so its eigenvalues are exactly the diagonal -(1..N):
    real, negative, well separated. That is the stability guarantee for free.
    """
    n = np.arange(N, dtype=np.float64)
    r = np.sqrt(2 * n + 1.0)
    # === YOUR CODE HERE ===
    A = np.tril(-np.outer(r, r), -1)
    A[np.arange(N), np.arange(N)] = -(n + 1.0)
    return A


def hippo_legs_B(N):
    """The matching input vector B[n] = sqrt(2n+1), as a column."""
    n = np.arange(N, dtype=np.float64)
    # === YOUR CODE HERE ===
    return np.sqrt(2 * n + 1.0).reshape(N, 1)


def discretize_bilinear(A, B, dt):
    """Bilinear / Tustin / trapezoidal discretization of dx/dt = A x + B u.

        A_bar = (I - dt/2 A)^-1 (I + dt/2 A)
        B_bar = (I - dt/2 A)^-1 (dt B)

    Equivalent to the trapezoid rule on the state ODE. The eigenvalue map it
    induces, lambda -> (1 + dt*lambda/2)/(1 - dt*lambda/2), is the Mobius
    transform taking the left half-plane onto the unit disc, which the checks
    below verify directly.
    """
    N = A.shape[0]
    I = np.eye(N)
    # === YOUR CODE HERE ===
    left = I - (dt / 2.0) * A
    A_bar = np.linalg.solve(left, I + (dt / 2.0) * A)
    B_bar = np.linalg.solve(left, dt * B)
    return A_bar, B_bar


def expm_via_eig(A, t):
    """exp(A t) by eigendecomposition — the exact solution operator of the ODE.

    Only valid for diagonalizable A; HiPPO-LegS is triangular with distinct
    diagonal entries, so it qualifies. Used purely as the ground truth against
    which the discretizations are graded.
    """
    # === YOUR CODE HERE ===
    w, V = np.linalg.eig(A)
    return np.real(V @ np.diag(np.exp(w * t)) @ np.linalg.inv(V))


# ---------------------------------------------------------------------------
# Three ways to run the same LTI system
# ---------------------------------------------------------------------------

def ssm_recurrent(A_bar, B_bar, C, u):
    """View 1: the sequential recurrence. O(T N^2) time, O(N) memory."""
    T = len(u)
    N = A_bar.shape[0]
    b = B_bar[:, 0]
    c = C[0]
    x = np.zeros(N)
    y = np.empty(T)
    # === YOUR CODE HERE ===
    for t in range(T):
        x = A_bar @ x + b * u[t]
        y[t] = c @ x
    return y


def s4_kernel(A_bar, B_bar, C, T):
    """View 2a: the explicit convolution kernel K[s] = C A_bar^s B_bar.

    Built by repeated matrix-vector products rather than matrix powers, which
    is O(T N^2) instead of O(T N^3). (Real S4 computes this in O((N+T) log T)
    from a diagonal-plus-low-rank factorization; the point here is the identity,
    not the fast kernel algorithm.)
    """
    c = C[0]
    v = B_bar[:, 0].copy()
    K = np.empty(T)
    # === YOUR CODE HERE ===
    for s in range(T):
        K[s] = c @ v
        v = A_bar @ v
    return K


def conv_direct(K, u):
    """View 2b: the causal convolution y[t] = sum_{s<=t} K[s] u[t-s], directly.

    O(T^2) multiply-adds, arranged as T vector updates so numpy does the inner
    loop. Correct and simple; the FFT path has to match it exactly.
    """
    T = len(u)
    y = np.zeros(T)
    # === YOUR CODE HERE ===
    for s in range(min(len(K), T)):
        y[s:] += K[s] * u[:T - s]
    return y


def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def conv_fft(K, u, pad=True):
    """View 3: the same convolution through the FFT, in O(T log T).

    THE detail: the FFT computes a CIRCULAR convolution of length n. The linear
    convolution of two length-T signals has 2T-1 samples, so unless n >= 2T-1
    the tail wraps around and lands on top of the beginning of the output. With
    `pad=False` you get exactly that bug, which the break-it section measures.
    """
    T = len(u)
    n = next_pow2(2 * T) if pad else T
    # === YOUR CODE HERE ===
    return np.fft.irfft(np.fft.rfft(K, n) * np.fft.rfft(u, n), n)[:T]


# ---------------------------------------------------------------------------
# What HiPPO is for: the Legendre memory basis
# ---------------------------------------------------------------------------

def state_kernel_matrix(A_bar, B_bar, T):
    """M[:, s] = A_bar^s B_bar — the map from history to state.

    Since x_t = sum_s A_bar^s B_bar u_{t-s}, the rows of M are exactly the T
    linear functionals of the past that the N-dimensional state retains. Any
    question of the form "can the state answer X about the history?" is the
    question "does X lie in the row space of M?".
    """
    N = A_bar.shape[0]
    M = np.empty((N, T))
    v = B_bar[:, 0].copy()
    # === YOUR CODE HERE ===
    for s in range(T):
        M[:, s] = v
        v = A_bar @ v
    return M


def legendre_memory_basis(N, T, dt):
    """The functionals HiPPO-LegS is designed to compute, in closed form.

    LegS integrates dx/dt = -(1/t)(A x + B u) so that x(t) holds the Legendre
    coefficients of u over [0, t]. Substituting t = exp(tau) removes the 1/t and
    leaves the LTI system dx/dtau = A x + B u, i.e. the fixed-A system S4
    actually runs — with real time warped exponentially. Written back in terms of
    the discrete lag s (so real-time position is exp(-s*dt) of the window), the
    coefficient functionals are

        L[n, s] = sqrt(2n+1) * P_n(2 exp(-s*dt) - 1) * exp(-s*dt) * dt

    which is precisely "project the history onto Legendre polynomials under an
    exponentially decaying memory measure". P_n is the Legendre polynomial;
    the exp(-s*dt) factor is the decaying measure and also the time warp.
    """
    s = np.arange(T)
    # === YOUR CODE HERE ===
    z = np.exp(-s * dt)
    L = np.empty((N, T))
    for n in range(N):
        coef = np.zeros(n + 1)
        coef[n] = 1.0
        P_n = np.polynomial.legendre.legval(2.0 * z - 1.0, coef)
        L[n] = np.sqrt(2 * n + 1.0) * P_n * z * dt
    return L


def memory_reconstruction_error(M, L):
    """Relative error of the BEST linear reconstruction of L's functionals from
    the state produced by M.

    Solve min_W ||L - W M||_F and report ||L - W M||_F / ||L||_F. Because the
    minimization is over all W, this is scale- and basis-free: it measures only
    whether the row space of M contains the row space of L, i.e. whether the
    state retains the information at all. For white-noise inputs it is exactly
    the relative RMS error of estimating the Legendre coefficients from x.
    """
    # === YOUR CODE HERE ===
    W, *_ = np.linalg.lstsq(M.T, L.T, rcond=None)
    resid = L - W.T @ M
    return float(np.linalg.norm(resid) / np.linalg.norm(L))


def random_stable_A(N, rng, max_real_part=-1.0):
    """A random matrix whose eigenvalues all have real part <= max_real_part.

    Shifted so its SLOWEST mode decays at exactly the same rate as HiPPO-LegS's
    slowest mode (-1). Without that matching the comparison would be a comparison
    of time constants, not of structure.
    """
    R = rng.standard_normal((N, N)) / np.sqrt(N)
    # === YOUR CODE HERE ===
    shift = float(np.max(np.linalg.eigvals(R).real))
    return R - (shift - max_real_part) * np.eye(N)


# ---------------------------------------------------------------------------
# Operation counts, for the FFT crossover
# ---------------------------------------------------------------------------

def _timed(fn):
    """Wall-clock seconds for one call of fn."""
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def ops_direct(T):
    """Multiply-adds in the direct causal convolution: T(T+1)/2."""
    # === YOUR CODE HERE ===
    return 0.5 * T * (T + 1.0)


def ops_fft(T):
    """Flops in the padded FFT convolution: two forward real FFTs, one inverse,
    and n/2 complex multiplies, with n = next power of two >= 2T. A real FFT of
    length n costs about 2.5 n log2(n) flops."""
    n = float(next_pow2(2 * T))
    # === YOUR CODE HERE ===
    return 3.0 * 2.5 * n * np.log2(n) + 6.0 * (n / 2.0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, T, dt = 16, 256, 0.05

    A = hippo_legs(N)
    B = hippo_legs_B(N)
    C = rng.standard_normal((1, N))
    u = rng.standard_normal(T)

    print("\n--- the HiPPO-LegS matrix ---")
    print(f"      N = {N}; A is {A.shape[0]}x{A.shape[1]}, "
          f"A[3,1] = {A[3, 1]:.6f} = -sqrt(7)*sqrt(3) = {-np.sqrt(7)*np.sqrt(3):.6f}")
    check("A is lower triangular", float(np.abs(np.triu(A, 1)).max()), 0.0, tol=0.0)
    check("diagonal is -(n+1)", np.diag(A), -(np.arange(N) + 1.0), tol=0.0)
    n_i, k_i = np.tril_indices(N, -1)
    check("strict lower part is -sqrt(2n+1)sqrt(2k+1)",
          A[n_i, k_i], -np.sqrt(2 * n_i + 1.0) * np.sqrt(2 * k_i + 1.0), tol=1e-12)
    eig_A = np.linalg.eigvals(A)
    print(f"      eigenvalues: {np.sort(eig_A.real)[:4]} ... {np.sort(eig_A.real)[-2:]}")
    check("every eigenvalue is real and negative (triangular => the diagonal)",
          np.sort(eig_A.real), -(np.arange(N)[::-1] + 1.0), tol=1e-9)
    check("slowest mode decays at rate 1", float(np.max(eig_A.real)), -1.0, tol=1e-9)

    print("\n--- discretization: the bilinear (Tustin) transform ---")
    A_bar, B_bar = discretize_bilinear(A, B, dt)
    # The whole content of Tustin is one Mobius transform of the spectrum.
    mobius = (1.0 + dt * eig_A / 2.0) / (1.0 - dt * eig_A / 2.0)
    eig_Abar = np.linalg.eigvals(A_bar)
    gap = float(np.abs(np.sort_complex(mobius) - np.sort_complex(eig_Abar)).max())
    rho = float(np.abs(eig_Abar).max())
    print(f"      dt = {dt}; spectral radius of A_bar = {rho:.6f} "
          f"(= (1-dt/2)/(1+dt/2) for the lambda=-1 mode = {(1-dt/2)/(1+dt/2):.6f})")
    check("eig(A_bar) == (1 + dt*lam/2)/(1 - dt*lam/2), exactly", gap, 0.0, tol=1e-12)
    check("stable A => spectral radius < 1", rho < 1.0)

    # The Mobius map is *the* reason "Re lambda < 0" is the stability condition:
    # it carries the open left half-plane onto the open unit disc, bijectively.
    z = rng.standard_normal(4000) + 1j * rng.standard_normal(4000)
    m = (1.0 + dt * z / 2.0) / (1.0 - dt * z / 2.0)
    agree = int(np.sum((z.real < 0) == (np.abs(m) < 1.0)))
    print(f"      over {len(z)} random complex lambda: "
          f"Re(lambda)<0 <=> |mobius(lambda)|<1 holds {agree}/{len(z)} times")
    check("left half-plane maps onto the unit disc, with no exceptions", agree, len(z))

    # Accuracy: bilinear is second order, forward Euler first. Grade both against
    # exp(A*t), the exact zero-input solution operator.
    x0 = rng.standard_normal(N)
    t_final = 1.0
    x_true = expm_via_eig(A, t_final) @ x0
    print(f"      {'dt':>9} {'bilinear err':>14} {'Euler err':>12}   error ratio vs previous dt")
    prev = None
    ratios = None
    for dt_h in (0.025, 0.0125, 0.00625):
        steps = int(round(t_final / dt_h))
        Ab_h, _ = discretize_bilinear(A, B, dt_h)
        Ae_h = np.eye(N) + dt_h * A
        e_bil = float(np.linalg.norm(np.linalg.matrix_power(Ab_h, steps) @ x0 - x_true)
                      / np.linalg.norm(x_true))
        e_eul = float(np.linalg.norm(np.linalg.matrix_power(Ae_h, steps) @ x0 - x_true)
                      / np.linalg.norm(x_true))
        tail = ""
        if prev is not None:
            ratios = (prev[0] / e_bil, prev[1] / e_eul)
            tail = f"   bilinear /{ratios[0]:.2f}   Euler /{ratios[1]:.2f}"
        print(f"      {dt_h:9.5f} {e_bil:14.4e} {e_eul:12.4e}{tail}")
        prev = (e_bil, e_eul)
    check("halving dt divides the bilinear error by ~4 (second order)",
          ratios[0], 4.0, tol=0.5)
    check("...and the Euler error by only ~2 (first order)", ratios[1] < 3.0)
    check("at the same dt, bilinear is far more accurate than Euler",
          prev[1] > 50 * prev[0])

    if HAVE_SCIPY:
        Ad, Bd, _, _, _ = scipy_signal.cont2discrete(
            (A, B, C, np.zeros((1, 1))), dt, method="bilinear")
        check("scipy cont2discrete(method='bilinear') gives the same A_bar",
              Ad, A_bar, tol=1e-12)
        check("...and the same B_bar", Bd, B_bar, tol=1e-12)
        # (scipy additionally rewrites C and D to preserve the transfer function;
        # S4 keeps C as-is and D = 0, which is a different output convention for
        # the same state recursion.)

    print("\n--- three views of one LTI system ---")
    y_rec = ssm_recurrent(A_bar, B_bar, C, u)
    K = s4_kernel(A_bar, B_bar, C, T)
    y_dir = conv_direct(K, u)
    y_fft = conv_fft(K, u)
    print(f"      K[:3] = {K[:3]}, K[-1] = {K[-1]:.3e}  (|K| decays like rho^s)")
    check("K[0] == C B_bar", float(K[0]), float((C @ B_bar)[0, 0]), tol=1e-15)
    check("K[2] == C A_bar^2 B_bar",
          float(K[2]), float((C @ np.linalg.matrix_power(A_bar, 2) @ B_bar)[0, 0]), tol=1e-12)
    e_dir = float(np.abs(y_rec - y_dir).max())
    e_fft = float(np.abs(y_rec - y_fft).max())
    e_df = float(np.abs(y_dir - y_fft).max())
    print(f"      |y| max = {np.abs(y_rec).max():.4f};  recurrence vs direct conv "
          f"{e_dir:.2e},  recurrence vs FFT {e_fft:.2e},  direct vs FFT {e_df:.2e}")
    check("recurrence == direct convolution", y_dir, y_rec, tol=1e-10)
    check("recurrence == FFT convolution", y_fft, y_rec, tol=1e-10)
    check("direct convolution == FFT convolution", y_fft, y_dir, tol=1e-10)
    # The recurrence is O(N) memory and strictly sequential; the convolution
    # touches every timestep in parallel. Same numbers, opposite execution model.
    check("the FFT was padded to at least 2T", next_pow2(2 * T) >= 2 * T)

    if HAVE_SCIPY:
        y_scipy = scipy_signal.fftconvolve(K, u)[:T]
        check("scipy.signal.fftconvolve agrees", y_scipy, y_rec, tol=1e-10)
    if HAVE_TORCH:
        n_t = next_pow2(2 * T)
        yk = torch.fft.rfft(torch.tensor(K), n=n_t)
        yu = torch.fft.rfft(torch.tensor(u), n=n_t)
        y_torch = torch.fft.irfft(yk * yu, n=n_t)[:T].numpy()
        check("torch.fft agrees", y_torch, y_rec, tol=1e-10)

    print("\n--- what HiPPO is FOR: the state as a Legendre memory ---")
    L = legendre_memory_basis(N, T, dt)
    M = state_kernel_matrix(A_bar, B_bar, T)
    err_hippo = memory_reconstruction_error(M, L)
    errs_rand = []
    worst_shift = 0.0
    for _ in range(5):
        A_r = random_stable_A(N, rng, max_real_part=-1.0)
        B_r = rng.standard_normal((N, 1))
        Ar_bar, Br_bar = discretize_bilinear(A_r, B_r, dt)
        errs_rand.append(memory_reconstruction_error(state_kernel_matrix(Ar_bar, Br_bar, T), L))
        worst_shift = max(worst_shift, abs(float(np.max(np.linalg.eigvals(A_r).real)) + 1.0))
    best_rand = min(errs_rand)
    check("every random comparison matrix has the same slowest decay rate as HiPPO",
          worst_shift, 0.0, tol=1e-9)
    print(f"      best linear reconstruction of the {N} Legendre memory coefficients:")
    print(f"        HiPPO-LegS state:            {err_hippo*100:7.3f}% relative error")
    print(f"        random stable A (5 draws):   " +
          " ".join(f"{e*100:.1f}%" for e in errs_rand))
    print(f"      HiPPO beats the BEST random draw by {best_rand/err_hippo:.0f}x")
    check("HiPPO's state reconstructs the Legendre coefficients to within 2%",
          err_hippo < 0.02)
    check("no random stable A of the same size and decay rate comes close",
          best_rand > 10 * err_hippo)

    # And the residual is discretization error, not a modelling gap: refine dt
    # (holding the time horizon T*dt fixed) and it shrinks toward zero.
    print(f"      {'dt':>9} {'T':>6}   reconstruction error")
    errs_dt = []
    for dt_h, T_h in ((0.05, 256), (0.0125, 1024)):
        Ab_h, Bb_h = discretize_bilinear(A, B, dt_h)
        e = memory_reconstruction_error(state_kernel_matrix(Ab_h, Bb_h, T_h),
                                        legendre_memory_basis(N, T_h, dt_h))
        errs_dt.append(e)
        print(f"      {dt_h:9.5f} {T_h:6d}   {e*100:.4f}%")
    check("refining dt 4x shrinks the reconstruction error (it is quadrature error)",
          errs_dt[0] / errs_dt[1] > 3.0)

    print("\n--- the FFT crossover ---")
    print(f"      {'T':>7} {'direct ms':>11} {'FFT ms':>9} {'direct/FFT':>11}"
          f" {'predicted':>10}")
    sweep = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
    measured = []
    for T_b in sweep:
        # Plain gaussian taps on purpose: a geometrically decaying kernel drops
        # into SUBNORMAL floats past ~1e-308, and subnormal arithmetic is orders
        # of magnitude slower on x86 — which showed up here as a 3.1 s direct
        # convolution at T=16384 and would have been read as "O(T^2) bites".
        Kb = rng.standard_normal(T_b)
        ub = rng.standard_normal(T_b)
        # np.convolve is the same O(T^2) algorithm as conv_direct, but with the
        # inner loop in C — the fair baseline. (conv_direct's python-level loop
        # over lags would lose on interpreter overhead, not on arithmetic.)
        reps = 100 if T_b <= 1024 else (20 if T_b <= 4096 else 8)

        def best_of(fn):
            """Minimum over reps, not the mean: a scheduler stall can inflate one
            call 7x and drag a mean with it, and there is no such thing as a
            spuriously FAST run."""
            fn()                                   # warm up / plan the FFT
            return min(_timed(fn) for _ in range(reps))

        t_dir = best_of(lambda: np.convolve(Kb, ub)[:T_b])
        t_ff = best_of(lambda: conv_fft(Kb, ub))
        measured.append((T_b, t_dir, t_ff))
        print(f"      {T_b:7d} {t_dir*1e3:11.4f} {t_ff*1e3:9.4f} {t_dir/t_ff:11.2f}"
              f" {ops_direct(T_b)/ops_fft(T_b):10.2f}")
    check("conv_direct and np.convolve compute the same thing",
          conv_direct(K, u), np.convolve(K, u)[:T], tol=1e-12)

    # Where the two op counts cross, versus where the two clocks cross.
    predicted_cross = next(t for t in sweep if ops_fft(t) < ops_direct(t))
    measured_cross = next(t for t, td, tf in measured if td > tf)
    ratio_small = measured[0][1] / measured[0][2]
    ratio_large = measured[-1][1] / measured[-1][2]
    print(f"      predicted crossover (flop counts): T = {predicted_cross}")
    print(f"      measured  crossover (wall clock):  T = {measured_cross}")
    print(f"      direct/FFT time ratio: {ratio_small:.2f} at T={sweep[0]}, "
          f"{ratio_large:.1f} at T={sweep[-1]}")
    check("at small T the direct convolution is genuinely faster", ratio_small < 1.0)
    check("at large T the FFT wins by more than an order of magnitude", ratio_large > 10.0)
    check("the measured crossover lands within a factor of 4 of the predicted one",
          0.25 <= measured_cross / predicted_cross <= 4.0)
    # Does the wall clock actually follow O(T^2) and O(T log T)? Fit log-log
    # slopes over the large-T end of the sweep, where interpreter and FFT-plan
    # overheads are negligible. d log t / d log T should be 2 for the direct
    # convolution and just over 1 for the FFT (the log T factor lifts it above 1).
    tail = measured[-4:]     # T = 2048 .. 16384
    lg = np.log2

    def loglog_slope(xs, ys):
        return float(np.polyfit(lg(np.array(xs, float)), lg(np.array(ys, float)), 1)[0])

    Ts = [r[0] for r in tail]
    slope_dir = loglog_slope(Ts, [r[1] for r in tail])
    slope_fft = loglog_slope(Ts, [r[2] for r in tail])
    slope_pred_fft = loglog_slope(Ts, [ops_fft(t) for t in Ts])
    print(f"      log-log slope over T={Ts[0]}..{Ts[-1]}: "
          f"direct {slope_dir:.2f} (theory 2.00), FFT {slope_fft:.2f} "
          f"(theory {slope_pred_fft:.2f} from the flop count)")
    check("direct convolution measures as O(T^2)", slope_dir, 2.0, tol=0.4)
    check("FFT convolution measures as O(T log T), not O(T^2)",
          slope_fft, slope_pred_fft, tol=0.4)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) Forget the zero-padding. The FFT convolves circularly with period n;
    #     with n = T the samples of the linear convolution at indices T..2T-2
    #     wrap around and are ADDED to outputs 0..T-2. The contamination of y[t]
    #     is exactly sum_{s>t} K[s] u[T+t-s]: heaviest at t=0 (where every lag
    #     past the first contributes) and zero at t=T-1.
    y_wrap = conv_fft(K, u, pad=False)
    err = np.abs(y_wrap - y_dir)
    scale = float(np.abs(y_dir).max())
    print(f"      unpadded FFT: max error {err.max():.4f} = {err.max()/scale*100:.1f}% of |y|max")
    print(f"        first 32 outputs: max error {err[:32].max():.3e}")
    print(f"        last  32 outputs: max error {err[-32:].max():.3e}")
    check("wraparound is a first-order corruption of the output, not a rounding issue",
          err.max() > 0.1 * scale)
    check("and it is concentrated at the START of the sequence",
          err[:32].max() > 1e3 * err[-32:].max())
    check("padding to next_pow2(2T) removes it entirely", y_fft, y_dir, tol=1e-10)

    # When is it invisible? Exactly when the signal has already decayed to zero
    # before the end of the window, because then the wrapped terms u[T+t-s] that
    # pair with the LARGE early taps of K are all zero. Verify that condition is
    # really what hides the bug.
    u_decay = u * np.exp(-np.arange(T) / (T / 16.0))
    y_dec_dir = conv_direct(K, u_decay)
    y_dec_wrap = conv_fft(K, u_decay, pad=False)
    err_dec = float(np.abs(y_dec_wrap - y_dec_dir).max())
    scale_dec = float(np.abs(y_dec_dir).max())
    print(f"      same bug on a signal that decays to {np.abs(u_decay[T//2:]).max():.1e} "
          f"by the halfway point:")
    print(f"        max error {err_dec:.3e} = {err_dec/scale_dec*100:.5f}% of |y|max "
          f"({err.max()/scale / (err_dec/scale_dec):.0f}x smaller, relatively)")
    check("a decaying signal hides the missing padding almost completely",
          err_dec / scale_dec < 1e-3 * (err.max() / scale))
    # It is only ALMOST invisible: what remains is set by the kernel's own tail,
    # since the surviving terms pair small u-indices with large lags of K.
    tail_bound = float(np.abs(K[T // 2:]).max() * np.abs(u_decay).sum())
    print(f"        residual {err_dec:.3e} is bounded by max|K[s>T/2]| * sum|u| "
          f"= {tail_bound:.3e} — it is the KERNEL tail, not the signal")
    check("the surviving error is explained by the kernel tail", err_dec < tail_bound)

    # (b) An unstable A. Flip the sign of HiPPO-LegS: eigenvalues become +1..+N,
    #     the Mobius map sends them outside the unit disc, and the state norm
    #     grows geometrically at exactly the spectral radius.
    A_unstable = -A
    Au_bar, Bu_bar = discretize_bilinear(A_unstable, B, dt)
    rho_u = float(np.abs(np.linalg.eigvals(Au_bar)).max())
    lam_worst = float(np.max(np.linalg.eigvals(A_unstable).real))
    print(f"      unstable A: max Re(lambda) = {lam_worst:+.1f} "
          f"-> spectral radius of A_bar = {rho_u:.6f} > 1")
    check("a positive-real-part eigenvalue maps OUTSIDE the unit disc", rho_u > 1.0)

    T_u = 200
    u_u = rng.standard_normal(T_u)
    x = np.zeros(N)
    norms = np.empty(T_u)
    for t in range(T_u):
        x = Au_bar @ x + Bu_bar[:, 0] * u_u[t]
        norms[t] = np.linalg.norm(x)
    # Stable system, same driving input, for contrast.
    xs = np.zeros(N)
    norms_s = np.empty(T_u)
    for t in range(T_u):
        xs = A_bar @ xs + B_bar[:, 0] * u_u[t]
        norms_s[t] = np.linalg.norm(xs)
    growth = float((norms[-1] / norms[T_u // 2]) ** (1.0 / (T_u - 1 - T_u // 2)))
    print(f"      ||x_t||: t=50 {norms[50]:.3e}, t=100 {norms[100]:.3e}, "
          f"t={T_u-1} {norms[-1]:.3e}")
    print(f"      measured geometric rate {growth:.6f} vs predicted spectral radius "
          f"{rho_u:.6f}  ({abs(growth-rho_u)/rho_u*100:.3f}% off)")
    print(f"      the stable system on the same input stays at "
          f"||x||max = {norms_s.max():.4f}")
    check("the state explodes geometrically at the predicted spectral radius",
          growth, rho_u, tol=0.01 * rho_u)
    # The engineering consequence, stated as a step COUNT rather than a magnitude:
    # extrapolate ||x|| ~ rho^t from a reference step already inside the geometric
    # regime and predict when the state leaves float32 range. (Extrapolating from
    # t=0 misses by ~15 steps: HiPPO-LegS is strongly non-normal, so the first
    # dozen steps are transient, not yet rho^t.)
    t_ref = T_u // 4
    t_overflow = int(np.argmax(norms > 3.4e38))
    t_pred = t_ref + np.log(3.4e38 / norms[t_ref]) / np.log(rho_u)
    print(f"      leaves float32 range at step {t_overflow}; extrapolating rho^t "
          f"from step {t_ref} predicts {t_pred:.1f}")
    check("float32 overflow arrives at the step the spectral radius predicts",
          float(t_overflow), t_pred, tol=1.5)
    check("the stable system on the same input stays O(1)", float(norms_s.max()) < 10.0)

    summary()
