"""Artifact 28.1 -- A test-time-compute harness on a toy task with an exact verifier.

Task: given (a, b, c, m), produce y = (a*b + c) mod m.  The verifier is closed-form,
so "verified" and "correct" coincide -- the clean-RLVR regime with no reward hacking.

We simulate the sampler instead of calling a model: each problem carries fixed logits
over 10 candidate answers; sampling at temperature T induces a categorical
distribution.  Lowering T ("sharpening") mimics what RLVR does to a policy: mass
concentrates on each problem's mode.  Where the mode is correct, accuracy rises;
where it is wrong, the tail probability of EVER sampling the right answer collapses.

Asserted demonstrations:
  1. analytic pass@k = mean_i [1 - (1-p_i)^k] matches the unbiased empirical
     estimator of Chen et al. (2021) drawn from real samples;
  2. sharpening (T=0.25) raises pass@1 but strictly lowers pass@64 vs. the base
     sampler (T=1.0), with a crossover at small k -- the pass@k inversion;
  3. majority voting beats single-sample accuracy when the correct answer is modal
     (a plurality) but has per-sample probability < 1/2;
  4. at n=16: verifier-guided selection == empirical pass@16, and best-of-n under a
     NOISY reranker lands strictly between pass@1 and pass@16.
"""
import numpy as np

rng = np.random.default_rng(0)
M, C = 200, 10          # problems, candidate answers per problem
N_SAMP = 256            # samples per problem for empirical estimates
K_MAX = 64

def softmax_T(z, T):
    """Temperature softmax, numerically stable (Ch. 5)."""
    a = z / T
    a = a - a.max(axis=-1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=-1, keepdims=True)

def make_problems():
    """(a*b+c) mod m instances, candidate answer values, per-problem logits.

    70% of problems are 'aligned': the argmax-logit candidate is the true answer.
    30% are 'misaligned': a distractor is the mode; the truth sits mid-distribution.
    This is the only structural knob -- everything downstream is measured.
    """
    a = rng.integers(2, 50, M); b = rng.integers(2, 50, M)
    c = rng.integers(0, 50, M); m = rng.integers(51, 400, M)
    truth = (a * b + c) % m
    cands = np.empty((M, C), dtype=np.int64)
    logits = rng.normal(0.0, 1.0, (M, C))
    aligned = rng.random(M) < 0.7
    for i in range(M):
        # candidate 0 is the truth; distractors are plausible wrong residues
        pool = {int(truth[i])}
        row = [int(truth[i])]
        for v in [(a[i]*b[i]-c[i]) % m[i], (a[i]+b[i]*c[i]) % m[i], (a[i]*b[i]) % m[i]]:
            v = int(v)
            if v not in pool:
                row.append(v); pool.add(v)
        while len(row) < C:                      # fill with distinct random residues
            v = int(rng.integers(0, m[i]))
            if v not in pool:
                row.append(v); pool.add(v)
        cands[i] = row
        top = logits[i].max()
        if aligned[i]:   # truth gets the argmax logit by a clear margin
            logits[i, 0] = top + rng.uniform(0.3, 1.2)
        else:            # a distractor gets the argmax; truth stays unboosted
            j = int(rng.integers(1, C))
            logits[i, j] = top + rng.uniform(0.3, 1.2)
    return a, b, c, m, truth, cands, logits, aligned

def verify(a, b, c, m, y):
    """The exact verifier: closed-form check, no learned component."""
    return (a * b + c) % m == y

def sample_answers(logits, cands, T, n):
    """Draw n candidate-answer VALUES per problem from softmax(logits/T)."""
    P = softmax_T(logits, T)
    cum = P.cumsum(axis=1)
    idx = (rng.random((M, n, 1)) < cum[:, None, :]).argmax(axis=2)
    return np.take_along_axis(cands, idx.reshape(M, -1), axis=1).reshape(M, n), idx

def pass_at_k_unbiased(correct_counts, n, ks):
    """Chen et al. (2021): pass@k = 1 - C(n-c,k)/C(n,k), averaged over problems.
    Computed as a running product to avoid factorial overflow."""
    out = []
    for k in ks:
        frac = np.ones(M)
        for j in range(k):                       # prod (n-c-j)/(n-j)
            frac *= np.clip(n - correct_counts - j, 0, None) / (n - j)
        out.append(np.mean(1.0 - frac))
    return np.array(out)

def run():
    a, b, c, m, truth, cands, logits, aligned = make_problems()
    ks = np.arange(1, K_MAX + 1)
    curves = {}
    for name, T in [("base  T=1.00", 1.0), ("sharp T=0.25", 0.25)]:
        P = softmax_T(logits, T)
        p_i = P[:, 0]                            # analytic per-draw correctness prob
        analytic = np.array([np.mean(1 - (1 - p_i) ** k) for k in ks])
        ans, _ = sample_answers(logits, cands, T, N_SAMP)
        ok = verify(a[:, None], b[:, None], c[:, None], m[:, None], ans)
        empirical = pass_at_k_unbiased(ok.sum(axis=1), N_SAMP, ks)
        resid = np.abs(analytic - empirical).max()
        curves[name] = (analytic, empirical, ok, p_i)
        print(f"[{name}] pass@1={analytic[0]:.4f}  pass@64={analytic[-1]:.4f}  "
              f"max|analytic-empirical| over k=1..64: {resid:.4f}")
        assert resid < 0.03, "unbiased estimator should match closed form"
    base_a, base_e = curves["base  T=1.00"][0], curves["base  T=1.00"][1]
    shrp_a, shrp_e = curves["sharp T=0.25"][0], curves["sharp T=0.25"][1]

    print("\n   k   base(analytic)  sharp(analytic)  base(empirical)  sharp(empirical)")
    for k in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]:
        print(f"  {k:3d}   {base_a[k-1]:14.4f}  {shrp_a[k-1]:15.4f}"
              f"  {base_e[k-1]:15.4f}  {shrp_e[k-1]:16.4f}")

    # --- THE CENTRAL ASSERTION: pass@k inversion under sharpening ---------------
    assert shrp_a[0] > base_a[0] and shrp_e[0] > base_e[0], "sharpening must raise pass@1"
    assert shrp_a[-1] < base_a[-1] and shrp_e[-1] < base_e[-1], "and strictly lower pass@64"
    k_star = int(np.argmax(base_a > shrp_a)) + 1   # first k where base overtakes
    assert 1 < k_star <= K_MAX
    print(f"\nCROSSOVER: sharpened wins pass@k for k<{k_star}, base wins for k>={k_star} "
          f"(analytic; empirical crossover at k={int(np.argmax(base_e > shrp_e)) + 1})")
    p_mis = curves['sharp T=0.25'][3][~aligned]
    print(f"sharpened per-draw P(correct) on misaligned problems: "
          f"mean {p_mis.mean():.4f} (base: {curves['base  T=1.00'][3][~aligned].mean():.4f})"
          f" -- coverage lost, never recoverable by more samples")

    # --- majority voting: correct answer modal but not majority per draw --------
    # Construct distributions with P(correct)~0.35 and 6+ wrong answers splitting
    # the rest: the truth is the PLURALITY answer but loses to 'any wrong answer'.
    p_true = rng.uniform(0.30, 0.40, M)
    maj_logits = np.log(np.column_stack(
        [p_true] + [(1 - p_true) / (C - 1)] * (C - 1)) + 1e-12)
    ans, _ = sample_answers(maj_logits, cands, 1.0, 33)
    ok = verify(a[:, None], b[:, None], c[:, None], m[:, None], ans)
    single = ok.mean()                            # pass@1 for this sampler
    voted = np.array([np.bincount(row).argmax() for row in ans])  # plurality answer
    maj = verify(a, b, c, m, voted).mean()
    print(f"\n[majority] per-sample accuracy {single:.4f} -> majority@33 {maj:.4f}")
    assert maj > single + 0.2, "plurality-but-not-majority truth: voting must win big"

    # --- selection strategies at n=16 on the base sampler -----------------------
    n = 16
    ans, idx = sample_answers(logits, cands, 1.0, n)
    ok = verify(a[:, None], b[:, None], c[:, None], m[:, None], ans)
    verifier_sel = ok.any(axis=1).mean()          # exact verifier: any hit counts
    scores = np.take_along_axis(logits, idx, axis=1) + rng.normal(0, 1.5, (M, n))
    best_of_n = ok[np.arange(M), scores.argmax(axis=1)].mean()  # noisy reranker
    voted = np.array([np.bincount(row).argmax() for row in ans])
    maj16 = verify(a, b, c, m, voted).mean()
    p1 = ok[:, 0].mean()
    print(f"[n=16, base sampler] pass@1 {p1:.4f} | majority {maj16:.4f} | "
          f"noisy best-of-16 {best_of_n:.4f} | verifier-guided {verifier_sel:.4f}")
    assert p1 < best_of_n < verifier_sel, "reranker sits between single-draw and oracle"

    # --- cross-check our softmax against scipy (optional dependency) ------------
    try:
        from scipy.special import softmax as sp_softmax
        d = np.abs(softmax_T(logits, 0.25) - sp_softmax(logits / 0.25, axis=1)).max()
        print(f"[cross-check] max|ours - scipy.softmax| = {d:.2e}")
        assert d < 1e-12
    except ImportError:
        print("[skipped: scipy not installed]")
    print("\nAll assertions passed.")

if __name__ == "__main__":
    run()
