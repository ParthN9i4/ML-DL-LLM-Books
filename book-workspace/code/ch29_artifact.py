"""Artifact 29.1 -- The measurement crisis in three experiments (pure NumPy).

(1) Three standard multiple-choice scoring protocols applied to the SAME
    simulated models on the SAME items produce DIFFERENT rankings
    (Kendall tau strictly below 1).
(2) A Min-K%-style contamination detector separates planted memorized
    documents from clean ones (AUC above threshold), and beats naive
    mean-log-probability scoring.
(3) A SMOOTH per-token accuracy curve, viewed through exact-match over a
    multi-token answer, produces a sharp apparent "emergence" jump.

Log-probabilities are simulated; no model is called. Runs in seconds.
"""
import numpy as np

rng = np.random.default_rng(0)

# ----------------------------------------------------------------------
# Part 1: three scoring protocols, three models, one item set
# ----------------------------------------------------------------------
# Each item has 4 options; option j of item i is a token sequence of
# length L[i, j]. Correct options are drawn ~1.5 tokens longer on average
# (as in real benchmarks, where the right answer is often the wordiest).
N_ITEMS, N_OPT = 800, 4
L = rng.integers(3, 12, size=(N_ITEMS, N_OPT)).astype(float)
correct = rng.integers(0, N_OPT, size=N_ITEMS)
L[np.arange(N_ITEMS), correct] += rng.integers(0, 4, size=N_ITEMS)

# Model profiles: letter_skill (does the model map its knowledge onto the
# 'answer with A/B/C/D' format?), cont (per-token log-prob advantage it
# gives the correct option text), base (its typical per-token log-prob,
# i.e. fluency). base CANCELS across options under length normalization
# but NOT under raw summation, where every extra token costs |base| nats:
# unnormalized scoring confounds fluency with correctness.
# A: instruction-tuned, letter-fluent, mediocre judge of option text.
# B: strong base model: best per-token judgment, letter-clumsy.
# C: middling judge but very fluent (high base), letter-mediocre.
MODELS = {
    "Model-A": dict(letter=0.85, cont=0.22, base=-3.0),
    "Model-B": dict(letter=0.15, cont=0.55, base=-2.6),
    "Model-C": dict(letter=0.45, cont=0.30, base=-1.2),
}

def score_protocols(p):
    """Return accuracy under (letter, length-normalized, unnormalized)."""
    # Letter protocol: logit over the 4 letter tokens; correct letter gets
    # a boost of size `letter`, all letters get unit Gaussian noise.
    lg = rng.normal(0, 1, size=(N_ITEMS, N_OPT))
    lg[np.arange(N_ITEMS), correct] += p["letter"] * 3.0
    acc_letter = np.mean(np.argmax(lg, axis=1) == correct)
    # Continuation protocols: mean per-token log-prob of option j is
    # base + cont*[j correct] + noise shrinking with length.
    mu = p["base"] + rng.normal(0, 1.0, size=L.shape) / np.sqrt(L)
    mu[np.arange(N_ITEMS), correct] += p["cont"]
    total = mu * L                      # summed (unnormalized) log-prob
    acc_norm = np.mean(np.argmax(mu, axis=1) == correct)     # per-token mean
    acc_unnorm = np.mean(np.argmax(total, axis=1) == correct)
    return np.array([acc_letter, acc_norm, acc_unnorm])

acc = np.stack([score_protocols(p) for p in MODELS.values()])  # (3 models, 3 protos)
PROTOS = ["letter log-prob", "length-normalized", "unnormalized sum"]

def kendall_tau(a, b):
    """Kendall tau-a between two score vectors (rank by descending score)."""
    n = len(a); s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / (n * (n - 1) / 2)

print("=== Part 1: protocol choice reorders the leaderboard ===")
for m, name in enumerate(MODELS):
    print(f"  {name}:  " + "  ".join(f"{p}={acc[m, k]:.3f}" for k, p in
                                     zip(range(3), PROTOS)))
taus = {}
for k1 in range(3):
    for k2 in range(k1 + 1, 3):
        t = kendall_tau(acc[:, k1], acc[:, k2])
        taus[(PROTOS[k1], PROTOS[k2])] = t
        print(f"  Kendall tau ({PROTOS[k1]} vs {PROTOS[k2]}) = {t:+.3f}")
min_tau = min(taus.values())
assert min_tau < 1.0, "rankings agree everywhere; construction failed"
# every protocol crowns a different winner in this construction:
winners = [list(MODELS)[np.argmax(acc[:, k])] for k in range(3)]
print(f"  winners per protocol: {winners}")
print(f"  min pairwise tau = {min_tau:+.3f}  (< 1: rankings disagree)  OK")

try:                                   # cross-check tau against scipy
    from scipy.stats import kendalltau
    ref = kendalltau(acc[:, 1], acc[:, 2]).statistic
    ours = taus[(PROTOS[1], PROTOS[2])]
    assert abs(ref - ours) < 1e-12
    print(f"  scipy cross-check: |tau_ours - tau_scipy| = {abs(ref-ours):.1e}")
except ImportError:
    print("  [skipped: scipy not installed]")

# ----------------------------------------------------------------------
# Part 2: Min-K%-style contamination detection
# ----------------------------------------------------------------------
# Clean documents: mostly ordinary tokens, but ~15% genuinely surprising
# tokens (rare names, numbers) with very low log-prob. Memorized documents:
# the model has seen them, so the surprising tail is GONE -- every token,
# including the "hard" ones, gets comfortable log-prob. Every document
# also carries a difficulty offset delta (some text is just harder), which
# confounds the naive mean-log-prob detector but not the tail statistic.
# Min-K% scores a document by the mean of its bottom-k% token log-probs:
# memorized text has no bad tail, so its Min-K% score is anomalously high.
N_CLEAN, N_MEM, DOC_LEN, K_PCT = 300, 100, 300, 20

def sample_doc(memorized):
    delta = rng.normal(0, 0.45)         # per-document difficulty offset
    if memorized:
        return rng.normal(-3.1 + delta, 0.75, size=DOC_LEN)
    hard = rng.random(DOC_LEN) < 0.15
    lp = rng.normal(-2.6 + delta, 0.9, size=DOC_LEN)
    lp[hard] = rng.normal(-6.5 + delta, 1.2, size=hard.sum())
    return lp

docs = [sample_doc(False) for _ in range(N_CLEAN)] + \
       [sample_doc(True) for _ in range(N_MEM)]
labels = np.r_[np.zeros(N_CLEAN), np.ones(N_MEM)]

def min_k_score(lp, k_pct):
    k = max(1, int(len(lp) * k_pct / 100))
    return np.mean(np.sort(lp)[:k])     # mean of the k% LOWEST log-probs

def auc(scores, labels):
    """AUC via the rank-sum (Mann-Whitney) identity, ties averaged."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    n1, n0 = labels.sum(), (1 - labels).sum()
    return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

mink = np.array([min_k_score(d, K_PCT) for d in docs])
meanlp = np.array([d.mean() for d in docs])
auc_mink, auc_mean = auc(mink, labels), auc(meanlp, labels)
print("\n=== Part 2: Min-K% contamination detector ===")
print(f"  planted rate: {N_MEM}/{N_CLEAN + N_MEM} = {N_MEM/(N_CLEAN+N_MEM):.3f}")
print(f"  AUC, Min-{K_PCT}% score      = {auc_mink:.3f}")
print(f"  AUC, mean log-prob (naive) = {auc_mean:.3f}")
assert auc_mink > 0.90, f"Min-K% AUC {auc_mink:.3f} below threshold 0.90"
assert auc_mink > auc_mean, "tail statistic should beat the naive mean here"
print(f"  Min-K% AUC > 0.90 threshold and beats naive mean  OK")

try:                                   # cross-check AUC against sklearn
    from sklearn.metrics import roc_auc_score
    ref = roc_auc_score(labels, mink)
    assert abs(ref - auc_mink) < 1e-12
    print(f"  sklearn cross-check: |AUC_ours - AUC_sklearn| = {abs(ref-auc_mink):.1e}")
except ImportError:
    print("  [skipped: sklearn not installed]")

# ----------------------------------------------------------------------
# Part 3: emergence as a metric artifact
# ----------------------------------------------------------------------
# Underlying capability: per-token accuracy p(s) improves LINEARLY in
# normalized log-compute s -- maximally smooth, constant slope, nothing
# emergent about it. Reported metric: exact match on a T-token answer,
# i.e. all T tokens correct => EM(s) = p(s)^T under independence. We
# also estimate EM empirically by Monte Carlo over N_EVAL items.
T_ANS, N_EVAL = 12, 2000
s = np.linspace(0, 1, 25)                       # normalized log-compute grid
p = np.linspace(0.25, 0.99, 25)                 # smooth: constant increments
em_exact = p ** T_ANS
em_mc = np.array([np.mean(np.all(rng.random((N_EVAL, T_ANS)) < pi, axis=1))
                  for pi in p])

d_p = np.max(np.abs(np.diff(p)))                # largest per-step jump
d_em = np.max(np.abs(np.diff(em_exact)))
d_em_mc = np.max(np.abs(np.diff(em_mc)))
amp_bound = T_ANS * np.max(p[1:] ** (T_ANS - 1) * np.abs(np.diff(p)))
print("\n=== Part 3: smooth curve + exact match = apparent emergence ===")
print(f"  per-token curve:  max per-step jump = {d_p:.4f}  (smooth)")
print(f"  exact-match (T={T_ANS}): max per-step jump = {d_em:.4f} "
      f"({d_em/d_p:.1f}x amplification)")
print(f"  Monte-Carlo EM ({N_EVAL} items):  max per-step jump = {d_em_mc:.4f}")
print(f"  chain-rule bound T*p^(T-1)*dp accounts for it: {amp_bound:.4f} >= {d_em:.4f}")
# EM is negligible until p is already large: EM < 0.05 requires p < 0.05^(1/T)
p_thresh = 0.05 ** (1 / T_ANS)
frac_hidden = np.mean(em_exact < 0.05)
print(f"  EM stays < 0.05 until per-token p exceeds {p_thresh:.3f}; "
      f"{frac_hidden:.0%} of the grid looks like 'no ability'")
assert d_p < 0.04, "underlying curve is not smooth; construction failed"
assert d_em > 0.15 and d_em_mc > 0.15, "metric curve failed to jump"
assert d_em > 4 * d_p, "metric jump should dwarf the underlying increments"
assert amp_bound >= d_em - 1e-9
print("  underlying smooth (jump < 0.04), metric sharp (jump > 0.15, "
      "> 4x underlying)  OK")

if __name__ == "__main__":
    print("\nAll assertions passed: same models + same items, different "
          "protocol => different leaderboard; memorized text is detectable "
          "from its missing surprisal tail; sharp 'emergence' can be "
          "manufactured from a smooth curve by the metric alone.")
