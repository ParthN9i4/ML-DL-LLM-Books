"""
ex26 — Speculative decoding, with the proof as an assertion.  (Book: Chapter 33)

A small DRAFT model proposes gamma tokens; the big TARGET model verifies them
all in one parallel pass. The magic is the acceptance rule:

    accept draft token a with probability  min(1, p(a) / q(a))
    on rejection, resample from the RESIDUAL  max(0, p - q) / sum(max(0, p - q))

With this rule the output distribution is EXACTLY p — not approximately, not
"close for a good draft": exactly. The draft model can be arbitrarily bad and
you still sample from the target; a bad draft only costs SPEED (more
rejections), never correctness.

That claim is a theorem with a three-line proof:

    P(output = a) = q(a) * min(1, p(a)/q(a))                    [accepted]
                  + P(reject) * residual(a)                     [resampled]
    = min(q(a), p(a)) + (1 - sum_b min(p(b), q(b))) * (p(a) - min(p(a), q(a))) / (1 - sum_b min(p(b), q(b)))
    = min(q(a), p(a)) + p(a) - min(p(a), q(a)) = p(a).

And because it is exact, it is TESTABLE: sample a few hundred thousand tokens
and run a chi-square test against direct target sampling. That is what the
checks below do. The expected number of tokens per verification round is also
closed-form: (1 - alpha^(gamma+1)) / (1 - alpha) where alpha = sum_a min(p, q)
is the acceptance rate. Measured against the formula below.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


def acceptance_rate(p, q):
    """alpha = sum_a min(p(a), q(a)) — the total-variation complement.

    This is THE quantity: it is 1 - TV(p, q), so the speedup is a direct
    function of how close the draft is to the target.
    """
    # === YOUR CODE HERE ===
    return float(np.minimum(p, q).sum())


def residual_dist(p, q):
    """The resampling distribution max(0, p - q), normalized.

    Guard the degenerate case p == q (residual mass 0): any distribution works
    because it is never sampled; return p for definiteness.
    """
    # === YOUR CODE HERE ===
    r = np.maximum(0.0, p - q)
    s = r.sum()
    if s <= 0:
        return p.copy()
    return r / s


def speculative_step(p, q, gamma, rng):
    """One draft-and-verify round with a STATIC pair (p, q) per position.

    Draft samples gamma tokens i.i.d. from q; the target verifies each in
    sequence with the min(1, p/q) rule; the first rejection resamples from the
    residual and ends the round; if all gamma are accepted, one bonus token is
    sampled directly from p (the "free" token of the parallel verification).

    Returns the list of emitted tokens (length 1..gamma+1).
    """
    out = []
    # === YOUR CODE HERE ===
    for _ in range(gamma):
        a = rng.choice(len(q), p=q)
        if rng.random() < min(1.0, p[a] / max(q[a], 1e-300)):
            out.append(a)
        else:
            out.append(rng.choice(len(p), p=residual_dist(p, q)))
            return out
    out.append(rng.choice(len(p), p=p))
    return out


def expected_tokens_per_round(alpha, gamma):
    """E[tokens emitted per round] = (1 - alpha^(gamma+1)) / (1 - alpha)."""
    # === YOUR CODE HERE ===
    if abs(1.0 - alpha) < 1e-12:
        return gamma + 1.0
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def chi_square_stat(counts, probs):
    """Pearson chi-square of observed counts against expected probabilities."""
    n = counts.sum()
    expected = n * probs
    mask = expected > 0
    # === YOUR CODE HERE ===
    return float(((counts[mask] - expected[mask]) ** 2 / expected[mask]).sum())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K = 20

    def rand_dist(temp=1.0):
        z = rng.standard_normal(K) / temp
        e = np.exp(z - z.max())
        return e / e.sum()

    p = rand_dist()
    q_close = 0.8 * p + 0.2 * rand_dist()      # a good draft
    q_close /= q_close.sum()
    q_far = rand_dist(temp=0.4)                # a bad draft, sharply different

    print("\n--- the machinery ---")
    check("acceptance rate is 1 - TV(p, q)",
          acceptance_rate(p, q_close), 1.0 - 0.5 * float(np.abs(p - q_close).sum()), tol=1e-12)
    check("residual is a distribution", float(residual_dist(p, q_far).sum()), 1.0, tol=1e-12)
    check("residual is zero wherever q >= p",
          float(residual_dist(p, q_far)[q_far >= p].sum()), 0.0, tol=1e-12)
    check("identical draft accepts everything", acceptance_rate(p, p), 1.0, tol=1e-12)

    print("\n--- THE theorem: output distribution is exactly p ---")
    # Collect a large sample of emitted tokens through the speculative pipeline
    # and chi-square it against p. Do it for BOTH drafts — the bad one included:
    # correctness must not depend on draft quality.
    for name, q in (("good draft", q_close), ("bad draft", q_far)):
        tokens = []
        while len(tokens) < 200_000:
            tokens.extend(speculative_step(p, q, gamma=4, rng=rng))
        tokens = np.array(tokens[:200_000])
        counts = np.bincount(tokens, minlength=K)
        stat = chi_square_stat(counts, p)
        # 19 dof: 95th percentile is 30.14. A correct sampler lands under it
        # ~95% of the time; a subtly wrong one (see break-it) lands in the
        # thousands. Use a generous threshold to keep the seed-flake rate low.
        alpha = acceptance_rate(p, q)
        print(f"      {name}: alpha = {alpha:.3f}, chi-square(19 dof) = {stat:.1f}")
        check(f"{name}: output matches target (chi-square < 43, i.e. p > 0.001)", stat < 43.0)

    print("\n--- the speedup formula ---")
    for name, q in (("good draft", q_close), ("bad draft", q_far)):
        alpha = acceptance_rate(p, q)
        gamma = 4
        rounds = 40_000
        emitted = sum(len(speculative_step(p, q, gamma, rng)) for _ in range(rounds))
        measured = emitted / rounds
        predicted = expected_tokens_per_round(alpha, gamma)
        print(f"      {name}: measured {measured:.3f} tokens/round, "
              f"predicted {predicted:.3f}")
        check(f"{name}: measured rate matches (1-a^(g+1))/(1-a) within 2%",
              measured, predicted, tol=0.02 * predicted)
    # And the punchline in words: the good draft emits more tokens per
    # expensive target pass; the bad draft fewer — but both are EXACT.

    print("\n--- gamma tuning is a real trade ---")
    # More drafting only helps while the draft keeps getting accepted:
    # tokens/round saturates at 1/(1-alpha) as gamma grows.
    alpha = acceptance_rate(p, q_close)
    rates = [expected_tokens_per_round(alpha, g) for g in (1, 2, 4, 8, 16, 64)]
    print("      gamma:        " + " ".join(f"{g:>6}" for g in (1, 2, 4, 8, 16, 64)))
    print("      tokens/round: " + " ".join(f"{r:6.2f}" for r in rates))
    print(f"      asymptote 1/(1-alpha) = {1/(1-alpha):.2f}")
    check("tokens/round increases in gamma", all(rates[i] < rates[i+1] for i in range(len(rates)-1)))
    check("but saturates below 1/(1-alpha)", rates[-1] < 1 / (1 - alpha))

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) The tempting shortcut: on rejection, resample from the TARGET p
    #     instead of the residual. Feels harmless — p is the right distribution,
    #     surely? No: accepted tokens are already biased toward min(p, q), and
    #     only the residual corrects that bias. Chi-square catches it instantly.
    def speculative_wrong(p_, q_, gamma, rng_):
        out = []
        for _ in range(gamma):
            a = rng_.choice(len(q_), p=q_)
            if rng_.random() < min(1.0, p_[a] / max(q_[a], 1e-300)):
                out.append(a)
            else:
                out.append(rng_.choice(len(p_), p=p_))     # WRONG: not the residual
                return out
        out.append(rng_.choice(len(p_), p=p_))
        return out

    tokens = []
    while len(tokens) < 200_000:
        tokens.extend(speculative_wrong(p, q_far, 4, rng))
    counts = np.bincount(np.array(tokens[:200_000]), minlength=K)
    stat_wrong = chi_square_stat(counts, p)
    print(f"      resampling from p instead of the residual: chi-square = {stat_wrong:.0f}")
    check("the wrong resampler is detected decisively (chi-square > 200)", stat_wrong > 200.0)

    # (b) Zero-probability draft support: if q(a) = 0 where p(a) > 0, the
    #     acceptance rule alone can NEVER emit a — only the residual can. Drop
    #     the residual and the output support is silently wrong.
    q_hole = p.copy()
    q_hole[3] = 0.0
    q_hole /= q_hole.sum()
    tokens = []
    while len(tokens) < 50_000:
        tokens.extend(speculative_step(p, q_hole, 4, rng))
    counts = np.bincount(np.array(tokens[:50_000]), minlength=K)
    frac = counts[3] / 50_000
    print(f"      token 3 has q = 0, p = {p[3]:.4f}; emitted fraction with residual = {frac:.4f}")
    check("the residual restores support the draft cannot propose",
          frac, float(p[3]), tol=0.15 * float(p[3]) + 0.002)

    summary()
