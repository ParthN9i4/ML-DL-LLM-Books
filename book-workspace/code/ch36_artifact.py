"""Artifact 36.1 -- Horizon compounding in a deterministic sandbox agent loop.

A tiny agent environment with three in-process tools (calculator, key-value
store, append-only file) driven by a SCRIPTED policy with an injectable
per-step error rate.  No LLM calls anywhere: the point is the reliability
arithmetic, measured through the environment rather than assumed.

Task: maintain a running sum acc_t = sum_{i<=t} (2i+1) = t^2 + 2t and log
"t:acc_t" to the file each step.  An episode of horizon H succeeds iff the
final file equals the ground-truth log exactly.

Three regimes:
  (a) independent  -- each step re-derives its value from the plan; an injected
      error corrupts only that step's line.  Prediction: success = p^H.
  (b) corrupt      -- the policy reads the accumulator back from the KV store,
      and glitches leave junk keys that persist; the more junk, the likelier a
      later read grabs junk (state corruption raising the later hazard).
      Prediction: success measurably WORSE than p^H at the same injected rate.
  (c) verify       -- like (a), plus a per-step verifier (a unit test for the
      segment) that catches a bad step and retries it up to r times.
      Prediction: success = q^H with q = 1-(1-p)^(r+1) (geometric retry).
"""
import numpy as np

# ------------------------------ sandbox tools -------------------------------
class Calculator:                                   # pure, deterministic
    def apply(self, op, a, b):
        if op == "add": return a + b
        if op == "mul": return a * b
        raise ValueError(op)

class KVStore:                                      # mutable state that persists
    def __init__(self): self.d = {}
    def set(self, k, v): self.d[k] = v
    def get(self, k, default=None): return self.d.get(k, default)
    def junk_count(self): return sum(1 for k in self.d if k.startswith("tmp"))

class FileTool:                                     # append-only log
    def __init__(self): self.lines = []
    def append(self, line): self.lines.append(line)

def truth_lines(H):
    return [f"{t}:{t * t + 2 * t}" for t in range(1, H + 1)]

# ------------------------------ one episode ---------------------------------
def run_episode(H, p, rng, mode, truth, retries=0, glitch_p=0.10, snag_p=0.05):
    calc, kv, log = Calculator(), KVStore(), FileTool()
    for t in range(1, H + 1):
        if mode == "independent":
            # Stateless re-derivation: acc_t = t*t + 2t computed fresh from the
            # plan, so an error here cannot propagate to later steps.
            acc = calc.apply("add", calc.apply("mul", t, t), 2 * t)
            if rng.random() > p:                    # injected action error
                acc += int(rng.integers(1, 10))
        elif mode == "corrupt":
            # Stateful accumulation: read acc back from the store.  Each junk
            # key snags the read with prob snag_p (independently), so the
            # per-step hazard grows with accumulated corruption.
            j = kv.junk_count()
            prev = kv.get("acc", 0)
            if j and rng.random() > (1.0 - snag_p) ** j:
                prev = 999_983                      # grabbed a junk entry
            acc = calc.apply("add", prev, 2 * t + 1)
            if rng.random() > p:                    # same injected rate as (a)
                acc += int(rng.integers(1, 10))
            if rng.random() < glitch_p:             # glitch leaves junk behind
                kv.set(f"tmp{t}", 999_983)
        elif mode == "verify":
            # Verifiable segment: the environment can unit-test each step's
            # output; a failed check rolls the step back and retries.
            expected = t * t + 2 * t
            for _ in range(retries + 1):
                acc = calc.apply("add", calc.apply("mul", t, t), 2 * t)
                if rng.random() > p:
                    acc += int(rng.integers(1, 10))
                if acc == expected:
                    break                           # verifier accepts
        kv.set("acc", acc)
        log.append(f"{t}:{acc}")
    ok_lines = sum(a == b for a, b in zip(log.lines, truth))
    return log.lines == truth, ok_lines            # success judged via the env

def simulate(H, p, n, rng, mode, retries=0):
    truth = truth_lines(H)
    wins = tot_ok = 0
    for _ in range(n):
        w, ok = run_episode(H, p, rng, mode, truth, retries)
        wins += w; tot_ok += ok
    return wins / n, tot_ok / (n * H)              # success rate, line accuracy

# --------------------------------- driver -----------------------------------
if __name__ == "__main__":
    rng  = np.random.default_rng(0)
    p, r, N = 0.90, 2, 3000                        # per-step accuracy, retries, episodes
    q    = 1.0 - (1.0 - p) ** (r + 1)              # geometric-retry effective accuracy
    grid = [1, 2, 4, 8, 16, 24, 32, 48, 64]
    sd   = lambda s: (max(s * (1 - s), 1e-12) / N) ** 0.5

    print(f"p = {p}  retries = {r}  q = 1-(1-p)^{r+1} = {q:.6f}  N = {N} episodes/point")
    print(f"{'H':>3} {'p^H':>8} {'indep':>8} {'corrupt':>8} {'verify':>8} {'q^H':>8}")
    res = {}
    for H in grid:
        sA, lA = simulate(H, p, N, rng, "independent")
        sB, lB = simulate(H, p, N, rng, "corrupt")
        sC, lC = simulate(H, p, N, rng, "verify", retries=r)
        res[H] = (sA, lA, sB, lB, sC)
        print(f"{H:>3} {p**H:>8.4f} {sA:>8.4f} {sB:>8.4f} {sC:>8.4f} {q**H:>8.4f}")

    # (a) independent errors track p^H within binomial tolerance at every H
    worst = max(abs(res[H][0] - p ** H) - 4.5 * sd(p ** H) for H in grid)
    print(f"\n[a] max (|indep - p^H| - 4.5*sigma) over H grid: {worst:+.4f}  (<= 0.004 required)")
    for H in grid:
        assert abs(res[H][0] - p ** H) <= 4.5 * sd(p ** H) + 0.004, f"(a) fails at H={H}"
    phatA = res[16][1]                              # measured per-step accuracy, mode (a)
    print(f"[a] measured per-step line accuracy at H=16: {phatA:.4f} (injected p = {p})")
    assert abs(phatA - p) < 0.01

    # (b) persistent corruption: worse than p^H, by >3 pooled-sigma at H=16,32
    for H in (16, 32):
        sA, _, sB, _, _ = res[H]
        gap  = p ** H - sB
        pool = ((p**H * (1 - p**H) + sB * (1 - sB)) / N) ** 0.5
        print(f"[b] H={H}: p^H - corrupt = {gap:.4f}  ({gap/pool:.1f} pooled sigma)")
        assert gap > 3 * pool, f"(b) not separated at H={H}"
    sB16, lB16 = res[16][2], res[16][3]             # independence fails both ways:
    print(f"[b] H=16 marginal line accuracy {lB16:.4f}: naive {lB16:.3f}^16 = "
          f"{lB16**16:.2e} << measured success {sB16:.4f} < p^16 = {p**16:.4f}")
    assert lB16 ** 16 < sB16 < p ** 16              # under- and over-prediction

    # (c) verifier+retry matches the geometric prediction q^H and beats (a)
    worst = max(abs(res[H][4] - q ** H) - 4.5 * sd(q ** H) for H in grid)
    print(f"[c] max (|verify - q^H| - 4.5*sigma) over H grid: {worst:+.4f}  (<= 0.004 required)")
    for H in grid:
        assert abs(res[H][4] - q ** H) <= 4.5 * sd(q ** H) + 0.004, f"(c) fails at H={H}"
        if H >= 8:
            assert res[H][4] - res[H][0] > 0.2, f"(c) no lift at H={H}"
    print(f"[c] lift at H=64: verify {res[64][4]:.4f} vs indep {res[64][0]:.4f} "
          f"(analytic {q**64:.4f} vs {p**64:.4f})")

    # required per-step accuracy for 90% end-to-end success
    req = ", ".join(f"H={H}:{0.9 ** (1.0 / H):.5f}" for H in (8, 16, 32, 64))
    print(f"\nper-step accuracy required for 90% end-to-end success: {req}")
    print("ALL ASSERTIONS PASSED")
