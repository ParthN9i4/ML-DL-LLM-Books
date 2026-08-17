"""Artifact 33.1 -- Speculative decoding from scratch + a paged KV allocator.

Speculative decoding is a statement about SAMPLING, not about language models:
given a target distribution p and a cheap draft distribution q over a vocabulary,
the modified rejection rule emits tokens whose distribution is EXACTLY p while
letting q do most of the proposing.  We use fixed (context-independent) discrete
distributions so every quantity has a closed form to assert against:
  * output distribution == p          (chi-square + total-variation test)
  * acceptance rate     == sum_x min(p(x), q(x)) = 1 - TV(p, q)
  * tokens/verification == (1 - alpha^(gamma+1)) / (1 - alpha)
Part 2 is a block-granular ("paged") KV allocator measured against contiguous
max-length pre-allocation on random-length requests.
"""
import numpy as np
from scipy import stats

try:
    import torch
except ImportError:
    torch = None


# ----------------------------------------------------------------------------
# Part 1: speculative decoding over explicit distributions
# ----------------------------------------------------------------------------
def sample(cdf, u):
    """Inverse-CDF sampling: map uniforms u in [0,1) to vocabulary indices."""
    return np.searchsorted(cdf, u, side="right")


def spec_decode(p, q, gamma, n_cycles, rng):
    """Run n_cycles of speculative decoding, vectorized over cycles.

    Each cycle: q proposes gamma tokens; the target verifies them left to
    right, accepting draft x with prob min(1, p(x)/q(x)); on first rejection
    it resamples from the residual max(0, p-q)/(1-alpha) and the cycle ends;
    if all gamma are accepted it draws one bonus token directly from p.
    Returns (token counts, #accepted drafts, #examined drafts, #tokens).
    """
    V = p.size
    q_cdf, p_cdf = np.cumsum(q), np.cumsum(p)
    residual = np.maximum(p - q, 0.0)
    residual /= residual.sum()                       # = (p-q)^+ / (1-alpha)
    r_cdf = np.cumsum(residual)

    drafts = sample(q_cdf, rng.random((n_cycles, gamma)))       # proposals ~ q
    accept_p = np.minimum(1.0, p[drafts] / q[drafts])           # min(1, p/q)
    acc = rng.random((n_cycles, gamma)) < accept_p
    # position of first rejection in each cycle; gamma if none
    first_rej = np.where(acc.all(axis=1), gamma, acc.argmin(axis=1))

    keep = np.arange(gamma)[None, :] < first_rej[:, None]       # accepted prefix
    kept_tokens = drafts[keep]
    # last token of the cycle: residual resample on rejection, else bonus ~ p
    u = rng.random(n_cycles)
    final = np.where(first_rej < gamma, sample(r_cdf, u), sample(p_cdf, u))

    counts = np.bincount(kept_tokens, minlength=V) + np.bincount(final, minlength=V)
    accepted = int(keep.sum())
    examined = accepted + int((first_rej < gamma).sum())  # verify stops at 1st reject
    return counts, accepted, examined, accepted + n_cycles


# ----------------------------------------------------------------------------
# Part 2: paged KV allocator vs contiguous pre-allocation
# ----------------------------------------------------------------------------
class PagedKVAllocator:
    """vLLM-style block allocator: KV slots are handed out in fixed-size
    blocks from a free list; a request's 'block table' maps its logical
    token positions to physical blocks.  A request of length T holds
    ceil(T/B) blocks, so waste is < B slots per request."""

    def __init__(self, n_blocks, block_size):
        self.B = block_size
        self.free = list(range(n_blocks))
        self.tables = {}                      # request id -> list of block ids
        self.lens = {}                        # request id -> tokens written

    def append_token(self, rid):
        n = self.lens.get(rid, 0)
        if n % self.B == 0:                   # current block full (or first token)
            self.tables.setdefault(rid, []).append(self.free.pop())
        self.lens[rid] = n + 1

    def allocated_slots(self):
        return sum(len(t) for t in self.tables.values()) * self.B


def fragmentation_experiment(rng, n_req=1000, max_len=2048, block=16):
    """All requests resident concurrently; lengths ~ Uniform{1..max_len}.
    Contiguous serving reserves max_len slots per request up front (it cannot
    know the final length); paging allocates blocks on demand."""
    lengths = rng.integers(1, max_len + 1, size=n_req)
    alloc = PagedKVAllocator(n_blocks=(n_req * (max_len // block + 1)), block_size=block)
    for rid, T in enumerate(lengths):        # feed tokens one at a time
        for _ in range(int(T)):
            alloc.append_token(rid)
    used = int(lengths.sum())
    paged_alloc = alloc.allocated_slots()
    # cross-check the allocator against the closed form ceil(T/B)*B
    assert paged_alloc == int((np.ceil(lengths / block) * block).sum())
    contig_alloc = n_req * max_len
    return used, paged_alloc, contig_alloc


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    V, gamma, n_cycles = 32, 4, 200_000
    p = rng.dirichlet(np.ones(V))                     # target distribution
    q = rng.dirichlet(np.ones(V))                     # draft distribution
    alpha = np.minimum(p, q).sum()                    # analytic acceptance rate

    counts, accepted, examined, n_tok = spec_decode(p, q, gamma, n_cycles, rng)

    # (a) output distribution == target, tested two ways
    emp = counts / n_tok
    tv = 0.5 * np.abs(emp - p).sum()
    chi2, pval = stats.chisquare(counts, f_exp=n_tok * p)
    print(f"[a] tokens={n_tok}  TV(empirical, p)={tv:.5f}  "
          f"chi2={chi2:.1f} (df={V-1})  p-value={pval:.3f}")
    # reference: direct i.i.d. sampling from p at the same sample size
    direct = np.bincount(sample(np.cumsum(p), rng.random(n_tok)), minlength=V)
    tv_direct = 0.5 * np.abs(direct / n_tok - p).sum()
    print(f"    reference TV for direct target sampling, same n: {tv_direct:.5f}")
    assert tv < 0.01, "speculative output drifted from target distribution"
    assert pval > 1e-3, "chi-square rejects distributional equivalence"

    # (b) measured acceptance rate vs analytic sum min(p, q)
    meas_alpha = accepted / examined
    print(f"[b] acceptance: measured={meas_alpha:.5f}  analytic={alpha:.5f}  "
          f"|diff|={abs(meas_alpha - alpha):.5f}")
    assert abs(meas_alpha - alpha) < 5e-3

    # (c) tokens per verification cycle vs (1 - a^(g+1)) / (1 - a)
    meas_tok = n_tok / n_cycles
    exp_tok = (1 - alpha ** (gamma + 1)) / (1 - alpha)
    print(f"[c] tokens/cycle: measured={meas_tok:.4f}  analytic={exp_tok:.4f}  "
          f"|diff|={abs(meas_tok - exp_tok):.4f}")
    assert abs(meas_tok - exp_tok) < 0.02

    # cross-check (a) with torch.multinomial as an independent sampler
    if torch is not None:
        tt = torch.multinomial(torch.from_numpy(p), n_tok, replacement=True,
                               generator=torch.Generator().manual_seed(0))
        tv_torch = 0.5 * np.abs(np.bincount(tt.numpy(), minlength=V) / n_tok - p).sum()
        print(f"    torch.multinomial cross-check TV: {tv_torch:.5f}")
        assert tv_torch < 0.01
    else:
        print("    [skipped: torch not installed]")

    # (d) paged allocator vs contiguous pre-allocation
    used, paged_alloc, contig_alloc = fragmentation_experiment(rng)
    frag_paged = 100 * (1 - used / paged_alloc)
    frag_contig = 100 * (1 - used / contig_alloc)
    print(f"[d] KV slots used={used}  paged alloc={paged_alloc}  "
          f"contiguous alloc={contig_alloc}")
    print(f"    fragmentation: contiguous={frag_contig:.2f}%  "
          f"paged (B=16)={frag_paged:.3f}%")
    assert frag_paged < frag_contig / 10, "paging should cut waste >10x here"

    print("All assertions passed.")
