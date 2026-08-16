"""
ex02 — Probability, information, and why cross-entropy is the only loss.
(Book: Chapter 2)

The single most useful identity in this book:

    minimizing cross-entropy  ==  maximizing likelihood  ==  minimizing KL
                              ==  minimizing compressed file size

They are four descriptions of one procedure.  Once you believe that, "next-token
prediction" stops looking like a trick and starts looking like a compression
bound — which is the frame Chapter 24 uses to talk about scaling.

Facts worth having in your hands rather than your notes:

  * H(p) is the expected code length of the BEST possible code for p.
  * H(p, q) is what you actually pay when you code p-distributed data with a
    code built for q.  The excess, H(p,q) - H(p), is exactly D_KL(p || q) >= 0.
  * So KL is not an abstract "distance". It is a bill, denominated in bits,
    for using the wrong model.
  * Perplexity is just exp(cross-entropy in nats).  It is NOT comparable across
    tokenizers; bits-per-byte is.  Chapter 23 belabours this because the
    literature still gets it wrong.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


# ---------------------------------------------------------------------------
# 1. The information-theoretic quantities
# ---------------------------------------------------------------------------

def entropy(p, base=2):
    """H(p) = -sum p_i log p_i.  Terms with p_i = 0 contribute 0 (0 log 0 := 0)."""
    p = np.asarray(p, dtype=np.float64)
    # === YOUR CODE HERE ===
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(p[nz]) / np.log(base)))


def cross_entropy(p, q, base=2):
    """H(p, q) = -sum p_i log q_i.  Infinite if q puts zero mass where p does not."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # === YOUR CODE HERE ===
    nz = p > 0
    if np.any(q[nz] == 0):
        return np.inf
    return float(-np.sum(p[nz] * np.log(q[nz]) / np.log(base)))


def kl_divergence(p, q, base=2):
    """D_KL(p || q) = sum p_i log(p_i / q_i) = H(p, q) - H(p).  Never negative."""
    # === YOUR CODE HERE ===
    return cross_entropy(p, q, base) - entropy(p, base)


def softmax(z):
    """Numerically stable softmax: subtract the max before exponentiating.

    Without the shift, exp overflows at z ~ 89 in float32.  With it, the largest
    exponent is exactly 0, so exp never exceeds 1.  See ex04.
    """
    z = np.asarray(z, dtype=np.float64)
    # === YOUR CODE HERE ===
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# 2. Maximum likelihood == cross-entropy minimization
# ---------------------------------------------------------------------------

def nll_from_logits(logits, labels):
    """Mean negative log-likelihood in NATS, from raw logits and integer labels.

    This is exactly torch's cross_entropy.  Compute it via log-sum-exp rather
    than log(softmax(z)) — the latter loses precision for confident predictions.
    """
    logits = np.asarray(logits, dtype=np.float64)
    # === YOUR CODE HERE ===
    m = np.max(logits, axis=-1, keepdims=True)
    logZ = m[..., 0] + np.log(np.sum(np.exp(logits - m), axis=-1))
    return float(np.mean(logZ - logits[np.arange(len(labels)), labels]))


# ---------------------------------------------------------------------------
# 3. Perplexity, bits per token, bits per byte
# ---------------------------------------------------------------------------

def perplexity(nll_nats):
    """PPL = exp(mean NLL in nats)."""
    # === YOUR CODE HERE ===
    return float(np.exp(nll_nats))


def bits_per_byte(total_nats, n_bytes):
    """Total loss in nats over a corpus, converted to bits per BYTE of raw text.

    This is the only cross-tokenizer-comparable number. A tokenizer that packs
    more text into each token gets a lower loss per token for free; dividing by
    the byte count of the underlying text removes that advantage.
    """
    # === YOUR CODE HERE ===
    return float(total_nats / np.log(2) / n_bytes)


# ---------------------------------------------------------------------------
# Checks — these ARE the specification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("\n--- entropy ---")
    check("uniform over 8 has entropy 3 bits", entropy([1 / 8] * 8), 3.0, tol=1e-12)
    check("a point mass has entropy 0", entropy([0, 1, 0]), 0.0, tol=1e-12)
    check("fair coin = 1 bit", entropy([0.5, 0.5]), 1.0, tol=1e-12)
    check("0 log 0 handled, not nan", np.isfinite(entropy([0.0, 0.5, 0.5])))
    # Uniform maximizes entropy over a fixed alphabet.
    check("uniform maximizes entropy",
          all(entropy(rng.dirichlet(np.ones(8))) <= 3.0 + 1e-12 for _ in range(200)))

    print("\n--- cross-entropy and KL ---")
    p = np.array([0.5, 0.25, 0.125, 0.125])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    # H(p) = 1.75 bits: this p is exactly a dyadic (Huffman-friendly) distribution.
    check("H(p) = 1.75 bits", entropy(p), 1.75, tol=1e-12)
    check("H(p,q) = 2 bits under uniform q", cross_entropy(p, q), 2.0, tol=1e-12)
    check("KL = H(p,q) - H(p) = 0.25 bits", kl_divergence(p, q), 0.25, tol=1e-12)
    check("KL(p||p) = 0", kl_divergence(p, p), 0.0, tol=1e-12)
    check("KL is non-negative (Gibbs)",
          all(kl_divergence(rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))) >= -1e-12
              for _ in range(300)))
    # Asymmetry is the whole reason forward vs reverse KL behave differently.
    a, b = np.array([0.9, 0.1]), np.array([0.5, 0.5])
    check("KL is asymmetric", abs(kl_divergence(a, b) - kl_divergence(b, a)) > 1e-3)
    check("KL is infinite when q misses support of p",
          np.isinf(kl_divergence([0.5, 0.5], [1.0, 0.0])))

    print("\n--- softmax and NLL ---")
    z = np.array([1.0, 2.0, 3.0])
    s = softmax(z)
    check("softmax sums to 1", float(s.sum()), 1.0, tol=1e-12)
    check("softmax is shift-invariant", softmax(z + 1000.0), s, tol=1e-12)
    check("softmax does not overflow at z=1000", bool(np.all(np.isfinite(softmax([1000.0, 0.0])))))

    logits = rng.standard_normal((256, 10))
    labels = rng.integers(0, 10, size=256)
    nll = nll_from_logits(logits, labels)
    # The identity: mean NLL == mean cross-entropy of the one-hot target vs softmax.
    manual = np.mean([
        cross_entropy(np.eye(10)[labels[i]], softmax(logits[i]), base=np.e)
        for i in range(256)
    ])
    check("NLL == cross-entropy against one-hot targets", nll, manual, tol=1e-10)

    try:
        import torch
        t_nll = torch.nn.functional.cross_entropy(
            torch.tensor(logits), torch.tensor(labels)
        ).item()
        check("NLL matches torch.nn.functional.cross_entropy", nll, t_nll, tol=1e-9)
    except ImportError:
        print("  ....  [skipped: torch not installed] torch cross-entropy cross-check")

    print("\n--- perplexity and bits-per-byte ---")
    # A model that is exactly uniform over V tokens has perplexity V.
    V = 50257
    uniform_nll = np.log(V)
    check("uniform model has perplexity = vocab size", perplexity(uniform_nll), V, tol=1e-6)
    check("perplexity of a perfect model is 1", perplexity(0.0), 1.0, tol=1e-12)

    # THE POINT: two tokenizers, same underlying text, same true model quality.
    # Tokenizer A emits 1000 tokens; tokenizer B packs the same bytes into 700.
    # Their perplexities differ wildly. Their bits-per-byte agree.
    n_bytes = 4000
    total_nats = 2200.0          # the model's total loss on this text, fixed
    ppl_a = perplexity(total_nats / 1000)
    ppl_b = perplexity(total_nats / 700)
    bpb_a = bits_per_byte(total_nats, n_bytes)
    bpb_b = bits_per_byte(total_nats, n_bytes)
    print(f"      tokenizer A: {1000} tokens -> perplexity {ppl_a:8.3f}, {bpb_a:.4f} bits/byte")
    print(f"      tokenizer B: {700} tokens -> perplexity {ppl_b:8.3f}, {bpb_b:.4f} bits/byte")
    check("perplexity differs across tokenizers on identical text", abs(ppl_a - ppl_b) > 5.0)
    check("bits-per-byte is identical across tokenizers", bpb_a, bpb_b, tol=1e-12)

    print("\n--- compression equals prediction ---")
    # Shannon's source coding theorem, empirically: the ideal code length for a
    # message under model q is -sum log2 q(symbol). A better model is a shorter
    # file, and the floor is the true entropy.
    probs = np.array([0.6, 0.2, 0.15, 0.05])
    data = rng.choice(4, size=20000, p=probs)
    counts = np.bincount(data, minlength=4)
    empirical = counts / counts.sum()

    true_bits = sum(-np.log2(probs[s]) for s in data)
    wrong_bits = sum(-np.log2(0.25) for s in data)
    print(f"      coded with the true model : {true_bits/8:8.0f} bytes")
    print(f"      coded with a uniform model: {wrong_bits/8:8.0f} bytes")
    print(f"      excess predicted by KL    : {len(data)*kl_divergence(probs,[0.25]*4)/8:8.0f} bytes")

    check("ideal code length / n approaches H(p)",
          true_bits / len(data), entropy(probs), tol=0.02)
    check("excess bytes == n * KL(p || uniform)",
          (wrong_bits - true_bits) / 8,
          len(data) * kl_divergence(probs, [0.25] * 4) / 8,
          tol=len(data) * 0.02 / 8)
    check("the better model gives the strictly smaller file", true_bits < wrong_bits)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) Naive softmax overflows. This is not hypothetical: unshifted logits in
    #     float32 overflow at ~88.7, and attention logits get large.
    big = np.array([1000.0, 0.0])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(big) / np.sum(np.exp(big))
    print(f"      naive softmax([1000, 0]) = {naive}  <- nan from inf/inf")
    check("naive softmax produces nan", bool(np.any(np.isnan(naive))))
    check("stable softmax does not", bool(np.all(np.isfinite(softmax(big)))))

    # (b) log(softmax(z)) vs the log-sum-exp form. For a confident prediction the
    #     probability underflows to 0 and the log is -inf, destroying the gradient.
    conf = np.array([[0.0, -800.0]])
    with np.errstate(divide="ignore"):
        naive_logp = np.log(softmax(conf)[0, 1])
    m = conf.max()
    stable_logp = float(conf[0, 1] - (m + np.log(np.sum(np.exp(conf - m)))))
    print(f"      log(softmax) = {naive_logp}, log-sum-exp form = {stable_logp:.1f}")
    check("log(softmax) underflows to -inf", np.isinf(naive_logp))
    check("log-sum-exp stays finite", np.isfinite(stable_logp))

    summary()
