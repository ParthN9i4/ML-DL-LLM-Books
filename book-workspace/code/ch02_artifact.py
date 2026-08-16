"""
Artifact 2.1 - Compression IS prediction.

An integer (range/arithmetic) coder driven by an order-k interpolated Markov model
over a synthetic-but-realistic text corpus. Three claims are verified numerically:
  (1) IDENTITY:  compressed bits == sum_t -log2 q(x_t | ctx) == N * H(p,q), to < 1%.
  (2) ROUNDTRIP: the decoder reconstructs the test text exactly.
  (3) MONOTONE:  a model with lower cross-entropy yields a strictly smaller file.
The coder is integer arithmetic on a 32-bit register pair; no float touches the
codeword. Floats live only in the model and are quantized to integer frequencies
out of TOTAL = 2**14 before reaching the coder -- which is why (1) holds to ~1e-5
rather than merely approximately. Pure NumPy core; torch is an optional cross-check.
"""
from __future__ import annotations
import math
from collections import defaultdict
import numpy as np

try:
    import torch
except ImportError:  # the artifact must run anywhere
    torch = None

RNG = np.random.default_rng(0)

# --- Arithmetic coder: 32-bit integer registers, Witten-Neal-Cleary underflow ---
PREC = 32
TOP = 1 << PREC                      # 2^32
MASK, HALF, QTR = TOP - 1, TOP >> 1, TOP >> 2
THREE_QTR = 3 * QTR
TOTAL = 1 << 14                      # frequency denominator (must satisfy TOTAL << TOP)

class Encoder:
    """Emits a bitstream. low/high are integers in [0, 2^32); never floats."""
    def __init__(self):
        self.low, self.high, self.pending, self.bits = 0, MASK, 0, []

    def _emit(self, bit: int) -> None:
        self.bits.append(bit)
        self.bits.extend([1 - bit] * self.pending)  # resolve straddling bits
        self.pending = 0

    def encode(self, cum_lo: int, cum_hi: int, total: int) -> None:
        rng = self.high - self.low + 1
        self.high = self.low + (rng * cum_hi) // total - 1
        self.low = self.low + (rng * cum_lo) // total
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QTR and self.high < THREE_QTR:
                self.pending += 1          # underflow: interval straddles the midpoint
                self.low -= QTR
                self.high -= QTR
            else:
                break
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) | 1) & MASK

    def finish(self) -> list[int]:
        self.pending += 1
        self._emit(0 if self.low < QTR else 1)
        return self.bits

class Decoder:
    def __init__(self, bits: list[int]):
        self.bits, self.pos = bits, 0
        self.low, self.high, self.value = 0, MASK, 0
        for _ in range(PREC):
            self.value = (self.value << 1) | self._bit()

    def _bit(self) -> int:
        b = self.bits[self.pos] if self.pos < len(self.bits) else 0
        self.pos += 1
        return b

    def target(self, total: int) -> int:
        """The scaled cumulative-frequency value the next symbol must contain."""
        rng = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // rng

    def consume(self, cum_lo: int, cum_hi: int, total: int) -> None:
        rng = self.high - self.low + 1
        self.high = self.low + (rng * cum_hi) // total - 1
        self.low = self.low + (rng * cum_lo) // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QTR and self.high < THREE_QTR:
                self.low -= QTR
                self.high -= QTR
                self.value -= QTR
            else:
                break
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) | 1) & MASK
            self.value = ((self.value << 1) & MASK) | self._bit()

# --- Corpus: Zipfian pseudo-word text with within-sentence repetition, so that
# --- higher-order context genuinely carries information (unlike i.i.d. noise).
def make_corpus(n_chars: int = 150_000) -> str:
    onsets = ["ka", "ro", "mi", "te", "lan", "pro", "dan", "vi", "sto", "nel",
              "ar", "bek", "tri", "mos", "hal", "sen", "dul", "fer"]
    codas = ["tion", "ing", "ent", "al", "ic", "um", "er", "os", "ade", "ly", "is"]
    vocab = sorted({onsets[i % len(onsets)] + onsets[(3 * i + 5) % len(onsets)] * (i % 2)
                    + codas[(7 * i) % len(codas)] for i in range(160)})
    ranks = np.arange(1, len(vocab) + 1)
    p = 1.0 / (ranks + 2.7) ** 1.05          # Zipf-Mandelbrot
    p /= p.sum()
    out = []
    while sum(len(s) for s in out) < n_chars:
        n_words = 4 + int(RNG.poisson(8))
        sent = []
        for _ in range(n_words):
            if sent and RNG.random() < 0.30:
                sent.append(sent[int(RNG.integers(len(sent)))])   # local repetition
            else:
                sent.append(vocab[int(RNG.choice(len(vocab), p=p))])
        text = " ".join(sent)
        if RNG.random() < 0.35:
            k = int(RNG.integers(1, max(2, len(sent))))
            text = " ".join(sent[:k]) + ", " + " ".join(sent[k:])
        out.append(text + ". ")
    return "".join(out)[:n_chars]

# --- Order-k model with Jelinek-Mercer style interpolation down to uniform. ---
def train_counts(idx: np.ndarray, K: int, A: int) -> list[dict]:
    models = [defaultdict(lambda: np.zeros(A, dtype=np.int64)) for _ in range(K + 1)]
    for t in range(len(idx)):
        s = int(idx[t])
        for k in range(K + 1):
            if t >= k:
                models[k][tuple(int(v) for v in idx[t - k:t])][s] += 1
    return models

def cond_probs(models: list[dict], ctx: tuple, A: int, alpha: float = 8.0) -> np.ndarray:
    """p_k(.|ctx) = (n_k + alpha * p_{k-1}) / (sum n_k + alpha), p_{-1} = uniform."""
    p = np.full(A, 1.0 / A)
    for k in range(len(models)):
        c = models[k].get(ctx[len(ctx) - k:] if k else ())
        if c is None:
            break                      # deeper contexts are unseen too; stop backing up
        p = (c + alpha * p) / (c.sum() + alpha)
    return p

def quantize(p: np.ndarray, total: int = TOTAL) -> np.ndarray:
    """Integer frequencies summing exactly to `total`, all >= 1 (no zero-probability
    symbol can ever be coded, so this is what makes the code universal)."""
    f = np.maximum(1, np.floor(p * total).astype(np.int64))
    f[int(np.argmax(f))] += total - int(f.sum())
    assert f.sum() == total and f.min() >= 1
    return f

class FreqSource:
    """Deterministic map context -> (freqs, cumulative). Encoder and decoder call
    this identically, so the decoder needs no side information."""
    def __init__(self, models, A, K):
        self.models, self.A, self.K, self.cache = models, A, K, {}

    def get(self, ctx: tuple):
        hit = self.cache.get(ctx)
        if hit is None:
            f = quantize(cond_probs(self.models, ctx, self.A))
            hit = (f, np.concatenate(([0], np.cumsum(f))))
            self.cache[ctx] = hit
        return hit

def code_and_check(idx: np.ndarray, src: FreqSource, K: int):
    """Encode, decode, and accumulate the ideal code length sum -log2 q."""
    enc, ideal_bits = Encoder(), 0.0
    for t in range(K, len(idx)):
        f, cum = src.get(tuple(int(v) for v in idx[t - K:t]))
        s = int(idx[t])
        enc.encode(int(cum[s]), int(cum[s + 1]), TOTAL)
        ideal_bits += -math.log2(f[s] / TOTAL)
    bits = enc.finish()
    dec, rec = Decoder(bits), list(int(v) for v in idx[:K])
    for _ in range(K, len(idx)):
        f, cum = src.get(tuple(rec[-K:]) if K else ())
        s = int(np.searchsorted(cum, dec.target(TOTAL), side="right") - 1)
        dec.consume(int(cum[s]), int(cum[s + 1]), TOTAL)
        rec.append(s)
    return len(bits), ideal_bits, np.array(rec, dtype=idx.dtype)

if __name__ == "__main__":
    text = make_corpus()
    alphabet = sorted(set(text))
    A, stoi = len(alphabet), {c: i for i, c in enumerate(alphabet)}
    idx, cut = np.array([stoi[c] for c in text], dtype=np.int16), int(0.70 * len(text))
    train, test = idx[:cut], idx[cut:cut + 40_000]
    print(f"corpus: {len(idx)} chars, alphabet A={A}, train={len(train)}, test={len(test)}")
    print(f"raw test size: {len(test)} bytes = {8*len(test)} bits\n")
    print(f"{'order k':>8} {'H_q bits/sym':>13} {'coded bits':>11} "
          f"{'ideal bits':>11} {'rel. gap':>10} {'bytes':>8} {'exact':>6}")
    results, src3 = {}, None
    for K in (0, 1, 2, 3, 4):
        src = FreqSource(train_counts(train, K, A), A, K)
        n_bits, ideal, rec = code_and_check(test, src, K)
        H, gap = ideal / (len(test) - K), abs(n_bits - ideal) / ideal
        ok = bool(np.array_equal(rec, test))
        src3 = src if K == 3 else src3
        results[K] = (n_bits, H, gap, ok)
        print(f"{K:>8} {H:>13.4f} {n_bits:>11} {ideal:>11.1f} {gap:>9.4%} "
              f"{math.ceil(n_bits/8):>8} {str(ok):>6}")
        assert ok, f"round-trip FAILED at order {K}"
        assert gap < 0.01, f"identity violated at order {K}: {gap:.4%}"
    # (3) strict monotone improvement in file size as the model improves
    sizes = [results[K][0] for K in (0, 1, 2, 3)]
    assert all(a > b for a, b in zip(sizes, sizes[1:])), sizes
    print(f"\nstrict improvement order0 -> order3: {sizes[0]} -> {sizes[3]} bits "
          f"({100*(1-sizes[3]/sizes[0]):.1f}% smaller); compression ratio "
          f"{8*len(test)/sizes[3]:.2f}x vs raw 8 bits/char")
    # ---- perplexity / bits-per-token / bits-per-byte (Section 2.5) ----
    best_bits, best_H = results[3][0], results[3][1]
    n_bytes = len(test)                      # 1 char == 1 byte for this alphabet
    n_words = len(bytes(text[cut:cut + 40_000], "ascii").split())
    print(f"\ncharacter tokenizer: {best_H:.4f} bits/token, ppl "
          f"{2**best_H:.3f}, {best_bits/n_bytes:.4f} bits/byte")
    print(f"word tokenizer     : {best_bits/n_words:.4f} bits/token, ppl "
          f"{2**(best_bits/n_words):.1f}, {best_bits/n_bytes:.4f} bits/byte")
    print("  same file, same bits/byte, perplexities differ by "
          f"{2**(best_bits/n_words)/2**best_H:.0f}x -- bits/byte is the comparable unit")
    # ---- self-verification against an independent implementation ----
    P = np.stack([src3.get(tuple(int(v) for v in test[t-3:t]))[0] for t in range(3, 2003)]) / TOTAL
    tgt = test[3:2003].astype(np.int64)
    ce_np = float(-np.mean(np.log2(P[np.arange(len(tgt)), tgt])))
    if torch is not None:
        ce_t = torch.nn.functional.cross_entropy(
            torch.log(torch.from_numpy(P)), torch.from_numpy(tgt)).item() / math.log(2)
        print(f"\ncross-entropy cross-check (first 2000 test symbols): "
              f"numpy {ce_np:.6f} vs torch {ce_t:.6f}, |diff| {abs(ce_np-ce_t):.2e} bits")
        assert abs(ce_np - ce_t) < 1e-9
    else:
        print("\n[skipped: torch not installed] numpy cross-entropy "
              f"{ce_np:.6f} bits/symbol")
    # Gibbs' inequality, empirically: the model code beats the uniform code.
    ce_wrong = float(-np.mean(np.log2(np.full(A, 1.0 / A)[tgt])))
    print(f"Gibbs check: model code {ce_np:.4f} bits <= uniform code "
          f"{ce_wrong:.4f} bits, excess (KL) = {ce_wrong-ce_np:.4f} bits/symbol")
    assert ce_np < ce_wrong
    print("\nall assertions passed")
