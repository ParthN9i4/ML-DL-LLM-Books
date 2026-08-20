"""
ex22 — Pretraining a tiny language model.  (Book: Chapters 23 and 24)

A language model is a compressor with a training loop bolted on. Everything it
does is next-token cross-entropy, and almost every catastrophic bug in a first
pretraining run is a bug in what "next" means.

So this exercise starts by checking the objective rather than the model. At
initialization a transformer's logits are near zero, so its predicted
distribution is near uniform, so its cross-entropy must be ln(V) nats. That one
line — measure the loss before the first optimizer step and compare it to
ln(vocab_size) — catches a wrong vocabulary size, a broken softmax axis and an
output layer initialized too large, all before wasting an hour of compute.
What it can NOT catch is any bug in the targets: a uniform predictor pays
ln(V) on every labelling (asserted below), so a mislabelled or shuffled batch
sails straight through — which is exactly what the entropy floor and the
break-it section are for. Measured here: 3.3017 against ln(27) = 3.2958, a
gap of 0.0058 nats, entirely accounted for by the initialization scale of
the output weights.

The corpus is 130 sentences generated from a fixed 8x8x8x8 slot grammar and
embedded literally below, which buys something a real corpus cannot: the true
entropy of the source is known in closed form. Each sentence draws a name, a
verb, an adjective, a noun and one of two templates, so it carries exactly
4*ln(8) + ln(2) = 9.0109 nats no matter how many characters it takes to write
down. Divide by the character count and you get an entropy floor of 0.385
bits/byte that no model can beat on held-out text. That floor turns vague
claims into arithmetic: the ladder uniform (4.755 bpb) > unigram (4.164 bpb) >
model (0.934 bpb final, 0.738 bpb at its step-800 best) > source (0.375 bpb
over the scored held-out span) is measured end to end, and a loss BELOW the
floor is proof of a bug rather than a breakthrough. Training deliberately
runs past the point of usefulness, so the val curve turns around
at step 800 and the train/val gap opens to 0.39 nats — fifty times the gap of
a unigram model, which has no capacity to memorize anything.

Which is exactly how the two break-it cases are caught. Shifting the targets by
zero instead of one (0.0035 nats) and forgetting the causal mask (0.0772 nats)
both drive the loss under the floor in 400 steps, and both look like a miracle.
They are told apart by their per-position loss profile: the shift bug leaks at
every position equally (last position 0.9x the rest), while the missing mask
leaks everywhere EXCEPT the last position of the window, whose target lies
outside it (last position 133x the rest). Nothing else here distinguishes them.

To learn: replace each function body with `pass` and reimplement.
"""

import copy
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
    torch.set_num_threads(2)        # a 20k-parameter model does not want 64 threads
except ImportError:                 # pragma: no cover
    HAVE_TORCH = False

LN2 = math.log(2.0)

# ---------------------------------------------------------------------------
# The corpus. Embedded, not downloaded — and synthetic on purpose, because a
# synthetic source has a KNOWN entropy and therefore a computable floor.
#
# Grammar (each sentence, choices independent and uniform):
#     name  in {ada, alan, grace, linus, edsger, barbara, donald, john}
#     verb  in {found, kept, lost, sold, named, hid, carried, traded}
#     adj   in {small, red, quiet, broken, golden, hollow, bright, heavy}
#     noun  in {stone, river, letter, garden, engine, mirror, ladder, harbor}
#     template in {"<name> <verb> the <adj> <noun> .",
#                  "the <adj> <noun> was <verb> by <name> ."}
# so H(sentence) = 4*ln(8) + ln(2) nats exactly.
# ---------------------------------------------------------------------------
SENTENCE_ENTROPY_NATS = 4.0 * math.log(8.0) + math.log(2.0)   # 9.010913...

CORPUS = """\
barbara lost the bright stone .
alan hid the small garden .
donald carried the red garden .
the red garden was found by donald .
ada carried the small garden .
the bright letter was named by grace .
edsger lost the red garden .
the broken harbor was found by alan .
donald hid the heavy harbor .
the broken river was lost by linus .
the heavy engine was hid by john .
the bright letter was kept by alan .
the bright stone was traded by grace .
the hollow mirror was hid by alan .
john kept the red engine .
alan found the golden harbor .
donald hid the small harbor .
the small garden was traded by alan .
the bright ladder was sold by grace .
john kept the quiet harbor .
the bright engine was lost by edsger .
barbara carried the broken letter .
grace sold the broken stone .
grace named the golden stone .
barbara hid the quiet stone .
donald carried the bright ladder .
the broken river was found by donald .
the red mirror was lost by john .
the quiet river was found by alan .
ada kept the broken ladder .
edsger hid the hollow harbor .
john traded the heavy harbor .
grace kept the hollow engine .
grace found the broken mirror .
the red engine was named by ada .
the broken mirror was hid by grace .
linus sold the bright garden .
the small stone was hid by john .
john named the broken mirror .
barbara hid the red garden .
john sold the hollow garden .
the hollow river was traded by ada .
the broken harbor was carried by alan .
donald hid the red ladder .
alan lost the quiet letter .
john lost the heavy mirror .
the small river was found by grace .
grace carried the broken garden .
linus named the broken mirror .
the small mirror was lost by donald .
the quiet stone was lost by donald .
grace found the quiet letter .
the hollow harbor was found by alan .
alan found the broken garden .
alan traded the small river .
the heavy harbor was named by linus .
linus named the broken harbor .
alan carried the heavy mirror .
the red garden was carried by linus .
alan lost the hollow letter .
grace traded the broken river .
the broken letter was lost by john .
donald hid the bright garden .
the small mirror was hid by alan .
the bright mirror was found by john .
the red garden was kept by edsger .
alan kept the golden engine .
the quiet ladder was named by grace .
the quiet harbor was carried by edsger .
alan named the small letter .
the small river was named by alan .
the red engine was sold by alan .
the hollow ladder was found by john .
the small garden was lost by edsger .
grace named the small letter .
edsger named the broken engine .
the hollow stone was named by grace .
the small garden was found by ada .
the red ladder was traded by linus .
donald named the broken garden .
the hollow stone was carried by grace .
ada kept the golden ladder .
the golden garden was carried by alan .
ada traded the quiet letter .
the hollow mirror was named by ada .
barbara sold the small engine .
grace found the hollow ladder .
edsger sold the broken stone .
alan lost the bright stone .
the broken river was named by edsger .
grace carried the hollow harbor .
the bright letter was found by grace .
ada sold the red stone .
the bright harbor was kept by barbara .
ada sold the heavy engine .
alan kept the red harbor .
alan named the broken garden .
john traded the bright river .
the broken river was found by edsger .
barbara named the golden letter .
the golden river was traded by ada .
john named the golden harbor .
the golden river was sold by alan .
the heavy river was named by ada .
the bright garden was named by john .
the red letter was kept by linus .
the quiet engine was hid by edsger .
barbara sold the heavy harbor .
grace found the heavy harbor .
grace carried the hollow ladder .
the hollow mirror was found by barbara .
alan sold the small engine .
alan carried the bright river .
donald named the small engine .
edsger lost the broken engine .
the hollow ladder was sold by barbara .
the red stone was sold by donald .
donald traded the quiet engine .
grace lost the heavy ladder .
the golden ladder was named by edsger .
edsger traded the bright river .
the broken harbor was kept by grace .
john hid the heavy ladder .
linus sold the red letter .
alan hid the broken mirror .
linus found the bright ladder .
the golden mirror was carried by linus .
the hollow letter was named by john .
linus kept the golden garden .
john carried the golden stone .
"""


# ---------------------------------------------------------------------------
# The objective and the metrics, in numpy. No library helper does any of this:
# the whole point is that you can only trust cross-entropy you wrote yourself.
# ---------------------------------------------------------------------------

def softmax_np(z, axis=-1):
    """Numerically stable softmax. -inf entries map to exactly 0."""
    # === YOUR CODE HERE ===
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def entropy_nats(p, axis=-1):
    """H(p) = -sum p log p, in nats, with 0*log0 = 0 handled exactly."""
    p = np.asarray(p, dtype=np.float64)
    # === YOUR CODE HERE ===
    logp = np.zeros_like(p)
    np.log(p, out=logp, where=p > 0)
    return -(p * logp).sum(axis=axis) + 0.0


def cross_entropy_nats(logits, targets):
    """Per-token cross-entropy, -log p(target), computed via log-sum-exp.

    Never form the probabilities and then take their log: that is the same
    catastrophic cancellation ex02 and ex04 spend their time on. Work in log
    space from the start.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets)
    # === YOUR CODE HERE ===
    m = np.max(logits, axis=-1, keepdims=True)
    logZ = (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True)))[..., 0]
    z_target = np.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return logZ - z_target


def bits_per_byte(nats_per_token, bytes_per_token=1.0):
    """Convert a cross-entropy in nats/token into bits/byte.

        bits/byte = (nats/token) / ln(2) / (bytes/token)

    The division by ln(2) is the nats->bits change of base; the division by
    bytes/token is what makes the number comparable ACROSS tokenizers. A
    BPE model with 3.6 bytes/token and loss 1.9 is at 0.76 bpb, better than a
    character model at loss 0.9 (= 1.30 bpb), and loss alone hides that.
    Here the tokens ARE bytes, so bytes_per_token = 1.
    """
    # === YOUR CODE HERE ===
    return nats_per_token / LN2 / bytes_per_token


def unigram_log_probs(train_ids, vocab_size):
    """Laplace-smoothed unigram model fitted on the training split.

    What smoothing guards against: an unsmoothed unigram assigns probability 0
    to any character it never saw, and a single such character in the
    validation split makes the whole cross-entropy +inf. On THIS corpus every
    character happens to occur in both splits, so here the smoothing shifts
    the val loss by only 3.8e-05 nats — the main block therefore manufactures
    the failure instead, by scoring a symbol the training data never contains.
    """
    # === YOUR CODE HERE ===
    counts = np.bincount(train_ids, minlength=vocab_size).astype(np.float64) + 1.0
    return np.log(counts / counts.sum())


def source_entropy_nats_per_char(n_sentences, n_chars):
    """The true entropy rate of the grammar, in nats per character.

    Every sentence carries SENTENCE_ENTROPY_NATS regardless of how many
    characters spell it out, and the mapping from choices to characters is
    injective, so the entropy of a span of text is just (#sentences) *
    H(sentence). This is the floor: no predictor, however large, can average
    below it on held-out text drawn from this source.
    """
    # === YOUR CODE HERE ===
    return n_sentences * SENTENCE_ENTROPY_NATS / n_chars


# ---------------------------------------------------------------------------
# Sampling: temperature, top-k, and the draw itself.
# ---------------------------------------------------------------------------

def apply_temperature(logits, temperature):
    """Divide logits by T. T<1 sharpens, T>1 flattens, T->inf gives uniform."""
    # === YOUR CODE HERE ===
    return np.asarray(logits, dtype=np.float64) / temperature


def top_k_filter(logits, k):
    """Keep the k largest logits, set the rest to -inf (probability exactly 0).

    Ties at the boundary keep every tied entry, so the surviving set can be
    larger than k — which is the honest behaviour, since breaking a tie
    arbitrarily would make the output depend on argsort's tie order.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if k is None or k >= logits.shape[-1]:
        return logits.copy()
    # === YOUR CODE HERE ===
    kth = np.sort(logits, axis=-1)[..., -k][..., None]
    return np.where(logits >= kth, logits, -np.inf)


def sample_from_probs(p, rng):
    """One categorical draw by inverse-CDF: cumsum, uniform, searchsorted."""
    # === YOUR CODE HERE ===
    cdf = np.cumsum(p)
    return int(np.searchsorted(cdf, rng.random() * cdf[-1]))


# ---------------------------------------------------------------------------
# The model. torch carries the parameters and the autograd, but the loss, the
# mask, the metrics and the sampler are all written out above and below.
# ---------------------------------------------------------------------------

if HAVE_TORCH:

    class Block(nn.Module):
        """Pre-norm attention + MLP, with the causal mask made removable so the
        break-it section can actually remove it."""

        def __init__(self, d_model, n_head):
            super().__init__()
            self.n_head = n_head
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.proj = nn.Linear(d_model, d_model, bias=False)
            self.fc1 = nn.Linear(d_model, 2 * d_model)
            self.fc2 = nn.Linear(2 * d_model, d_model)

        def forward(self, x, causal=True):
            B, T, D = x.shape
            H, dh = self.n_head, D // self.n_head
            # === YOUR CODE HERE ===
            q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
            q = q.view(B, T, H, dh).transpose(1, 2)
            k = k.view(B, T, H, dh).transpose(1, 2)
            v = v.view(B, T, H, dh).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
            if causal:
                keep = torch.tril(torch.ones(T, T, dtype=torch.bool))
                att = att.masked_fill(~keep, float("-inf"))
            w = torch.softmax(att, dim=-1)
            y = (w @ v).transpose(1, 2).reshape(B, T, D)
            x = x + self.proj(y)
            x = x + self.fc2(torch.tanh(self.fc1(self.ln2(x))))
            return x

    class TinyLM(nn.Module):
        """~20k parameters. Small enough to train to convergence in seconds,
        large enough that overfitting a 3.5k-character corpus is inevitable."""

        def __init__(self, vocab_size, d_model=32, n_head=4, n_layer=2, block=32):
            super().__init__()
            self.block = block
            self.tok = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(block, d_model)
            self.blocks = nn.ModuleList(Block(d_model, n_head) for _ in range(n_layer))
            self.lnf = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            # Small init is what makes the ln(V) check below meaningful: it is
            # the reason the logits start near zero.
            nn.init.normal_(self.tok.weight, std=0.02)
            nn.init.normal_(self.pos.weight, std=0.02)
            nn.init.normal_(self.head.weight, std=0.02)

        def forward(self, idx, causal=True):
            B, T = idx.shape
            # === YOUR CODE HERE ===
            x = self.tok(idx) + self.pos(torch.arange(T))
            for blk in self.blocks:
                x = blk(x, causal)
            return self.head(self.lnf(x))

    def torch_cross_entropy(logits, targets):
        """The same formula as cross_entropy_nats, in torch so it differentiates.

        Written out rather than calling F.cross_entropy — if you cannot write
        it you cannot debug it, and the cross-check below shows they agree.
        """
        # === YOUR CODE HERE ===
        logZ = torch.logsumexp(logits, dim=-1)
        z_target = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return logZ - z_target

    def make_batch(data, batch_size, block, rng, shift=1):
        """Random contiguous windows. `shift` is the target offset — 1 is
        correct, and 0 is break-it case (a)."""
        # === YOUR CODE HERE ===
        ix = rng.integers(0, len(data) - block - 1, size=batch_size)
        x = np.stack([data[i:i + block] for i in ix])
        y = np.stack([data[i + shift:i + shift + block] for i in ix])
        return torch.from_numpy(x), torch.from_numpy(y)

    def eval_split(model, data, block, causal=True, shift=1):
        """Non-overlapping tiling of a split. Returns (mean nats/token,
        per-position nats, the target ids actually scored).

        The per-position breakdown is the diagnostic that separates the two
        failure modes at the end of this file.
        """
        starts = range(0, len(data) - block - shift, block)
        x = np.stack([data[i:i + block] for i in starts])
        y = np.stack([data[i + shift:i + shift + block] for i in starts])
        # === YOUR CODE HERE ===
        with torch.no_grad():
            logits = model(torch.from_numpy(x), causal)
            per_tok = torch_cross_entropy(logits, torch.from_numpy(y)).numpy()
        return float(per_tok.mean()), per_tok.mean(axis=0), y.reshape(-1)

    def train_lm(model, data, steps, block, batch_size=32, lr=3e-3,
                 causal=True, shift=1, seed=0, probe=None, probe_every=0):
        """Plain AdamW loop. `probe` is called with (step, model) to record the
        train/val curves without a second pass over the data."""
        rng = np.random.default_rng(seed)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        # === YOUR CODE HERE ===
        for step in range(steps + 1):
            if probe is not None and probe_every and step % probe_every == 0:
                probe(step, model)
            if step == steps:
                break
            x, y = make_batch(data, batch_size, block, rng, shift)
            loss = torch_cross_entropy(model(x, causal), y).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return model

    def next_token_logits(model, context_ids, block):
        """Logits for the token AFTER the given context, cropped to the window."""
        # === YOUR CODE HERE ===
        x = torch.from_numpy(np.array([context_ids[-block:]], dtype=np.int64))
        with torch.no_grad():
            return model(x)[0, -1, :].numpy().astype(np.float64)

    def generate(model, prompt_ids, n_new, block, rng, temperature=1.0, k=None):
        """Autoregressive sampling: logits -> top-k -> temperature -> draw."""
        ctx = list(prompt_ids)
        # === YOUR CODE HERE ===
        for _ in range(n_new):
            z = next_token_logits(model, ctx, block)
            z = top_k_filter(z, k)
            p = softmax_np(apply_temperature(z, temperature))
            ctx.append(sample_from_probs(p, rng))
        return ctx


if __name__ == "__main__":
    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    print("\n--- the corpus, its vocabulary, and its true entropy ---")
    chars = sorted(set(CORPUS))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in CORPUS], dtype=np.int64)

    # Character-level == byte-level only if every character is one UTF-8 byte.
    # Check it rather than assume it, because the bits/byte number below is
    # meaningless otherwise.
    n_bytes = len(CORPUS.encode("utf-8"))
    check("every character is exactly one UTF-8 byte", n_bytes == len(CORPUS))

    n_sent = CORPUS.count("\n")
    print(f"      {len(CORPUS)} characters, {n_sent} sentences, vocab size V = {V}")
    print(f"      H(sentence) = 4*ln(8) + ln(2) = {SENTENCE_ENTROPY_NATS:.4f} nats")
    floor_all = source_entropy_nats_per_char(n_sent, len(CORPUS))
    print(f"      source entropy rate = {floor_all:.4f} nats/char "
          f"= {bits_per_byte(floor_all):.4f} bits/byte")
    check("the entropy floor is below the uniform baseline", floor_all < math.log(V))

    # Split on a sentence boundary so the held-out text contains whole sentences.
    cut = CORPUS.index("\n", int(0.8 * len(CORPUS))) + 1
    train_ids, val_ids = ids[:cut], ids[cut:]
    print(f"      train {len(train_ids)} chars, val {len(val_ids)} chars "
          f"({int((val_ids == stoi[chr(10)]).sum())} held-out sentences)")

    # -----------------------------------------------------------------------
    print("\n--- the objective: cross-entropy of a uniform predictor is ln(V) ---")
    # The exact identity first, with no model and no sampling noise in it.
    uniform_logits = np.zeros((len(val_ids), V))
    ce_uniform = float(cross_entropy_nats(uniform_logits, val_ids).mean())
    print(f"      uniform-logit cross-entropy = {ce_uniform:.12f} nats")
    print(f"      ln(V) = ln({V})              = {math.log(V):.12f} nats")
    check("uniform logits give exactly ln(V)", ce_uniform, math.log(V), tol=1e-12)
    # The value is INVARIANT to the targets — asserted next — and that cuts both
    # ways. It makes ln(V) a floor for any uninformed predictor, and it means
    # the init check can never catch a label bug: it is a check on the MODEL
    # and the loss plumbing (vocab size, softmax axis, init scale), not on the
    # labels. Break-it case (a) below is a label bug, and it sails through.
    rng0 = np.random.default_rng(0)
    junk = rng0.integers(0, V, size=len(val_ids))
    check("and ln(V) whatever the targets are",
          float(cross_entropy_nats(uniform_logits, junk).mean()), math.log(V), tol=1e-12)
    check("entropy of the uniform distribution is also ln(V)",
          float(entropy_nats(np.full(V, 1.0 / V))), math.log(V), tol=1e-12)

    # -----------------------------------------------------------------------
    print("\n--- baselines, computed ---")
    uniform_nats = math.log(V)
    logp_uni = unigram_log_probs(train_ids, V)
    unigram_nats = float(-logp_uni[val_ids].mean())
    unigram_train_nats = float(-logp_uni[train_ids].mean())
    print(f"      {'uniform':>10}: {uniform_nats:.4f} nats/char = "
          f"{bits_per_byte(uniform_nats):.4f} bits/byte")
    print(f"      {'unigram':>10}: {unigram_nats:.4f} nats/char = "
          f"{bits_per_byte(unigram_nats):.4f} bits/byte")
    print(f"      {'source':>10}: {floor_all:.4f} nats/char = "
          f"{bits_per_byte(floor_all):.4f} bits/byte   <- the floor")
    check("unigram beats uniform on held-out text", unigram_nats < uniform_nats)
    check("and the unigram model still sits above the source floor",
          unigram_nats > floor_all)
    # A unigram model's loss on its OWN training data is the entropy of the
    # character marginal, exactly — the only gap is the Laplace smoothing.
    train_freq = np.bincount(train_ids, minlength=V) / len(train_ids)
    print(f"      unigram train loss {unigram_train_nats:.6f} vs entropy of the "
          f"character marginal {float(entropy_nats(train_freq)):.6f}")
    check("unigram train loss == H(character marginal), up to smoothing",
          unigram_train_nats, float(entropy_nats(train_freq)), tol=1e-3)
    # bits/byte is just a change of base here, and that is worth pinning down.
    check("bits/byte = nats / ln(2) when 1 token = 1 byte",
          bits_per_byte(unigram_nats), unigram_nats / LN2, tol=1e-15)
    check("uniform bits/byte = log2(V)", bits_per_byte(uniform_nats), math.log2(V), tol=1e-12)
    # The bytes/token normalization, exercised on the docstring's own example
    # rather than left at its default of 1: a BPE model at 1.9 nats/token and
    # 3.6 bytes/token must come out BELOW a character model at 0.9 nats/token,
    # and inverting the division (times instead of over) gives 9.87 bpb, not
    # 0.76 — both checks catch that.
    bpe_bpb = bits_per_byte(1.9, 3.6)
    print(f"      BPE example: 1.9 nats at 3.6 bytes/token = {bpe_bpb:.4f} bpb, "
          f"char model at 0.9 nats = {bits_per_byte(0.9):.4f} bpb")
    check("bits/byte divides by bytes/token: 1.9 nats at 3.6 B/tok = 0.7614 bpb",
          bpe_bpb, 1.9 / LN2 / 3.6, tol=1e-12)
    check("once normalized, the BPE model beats the char model",
          bpe_bpb < bits_per_byte(0.9, 1.0))

    # On this corpus the smoothing is defensive only: every character occurs
    # in both splits, so removing it barely moves the number. Measure that.
    raw_cnt = np.bincount(train_ids, minlength=V).astype(np.float64)
    unsm_val_nats = float(-np.log(raw_cnt / raw_cnt.sum())[val_ids].mean())
    print(f"      smoothing shifts the unigram val loss by "
          f"{abs(unigram_nats - unsm_val_nats):.1e} nats on this corpus")

    # And the failure the Laplace smoothing exists for, manufactured: extend
    # the vocabulary by one symbol the training data never contains. The
    # unsmoothed model prices it at +inf nats; the smoothed model at exactly
    # ln(N + V + 1) — a closed form the implementation does not get to see.
    raw_ext = np.bincount(train_ids, minlength=V + 1).astype(np.float64)
    check("the extension symbol never occurs in train", raw_ext[V] == 0.0)
    with np.errstate(divide="ignore"):
        unsmoothed_ext = np.log(raw_ext / raw_ext.sum())
    smoothed_ext = unigram_log_probs(train_ids, V + 1)
    print(f"      unseen symbol: unsmoothed {-unsmoothed_ext[V]:.4f} nats, "
          f"smoothed {float(-smoothed_ext[V]):.4f} nats (= ln(N+V+1) = "
          f"{math.log(len(train_ids) + V + 1):.4f})")
    check("unsmoothed: one unseen character costs +inf nats",
          math.isinf(float(-unsmoothed_ext[V])))
    check("smoothed: the same character costs exactly ln(N+V+1) nats",
          float(-smoothed_ext[V]), math.log(len(train_ids) + V + 1), tol=1e-12)

    # -----------------------------------------------------------------------
    print("\n--- temperature and top-k, on synthetic logits (no model needed) ---")
    z_demo = np.array([4.0, 2.0, 1.5, 0.0, -1.0, -3.0])
    temps = [0.1, 0.25, 0.5, 1.0, 2.0, 8.0, 1000.0]
    H_of_T = [float(entropy_nats(softmax_np(apply_temperature(z_demo, T)))) for T in temps]
    print("      T:  " + " ".join(f"{T:>8g}" for T in temps))
    print("      H:  " + " ".join(f"{h:8.4f}" for h in H_of_T))
    print(f"      ln(6) = {math.log(6):.4f}  <- the T -> infinity limit")
    # Mechanism: softmax(z/T) is a Boltzmann distribution in beta = 1/T, and
    # dH/dbeta = -beta*Var(z) <= 0. So H is nondecreasing in T, strictly so
    # unless every logit is equal. This is a theorem, not a tendency.
    check("entropy is strictly increasing in temperature",
          all(H_of_T[i] < H_of_T[i + 1] for i in range(len(H_of_T) - 1)))
    check("T -> infinity reaches the uniform entropy ln(6)",
          H_of_T[-1], math.log(6), tol=1e-4)
    # The T -> 0 end, with the rate rather than a threshold. The mass off the
    # argmax is dominated by the runner-up, so q = 1 - p_max ~ exp(-gap/T)
    # where gap is the distance between the top two logits; and the entropy is
    # squeezed by H <= q + q*ln((V-1)/q), which is rigorous for any q.
    p_cold = softmax_np(apply_temperature(z_demo, 0.1))
    q_off = float(1.0 - p_cold.max())
    logit_gap = float(np.sort(z_demo)[-1] - np.sort(z_demo)[-2])
    H_bound = q_off + q_off * math.log((len(z_demo) - 1) / q_off)
    print(f"      T = 0.1: off-argmax mass {q_off:.6e}, exp(-gap/T) = "
          f"{math.exp(-logit_gap / 0.1):.6e}")
    print(f"      T = 0.1: H = {H_of_T[0]:.6e} <= q + q*ln((V-1)/q) = {H_bound:.6e}")
    check("T -> 0: the off-argmax mass decays as exp(-gap/T) (within 2%)",
          q_off, math.exp(-logit_gap / 0.1), tol=0.02 * math.exp(-logit_gap / 0.1))
    check("T -> 0: entropy is squeezed to zero by q + q*ln((V-1)/q)",
          H_of_T[0] <= H_bound)
    for k in (1, 2, 3, 6):
        p_k = softmax_np(top_k_filter(z_demo, k))
        Hk = float(entropy_nats(p_k))
        print(f"      top-{k}: H = {Hk:.4f} nats, ln(k) = {math.log(k):.4f}, "
              f"support = {int((p_k > 0).sum())}")
        check(f"top-{k} entropy cannot exceed ln(k)", Hk <= math.log(k) + 1e-12)
        check(f"top-{k} keeps exactly {k} tokens in support", int((p_k > 0).sum()) == k)
    check("top-k with k >= V is the identity",
          softmax_np(top_k_filter(z_demo, 6)), softmax_np(z_demo), tol=1e-15)

    print("\n--- the sampler draws from the distribution it is given ---")
    p_test = softmax_np(z_demo)
    rng_s = np.random.default_rng(1)
    draws = np.bincount([sample_from_probs(p_test, rng_s) for _ in range(40_000)],
                        minlength=6) / 40_000
    print("      target: " + " ".join(f"{v:6.4f}" for v in p_test))
    print("      drawn:  " + " ".join(f"{v:6.4f}" for v in draws))
    # 40k draws: the standard error of each frequency is at most 0.5/sqrt(40000)
    # = 0.0025, so 5 standard errors is 0.0125.
    se = 0.5 / math.sqrt(40_000)
    check("empirical frequencies match p within 5 standard errors",
          float(np.abs(draws - p_test).max()) < 5 * se)

    if not HAVE_TORCH:
        print("\n  ....  [skipped: torch not installed] model training, "
              "sampling and the break-it section")
        summary()

    # -----------------------------------------------------------------------
    print("\n--- the check nobody runs: loss at initialization ---")
    BLOCK = 32
    torch.manual_seed(0)
    model = TinyLM(V, block=BLOCK)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      TinyLM: {n_params} parameters, context {BLOCK}")

    init_nats, _, _ = eval_split(model, val_ids, BLOCK)
    print(f"      loss at init = {init_nats:.6f} nats")
    print(f"      ln(V)        = {math.log(V):.6f} nats")
    print(f"      difference   = {init_nats - math.log(V):+.6f} nats "
          f"({abs(init_nats - math.log(V)) / math.log(V) * 100:.2f}%)")
    # Where the 0.0058-nat gap lives, exactly. Writing d_i for a position's
    # logit deviations from their own mean, convexity sandwiches
    # log-sum-exp(z) - mean(z) - ln(V) into [0, max_i d_i], hence
    # |CE - ln(V)| <= 2*max|d|. Both sides are computed over ALL the windows
    # the eval loss averages (max|d| = 0.4140 here), and the log-sum-exp is
    # recovered from the hand-written loss (lse = CE + z_target), so a broken
    # normalization axis in cross_entropy_nats breaks the sandwich.
    starts_w = range(0, len(val_ids) - BLOCK - 1, BLOCK)
    xw = np.stack([val_ids[i:i + BLOCK] for i in starts_w])
    yw = np.stack([val_ids[i + 1:i + 1 + BLOCK] for i in starts_w])
    with torch.no_grad():
        z_init = model(torch.from_numpy(xw)).numpy().astype(np.float64)
    dev = z_init - z_init.mean(axis=-1, keepdims=True)
    max_dev = float(np.abs(dev).max())
    z_tgt = np.take_along_axis(z_init, yw[..., None], axis=-1)[..., 0]
    lse_gap = cross_entropy_nats(z_init, yw) + z_tgt \
        - z_init.mean(axis=-1) - math.log(V)
    print(f"      logit spread at init: max|z - mean(z)| = {max_dev:.4f} "
          f"over all {xw.shape[0]} eval windows")
    check("convexity sandwich: lse(z) - mean(z) - ln(V) >= 0 at every position",
          float(lse_gap.min()) >= -1e-9)
    check("...and <= max_i d_i at every position",
          float((dev.max(axis=-1) - lse_gap).min()) >= -1e-9)
    check("hence |loss - ln(V)| <= 2*max|d|  (0.0058 vs 0.8280 here)",
          abs(init_nats - math.log(V)) <= 2 * max_dev)
    # That sandwich is SCALE-COVARIANT: max|d| grows with the logits, so it
    # holds at ANY init scale and can never catch an over-scaled output layer.
    # Demonstrate: multiply the head by 25 (std 0.02 -> 0.5). The bound still
    # holds while the loss lands miles from ln(V) — measured 6.37 nats, a
    # 3.07-nat miss — and only the absolute tolerance below catches it.
    over = copy.deepcopy(model)
    with torch.no_grad():
        over.head.weight.mul_(25.0)
    over_nats, _, _ = eval_split(over, val_ids, BLOCK)
    with torch.no_grad():
        z_over = over(torch.from_numpy(xw)).numpy().astype(np.float64)
    over_dev = float(np.abs(z_over - z_over.mean(axis=-1, keepdims=True)).max())
    print(f"      head x25: loss {over_nats:.4f} vs ln(V) {math.log(V):.4f}, "
          f"yet 2*max|d| = {2 * over_dev:.4f} still covers the miss")
    check("the covariant bound also holds for the over-scaled init — useless as a guard",
          abs(over_nats - math.log(V)) <= 2 * over_dev)
    check("the over-scaled init misses ln(V) by more than 1 nat (measured 3.07)",
          abs(over_nats - math.log(V)) > 1.0)
    # The scale-SENSITIVE check is the absolute one: with std-0.02 output
    # weights the deviation measured here is 0.0058 nats, 0.18% of ln(V).
    check("loss at init equals ln(V) to within 0.01 nats",
          init_nats, math.log(V), tol=0.01)

    # Cross-check the hand-written loss against torch's, since everything above
    # and below depends on it being right.
    import torch.nn.functional as F  # noqa: E402  (cross-check only)
    xb = torch.from_numpy(val_ids[:BLOCK][None, :])
    yb = torch.from_numpy(val_ids[1:BLOCK + 1][None, :])
    with torch.no_grad():
        lg = model(xb)
        mine = torch_cross_entropy(lg, yb).mean().item()
        theirs = F.cross_entropy(lg.reshape(-1, V), yb.reshape(-1)).item()
        numpy_version = float(cross_entropy_nats(lg.numpy(), yb.numpy()).mean())
    print(f"      hand-written {mine:.9f} | F.cross_entropy {theirs:.9f} | "
          f"numpy {numpy_version:.9f}")
    check("hand-written torch loss == F.cross_entropy", mine, theirs, tol=1e-6)
    check("hand-written numpy loss == F.cross_entropy", numpy_version, theirs, tol=1e-6)

    # -----------------------------------------------------------------------
    print("\n--- training, with both curves measured ---")
    STEPS = 2000
    curve = []

    def probe(step, m):
        tr_l, _, _ = eval_split(m, train_ids, BLOCK)
        va_l, _, _ = eval_split(m, val_ids, BLOCK)
        curve.append((step, tr_l, va_l))

    t0 = time.perf_counter()
    train_lm(model, train_ids, STEPS, BLOCK, probe=probe, probe_every=200)
    train_secs = time.perf_counter() - t0
    print(f"      {'step':>6} {'train':>8} {'val':>8} {'gap':>8}   (nats/char)")
    for step, tr_l, va_l in curve:
        print(f"      {step:6d} {tr_l:8.4f} {va_l:8.4f} {va_l - tr_l:+8.4f}")
    print(f"      trained {STEPS} steps in {train_secs:.1f}s")

    steps_arr = [c[0] for c in curve]
    tr_curve = [c[1] for c in curve]
    va_curve = [c[2] for c in curve]
    best_i = int(np.argmin(va_curve))
    print(f"      best val {va_curve[best_i]:.4f} at step {steps_arr[best_i]} "
          f"(= {bits_per_byte(va_curve[best_i]):.4f} bits/byte), "
          f"final val {va_curve[-1]:.4f}")

    # "The loss went down" is worth nothing unless it went below a model that
    # ignores context entirely, so both claims are anchored to the unigram.
    print(f"      train {tr_curve[0]:.4f} -> {tr_curve[-1]:.4f} nats, "
          f"{unigram_train_nats / tr_curve[-1]:.1f}x better than unigram")
    check("training beat the context-free unigram model on train",
          tr_curve[-1] < unigram_train_nats)
    check("the model beats the uniform baseline on held-out text",
          va_curve[-1] < uniform_nats)
    check("the model beats the unigram baseline on held-out text",
          va_curve[-1] < unigram_nats)
    # The overfitting signature is not "val goes up" alone — it is val going up
    # WHILE train keeps going down. Both halves are asserted, and the gap they
    # imply is compared against a model that CANNOT memorize: the 27-parameter
    # unigram, whose own train/val gap is the sampling noise of this corpus.
    check("the gap starts at zero for an untrained model",
          abs(va_curve[0] - tr_curve[0]) < 0.01)
    check("past the val minimum, val gets worse", va_curve[-1] > va_curve[best_i])
    check("...while train keeps getting better", tr_curve[-1] < tr_curve[best_i])
    gap_final = va_curve[-1] - tr_curve[-1]
    unigram_gap = unigram_nats - unigram_train_nats
    print(f"      train/val gap: transformer {gap_final:+.4f} nats, "
          f"unigram {unigram_gap:+.4f} nats ({gap_final / unigram_gap:.0f}x)")
    check("so the gap grows once memorization starts",
          gap_final > va_curve[best_i] - tr_curve[best_i])
    check("and dwarfs the gap of a model with no capacity to memorize",
          gap_final > 10 * unigram_gap)

    print("\n--- bits per byte, against the floor ---")
    final_train, _, _ = eval_split(model, train_ids, BLOCK)
    final_val, _, val_targets = eval_split(model, val_ids, BLOCK)
    # Count the sentences inside exactly the span that was scored, so the floor
    # and the loss are computed over the same characters.
    val_sent_scored = int((val_targets == stoi["\n"]).sum())
    floor_val = source_entropy_nats_per_char(val_sent_scored, len(val_targets))
    rows = [("uniform", uniform_nats), ("unigram", unigram_nats),
            ("model (val)", final_val), ("model (train)", final_train),
            ("source floor", floor_val)]
    for name, nats in rows:
        print(f"      {name:>14}: {nats:7.4f} nats/char  {bits_per_byte(nats):7.4f} bits/byte")
    check("the ladder holds: uniform > unigram > model > floor",
          uniform_nats > unigram_nats > final_val > floor_val)
    # The train loss, on memorized text, is allowed to sit below the floor —
    # memorization is not compression of the SOURCE. This is measured, and it
    # is why the floor is only a floor for HELD-OUT data.
    print(f"      (train loss is {'below' if final_train < floor_val else 'above'} "
          f"the source floor — memorization, not compression)")
    check("held-out loss respects the floor", final_val > floor_val)

    # -----------------------------------------------------------------------
    print("\n--- what temperature does to the model's own distribution ---")
    prompt = "ada found the "
    ctx = [stoi[c] for c in prompt]
    z_model = next_token_logits(model, ctx, BLOCK)
    top5 = np.argsort(-z_model)[:5]
    print(f"      after {prompt!r} the model's top continuations are: "
          + ", ".join(f"{chars[i]!r}" for i in top5))
    rng_t = np.random.default_rng(3)
    print(f"      {'T':>7} {'H(p) analytic':>14} {'H empirical':>12}")
    Hs = []
    for T in (0.25, 0.5, 1.0, 2.0, 8.0):
        p = softmax_np(apply_temperature(z_model, T))
        d = np.bincount([sample_from_probs(p, rng_t) for _ in range(8000)],
                        minlength=V) / 8000
        H_a, H_e = float(entropy_nats(p)), float(entropy_nats(d))
        Hs.append(H_a)
        print(f"      {T:7.2f} {H_a:14.4f} {H_e:12.4f}")
        check(f"T={T}: the sampler reproduces the analytic entropy", H_a, H_e, tol=0.05)
    check("model entropy is strictly increasing in T",
          all(Hs[i] < Hs[i + 1] for i in range(len(Hs) - 1)))

    print("\n      generated text (top-k=5, T=0.8):")
    gen = generate(model, [stoi[c] for c in "the "], 200, BLOCK,
                   np.random.default_rng(4), temperature=0.8, k=5)
    for line in "".join(chars[i] for i in gen).split("\n")[:5]:
        print(f"        | {line}")

    # Temperature has a measurable consequence beyond entropy: the fraction of
    # generated words that are real words in the corpus.
    corpus_words = set(w for w in CORPUS.replace("\n", " ").split(" ") if w)
    print(f"      {'T':>7} {'in-vocab words':>15}")
    fracs = []
    for T in (0.5, 1.0, 1.5, 2.0):
        g = generate(model, [stoi[c] for c in "the "], 700, BLOCK,
                     np.random.default_rng(5), temperature=T)
        txt = "".join(chars[i] for i in g)
        ws = [w for w in txt.replace("\n", " ").split(" ") if w][1:-1]
        frac = sum(w in corpus_words for w in ws) / len(ws)
        fracs.append(frac)
        print(f"      {T:7.2f} {frac:15.3f}   ({len(ws)} words)")
    check("hot sampling produces far more non-words than cold sampling",
          fracs[0] - fracs[-1] > 0.2)

    # -----------------------------------------------------------------------
    # BREAK IT — two bugs that both look like a triumph
    # -----------------------------------------------------------------------
    print("\n--- break it ---")
    BUG_STEPS = 400

    # (a) The off-by-one: targets are the INPUT tokens, not the next ones. The
    #     model is asked to predict token t from a context that already
    #     contains token t. It is a copy, so the loss goes to zero.
    torch.manual_seed(0)
    bug_shift = train_lm(TinyLM(V, block=BLOCK), train_ids, BUG_STEPS, BLOCK, shift=0)
    shift_nats, shift_pos, _ = eval_split(bug_shift, val_ids, BLOCK, shift=0)

    # (b) The missing causal mask: targets are correct, but every position can
    #     attend to the future, and the target of position t is sitting at
    #     input position t+1. Also a copy, also near zero.
    torch.manual_seed(0)
    bug_mask = train_lm(TinyLM(V, block=BLOCK), train_ids, BUG_STEPS, BLOCK, causal=False)
    mask_nats, mask_pos, _ = eval_split(bug_mask, val_ids, BLOCK, causal=False)

    print(f"      honest model, {STEPS} steps : val {final_val:.4f} nats "
          f"= {bits_per_byte(final_val):.4f} bpb")
    print(f"      (a) target shift = 0, {BUG_STEPS} steps: val {shift_nats:.4f} nats "
          f"= {bits_per_byte(shift_nats):.4f} bpb")
    print(f"      (b) no causal mask,   {BUG_STEPS} steps: val {mask_nats:.4f} nats "
          f"= {bits_per_byte(mask_nats):.4f} bpb")
    print(f"      source entropy floor              : {floor_val:.4f} nats "
          f"= {bits_per_byte(floor_val):.4f} bpb")

    # Both bugs are caught by the same argument and it is not "the loss looks
    # low": it is that the loss is below a floor that no predictor can cross.
    check("(a) claims to compress held-out text below the source entropy",
          shift_nats < floor_val)
    check("(b) claims to compress held-out text below the source entropy",
          mask_nats < floor_val)
    check("(a) is >100x below the honest model's loss", shift_nats < final_val / 100)
    check("(b) is >5x below the honest model's loss", mask_nats < final_val / 5)

    # Now tell them apart. Under the shift bug EVERY position can copy, so the
    # per-position loss is flat. Under the missing mask every position can copy
    # its target from the input EXCEPT the last one, whose target lies outside
    # the window — so exactly one position is left holding the real problem.
    def profile(pos):
        return float(pos[-1]), float(pos[:-1].mean())

    a_last, a_rest = profile(shift_pos)
    b_last, b_rest = profile(mask_pos)
    h_last, h_rest = profile(eval_split(model, val_ids, BLOCK)[1])
    print(f"      {'':>22} {'positions 0..30':>16} {'position 31':>13} {'ratio':>8}")
    for name, rest, last in (("honest", h_rest, h_last),
                             ("(a) shift bug", a_rest, a_last),
                             ("(b) no mask", b_rest, b_last)):
        print(f"      {name:>22} {rest:16.4f} {last:13.4f} {last / rest:7.1f}x")
    check("(a) leaks uniformly: the last position is no harder than the rest",
          a_last < 2.0 * a_rest)
    check("(b) leaks everywhere but the last position, which stays hard",
          b_last > 20.0 * b_rest)
    check("(b)'s surviving position is as hard as honest language modelling",
          b_last > h_rest)
    check("the two signatures are opposite",
          (a_last / a_rest) < 2.0 < 20.0 < (b_last / b_rest))

    # The honest model's own per-position profile is the sanity check that the
    # diagnostic means anything: with a real causal model, position 0 has no
    # context at all and is the hardest position in the window.
    honest_pos = eval_split(model, val_ids, BLOCK)[1]
    print(f"      honest per-position loss: pos 0 = {honest_pos[0]:.4f}, "
          f"mean of pos 1..31 = {honest_pos[1:].mean():.4f}")
    check("with a correct mask, more context genuinely helps",
          float(honest_pos[0]) > 2 * float(honest_pos[1:].mean()))

    print(f"\n      total runtime {time.perf_counter() - t_start:.1f}s")
    summary()
