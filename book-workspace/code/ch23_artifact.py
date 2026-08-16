"""
Artifact 23.1 -- Byte-level BPE and a bits-per-byte evaluator, from scratch.

Trains byte-level BPE tokenizers (base vocab = the 256 raw bytes, so ANY
UTF-8 input is encodable -- byte fallback is structural, not a special case),
verifies that the encoder exactly reproduces training-time segmentation, and
demonstrates the chapter's thesis with hard assertions: perplexity is NOT
comparable across tokenizers; bits-per-byte IS. Pure stdlib + numpy.
"""
import math, re, time
from collections import Counter
import numpy as np

# --- Pre-tokenization: a simplified GPT-2-style regex. It partitions the ---
# --- text completely (word runs, punctuation runs, whitespace runs), so ----
# --- no byte is ever dropped. Real tokenizers use \p{L} classes (regex lib).
PRETOK = re.compile(r" ?\w+| ?[^\w\s]+|\s+", re.UNICODE)

def pretokenize(text):
    """Split text into pre-tokens, each returned as raw UTF-8 bytes."""
    return [m.group().encode("utf-8") for m in PRETOK.finditer(text)]

def merge_seq(seq, pair, new_id):
    """One left-to-right, non-overlapping pass replacing `pair` by `new_id`.
    The SAME routine is used in training and inference -- that identity is
    what makes assertion (b) hold exactly."""
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id); i += 2
        else:
            out.append(seq[i]); i += 1
    return out

class ByteBPE:
    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)}  # byte fallback base
        self.merges = []                                  # [((a,b), new_id)]
        self._cache = {}

    def train(self, text, vocab_size):
        """Greedy BPE: repeatedly merge the most frequent adjacent pair.
        Returns the final training-time segmentation of every unique word."""
        words = Counter(pretokenize(text))          # word bytes -> frequency
        segs = {w: list(w) for w in words}          # current id sequences
        while len(self.vocab) < vocab_size:
            pairs = Counter()
            for w, cnt in words.items():
                s = segs[w]
                for a, b in zip(s, s[1:]):
                    pairs[(a, b)] += cnt
            if not pairs:
                break
            # deterministic tie-break: highest count, then smallest pair ids
            best = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            self.merges.append((best, new_id))
            for w in segs:                          # apply merge everywhere
                segs[w] = merge_seq(segs[w], best, new_id)
        return segs

    def encode_word(self, wb):
        """Apply the learned merges IN TRAINING ORDER. This replays training
        exactly, so encoding a training word reproduces its training seg."""
        if wb in self._cache:
            return self._cache[wb]
        s = list(wb)
        for pair, nid in self.merges:
            if len(s) < 2:
                break
            if any(s[i] == pair[0] and s[i+1] == pair[1] for i in range(len(s)-1)):
                s = merge_seq(s, pair, nid)
        self._cache[wb] = s
        return s

    def encode(self, text):
        return [t for w in pretokenize(text) for t in self.encode_word(w)]

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8")

class ByteTrigramLM:
    """Interpolated byte trigram LM. Defines ONE distribution over byte
    strings; both tokenizers below inherit their token probabilities from it
    (a token's probability = product of its bytes' probabilities), so total
    bits MUST agree while per-token perplexity does not."""
    def __init__(self, data):
        self.n = len(data)
        self.uni = Counter(data)
        self.bi = Counter(zip(data, data[1:]))
        self.tri = Counter(zip(data, data[1:], data[2:]))

    def logp2(self, a, b, c):  # log2 p(c | a, b)
        p1 = (self.uni[c] + 1) / (self.n + 256)          # add-1 smoothed
        p2 = self.bi[(b, c)] / self.uni[b] if self.uni[b] else 0.0
        p3 = self.tri[(a, b, c)] / self.bi[(a, b)] if self.bi[(a, b)] else 0.0
        return math.log2(0.6 * p3 + 0.25 * p2 + 0.15 * p1)

    def byte_bits(self, text_bytes):
        pad = b"\x00\x00" + text_bytes
        return [-self.logp2(pad[i], pad[i+1], pad[i+2]) for i in range(len(text_bytes))]

class TokenBigramLM:
    """Independently trained token-level bigram LM (the empirical check)."""
    def __init__(self, tokens, V):
        prev = [-1] + tokens[:-1]
        self.V, self.N = V, len(tokens)
        self.uni, self.bi, self.ctx = Counter(tokens), Counter(zip(prev, tokens)), Counter(prev)

    def bits(self, tokens):
        total, prev = 0.0, -1
        for t in tokens:
            p2 = self.bi[(prev, t)] / self.ctx[prev] if self.ctx[prev] else 0.0
            p = 0.90 * p2 + 0.09 * self.uni[t] / self.N + 0.01 / self.V
            total += -math.log2(p); prev = t
        return total

WORDS = ("the model token byte pair encoding merge vocabulary corpus text language "
         "compression ratio perplexity bits per entropy likelihood objective next "
         "prediction training inference sequence context window frequency adjacent "
         "greedy subword unit segmentation boundary whitespace digit code python "
         "function return loop count print assert vector matrix gradient loss "
         "optimizer batch layer attention scale data sample probability estimate "
         "measure evaluate compare across different sizes cost price effective "
         "length script latin multibyte unicode fallback rare frequent common "
         "stream buffer parse split join encode decode exact round trip").split()
ZH = list("语言模型把文本切分成子词单元分词器决定世界每个汉字通常需要三字节压缩率因而异")
CODE = "def f(x):\n    return x + 1  # 42\n"

def build_corpus(rng):
    """Random sentences over a fixed word pool: English-dominated with a rare
    CJK long tail and occasional code, mimicking web-corpus composition."""
    def sentence():
        ws = [WORDS[i] for i in rng.integers(0, len(WORDS), rng.integers(4, 11))]
        r = rng.random()
        if r < 0.06:   # rare CJK run: the multilingual long tail
            ws.append("".join(ZH[i] for i in rng.integers(0, len(ZH), 4)))
        elif r < 0.10: # occasional code
            ws.append(CODE)
        return " ".join(ws) + ". "
    train = "".join(sentence() for _ in range(2500))
    test = "".join(sentence() for _ in range(450))
    en_sample = ("the model must encode text and estimate the next token "
                 "probability across a common vocabulary of frequent subword units. ") * 4
    zh_sample = "语言模型把文本切分成子词单元。分词器决定模型的世界。每个汉字通常需要三字节。" * 4
    unseen = "이 문장은 학습 데이터에 없다 وهذه جملة غير مرئية 🦑"  # scripts NEVER trained on
    return train, test, en_sample, zh_sample, unseen

if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(0)
    train_text, test_text, en_sample, zh_sample, unseen = build_corpus(rng)
    print(f"corpus: train={len(train_text.encode())} bytes, test={len(test_text.encode())} bytes")
    assert b"".join(pretokenize(test_text)) == test_text.encode()  # regex partitions fully

    tokA, tokB = ByteBPE(), ByteBPE()
    segsA = tokA.train(train_text, 300)
    segsB = tokB.train(train_text, 800)

    # (a) HARD ASSERTION: exact round-trip on ASCII + code + CJK + unseen scripts.
    for s in (train_text[:5000], test_text, zh_sample, unseen):
        for tok in (tokA, tokB):
            assert tok.decode(tok.encode(s)) == s
    print("[a] round-trip exact on ASCII/code/UTF-8 incl. never-seen scripts: OK")

    # (b) HARD ASSERTION: encoder reproduces training-time segmentation.
    checked = 0
    for tok, segs in ((tokA, segsA), (tokB, segsB)):
        for w, seg in segs.items():
            assert tok.encode_word(w) == seg; checked += 1
    print(f"[b] encoder == training segmentation on {checked} unique pre-tokens: OK")

    # (c) HARD ASSERTION (thesis): one byte-level distribution, two tokenizers.
    lm = ByteTrigramLM(train_text.encode())
    bb = lm.byte_bits(test_text.encode())
    n_bytes, total_bits = len(bb), sum(bb)
    stats = {}
    for name, tok in (("V=300", tokA), ("V=800", tokB)):
        ids = tok.encode(test_text)
        bits, pos = 0.0, 0                      # regroup byte bits per token
        for t in ids:
            L = len(tok.vocab[t]); bits += sum(bb[pos:pos+L]); pos += L
        assert pos == n_bytes                   # tokens tile the byte stream
        stats[name] = (len(ids), 2 ** (bits/len(ids)), bits/n_bytes)
        print(f"[c] {name}: {len(ids):5d} tokens  PPL={stats[name][1]:8.3f}  BPB={stats[name][2]:.6f}")
    (nA, pplA, bpbA), (nB, pplB, bpbB) = stats["V=300"], stats["V=800"]
    assert abs(bpbA - bpbB) < 1e-9 and abs(bpbA - total_bits/n_bytes) < 1e-6
    assert pplB / pplA > 1.3
    print(f"[c] THESIS: |BPB diff|={abs(bpbA-bpbB):.2e} (<1e-9), PPL ratio={pplB/pplA:.2f}x")

    # (d) Worse: perplexity MISRANKS independently trained models. A bigram LM
    # over V=800 tokens sees ~4x more bytes of context per conditioning step
    # than one over V=300 tokens, so it is the genuinely better model -- and
    # BPB says so. Per-token perplexity says the opposite.
    for name, tok in (("V=300", tokA), ("V=800", tokB)):
        tr, te = tok.encode(train_text), tok.encode(test_text)
        bits = TokenBigramLM(tr, len(tok.vocab)).bits(te)
        stats[name] = (2 ** (bits/len(te)), bits/n_bytes)
        print(f"[d] {name}: PPL={stats[name][0]:8.3f}  BPB={stats[name][1]:.4f}")
    (pplA, bpbA), (pplB, bpbB) = stats["V=300"], stats["V=800"]
    assert pplB > 1.5 * pplA   # PPL claims the V=800 model is far worse...
    assert bpbB < bpbA         # ...BPB shows it is in fact the better model.
    print(f"[d] PPL misranks: V=800 looks {pplB/pplA:.1f}x worse by PPL, "
          f"yet is {bpbA/bpbB:.2f}x better by BPB")

    # (e) The token tax: tokens per byte, English vs Chinese, same tokenizer.
    for name, s in (("English", en_sample), ("Chinese", zh_sample)):
        ids, nb = tokB.encode(s), len(s.encode())
        print(f"[e] V=800 {name}: {len(ids)/nb:.3f} tokens/byte "
              f"({nb/len(ids):.2f} bytes/token, {len(ids)/len(s):.2f} tokens/char)")

    try:  # cross-check our entropy arithmetic against scipy
        from scipy.stats import entropy
        cnt = np.bincount(np.frombuffer(train_text.encode(), np.uint8), minlength=256)
        ours = -sum(c/cnt.sum() * math.log2(c/cnt.sum()) for c in cnt if c)
        ref = entropy(cnt, base=2)
        assert abs(ours - ref) < 1e-10
        print(f"[x] byte unigram entropy {ours:.4f} bits matches scipy (|diff|={abs(ours-ref):.1e})")
    except ImportError:
        print("[x] [skipped: scipy not installed]")

    longest = sorted(tokB.vocab.values(), key=len)[-3:]
    print(f"longest V=800 tokens: {[t.decode('utf-8', 'replace') for t in longest]}")
    print(f"all assertions passed in {time.time()-t0:.1f}s")
