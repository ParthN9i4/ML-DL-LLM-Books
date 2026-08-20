"""
ex21 — Training a BPE tokenizer from scratch.  (Book: Chapter 23)

A language model never sees text. It sees integers, and the map from text to
integers is byte-pair encoding: start from the 256 byte values, count adjacent
pairs, merge the most frequent pair into a new symbol, repeat. The ORDERED list
of merges is the whole tokenizer — the vocabulary is just that list replayed.

Working on bytes rather than characters is the design decision that carries the
most weight. There are exactly 256 of them, every string on earth is a sequence
of them, and so the tokenizer is total: no unknown token, no UNK, no failure
mode where an input is unrepresentable. Round-trip fidelity is therefore not a
property you test for and hope — it is structural, and if decode(encode(s))
differs from s by a single byte the implementation is wrong. The exercise
proves that on ASCII, accents, CJK, an emoji, whitespace runs and code
punctuation.

The insight this file is built to make undeniable is the second one: round-trip
fidelity proves NOTHING about whether a tokenizer is correct. Apply the very
same merges in a different order and every string still decodes back to itself
perfectly, while the integers handed to the model are different — a silent
train/serve skew that no round-trip test can see. Reordering merges only WITHIN
ties of the training count moves 722 of the 768 merges here, lengthens the
encoded corpus from 3326 to 3770 tokens (+13.3%), and leaves every round trip
byte-exact. What pins a tokenizer down is the order of the merge list, and the
check with teeth is that encoding the training corpus must reproduce the exact
token sequence training ended with.

Compression is measured, not asserted: on this 11507-byte corpus, bytes per
token goes 1.000 -> 1.598 -> 2.288 -> 3.460 at vocabulary 256, 320, 512, 1024.
The diminishing returns fall straight out of greedy selection: merging the top
pair can only lower every remaining count, so the counts consumed are
non-increasing forever — 5664 pair occurrences bought by the first 128 merges,
256 by the sixth block of 128. The bookkeeping is exact for pairs whose halves
differ, and an over-estimate for self-pairs, because the counter counts
overlapping occurrences while the merge is non-overlapping: the pair (space,
space) is counted 381 times here and removes 212 tokens.

The pathologies come free with the algorithm. Numbers fragment along whatever
digit pairs happened to be frequent — "2024" is one token here and "4202" is
two, and "2024" costs fewer tokens than "56" — so token cost is neither a
function of digit count nor monotone in it, and no token boundary lines up with
a decimal column. Spaces attach to the front of words, so " the" and "the" share
no token id at all, and the tokenization of a word therefore depends on the
character before it. And the vocabulary quietly spends itself on near-duplicates
that differ only by a leading space or by case.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


# ---------------------------------------------------------------------------
# The corpus. Deliberately mixed: prose with capitalized sentence openers,
# years and quantities, code punctuation, accented Latin, CJK and an emoji.
# Everything the tokenizer knows comes from these bytes and nothing else.
# ---------------------------------------------------------------------------
CORPUS_TEXT = """The model does not read text. The model reads integers, and the
tokenizer is the only thing standing between the two. A tokenizer that is wrong
is wrong everywhere at once: the embedding table is indexed by it, the loss is
computed over it, and the sampling loop emits into it.

Byte pair encoding starts from the 256 byte values and merges the most frequent
adjacent pair, over and over, until the vocabulary is the size you asked for.
The merges are ordered. The order is the tokenizer. Two tokenizers with the same
merges in a different order are different tokenizers, and a model trained with
one of them will quietly degrade when served with the other.

The training corpus for a real tokenizer is measured in terabytes. In 2019 the
vocabulary of the GPT-2 tokenizer was 50257 tokens; in 2023 a vocabulary of
100000 tokens was ordinary, and in 2024 and 2025 vocabularies of 128000 and
256000 tokens appeared in open models. Bigger vocabularies buy shorter
sequences, and shorter sequences buy attention that costs less, because the cost
of attention grows with the square of the sequence length.

The gain is not free. A vocabulary of 128000 tokens with a model width of 4096
spends 524288000 parameters on the embedding table alone, and the same number
again on the output projection when the two are not tied. Every token added is a
row that must be learned, and the rows for rare tokens are learned from almost
no examples. This is the tension the chapter keeps returning to: the tokenizer
trades sequence length against parameter count, and neither side is free.

Numbers are the classic failure. The tokenizer learned that the digit pair 20
was frequent, because the years 2019, 2020, 2021, 2022, 2023, 2024 and 2025 fill
the corpus, and so the number 2024 costs fewer tokens than the number 4202 even
though both have four digits. Arithmetic on such a representation is arithmetic
on a ragged grid: the model must learn that the token holding 20 sometimes means
twenty and sometimes means two thousand, depending on what follows it.

Whitespace is the second classic failure. The space attaches to the front of the
word that follows it, so the token for the word the and the token for the same
word with a leading space are two different integers with two different rows in
the embedding table. The model must learn twice that the word means the same
thing. The same happens with case: the word at the start of a sentence and the
word in the middle of one are different tokens, and The and the are as unrelated
to the embedding table as the words model and tokenizer.

Code makes the point sharper still. Consider the following:

    def encode(text, merges):
        ids = list(text.encode("utf-8"))
        for pair, idx in merges:
            ids = merge(ids, pair, idx)
        return ids

    def decode(ids, vocab):
        raw = b"".join(vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    class Tokenizer:
        def __init__(self, vocab_size=1024):
            self.vocab_size = vocab_size
            self.merges = []

        def train(self, data, verbose=False):
            ids = list(data)
            while len(self.merges) + 256 < self.vocab_size:
                stats = get_stats(ids)
                pair = max(stats, key=stats.get)
                ids = merge(ids, pair, 256 + len(self.merges))
                self.merges.append((pair, 256 + len(self.merges)))
            return self.merges

Indentation is spaces, and runs of spaces merge into tokens of their own, which
is why a code corpus and a prose corpus produce tokenizers that are bad at each
other's job. The parentheses, the brackets, the colons and the underscores all
compete for vocabulary slots against ordinary English words.

Text is not English. The word café is written with an accent, naïve keeps its
diaeresis, and the résumé of a model is not the same object as the resume of a
process. Tokyo is 東京 and Japan is 日本, and a single CJK character is three
bytes of UTF-8, so a byte level tokenizer that has never seen enough Chinese
will spend three tokens on one character. A character level tokenizer would
spend one token, right up until it meets a character it has never seen, and
then it has nothing to emit at all. The byte level tokenizer never has that
problem, which is the entire reason it is used. 🙂

The tokenizer is trained once and frozen forever. The model is trained around
it, the evaluation harness is built on it, and the serving stack ships it as a
data file next to the weights. Changing it is not an upgrade, it is a new model.
That is why the boring parts of this chapter matter: the order of the merges,
the exactness of the round trip, and the arithmetic of bytes per token.

The greedy rule is worth stating precisely, because the bookkeeping around it is
where the bugs live. At every step the trainer counts the adjacent pairs, takes
the pair with the largest count, and replaces every occurrence of it with a new
symbol. The replacement is left to right and non overlapping, so a run of three
identical symbols contains two counted pairs but admits only one replacement.
For every pair whose two halves differ, the number of tokens removed is exactly
the counted number of occurrences. For a pair whose two halves are identical the
count is an over estimate, and a run of spaces is exactly that case. A tokenizer
trained on indented code hits this constantly.

There is a second decision hidden in the loop, and it is the one nobody writes
down: what happens before the counting starts. A real implementation splits the
text with a regular expression first, so that a merge can never span a word
boundary, can never swallow a newline into the middle of a word, and can never
produce a token that holds the tail of one word and the head of the next. Drop
that split and the merges will happily cross the gaps, producing tokens that are
useless everywhere except in the corpus that produced them. The vocabulary fills
with fragments that encode the writing habits of the training set rather than
the structure of the language.

Counting is the expensive part. A naive trainer recounts every pair after every
merge, which costs the length of the corpus for each of the merges, and for a
vocabulary of 100000 merges over a corpus of a trillion bytes that is not a
program, it is a wish. Real trainers keep an index from each pair to the places
it occurs and update only the neighbourhoods that a merge touched, which turns
the inner loop into work proportional to the number of occurrences rather than
the length of the corpus. The version below is the naive one, because the naive
one is the one you can read.

Evaluation is where the tokenizer stops being an implementation detail. Bytes
per token on held out text is the number that decides how much of a document
fits in a context window of 8192 tokens, or 32768, or 131072. A tokenizer that
compresses English at 4 bytes per token and Japanese at 1 byte per token is a
tokenizer that charges Japanese users four times as much for the same document,
and that is not a hypothetical, it is the ordinary behaviour of a vocabulary fit
mostly to English. The fix is not clever code, it is a training mixture that
contains the languages you intend to serve.

The last property is the one that is easiest to test and easiest to trust too
much. Encode a string, decode it again, compare. If the two differ, something is
broken. If they agree, nothing at all has been established, because the identity
map on bytes is preserved by any ordering of any set of merges whatsoever. The
test that has teeth is the one that pins the integers themselves: encoding the
training corpus must reproduce the exact sequence of integers that training
finished with, token for token, and any reordering of the merge list breaks that
immediately.

Consider what a serving bug of that kind looks like from the outside. The model
loads, the prompts encode, the outputs decode, the logs look ordinary, and the
quality is merely somewhat worse than it was in evaluation. Nothing raises an
exception. The embedding rows that the model spent its training compute learning
are being indexed by integers that mean something slightly different, and the
damage is spread thinly across every token of every request. Tokenizer bugs are
quiet, and quiet bugs are expensive.

A short glossary, since the words are overloaded. A byte is one of 256 values. A
character, or code point, is one of 1114112 values, of which a few hundred
thousand are assigned. A token is an entry in the vocabulary, which is a byte
string of any length. A merge is a rule that rewrites a pair of tokens into one.
The vocabulary is the merge list replayed from the 256 base bytes, and nothing
else, which is why saving a tokenizer means saving an ordered list and not a
dictionary.

    def get_stats(ids):
        stats = {}
        for a, b in zip(ids, ids[1:]):
            stats[(a, b)] = stats.get((a, b), 0) + 1
        return stats

    def merge(ids, pair, new_id):
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def bytes_per_token(text, merges):
        raw = text.encode("utf-8")
        return len(raw) / len(encode(text, merges))

    if __name__ == "__main__":
        corpus = open("corpus.txt", encoding="utf-8").read()
        for size in [512, 1024, 2048, 4096, 8192, 16384]:
            merges = train(corpus.encode("utf-8"), vocab_size=size)
            print(size, round(bytes_per_token(corpus, merges), 3))

The numbers that come out of that loop always have the same shape. Doubling the
vocabulary from 512 to 1024 buys a lot, doubling from 8192 to 16384 buys very
little, and somewhere in between is the size that a team argues about for a week
and then picks by feel. The curve is not mysterious. Each merge removes as many
tokens as the pair it consumed had occurrences, greedy always takes the largest
remaining count, and so the returns fall by construction, merge after merge,
until the trainer is merging pairs that occur twice in the entire corpus.

日本語のテキストは、英語のテキストとは違う形で分割されます。一つの漢字は
三つのバイトになりますから、語彙が英語に偏っている場合、同じ意味の文章でも
トークン数が何倍にもなります。これは料金の問題であり、文脈長の問題でもあり、
そして評価の問題でもあります。中文的情况也一样：一个汉字通常是三个字节。

Los mismos problemas aparecen en español, en français, y in every language whose
orthography uses accented characters: café, naïve, résumé, Ångström, señor,
über, Grüße, ação. Each accented character costs two bytes where the unaccented
one costs a single byte, so a tokenizer that never saw them will spend two
tokens on every one of them, and the sequence length grows accordingly. 🙂 🚀 🎉

The chapter closes with the practical checklist. Save the merge list in order.
Version it with the weights. Test the round trip on ASCII, on accented Latin, on
CJK, on emoji, on runs of whitespace, and on source code. Then test the thing
the round trip cannot see: that encoding a fixed corpus reproduces a fixed
sequence of integers, byte for byte, merge for merge, forever.
"""

HELDOUT_TEXT = (
    "The serving stack loads the merge list from disk and replays it. "
    "A model trained in 2024 with a vocabulary of 32000 tokens cannot be "
    "served with the tokenizer from 2025, even when both round trip "
    "perfectly on every string you try. The café was quiet; 東京 was not. 🙂"
)


def get_stats(ids):
    """Count every adjacent pair in the id sequence.

    Insertion order matters: `max(stats, key=stats.get)` returns the FIRST
    maximal key, so ties are broken by earliest occurrence and training is
    deterministic.
    """
    stats = {}
    # === YOUR CODE HERE ===
    for a, b in zip(ids, ids[1:]):
        stats[(a, b)] = stats.get((a, b), 0) + 1
    return stats


def merge(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` with `new_id`.

    Left to right and non-overlapping: on [a, a, a] with pair (a, a) exactly
    one merge fires even though get_stats counted two occurrences — which is
    why the token-count bookkeeping is exact only for pairs whose two halves
    DIFFER, and an over-estimate for self-pairs (the main block measures both).
    """
    out = []
    i = 0
    # === YOUR CODE HERE ===
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(data, vocab_size):
    """Learn an ordered merge list from raw bytes.

    Returns (merges, log, ids): merges is [((a, b), new_id), ...] in learned
    order, log[k] records the pair count merge k consumed and the sequence
    length it left behind, and ids is the fully merged corpus — the token
    sequence that `encode` must reproduce exactly.
    """
    ids = list(data)
    merges = []
    log = []
    # === YOUR CODE HERE ===
    while 256 + len(merges) < vocab_size:
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        if stats[pair] < 2:
            break                     # nothing left worth merging
        new_id = 256 + len(merges)
        before = len(ids)
        ids = merge(ids, pair, new_id)
        merges.append((pair, new_id))
        log.append({"pair": pair, "count": stats[pair], "n_before": before,
                    "n_after": len(ids), "removed": before - len(ids)})
    return merges, log, ids


def build_vocab(merges):
    """Replay the merge list to get id -> bytes. The vocabulary is derived."""
    # === YOUR CODE HERE ===
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in merges:
        vocab[new_id] = vocab[a] + vocab[b]
    return vocab


def encode(text, merges):
    """Text -> ids, applying merges in the order they were LEARNED.

    Order is not a detail. A later merge may consume a token that an earlier
    merge created, so running them out of order produces a different (still
    decodable!) encoding.
    """
    # === YOUR CODE HERE ===
    ids = list(text.encode("utf-8"))
    for pair, new_id in merges:
        if len(ids) < 2:
            break
        ids = merge(ids, pair, new_id)
    return ids


def encode_greedy_rank(text, merges):
    """The same encoding by a DIFFERENT algorithm — the reference cross-check.

    Instead of sweeping the merge list once, repeatedly find the pair present in
    the sequence with the smallest merge rank and apply only that one. This is
    how production tokenizers encode (you never sweep 100000 merges per string),
    and it must agree with `encode` token for token. Two independent routes to
    the same integers is the strongest correctness evidence available here.
    """
    ranks = {pair: r for r, (pair, _) in enumerate(merges)}
    new_ids = dict(merges)
    ids = list(text.encode("utf-8"))
    # === YOUR CODE HERE ===
    while len(ids) >= 2:
        present = [p for p in get_stats(ids) if p in ranks]
        if not present:
            break
        best = min(present, key=lambda p: ranks[p])
        ids = merge(ids, best, new_ids[best])
    return ids


def decode(ids, vocab):
    """ids -> text. Concatenate the byte strings, then decode UTF-8 once.

    errors="replace" only ever fires on a TRUNCATED token sequence (streaming
    half of a multi-byte character); on a complete sequence the bytes are the
    original bytes exactly.
    """
    # === YOUR CODE HERE ===
    raw = b"".join(vocab[i] for i in ids)
    return raw.decode("utf-8", errors="replace")


def train_bpe_codepoints(text, vocab_size):
    """The same algorithm on Unicode CODE POINTS instead of bytes.

    The base alphabet is now whatever characters happened to appear in the
    training text, which makes the tokenizer partial: anything else has no
    representation at all. Returns (merges, vocab, char_to_id, unk_id).
    """
    chars = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(chars)}
    unk_id = len(chars)                       # the UNK this design forces on you
    vocab = {i: c for c, i in char_to_id.items()}
    vocab[unk_id] = "�"
    ids = [char_to_id[c] for c in text]
    merges = []
    next_id = unk_id + 1
    # === YOUR CODE HERE ===
    while len(vocab) < vocab_size:
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        if stats[pair] < 2:
            break
        ids = merge(ids, pair, next_id)
        vocab[next_id] = vocab[pair[0]] + vocab[pair[1]]
        merges.append((pair, next_id))
        next_id += 1
    return merges, vocab, char_to_id, unk_id


def encode_codepoints(text, merges, char_to_id, unk_id):
    """Code-point encoder. Unseen characters have nowhere to go but UNK."""
    # === YOUR CODE HERE ===
    ids = [char_to_id.get(c, unk_id) for c in text]
    for pair, new_id in merges:
        if len(ids) < 2:
            break
        ids = merge(ids, pair, new_id)
    return ids


def bytes_per_token(n_bytes, n_tokens):
    """The compression number that actually matters for context length."""
    return n_bytes / n_tokens


if __name__ == "__main__":
    t_start = time.perf_counter()
    data = CORPUS_TEXT.encode("utf-8")
    N_BYTES = len(data)

    print("\n--- the corpus ---")
    print(f"      {N_BYTES} bytes, {len(CORPUS_TEXT)} characters, "
          f"{len(set(data))} distinct byte values, {len(set(CORPUS_TEXT))} distinct characters")
    check("multi-byte UTF-8 really is present (bytes > characters)",
          N_BYTES > len(CORPUS_TEXT))

    VOCAB = 1024
    t0 = time.perf_counter()
    merges, log, train_ids = train_bpe(data, VOCAB)
    t_train = time.perf_counter() - t0
    vocab = build_vocab(merges)
    print(f"      trained {len(merges)} merges -> vocab {len(vocab)} in {t_train:.2f}s")
    check("vocab size is 256 + number of merges", len(vocab), 256 + len(merges))
    check("every merged token's bytes are its two parents concatenated",
          all(vocab[i] == vocab[p[0]] + vocab[p[1]] for p, i in merges))

    print("\n--- the exact bookkeeping of a greedy merge ---")
    # A merge removes exactly as many tokens as the pair had occurrences --- but
    # only when the pair's two halves differ. get_stats counts OVERLAPPING pairs
    # while merge() replaces non-overlapping ones, so a self-pair like (space,
    # space) on a run of three spaces is counted twice and replaced once.
    hetero = [r for r in log if r["pair"][0] != r["pair"][1]]
    homo = [r for r in log if r["pair"][0] == r["pair"][1]]
    print(f"      {len(hetero)} merges of unequal pairs, {len(homo)} of self-pairs")
    check("for unequal pairs, tokens removed == counted occurrences, exactly",
          all(r["removed"] == r["count"] for r in hetero))
    check("for self-pairs the overlapping count is an upper bound, never below",
          all(r["removed"] <= r["count"] for r in homo))
    over = [r for r in homo if r["removed"] < r["count"]]
    if over:
        r = over[0]
        piece = (vocab[r["pair"][0]] + vocab[r["pair"][1]]).decode("utf-8", "replace")
        print(f"      overlap witness: pair {piece!r} counted {r['count']} times, "
              f"removed {r['removed']} tokens")
    check("and at least one self-pair really does overlap in this corpus",
          len(over) > 0)
    check("training ended with the length the log predicts",
          len(train_ids), log[-1]["n_after"])
    # None of the other checks pin down the SELECTION rule: a trainer that
    # merges the SECOND-most-frequent pair at every step still has exact
    # bookkeeping, non-increasing counts, monotone compression and perfect
    # round trips (verified by mutation — it passes every other check in this
    # file). So replay training with an independent argmax: at every step the
    # pair actually merged must carry the largest count in a freshly
    # recomputed stats table.
    ids_k = list(data)
    greedy_ok = True
    for (pair_k, nid_k), r in zip(merges, log):
        st = get_stats(ids_k)
        greedy_ok &= (st.get(pair_k, 0) == max(st.values()) == r["count"])
        ids_k = merge(ids_k, pair_k, nid_k)
    check("every merge really did consume the most frequent pair available",
          greedy_ok)

    print("\n--- encode reproduces training exactly (the order check) ---")
    t0 = time.perf_counter()
    ids_corpus = encode(CORPUS_TEXT, merges)
    print(f"      encoded the corpus in {time.perf_counter() - t0:.2f}s -> "
          f"{len(ids_corpus)} tokens")
    check("encode(corpus) == the token sequence training ended with",
          ids_corpus, train_ids)
    # Cross-check against a genuinely different algorithm: lowest-rank-first,
    # the way a production tokenizer encodes. Same integers or one of them is
    # wrong.
    t0 = time.perf_counter()
    ids_rank = encode_greedy_rank(CORPUS_TEXT, merges)
    print(f"      lowest-rank-first encoder agrees on the corpus "
          f"({time.perf_counter() - t0:.2f}s)")
    check("merge-sweep and lowest-rank-first encoders agree on the corpus",
          ids_rank, ids_corpus)

    print("\n--- round trip: byte-level BPE cannot fail this ---")
    CASES = [
        ("empty string", ""),
        ("plain ascii", "the model reads integers"),
        ("accented latin", "café naïve résumé Ångström"),
        ("CJK", "東京都は日本の首都です。"),
        ("emoji (4-byte)", "ship it 🚀🙂 done"),
        ("whitespace runs", "a\t\tb\n\n\n    c  \r\n"),
        ("code punctuation", 'def f(x, *, k=2): return x**k  # {"a": [1,2]}'),
        ("unseen script", "Ωμέγα Привет שלום"),
        ("everything at once", "The naïve 東京 café — 🎉 costs $1,234.56!\tOK?"),
        ("nbsp + zero width", "a b​c"),
    ]
    all_exact = True
    agree = True
    for name, s in CASES:
        ids = encode(s, merges)
        back = decode(ids, vocab)
        exact_bytes = b"".join(vocab[i] for i in ids) == s.encode("utf-8")
        all_exact &= (back == s) and exact_bytes
        agree &= encode_greedy_rank(s, merges) == ids
        print(f"      {name:22s} {len(s.encode('utf-8')):4d}B -> {len(ids):4d} tok  "
              f"exact={back == s}")
    check("decode(encode(s)) == s exactly, byte for byte, in every case", all_exact)
    check("the two independent encoders agree on every case too", agree)
    # And the reason it is structural: every id expands to bytes, and the base
    # 256 ids cover every possible byte, so no input can be unrepresentable.
    check("all 256 byte values are in the vocabulary",
          all(bytes([i]) in set(vocab.values()) for i in range(256)))
    # ...and the totality claim exercised end to end: each of the 256 possible
    # byte values, pushed through encode and decode as a character (latin-1 is
    # a bijection between bytes and U+0000..U+00FF), comes back exactly. A
    # string compare, so a broken decode FAILs here instead of raising.
    check("every possible byte value round-trips through encode/decode",
          all(decode(encode(bytes([i]).decode("latin-1"), merges), vocab)
              == bytes([i]).decode("latin-1") for i in range(256)))

    print("\n--- compression: bytes per token as the vocabulary grows ---")
    # log[k]["n_after"] IS the corpus length at vocab size 257 + k, so the whole
    # curve comes from the single training run above.
    def tokens_at(vocab_size):
        if vocab_size <= 256:
            return N_BYTES
        return log[vocab_size - 257]["n_after"]

    sizes = [256] + [v for v in (320, 384, 512, 768, 1024) if v <= 256 + len(merges)]
    bpt = [bytes_per_token(N_BYTES, tokens_at(v)) for v in sizes]
    print(f"      {'vocab':>7} {'tokens':>8} {'bytes/token':>12} {'gain':>8}")
    for i, v in enumerate(sizes):
        gain = "" if i == 0 else f"{bpt[i] - bpt[i-1]:+8.3f}"
        print(f"      {v:7d} {tokens_at(v):8d} {bpt[i]:12.3f} {gain:>8}")
    check("with no merges at all, one token per byte",
          len(encode(CORPUS_TEXT, [])), N_BYTES)
    check("compression improves monotonically with vocabulary",
          all(bpt[i] < bpt[i + 1] for i in range(len(bpt) - 1)))

    # Diminishing returns are a THEOREM about greedy, not an observation.
    # After merging the top pair p into X, every other pair has lost
    # occurrences, and every pair involving X occurs at most as often as X
    # itself, which is at most count(p). So the max pair count can never rise:
    # the merges are consumed in non-increasing order of value, forever.
    counts = np.array([r["count"] for r in log])
    print(f"      pair count consumed: merge 1 -> {counts[0]}, "
          f"merge {len(counts)} -> {counts[-1]}, median {int(np.median(counts))}")
    check("the greedy pair count is non-increasing over training",
          bool(np.all(counts[:-1] >= counts[1:])))
    BLOCK = 128
    blocks = [int(counts[i:i + BLOCK].sum())
              for i in range(0, (len(counts) // BLOCK) * BLOCK, BLOCK)]
    print(f"      pair occurrences consumed per {BLOCK}-merge block: " +
          " ".join(str(b) for b in blocks))
    check("each block of merges buys strictly less than the one before",
          all(blocks[i] > blocks[i + 1] for i in range(len(blocks) - 1)))

    # Held-out text compresses too, but less — the merges were fit to the corpus.
    ho_bytes = len(HELDOUT_TEXT.encode("utf-8"))
    ho_tokens = len(encode(HELDOUT_TEXT, merges))
    print(f"      held-out: {ho_bytes}B -> {ho_tokens} tok = "
          f"{bytes_per_token(ho_bytes, ho_tokens):.3f} bytes/token "
          f"(train {bpt[-1]:.3f})")
    check("held-out text still compresses better than raw bytes",
          bytes_per_token(ho_bytes, ho_tokens) > 1.0)
    check("but not as well as the training corpus it was fit to",
          bytes_per_token(ho_bytes, ho_tokens) < bpt[-1])

    print("\n--- pathology (a): digits fragment inconsistently ---")
    probes = ["2024", "4202", "2020", "1998", "1234", "5678", "9999", "1000"]
    tok_counts = {p: len(encode(p, merges)) for p in probes}
    for p in probes:
        pieces = [decode([i], vocab) for i in encode(p, merges)]
        print(f"      {p} -> {len(encode(p, merges))} tokens {pieces}")
    # The mechanism: token count is NOT a function of digit count. Find the
    # actual witnesses rather than asserting a number.
    witness = None
    for a in probes:
        for b in probes:
            if len(a) == len(b) and tok_counts[a] != tok_counts[b]:
                witness = (a, b)
                break
        if witness:
            break
    # The check runs BEFORE anything dereferences `witness`, so a trainer
    # rewrite that kills the property produces a FAIL, not an IndexError.
    check("two equally long numbers cost different numbers of tokens",
          witness is not None)
    if witness:
        print(f"      same digit count, different token count: {witness} -> "
              f"{tok_counts[witness[0]]} vs {tok_counts[witness[1]]} tokens")
    # Sharper: token cost is not even MONOTONE in digit count. Find a longer
    # number that costs strictly fewer tokens than a shorter one.
    short = ["56", "78", "99", "42"]
    short_counts = {p: len(encode(p, merges)) for p in short}
    inversion = [(a, b) for a in probes for b in short
                 if len(a) > len(b) and tok_counts[a] < short_counts[b]]
    check("a longer number can cost strictly fewer tokens than a shorter one",
          len(inversion) > 0)
    if inversion:
        a, b = inversion[0]
        print(f"      inversion: {a} ({len(a)} digits) costs {tok_counts[a]} tokens, "
              f"{b} ({len(b)} digits) costs {short_counts[b]}")
    # And why arithmetic suffers: column alignment is destroyed. The byte
    # offsets where tokens start differ between two 4-digit numbers, so the
    # model cannot line up units with units.
    def boundaries(s):
        out, pos = [], 0
        for i in encode(s, merges):
            out.append(pos)
            pos += len(vocab[i])
        return out
    b1, b2 = boundaries("2024"), boundaries("4202")
    print(f"      token start offsets: 2024 {b1}, 4202 {b2}")
    check("digit-column boundaries do not line up between two 4-digit numbers",
          b1 != b2)

    print("\n--- pathology (b): the leading space is part of the token ---")
    ids_the, ids_sp_the = encode("the", merges), encode(" the", merges)
    print(f"      'the'  -> {ids_the}  {[decode([i], vocab) for i in ids_the]}")
    print(f"      ' the' -> {ids_sp_the}  {[decode([i], vocab) for i in ids_sp_the]}")
    check("' the' and 'the' share no token ids",
          set(ids_the).isdisjoint(set(ids_sp_the)))
    vocab_bytes = set(vocab.values())
    space_dups = sorted(v for v in vocab_bytes
                        if v.startswith(b" ") and v[1:] in vocab_bytes and len(v) > 1)
    print(f"      {len(space_dups)} vocab entries are 'space + an existing token', e.g. "
          + ", ".join(repr(v.decode('utf-8', 'replace')) for v in space_dups[:6]))
    check("the vocabulary really does hold space-prefixed duplicates",
          len(space_dups) > 0)
    # How often it matters, measured on ordinary held-out prose: for how many
    # words does the preceding space actually FUSE into the word's first token?
    words = sorted(set(re.findall(r"[A-Za-z]+", HELDOUT_TEXT)))
    fused = [w for w in words
             if len(vocab[encode(" " + w, merges)[0]]) > 1
             and vocab[encode(" " + w, merges)[0]].startswith(b" ")]
    occ = sum(len(re.findall(r"(?<= )" + w + r"\b", HELDOUT_TEXT)) for w in words)
    occ_fused = sum(len(re.findall(r"(?<= )" + w + r"\b", HELDOUT_TEXT)) for w in fused)
    print(f"      held-out prose: {len(fused)}/{len(words)} distinct words and "
          f"{occ_fused}/{occ} space-preceded word occurrences have the space fused in")
    print(f"      e.g. {fused[:8]}")
    check("the space really does fuse into word tokens in ordinary prose",
          len(fused) > 0)
    # The consequence, measured: for some of those words the two spellings share
    # NO token id at all — the model gets two disjoint sets of embedding rows for
    # one word, and which set it gets depends on the preceding character.
    disjoint = [w for w in fused
                if set(encode(" " + w, merges)).isdisjoint(set(encode(w, merges)))]
    print(f"      {len(disjoint)}/{len(fused)} of the fused words share NO token id "
          f"with their own bare spelling: {disjoint[:8]}")
    check("a word and the same word after a space can share zero embedding rows",
          len(disjoint) > 0)
    # A newline is left context too, and it does NOT reset the word to its
    # bare form: with no pre-tokenization the newline fuses into the following
    # word exactly like the space did, because the vocabulary learns tokens
    # that START with a newline. ANY left-context byte can change a word's
    # tokenization — which is the pathology pre-tokenization exists to stop.
    nl_tokens = sorted(v for v in vocab_bytes if v.startswith(b"\n") and len(v) > 1)
    print(f"      {len(nl_tokens)} learned tokens start with a newline, e.g. "
          + ", ".join(repr(v.decode('utf-8', 'replace')) for v in nl_tokens[-3:]))
    nl_fused = [w for w in disjoint
                if encode("\n" + w, merges)[1:] != encode(w, merges)]
    check("a newline fuses into the following word just like a space does",
          len(nl_fused) > 0)
    if nl_fused:
        w0 = nl_fused[0]
        ids_nl = encode("\n" + w0, merges)
        print(f"      {w0!r}: bare {encode(w0, merges)}, after a space "
              f"{encode(' ' + w0, merges)}, after a newline {ids_nl} "
              f"{[decode([i], vocab) for i in ids_nl]}")
        print(f"      {len(nl_fused)}/{len(disjoint)} of the disjoint words "
              f"tokenize differently after a newline than bare")

    print("\n--- pathology (c): near-duplicate tokens ---")
    case_dups = sorted(
        (v, w) for v in vocab_bytes for w in vocab_bytes
        if v != w and len(v) > 1 and v.lower() == w.lower() and v.lower() == v
    )
    print(f"      {len(case_dups)} vocab pairs differ only by case, e.g. "
          + ", ".join(f"{v!r}/{w!r}" for v, w in case_dups[:4]))
    check("the vocabulary spends slots on case duplicates", len(case_dups) > 0)
    # Count distinct TOKENS: each case pair contributes TWO learned tokens,
    # and a union guards against a token appearing in both lists.
    near_dup_tokens = set(space_dups) | {t for pr in case_dups for t in pr}
    print(f"      {len(near_dup_tokens)} of {len(merges)} learned tokens "
          f"({100 * len(near_dup_tokens) / len(merges):.1f}%) are near-duplicates of another token")
    # Also the pre-tokenization story: with no regex split, merges cross word
    # boundaries and produce tokens with an interior space. GPT-2's regex
    # forbids exactly these — but it deliberately PERMITS whole runs of
    # whitespace as single chunks (its real vocabulary holds multi-space
    # indentation tokens), so whitespace-only entries are counted separately.
    interior_space = sorted(v for v in vocab_bytes
                            if len(v) > 1 and b" " in v[1:] and v.strip() != b"")
    ws_runs = sorted(v for v in vocab_bytes
                     if len(v) > 1 and b" " in v[1:] and v.strip() == b"")
    print(f"      {len(interior_space)} tokens hold an interior space inside "
          f"non-space text (pre-tokenization would forbid these), e.g. "
          + ", ".join(repr(v.decode('utf-8', 'replace')) for v in interior_space[:4]))
    print(f"      plus {len(ws_runs)} whitespace-run tokens, which GPT-2's "
          f"regex would keep, e.g. "
          + ", ".join(repr(v.decode('utf-8', 'replace')) for v in ws_runs[:3]))
    check("without pre-tokenization, merges cross word boundaries",
          len(interior_space) > 0)
    check("whitespace runs merge into tokens of their own (these GPT-2 keeps)",
          len(ws_runs) > 0)

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) Serialize the tokenizer as {pair: count} and rebuild the order by
    #     sorting on the counts. This is the reordering that looks safe, and the
    #     first surprise is that it looks EVEN SAFER than it is: greedy already
    #     emits merges in non-increasing count order, so a stable sort on the
    #     counts is the identity. The bug hides entirely in the tie-break.
    stable = [merges[i] for i in np.argsort([-r["count"] for r in log], kind="stable")]
    check("sorting by count with a stable sort is a no-op (greedy was already sorted)",
          stable == merges)
    tie_groups = {}
    for r in log:
        tie_groups[r["count"]] = tie_groups.get(r["count"], 0) + 1
    tied = sum(n for n in tie_groups.values() if n > 1)
    print(f"      {tied}/{len(merges)} merges share their count with another merge")
    check("ties are the common case, so the tie-break decides almost everything",
          tied > len(merges) // 2)

    #     So: same counts, different (arbitrary) tie-break — a dict iteration
    #     order, a hash seed, a sort key that reads the pair instead of the id.
    counts_ = [r["count"] for r in log]
    merges_freq = [merges[i] for i in
                   sorted(range(len(merges)), key=lambda i: (-counts_[i], -i))]
    check("the tie-broken order is a genuine permutation of the same merges",
          sorted(merges_freq) == sorted(merges) and merges_freq != merges)
    moved = sum(1 for a, b in zip(merges_freq, merges) if a != b)
    print(f"      reversing the order WITHIN ties moves {moved}/{len(merges)} merges")

    diverged = 0
    rt_ok = True
    for name, s in CASES:
        ref = encode(s, merges)
        alt = encode(s, merges_freq)
        rt_ok &= (decode(alt, vocab) == s)
        diverged += (ref != alt)
    ref_corpus = encode(CORPUS_TEXT, merges)
    alt_corpus = encode(CORPUS_TEXT, merges_freq)
    print(f"      corpus: {len(ref_corpus)} tokens (learned order) vs "
          f"{len(alt_corpus)} tokens (tie-broken order) = "
          f"{100 * (len(alt_corpus) / len(ref_corpus) - 1):+.1f}% sequence length")
    print(f"      {diverged}/{len(CASES)} of the round-trip cases encode differently, "
          f"and round trip is still exact for ALL of them: {rt_ok}")
    check("the reordered merges produce a different encoding of the corpus",
          alt_corpus != ref_corpus)
    check("...yet decode back to the identical text, byte for byte",
          decode(alt_corpus, vocab), CORPUS_TEXT)
    check("round trip therefore does NOT validate a tokenizer",
          rt_ok and diverged > 0)
    # The check that DOES catch it: encode(corpus) must reproduce training.
    check("the training-reproduction check catches what round trip misses",
          alt_corpus != train_ids and ref_corpus == train_ids)
    # A cruder reordering for scale — "apply the longest tokens first", the other
    # classic wrong implementation. Same merges, same perfect round trip.
    merges_long = sorted(merges, key=lambda mi: (-len(vocab[mi[1]]), mi[1]))
    long_corpus = encode(CORPUS_TEXT, merges_long)
    print(f"      longest-token-first order: {len(long_corpus)} tokens "
          f"({len(long_corpus) / len(ref_corpus):.2f}x the learned order), "
          f"round trip exact: {decode(long_corpus, vocab) == CORPUS_TEXT}")
    check("longest-first is also perfectly reversible and also wrong",
          decode(long_corpus, vocab) == CORPUS_TEXT and long_corpus != ref_corpus)

    # (b) Do BPE on Unicode code points instead of bytes. Now the base alphabet
    #     is finite and corpus-dependent, so a character that never appeared is
    #     literally unrepresentable — there is no sequence of ids that decodes
    #     to it, and UNK is the only option left.
    cp_merges, cp_vocab, cp_char_to_id, cp_unk = train_bpe_codepoints(CORPUS_TEXT, VOCAB)
    print(f"      code-point tokenizer: {len(cp_char_to_id)} base characters "
          f"+ 1 UNK + {len(cp_merges)} merges")
    OOV = "Ω漢字 Привет"
    unseen = sorted(set(OOV) - set(CORPUS_TEXT))
    print(f"      characters in {OOV!r} never seen in training: "
          + " ".join(unseen))
    check("the probe really does contain unseen characters", len(unseen) > 0)

    cp_ids = encode_codepoints(OOV, cp_merges, cp_char_to_id, cp_unk)
    cp_back = "".join(cp_vocab[i] for i in cp_ids)
    n_unk = sum(1 for i in cp_ids if i == cp_unk)
    print(f"      code-point round trip: {OOV!r} -> {cp_back!r} "
          f"({n_unk} UNK tokens)")
    byte_back = decode(encode(OOV, merges), vocab)
    print(f"      byte-level round trip:  {OOV!r} -> {byte_back!r}")
    n_unseen_occurrences = sum(1 for c in OOV if c not in cp_char_to_id)
    check("code-point BPE loses information on unseen characters", cp_back != OOV)
    check("it emits one UNK per unseen character, because nothing else exists",
          n_unk, n_unseen_occurrences)
    check("byte-level BPE round-trips the very same string exactly", byte_back, OOV)
    # And the structural difference in one number: the byte tokenizer's base
    # alphabet covers every possible input; the code-point one covers what it saw.
    print(f"      base alphabet coverage: bytes 256/256 of all byte values; "
          f"code points {len(cp_char_to_id)} of 1114112 possible")
    check("the code-point base alphabet cannot cover Unicode",
          len(cp_char_to_id) < 0x110000)

    print(f"\n      total runtime {time.perf_counter() - t_start:.1f}s")
    summary()
