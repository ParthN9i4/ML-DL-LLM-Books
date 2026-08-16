"""Artifact 25.1 -- MinHash LSH near-duplicate detection + n-gram decontamination.

Pure NumPy. Three experiments, each self-verifying with hard assertions:
  (1) The measured LSH detection-probability curve matches the analytic
      S-curve  P(candidate) = 1 - (1 - s^r)^b  to a stated tolerance.
  (2) On a synthetic corpus with PLANTED near-duplicates, LSH candidate
      generation achieves stated recall/precision at the theoretical
      threshold s* = (1/b)^(1/r), judged against brute-force Jaccard.
  (3) An 8-gram decontamination checker recovers 100% of planted
      benchmark leaks with zero false positives.
"""
import time
import numpy as np

rng = np.random.default_rng(0)
U64 = np.uint64

# ---------------- MinHash configuration (FineWeb's: 14 bands x 8 rows) ----
B_BANDS, R_ROWS = 14, 8
K_HASH = B_BANDS * R_ROWS                       # 112 hash functions
S_STAR = (1.0 / B_BANDS) ** (1.0 / R_ROWS)      # ~0.719 similarity threshold

# One multiply-add hash per signature row: h_i(x) = a_i*x + c_i  (mod 2^64).
# a_i is forced odd so the map is a bijection on Z_{2^64}. Wraparound IS the
# modulus -- NumPy uint64 arithmetic wraps silently, which is exactly what
# we want. Never use Python's built-in hash() here: it is salted per process.
HASH_A = (rng.integers(1, 2**62, K_HASH, dtype=U64) << U64(1)) | U64(1)
HASH_C = rng.integers(0, 2**63, K_HASH, dtype=U64)


def minhash_batch(shingle_mat):
    """(n_docs, m) uint64 shingle IDs -> (n_docs, K_HASH) signatures.
    Row i of a signature is min over the doc's shingles of h_i(shingle)."""
    hv = shingle_mat[:, None, :] * HASH_A[None, :, None] + HASH_C[None, :, None]
    return hv.min(axis=2)


def band_fold(sigs):
    """Fold each band of R_ROWS rows into a single uint64 bucket key.
    (n, K_HASH) -> (n, B_BANDS). Two docs collide in band j iff all R_ROWS
    signature entries of that band agree (up to 2^-64 fold collisions)."""
    bands = sigs.reshape(sigs.shape[0], B_BANDS, R_ROWS)
    key = np.zeros((sigs.shape[0], B_BANDS), dtype=U64)
    for j in range(R_ROWS):                      # polynomial fold of the band
        key = key * U64(0x9E3779B97F4A7C15) + bands[:, :, j]
    return key


def analytic_curve(s):
    """P(>=1 band collision) for a pair with per-row match probability s."""
    return 1.0 - (1.0 - s ** R_ROWS) ** B_BANDS


# ================ Experiment 1: the S-curve is the content ================
def detection_curve(m=80, trials=600):
    """Pairs of m-element sets with controlled Jaccard: c shared random IDs,
    m-c private each => J = c/(2m-c). Measure band-collision frequency."""
    rows = []
    for s in np.linspace(0.05, 0.95, 19):
        c = int(round(2 * m * s / (1 + s)))     # shared count achieving ~s
        s_true = c / (2 * m - c)                # exact Jaccard actually built
        shared = rng.integers(0, 2**63, (trials, c), dtype=U64)
        A = np.concatenate([shared, rng.integers(0, 2**63, (trials, m - c), dtype=U64)], 1)
        Bm = np.concatenate([shared, rng.integers(0, 2**63, (trials, m - c), dtype=U64)], 1)
        ka, kb = band_fold(minhash_batch(A)), band_fold(minhash_batch(Bm))
        emp = (ka == kb).any(axis=1).mean()     # any band agrees -> candidate
        rows.append((s_true, emp, analytic_curve(s_true)))
    return np.array(rows)


# ============ Experiment 2: planted near-duplicates in a corpus ===========
VOCAB, DOC_LEN, N_BASE, N_PLANT = 2000, 160, 500, 100

def make_corpus():
    """500 random word-ID docs + 100 mutated copies (replacement rate 0..15%).
    A rate-rho copy keeps ~(1-rho)^3 of 3-word shingles => J ~ u/(2-u)."""
    docs = [rng.integers(0, VOCAB, DOC_LEN) for _ in range(N_BASE)]
    pairs = []
    for k in range(N_PLANT):
        rate = 0.15 * k / (N_PLANT - 1)
        d = docs[k].copy()
        mask = rng.random(DOC_LEN) < rate
        d[mask] = rng.integers(0, VOCAB, int(mask.sum()))
        docs.append(d)
        pairs.append((k, N_BASE + k))
    return docs, pairs


def shingle_sets(docs, n=3):
    """Pack each 3-word shingle into one uint64: x0*V^2 + x1*V + x2."""
    out = []
    for d in docs:
        d = d.astype(U64)
        sh = d[:-2] * U64(VOCAB) ** U64(2) + d[1:-1] * U64(VOCAB) + d[2:]
        out.append(np.unique(sh))
    return out


def jaccard(a, b):
    return len(np.intersect1d(a, b)) / len(np.union1d(a, b))


def lsh_candidates(shingles):
    """Bucket docs by (band, key); every same-bucket pair is a candidate."""
    m_min = min(len(s) for s in shingles)       # pad-free: truncate to m_min
    mat = np.stack([s[:m_min] for s in shingles])
    keys = band_fold(minhash_batch(mat))
    buckets, cands = {}, set()
    for i in range(keys.shape[0]):
        for bnd in range(B_BANDS):
            k = (bnd, int(keys[i, bnd]))
            for j in buckets.setdefault(k, []):
                cands.add((j, i))
            buckets[k].append(i)
    return cands


# ============ Experiment 3: n-gram decontamination checker ================
def ngram_hashes(seq, n=8):
    """Rolling polynomial hash of every n-gram, mod 2^64."""
    h = np.zeros(len(seq) - n + 1, dtype=U64)
    for j in range(n):
        h = h * U64(1099511628211) + seq[j:len(seq) - n + 1 + j].astype(U64)
    return h


def decontaminate(docs, benchmark, n=8):
    """Flag any doc sharing >=1 n-gram with any benchmark item."""
    bench = set()
    for item in benchmark:
        bench.update(ngram_hashes(item, n).tolist())
    flagged = {}
    for i, d in enumerate(docs):
        hits = sum(h in bench for h in ngram_hashes(d, n).tolist())
        if hits:
            flagged[i] = hits
    return flagged


if __name__ == "__main__":
    t0 = time.time()
    # ---- (1) S-curve --------------------------------------------------
    curve = detection_curve()
    dev = np.abs(curve[:, 1] - curve[:, 2]).max()
    print(f"[curve] b={B_BANDS} r={R_ROWS}  s* = (1/b)^(1/r) = {S_STAR:.4f}")
    print("[curve]   s_true   empirical   analytic 1-(1-s^r)^b")
    for s, e, a in curve[[4, 8, 12, 13, 14, 16]]:
        print(f"[curve]   {s:.3f}     {e:.3f}       {a:.3f}")
    print(f"[curve] max |empirical - analytic| over 19 points = {dev:.4f}")
    assert dev < 0.07, "S-curve deviates beyond tolerance"
    near = curve[np.argmin(np.abs(curve[:, 0] - S_STAR))]
    print(f"[curve] at s={near[0]:.3f} (nearest s*): measured {near[1]:.3f}, "
          f"1-(1-1/b)^b = {1-(1-1/B_BANDS)**B_BANDS:.3f}")

    # ---- (2) planted near-duplicates ---------------------------------
    docs, planted = make_corpus()
    sh = shingle_sets(docs)
    # brute-force ground truth over all 600*599/2 pairs
    J = {}
    for i in range(len(sh)):
        for j in range(i + 1, len(sh)):
            J[(i, j)] = jaccard(sh[i], sh[j])
    cands = lsh_candidates(sh)
    pos = {p for p, v in J.items() if v >= S_STAR}          # true positives set
    hi = {p for p, v in J.items() if v >= 0.80}
    rec_star = len(cands & pos) / len(pos)
    rec_hi = len(cands & hi) / len(hi)
    prec_05 = sum(J[p] >= 0.50 for p in cands) / len(cands)
    verified = {p for p in cands if J[p] >= S_STAR}          # verification pass
    print(f"[dedup] docs={len(docs)}  pairs={len(J)}  planted={len(planted)}  "
          f"pairs with J>=s*: {len(pos)}")
    print(f"[dedup] LSH candidates: {len(cands)}   "
          f"background (J<0.3) among them: {sum(J[p] < 0.3 for p in cands)}")
    print(f"[dedup] recall @ J>=s*: {rec_star:.3f}   recall @ J>=0.80: {rec_hi:.3f}")
    print(f"[dedup] candidate precision w.r.t. J>=0.50: {prec_05:.3f}   "
          f"after exact-Jaccard verification w.r.t. s*: "
          f"{len(verified & pos)/max(len(verified),1):.3f}")
    assert rec_star >= 0.75 and rec_hi >= 0.90, "LSH recall below threshold"
    assert prec_05 >= 0.90, "LSH candidate precision below threshold"
    assert verified <= pos and len(verified) == len(cands & pos)

    # ---- (3) decontamination -----------------------------------------
    bench = [rng.integers(0, VOCAB, 30) for _ in range(40)]
    corpus = [d.copy() for d in docs]
    leak_ids = rng.choice(len(corpus), 25, replace=False)
    for i in leak_ids:                       # splice item verbatim into doc
        item = bench[int(rng.integers(0, 40))]
        off = int(rng.integers(0, DOC_LEN - 30))
        corpus[i][off:off + 30] = item
    flagged = decontaminate(corpus, bench)
    tp = set(flagged) & set(leak_ids.tolist())
    fp = set(flagged) - set(leak_ids.tolist())
    print(f"[decon] planted leaks: {len(leak_ids)}  flagged: {len(flagged)}  "
          f"recovered: {len(tp)}  false positives: {len(fp)}")
    print(f"[decon] matching 8-grams per leaked doc: "
          f"min={min(flagged.values())} max={max(flagged.values())}")
    assert len(tp) == len(leak_ids) and not fp, "decontamination missed a leak"

    print(f"[done] all assertions passed in {time.time() - t0:.1f}s")
