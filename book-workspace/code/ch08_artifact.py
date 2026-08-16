"""Artifact 8.1 -- gradient-boosted regression trees from scratch (pure NumPy).

Exact-greedy and histogram split finders share one tree builder and one
predictor, so the only difference is how candidate splits are enumerated. Both
use the second-order gain and leaf weight of Chen & Guestrin (XGBoost):
    gain = 1/2 [ GL^2/(HL+lam) + GR^2/(HR+lam) - G^2/(H+lam) ] - gamma
    leaf = -G/(H+lam),   g_i = dL/dF_i,  h_i = d2L/dF_i^2
Self-checks: (a) sklearn GradientBoostingRegressor on matched hyperparameters,
(b) brute force reproducing the vectorized split and gain = 0.5 * SSE-reduction
at lam = 0, (c) histogram matching exact accuracy at n = 1e5, but faster.
"""
import time
import numpy as np
try:
    from sklearn.ensemble import GradientBoostingRegressor
except ImportError:
    GradientBoostingRegressor = None

def make_data(n, d, rng):
    """Friedman #1: signal in the first 5 coordinates, remaining d-5 pure noise."""
    X = rng.uniform(0.0, 1.0, size=(n, d))
    y = (10.0 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20.0 * (X[:, 2] - 0.5) ** 2
         + 10.0 * X[:, 3] + 5.0 * X[:, 4] + rng.normal(0.0, 1.0, size=n))
    return X, y

class Tree:
    """Struct-of-arrays tree; feature = -1 marks a leaf."""
    __slots__ = ("feature", "threshold", "left", "right", "value")
    def __init__(self):
        self.feature, self.threshold, self.left, self.right, self.value = ([] for _ in range(5))
    def add(self):                                   # append one unfilled leaf
        self.feature.append(-1); self.threshold.append(0.0)
        self.left.append(-1); self.right.append(-1); self.value.append(0.0)
        return len(self.feature) - 1
    def finish(self):                                # lists -> arrays, once built
        for k in ("feature", "left", "right"):
            setattr(self, k, np.asarray(getattr(self, k), dtype=np.int64))
        self.threshold = np.asarray(self.threshold, dtype=np.float64)
        self.value = np.asarray(self.value, dtype=np.float64)
        return self

def tree_predict(t, X):
    """Level-synchronous traversal: every row walks down together."""
    node = np.zeros(X.shape[0], dtype=np.int64)
    while True:
        idx = np.flatnonzero(t.feature[node] >= 0)
        if idx.size == 0:
            return t.value[node]
        nd = node[idx]
        left = X[idx, t.feature[nd]] <= t.threshold[nd]   # one gather per level
        node[idx] = np.where(left, t.left[nd], t.right[nd])

def gain_of(GL, HL, GR, HR, G, H, lam):
    return 0.5 * (GL * GL / (HL + lam) + GR * GR / (HR + lam) - G * G / (H + lam))

def best_split_exact(X, g, h, order, G, H, lam, min_child):
    """order: (d, m) sample indices sorted by each feature's value in this node.
    Returns (gain, feature, threshold); feature = -1 means no admissible split."""
    best = (0.0, -1, 0.0)
    for f in range(order.shape[0]):
        o = order[f]; xs = X[o, f]
        GL, HL = np.cumsum(g[o])[:-1], np.cumsum(h[o])[:-1]
        GR, HR = G - GL, H - HL
        # a cut is admissible only strictly between two DISTINCT feature values
        ok = (xs[:-1] < xs[1:]) & (HL >= min_child) & (HR >= min_child)
        if not ok.any():
            continue
        gains = np.where(ok, gain_of(GL, HL, GR, HR, G, H, lam), -np.inf)
        k = int(np.argmax(gains))
        if gains[k] > best[0]:
            best = (float(gains[k]), f, 0.5 * (xs[k] + xs[k + 1]))
    return best

def build_exact(X, g, h, max_depth, lam, gamma, min_child):
    order0 = np.argsort(X, axis=0, kind="stable").T.copy()   # (d, n): sort ONCE
    t = Tree(); stack = [(t.add(), order0, 0)]
    while stack:
        nid, order, depth = stack.pop()
        G, H = g[order[0]].sum(), h[order[0]].sum()
        gain, f, thr = 0.0, -1, 0.0
        if depth < max_depth and order.shape[1] >= 2:
            gain, f, thr = best_split_exact(X, g, h, order, G, H, lam, min_child)
        if f < 0 or gain <= gamma:
            t.value[nid] = -G / (H + lam)
            continue
        mask = X[:, f] <= thr                              # boolean over all rows
        lo = np.stack([r[mask[r]] for r in order])         # stable: stays sorted
        ro = np.stack([r[~mask[r]] for r in order])
        t.feature[nid], t.threshold[nid] = f, thr
        t.left[nid], t.right[nid] = t.add(), t.add()
        stack.append((t.left[nid], lo, depth + 1))
        stack.append((t.right[nid], ro, depth + 1))
    return t.finish()

def make_bins(X, max_bins):
    """Quantile bin edges. edges[f][b] is the UPPER bound of bin b, so 'bin <= b'
    and 'x <= edges[f][b]' are the same event: histogram trees emit real-valued
    thresholds and can share the exact-greedy predictor."""
    edges, Xb = [], np.empty(X.shape, dtype=np.uint8)
    q = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]       # interior quantile levels
    for f in range(X.shape[1]):
        e = np.unique(np.quantile(X[:, f], q))          # ties collapse: fewer bins
        edges.append(e)
        Xb[:, f] = np.searchsorted(e, X[:, f], side="left")
    return edges, Xb

def node_hist(Xb_node, gi, hi, nb):
    """(d, nb) gradient and hessian histograms: one linear pass per feature."""
    d = Xb_node.shape[1]
    Hg, Hh = np.empty((d, nb)), np.empty((d, nb))
    for f in range(d):
        col = Xb_node[:, f]                             # uint8, read sequentially
        Hg[f] = np.bincount(col, weights=gi, minlength=nb)[:nb]
        Hh[f] = np.bincount(col, weights=hi, minlength=nb)[:nb]
    return Hg, Hh

def best_split_hist(Hg, Hh, edges, G, H, lam, min_child):
    best = (0.0, -1, 0.0)
    for f in range(len(edges)):
        nb = len(edges[f])                   # only bins 0..nb-1 have a real edge
        if nb == 0:
            continue
        GL, HL = np.cumsum(Hg[f])[:nb], np.cumsum(Hh[f])[:nb]
        GR, HR = G - GL, H - HL
        ok = (HL >= min_child) & (HR >= min_child)
        if not ok.any():
            continue
        gains = np.where(ok, gain_of(GL, HL, GR, HR, G, H, lam), -np.inf)
        k = int(np.argmax(gains))
        if gains[k] > best[0]:
            best = (float(gains[k]), f, float(edges[f][k]))
    return best

def build_hist(X, Xb, edges, g, h, max_depth, lam, gamma, min_child, nb):
    t = Tree(); stack = [(t.add(), np.arange(X.shape[0]), 0, None)]
    while stack:
        nid, idx, depth, hists = stack.pop()
        gi, hi = g[idx], h[idx]; G, H = gi.sum(), hi.sum()
        gain, f, thr = 0.0, -1, 0.0
        if depth < max_depth and idx.size >= 2:
            if hists is None:
                hists = node_hist(Xb[idx], gi, hi, nb)
            gain, f, thr = best_split_hist(*hists, edges, G, H, lam, min_child)
        if f < 0 or gain <= gamma:
            t.value[nid] = -G / (H + lam)
            continue
        m = X[idx, f] <= thr
        li, ri = idx[m], idx[~m]
        small = li if li.size <= ri.size else ri            # subtraction trick:
        hs = node_hist(Xb[small], g[small], h[small], nb)   # build cheap child,
        hl = (hists[0] - hs[0], hists[1] - hs[1])           # subtract for other
        hli, hri = (hs, hl) if li.size <= ri.size else (hl, hs)
        t.feature[nid], t.threshold[nid] = f, thr
        t.left[nid], t.right[nid] = t.add(), t.add()
        stack.append((t.left[nid], li, depth + 1, hli))
        stack.append((t.right[nid], ri, depth + 1, hri))
    return t.finish()

class GBRT:
    """Stagewise fit F_m = F_{m-1} + eta * tree_m on the local quadratic model."""
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, lam=0.0,
                 gamma=0.0, min_child_weight=1.0, method="exact", max_bins=64):
        self.__dict__.update(locals()); del self.self
    def fit(self, X, y):
        self.base_ = float(y.mean())      # argmin of squared error, constant model
        F, self.trees_ = np.full(y.shape, self.base_), []
        if self.method == "hist":
            self.edges_, Xb = make_bins(X, self.max_bins)
        for _ in range(self.n_estimators):
            g, h = F - y, np.ones_like(F)                 # squared-error derivatives
            if self.method == "exact":
                t = build_exact(X, g, h, self.max_depth, self.lam, self.gamma,
                                self.min_child_weight)
            else:
                t = build_hist(X, Xb, self.edges_, g, h, self.max_depth, self.lam,
                               self.gamma, self.min_child_weight, self.max_bins)
            F += self.learning_rate * tree_predict(t, X)
            self.trees_.append(t)
        return self
    def predict(self, X):
        F = np.full(X.shape[0], self.base_)
        for t in self.trees_:
            F += self.learning_rate * tree_predict(t, X)
        return F

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def brute_force_split(X, g, h, lam, min_child):
    """Independent double loop over every (feature, midpoint) candidate, summing
    GL/HL from scratch. Returns (gain, f, thr, sse_reduction)."""
    G, H, best = g.sum(), h.sum(), (-np.inf, -1, 0.0, 0.0)
    sse0 = float(np.sum((g - g.mean()) ** 2))
    for f in range(X.shape[1]):
        v = np.unique(X[:, f])
        for a, b in zip(v[:-1], v[1:]):
            thr = 0.5 * (a + b)
            m = X[:, f] <= thr
            GL, HL = g[m].sum(), h[m].sum()
            GR, HR = G - GL, H - HL
            if HL < min_child or HR < min_child:
                continue
            gn = gain_of(GL, HL, GR, HR, G, H, lam)
            if gn > best[0]:
                sse = float(np.sum((g[m] - g[m].mean()) ** 2)
                            + np.sum((g[~m] - g[~m].mean()) ** 2))
                best = (float(gn), f, float(thr), sse0 - sse)
    return best

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # (b) second-order gain formula vs an independent brute-force search
    Xs, ys = make_data(300, 5, rng)
    gs, hs = ys.mean() - ys, np.ones(300)
    order = np.argsort(Xs, axis=0, kind="stable").T.copy()
    for lam in (0.0, 5.0):
        bf = brute_force_split(Xs, gs, hs, lam, 1.0)
        vec = best_split_exact(Xs, gs, hs, order, gs.sum(), hs.sum(), lam, 1.0)
        assert vec[1] == bf[1] and abs(vec[2] - bf[2]) < 1e-12, (vec, bf)
        print(f"[b] lam={lam:>4}: vectorized split (f={vec[1]}, thr={vec[2]:.6f}) "
              f"== brute force; |gain diff| = {abs(vec[0] - bf[0]):.3e}")
        if lam == 0.0:
            print(f"      gain={bf[0]:.6f}  0.5*SSE-reduction={0.5 * bf[3]:.6f}  "
                  f"|diff| = {abs(bf[0] - 0.5 * bf[3]):.3e}")
            assert abs(bf[0] - 0.5 * bf[3]) < 1e-9
    # (a) matched-hyperparameter cross-check against sklearn
    Xtr, ytr = make_data(2000, 8, rng)
    Xte, yte = make_data(2000, 8, rng)
    kw = dict(n_estimators=60, learning_rate=0.1, max_depth=3)
    p_ours = GBRT(**kw, method="exact").fit(Xtr, ytr).predict(Xte)
    r_ours = rmse(p_ours, yte)
    if GradientBoostingRegressor is None:
        print(f"[a] ours test RMSE = {r_ours:.9f}  [skipped: sklearn not installed]")
    else:
        p_sk = GradientBoostingRegressor(subsample=1.0, min_samples_leaf=1,
                                         random_state=0, **kw).fit(Xtr, ytr).predict(Xte)
        r_sk = rmse(p_sk, yte)
        rel = abs(r_ours - r_sk) / r_sk
        print(f"[a] test RMSE  ours={r_ours:.9f}  sklearn={r_sk:.9f}")
        print(f"    rel RMSE diff = {rel:.3e} (tol 2e-2);  max|pred diff| over "
              f"2000 test points = {np.max(np.abs(p_ours - p_sk)):.3e}")
        assert rel < 0.02, rel
    # (c) histogram vs exact-greedy at n = 1e5
    n, d = 100_000, 16
    Xb_tr, yb_tr = make_data(n, d, rng)
    Xb_te, yb_te = make_data(20_000, d, rng)
    cfg = dict(n_estimators=10, learning_rate=0.2, max_depth=6, lam=1.0)
    rows = []
    for meth, bins in (("exact", 64), ("hist", 255), ("hist", 64)):
        m = GBRT(**cfg, method=meth, max_bins=bins)
        t0 = time.perf_counter(); m.fit(Xb_tr, yb_tr)
        rows.append((meth if meth == "exact" else f"hist({bins})",
                     time.perf_counter() - t0, rmse(m.predict(Xb_te), yb_te)))
    print(f"\n[c] n={n}, d={d}, {cfg['n_estimators']} trees, depth "
          f"{cfg['max_depth']}, lambda={cfg['lam']}")
    print(f"    {'method':<12}{'fit (s)':>10}{'speedup':>10}{'test RMSE':>12}")
    base = rows[0][1]
    for name, s, r in rows:
        print(f"    {name:<12}{s:>10.3f}{base / s:>9.2f}x{r:>12.5f}")
    for name, s, r in rows[1:]:
        assert abs(r - rows[0][2]) < 0.05, (name, r, rows[0][2])
        assert s < base, (name, s, base)
    print(f"    max |RMSE_hist - RMSE_exact| = "
          f"{max(abs(r - rows[0][2]) for _, _, r in rows[1:]):.5f}  (tol 0.05)")
    print("\nall checks passed")
