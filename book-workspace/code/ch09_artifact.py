"""Artifact 9.1 - PCA by power iteration and Gaussian-mixture EM from scratch.
Pure NumPy core, sklearn only as a wrapped cross-check.  Asserts: (a) power
iteration matches np.linalg.svd to ~1e-10 up to sign and the variance /
reconstruction / SVD characterisations agree; (b) the GMM log-likelihood never
decreases; (c) as sigma^2 -> 0 the EM hard assignments become exactly the
k-means labels; (d) an isometric unrolling keeps k-NN sets while wrecking
global distances."""
import time
import numpy as np

try:
    import sklearn  # noqa: F401
    from sklearn.mixture import GaussianMixture
    from sklearn.manifold import TSNE
except ImportError:
    sklearn = None

RNG = np.random.default_rng(0)
# --- 9.1  PCA three ways ----------------------------------------------------
def power_iteration_pca(X, k, tol=1e-15, max_iter=5000):
    """Top-k eigenpairs of the sample covariance by power iteration, with
    Hotelling deflation C <- C - lam v v^T between components."""
    n = X.shape[0]
    Xc = X - X.mean(axis=0)
    C = (Xc.T @ Xc) / (n - 1)                    # d x d, symmetric PSD
    d = C.shape[0]
    lams, vecs, iters = [], [], []
    for _ in range(k):
        v = RNG.standard_normal(d)
        v, used = v / np.linalg.norm(v), 0
        for t in range(max_iter):
            w = C @ v
            nw = np.linalg.norm(w)
            if nw < 1e-300:                      # deflated into the null space
                break
            v, used = w / nw, t + 1
            lam = v @ C @ v                      # Rayleigh quotient
            # Stop on the EIGENVECTOR residual: the Rayleigh quotient is accurate
            # to the SQUARE of the vector error, so an eigenvalue test stops early.
            if np.linalg.norm(C @ v - lam * v) <= tol * max(1.0, abs(lam)):
                break
        lam = v @ C @ v
        v = v * np.sign(v[np.argmax(np.abs(v))])  # canonical sign
        lams.append(lam); vecs.append(v.copy()); iters.append(used)
        C = C - lam * np.outer(v, v)
    return np.array(lams), np.array(vecs), iters

def demo_pca():
    print("[1] PCA: power iteration vs SVD, and the three derivations")
    d, n, k = 8, 4000, 4
    true_spec = np.array([25.0, 12.0, 6.0, 3.0, 1.5, 0.8, 0.4, 0.2])
    Q, _ = np.linalg.qr(RNG.standard_normal((d, d)))     # random eigenbasis
    A = Q @ np.diag(np.sqrt(true_spec)) @ Q.T            # so Cov = Q diag Q^T
    X = RNG.standard_normal((n, d)) @ A + np.array([3.0, -1.0, 0, 0, 2, 0, 0, 0])
    lam_pi, V_pi, iters = power_iteration_pca(X, k)
    Xc = X - X.mean(axis=0)
    s, Vt = np.linalg.svd(Xc, full_matrices=False)[1:]
    lam_svd = s ** 2 / (n - 1)                   # singular values -> eigenvalues
    V_svd = Vt[:k] * np.sign(Vt[:k][np.arange(k), np.abs(Vt[:k]).argmax(1)])[:, None]
    err_vec, err_lam = (np.abs(np.abs(V_pi) - np.abs(V_svd)).max(),
                        np.abs(lam_pi - lam_svd[:k]).max())
    print(f"    eigenvalues (power iter) = {np.round(lam_pi, 6)}")
    print(f"    eigenvalues (SVD)        = {np.round(lam_svd[:k], 6)}")
    print(f"    sweeps per component = {iters};  max component error = {err_vec:.3e}"
          f";  max eigenvalue error = {err_lam:.3e}")
    assert err_vec < 1e-10 and err_lam < 1e-10
    proj_var = np.array([np.var(Xc @ V_pi[j], ddof=1) for j in range(k)])
    Z = Xc @ V_pi.T                              # scores
    recon, tail = ((Xc - Z @ V_pi) ** 2).sum() / (n - 1), lam_svd[k:].sum()
    print(f"    var of projections = {np.round(proj_var, 6)}   (= eigenvalues)")
    print(f"    rank-{k} reconstruction error = {recon:.10f}   eigen-tail = "
          f"{tail:.10f}   (diff {abs(recon - tail):.3e})")
    assert np.abs(proj_var - lam_pi).max() < 1e-9 and abs(recon - tail) < 1e-9
    Cw = np.cov((np.diag(lam_svd ** -0.5) @ Vt) @ Xc.T, ddof=1)   # PCA whitening
    print(f"    whitened covariance: max |C - I| = {np.abs(Cw - np.eye(d)).max():.3e}")
    assert np.abs(Cw - np.eye(d)).max() < 1e-10
# --- 9.3  Lloyd's algorithm ------------------------------------------------
def kmeans_lloyd(X, mu0, max_iter=200):
    """Lloyd from a given start -> labels, means, within-cluster SS per sweep."""
    mu, objs, labels = mu0.copy(), [], None
    for _ in range(max_iter):
        D = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(-1)      # n x K
        new = D.argmin(1)                        # assignment step
        objs.append(D[np.arange(len(X)), new].sum())
        for j in range(len(mu)):
            if (new == j).any():
                mu[j] = X[new == j].mean(0)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
    return labels, mu, np.array(objs)
# --- 9.4  Gaussian-mixture EM ----------------------------------------------
def logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)

def log_gauss(X, mu, Sigma):
    """log N(x | mu, Sigma) for every row of X, via Cholesky (never an inverse)."""
    L = np.linalg.cholesky(Sigma)
    z = np.linalg.solve(L, (X - mu).T)           # L z = x - mu
    return -0.5 * (X.shape[1] * np.log(2 * np.pi)
                   + 2.0 * np.log(np.diag(L)).sum() + (z ** 2).sum(0))

def em_gmm(X, pi, mu, Sigma, n_iter=60, reg=1e-8, update_cov=True):
    """Full-covariance EM.  The value recorded at step t is the exact incomplete-
    data log-likelihood of step t-1's M-step output: the true likelihood path,
    not the ELBO and not a post-hoc rescoring."""
    n, d, K = X.shape[0], X.shape[1], len(pi)
    pi, mu, Sigma = pi.copy(), mu.copy(), Sigma.copy()
    lls, singular = [], False
    for _ in range(n_iter):
        try:                                     # E step, in log space
            logp = np.stack([np.log(pi[j]) + log_gauss(X, mu[j], Sigma[j])
                             for j in range(K)], axis=1)         # n x K
        except np.linalg.LinAlgError:
            singular = True                      # a covariance lost rank
            break
        ll_row = logsumexp(logp, axis=1)
        lls.append(ll_row.sum())
        R = np.exp(logp - ll_row[:, None])       # responsibilities, rows sum to 1
        Nk = R.sum(0) + 1e-300                   # M step: weighted moments
        pi = Nk / n
        mu = (R.T @ X) / Nk[:, None]
        if update_cov:
            for j in range(K):
                Xd = X - mu[j]
                Sigma[j] = (Xd * R[:, j:j + 1]).T @ Xd / Nk[j] + reg * np.eye(d)
    return np.array(lls), pi, mu, Sigma, R, singular

def make_blobs(n_per, centers, scales):
    Xs = [RNG.standard_normal((n_per, len(c))) * s + c
          for c, s in zip(centers, scales)]
    return np.vstack(Xs), np.repeat(np.arange(len(centers)), n_per)

def demo_em():
    print("[2] Gaussian-mixture EM: monotone log-likelihood on every iteration")
    X, _ = make_blobs(300, np.array([[0.0, 0.0], [4.0, 1.0], [1.0, 5.0]]),
                      [0.8, 1.3, 0.6])
    K = 3
    mu0 = X[RNG.choice(len(X), K, replace=False)].copy()   # random data points
    S0 = np.stack([np.cov(X.T, ddof=1)] * K)               # global covariance
    lls, pi, mu, Sigma, R, _ = em_gmm(X, np.full(K, 1/K), mu0, S0, n_iter=60)
    diffs = np.diff(lls)                         # the object of Theorem 9.6
    print(f"    log-likelihood {lls[0]:.6f} -> {lls[-1]:.6f} over {len(lls)} iterations")
    print(f"    smallest per-iteration increment = {diffs.min():.3e}   (increments"
          f" below zero: {(diffs < 0).sum()} of {len(diffs)});  first five ="
          f" {np.array2string(diffs[:5], precision=6)}")
    assert np.all(diffs >= -1e-9), "EM log-likelihood decreased"
    print(f"    weights = {np.round(pi, 4)} (truth 1/3);  means = "
          f"{np.round(mu[np.lexsort(mu.T[::-1])].ravel(), 3)}")
    if sklearn is not None:
        ref = len(X) * GaussianMixture(
            K, covariance_type="full", reg_covar=1e-8, max_iter=400, tol=1e-10,
            random_state=0, init_params="random_from_data").fit(X).score(X)
        print(f"    sklearn GaussianMixture log-likelihood = {ref:.6f}   "
              f"(ours - theirs = {lls[-1] - ref:+.3e})")
        assert lls[-1] <= ref + 1e-4
    else:
        print("    [skipped: sklearn not installed]")
    return X

def demo_collapse(X):
    print("[3] The singularity: an unregularised component collapsing onto a point")
    Xd = np.vstack([X, X[0:1]])                  # one duplicated row: the usual culprit
    S, K = np.cov(Xd.T, ddof=1), 3
    mu0 = np.stack([Xd[0], Xd[1], Xd[2]])        # a mean sitting exactly on a datum
    S0 = np.stack([S, S, 1e-4 * np.eye(2)])      # started narrow: inside the basin
    lls, pi, mu, Sigma, _, sing = em_gmm(Xd, np.full(K, 1/K), mu0, S0, 40, reg=0.0)
    small = min(np.linalg.eigvalsh(Sg)[0] for Sg in Sigma)   # rank loss?
    print(f"    log-likelihood {lls[0]:.3f} -> {lls[-1]:.3f} in {len(lls)} iterations "
          f"  (monotone: {bool(np.all(np.diff(lls) >= -1e-9))})")
    print(f"    Cholesky then failed: {sing};  smallest covariance eigenvalue ="
          f" {small:.3e};  that component held {pi.min()*len(Xd):.4f} of {len(Xd)} points")
    assert small < 1e-6 and lls[-1] > lls[0] + 100 and sing
    print("    -> the likelihood is unbounded above; EM climbed a singular direction."
          "  A variance floor (reg_covar) is what keeps the problem well posed.")

def demo_zero_variance_limit():
    print("[4] sigma^2 -> 0: EM hard assignments become exactly k-means labels")
    X, _ = make_blobs(200, np.array([[0.0, 0.0], [3.5, 0.5], [1.0, 4.0]]), [0.7] * 3)
    K, d = 3, 2
    mu0 = X[RNG.choice(len(X), K, replace=False)].copy()   # shared with Lloyd
    km_labels, km_mu, km_obj = kmeans_lloyd(X, mu0)
    print(f"    Lloyd: within-cluster SS {km_obj[0]:.4f} -> {km_obj[-1]:.4f} in "
          f"{len(km_obj)} sweeps, monotone: {bool(np.all(np.diff(km_obj) <= 1e-9))}")
    assert np.all(np.diff(km_obj) <= 1e-9)   # Theorem 9.4: strict decrease or stop
    print("      sigma^2   mismatched labels   max||mu_EM - mu_kmeans||")
    for s2 in [4.0, 1.0, 0.25, 0.05, 0.01, 1e-3]:
        S0 = np.stack([s2 * np.eye(d)] * K)      # isotropic and FROZEN
        _, _, mu_em, _, R, _ = em_gmm(X, np.full(K, 1/K), mu0.copy(), S0, 120,
                                      update_cov=False)
        mism, gap = int((R.argmax(1) != km_labels).sum()), np.abs(mu_em - km_mu).max()
        print(f"      {s2:7.4g}   {mism:17d}   {gap:.3e}")
        if s2 <= 0.01:
            assert mism == 0 and gap < 1e-9, f"EM != Lloyd at sigma^2={s2}"
    print("    -> by sigma^2 = 0.01 responsibilities are one-hot in float64 and the"
          " M-step mean update is literally the Lloyd centroid update")

# --- 9.5  Neighbourhoods kept, global distances destroyed ------------------
def pdist(X):
    return np.sqrt(np.maximum(((X[:, None] - X[None]) ** 2).sum(-1), 0.0))

def knn_overlap(Da, Db, k):
    A = np.argsort(Da + np.eye(len(Da)) * 1e18, axis=1)[:, :k]
    B = np.argsort(Db + np.eye(len(Db)) * 1e18, axis=1)[:, :k]
    return np.mean([len(set(a) & set(b)) / k for a, b in zip(A, B)])

def demo_embedding_pitfall():
    print("[5] Manifold embeddings: neighbourhoods kept, global geometry lost")
    n, r0, b = 1200, 1.0, 0.30                   # 1200 points, 719400 pairs
    # Archimedean spiral r(t) = r0 + b t extruded along a height axis: an
    # intrinsically flat surface, so an EXACT 2-D isometric unrolling exists.
    t, h = np.sqrt(RNG.uniform(0.0, (6 * np.pi) ** 2, n)), RNG.uniform(0, 10.0, n)
    r = r0 + b * t
    Xa = np.stack([r * np.cos(t), h, r * np.sin(t)], axis=1)
    q = np.sqrt(r ** 2 + b ** 2)                 # arc length int sqrt(r^2+b^2) dt
    E = np.stack([(r * q + b ** 2 * np.log(r + q)) / (2 * b), h], axis=1)
    Da, De, iu = pdist(Xa), pdist(E), np.triu_indices(n, 1)
    a, e = Da[iu], De[iu]                        # all 719,400 pairs
    pear = np.corrcoef(a, e)[0, 1]
    rho = np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(e)))[0, 1]
    for k in (5, 10, 30):
        print(f"    {k:2d}-NN set overlap ambient vs embedded ="
              f" {knn_overlap(Da, De, k):.3f}")
    print(f"    Pearson r(ambient, embedded pairwise distance) = {pear:.4f};"
          f"  Spearman rho = {rho:.4f}")
    ratio = e / np.maximum(a, 1e-12)
    w, near = ratio.argmax(), a < 2.0            # a<2 = closer than one roll layer
    print(f"    worst stretch: ambient {a[w]:.4f} -> embedded {e[w]:.4f} (x{ratio[w]:.1f})")
    print(f"    of the {near.sum()} pairs within ambient distance 2.0, "
          f"{(near & (e > 15)).sum()} sit more than 15 apart in the embedding")
    assert knn_overlap(Da, De, 10) > 0.75 and pear < 0.8 and ratio.max() > 15
    if sklearn is None:
        print("    [skipped: sklearn not installed]")
        return
    C = np.zeros((3, 10))                        # blobs of very different radii
    C[1, 0], C[2, 1] = 14.0, 14.0
    Xb, yb = make_blobs(200, C, [0.15, 0.6, 1.8])
    Y = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Xb)
    rad = lambda Z, j: np.sqrt(((Z[yb == j] - Z[yb == j].mean(0)) ** 2).sum(1).mean())
    rin, rout = [rad(Xb, j) for j in range(3)], [rad(Y, j) for j in range(3)]
    print(f"    original cluster RMS radii = {np.round(rin, 3)} (ratio {max(rin)/min(rin):.1f}x)")
    print(f"    t-SNE cluster RMS radii    = {np.round(rout, 3)} (ratio {max(rout)/min(rout):.1f}x)")
    assert max(rin) / min(rin) > 8 and max(rout) / min(rout) < 4

if __name__ == "__main__":
    t0 = time.perf_counter(); demo_pca(); print(); Xg = demo_em()
    print(); demo_collapse(Xg)
    print(); demo_zero_variance_limit(); print(); demo_embedding_pitfall()
    print(f"\nall assertions passed in {time.perf_counter() - t0:.1f} s")
