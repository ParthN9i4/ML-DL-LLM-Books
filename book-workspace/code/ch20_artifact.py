"""
Chapter 20 Artifact -- Vision Transformers and Multimodal Encoders.

Part 1: a ViT forward pass from scratch (patchify as a strided convolution,
        class token, position embedding, pre-norm blocks), with parameters and
        MACs checked against closed forms.  ViT-Base/16 is counted by shape; a
        tiny ViT is executed with every matmul instrumented.
Part 2: InfoNCE and the pairwise sigmoid loss from scratch, gradients verified
        by central finite differences, plus numerical evidence that InfoNCE
        couples the batch while the sigmoid loss decomposes over pairs.

No timm, no pretrained weights, no downloads.  torch is an optional cross-check.
"""

import numpy as np
from scipy.special import erf, expit  # expit == logistic sigmoid

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

RNG = np.random.default_rng(0)
_MACS = [0]  # instrumented multiply-accumulate counter


def mm(a, b):
    """Matmul that records its multiply-accumulate count."""
    _MACS[0] += a.shape[0] * a.shape[1] * b.shape[1]
    return a @ b


# --------------------------------------------------------------------------
# 1. Closed-form parameter and MAC counts
# --------------------------------------------------------------------------
BASE = dict(img=224, patch=16, d=768, L=12, heads=12, ch=3, n_cls=1000)
TINY = dict(img=32, patch=8, d=48, L=3, heads=4, ch=3, n_cls=10)


def analytic_params(c):
    """Parameter count of a pre-norm ViT with MLP ratio 4, all biases present."""
    d, P, L = c["d"], c["patch"], c["L"]
    n_patch = (c["img"] // P) ** 2
    n = n_patch + 1                                  # + class token
    patch_embed = c["ch"] * P * P * d + d            # strided conv weight+bias
    # per block: qkv (3d^2+3d) + proj (d^2+d) + fc1 (4d^2+4d) + fc2 (4d^2+d)
    #            + two LayerNorms (4d)  =  12 d^2 + 13 d
    block = 12 * d * d + 13 * d
    return dict(
        n_patch=n_patch, n=n,
        patch_embed=patch_embed, cls=d, pos=n * d,
        blocks=L * block, final_ln=2 * d,
        backbone=patch_embed + d + n * d + L * block + 2 * d,
        head=d * c["n_cls"] + c["n_cls"],
    )


def analytic_macs(c):
    """Multiply-accumulates for one image, matmuls only (LN/softmax/GELU are O(nd))."""
    d, P, L = c["d"], c["patch"], c["L"]
    n_patch = (c["img"] // P) ** 2
    n = n_patch + 1
    pe = n_patch * (c["ch"] * P * P) * d              # patch projection
    # per block: 12 n d^2 (four projections + two MLP matmuls) + 2 n^2 d (QK^T, AV)
    blk = 12 * n * d * d + 2 * n * n * d
    return dict(patch_embed=pe, per_block=blk, blocks=L * blk,
                head=d * c["n_cls"], total=pe + L * blk + d * c["n_cls"])


# --------------------------------------------------------------------------
# 2. The model
# --------------------------------------------------------------------------
def init_vit(c, rng):
    d, P, L, ch = c["d"], c["patch"], c["L"], c["ch"]
    n = (c["img"] // P) ** 2 + 1
    s = 0.02
    p = {
        "pe_w": rng.normal(0, s, (d, ch * P * P)),   # == conv weight (d,ch,P,P)
        "pe_b": np.zeros(d),
        "cls": rng.normal(0, s, (1, d)),
        "pos": rng.normal(0, s, (n, d)),
        "ln_g": np.ones(d), "ln_b": np.zeros(d),
        "head_w": rng.normal(0, s, (c["n_cls"], d)), "head_b": np.zeros(c["n_cls"]),
        "blocks": [],
    }
    for _ in range(L):
        p["blocks"].append({
            "n1g": np.ones(d), "n1b": np.zeros(d),
            "qkv_w": rng.normal(0, s, (3 * d, d)), "qkv_b": np.zeros(3 * d),
            "pr_w": rng.normal(0, s, (d, d)), "pr_b": np.zeros(d),
            "n2g": np.ones(d), "n2b": np.zeros(d),
            "f1_w": rng.normal(0, s, (4 * d, d)), "f1_b": np.zeros(4 * d),
            "f2_w": rng.normal(0, s, (d, 4 * d)), "f2_b": np.zeros(d),
        })
    return p


def count_params(p):
    tot = sum(v.size for k, v in p.items() if k != "blocks")
    return tot + sum(a.size for blk in p["blocks"] for a in blk.values())


def patchify(x, W, b, P):
    """Patch embedding == convolution with kernel P and stride P. x: (ch,H,W)."""
    ch, H, Wd = x.shape
    t = x.reshape(ch, H // P, P, Wd // P, P).transpose(1, 3, 0, 2, 4)
    cols = t.reshape((H // P) * (Wd // P), ch * P * P)   # (n_patch, ch*P*P)
    return mm(cols, W.T) + b


def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def mha(x, blk, heads):
    n, d = x.shape
    dh = d // heads
    qkv = mm(x, blk["qkv_w"].T) + blk["qkv_b"]
    q, k, v = np.split(qkv, 3, axis=1)
    out = np.empty_like(x)
    for h in range(heads):
        sl = slice(h * dh, (h + 1) * dh)
        a = softmax(mm(q[:, sl], k[:, sl].T) / np.sqrt(dh))
        out[:, sl] = mm(a, v[:, sl])
    return mm(out, blk["pr_w"].T) + blk["pr_b"]


def vit_forward(img, p, c):
    """Pre-norm ViT. img: (ch,H,W). Returns (logits, tokens)."""
    z = patchify(img, p["pe_w"], p["pe_b"], c["patch"])
    z = np.concatenate([p["cls"], z], axis=0) + p["pos"]
    for blk in p["blocks"]:
        z = z + mha(layernorm(z, blk["n1g"], blk["n1b"]), blk, c["heads"])
        h = layernorm(z, blk["n2g"], blk["n2b"])
        z = z + mm(gelu(mm(h, blk["f1_w"].T) + blk["f1_b"]), blk["f2_w"].T) + blk["f2_b"]
    z = layernorm(z, p["ln_g"], p["ln_b"])
    return mm(z[:1], p["head_w"].T) + p["head_b"], z


# --------------------------------------------------------------------------
# 3. Contrastive objectives
# --------------------------------------------------------------------------
def l2n(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _unnorm_grad(X, G):
    """Push a gradient w.r.t. u = x/||x|| back through the normalization."""
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    U = X / nrm
    return (G - U * (G * U).sum(1, keepdims=True)) / nrm


def infonce(Xi, Xt, logit_scale):
    """Symmetric CLIP loss.  Returns (loss, dXi, dXt)."""
    B = Xi.shape[0]
    Zi, Zt = l2n(Xi), l2n(Xt)
    S = Zi @ Zt.T
    M = S * logit_scale
    Pr, Pc = softmax(M, axis=1), softmax(M, axis=0)
    I = np.eye(B)
    loss = -0.5 * (np.log(np.diag(Pr)).sum() + np.log(np.diag(Pc)).sum()) / B
    G = 0.5 / B * ((Pr - I) + (Pc - I))       # dL/dM
    dS = G * logit_scale
    return loss, _unnorm_grad(Xi, dS @ Zt), _unnorm_grad(Xt, dS.T @ Zi)


def siglip(Xi, Xt, logit_scale, bias):
    """Pairwise sigmoid loss.  Returns (loss, dXi, dXt)."""
    B = Xi.shape[0]
    Zi, Zt = l2n(Xi), l2n(Xt)
    S = Zi @ Zt.T
    Z = 2.0 * np.eye(B) - 1.0                 # +1 on the diagonal, -1 elsewhere
    A = logit_scale * S + bias
    loss = -np.log(expit(Z * A)).sum() / B
    dS = -(Z * expit(-Z * A)) * logit_scale / B
    return loss, _unnorm_grad(Xi, dS @ Zt), _unnorm_grad(Xt, dS.T @ Zi)


def fd_check(fn, Xi, Xt, rng, k=40, h=1e-5):
    """Central finite differences on k random coordinates of each input."""
    _, gi, gt = fn(Xi, Xt)
    worst = 0.0
    for side, g in ((0, gi), (1, gt)):
        for _ in range(k):
            i, j = int(rng.integers(g.shape[0])), int(rng.integers(g.shape[1]))
            A, Bm = Xi.copy(), Xt.copy()
            X = A if side == 0 else Bm
            X[i, j] += h; lp = fn(A, Bm)[0]
            X[i, j] -= 2 * h; lm = fn(A, Bm)[0]
            worst = max(worst, abs((lp - lm) / (2 * h) - g[i, j]))
    return worst


# --------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    print("=" * 74)
    print("PART 1  ViT geometry, parameters, FLOPs")
    print("=" * 74)
    ab, mb = analytic_params(BASE), analytic_macs(BASE)
    print(f"ViT-Base/16 @224: patches={ab['n_patch']}  sequence length n={ab['n']}")
    print(f"  patch embed {ab['patch_embed']:>12,}   pos embed {ab['pos']:>12,}")
    print(f"  12 blocks   {ab['blocks']:>12,}   backbone  {ab['backbone']:>12,}")
    print(f"  + 1000-way head -> {ab['backbone'] + ab['head']:,} total")
    print(f"  MACs/image  {mb['total']:,}  = {mb['total']/1e9:.2f} G"
          f"   ({2*mb['total']/1e9:.2f} GFLOPs at 2 FLOPs per MAC)")
    quad = 2 * ab["n"] ** 2 * BASE["d"]
    print(f"  attention-matmul share of a block: {quad/mb['per_block']*100:.2f}%"
          f"   (2n^2 d = 12 n d^2 at n = 6d = {6*BASE['d']} tokens)")
    assert ab["backbone"] == 85_798_656 and ab["n"] == 197
    assert ab["backbone"] + ab["head"] == 86_567_656

    # exact parameter count by construction, tiny config
    p = init_vit(TINY, RNG)
    at = analytic_params(TINY)
    built = count_params(p)
    print(f"\ntiny ViT (d=48,L=3,32px/patch8): built={built:,} "
          f"analytic={at['backbone']+at['head']:,}  diff={built-(at['backbone']+at['head'])}")
    assert built == at["backbone"] + at["head"]

    # instrumented MAC count of a real forward pass vs the closed form
    img = RNG.normal(size=(TINY["ch"], TINY["img"], TINY["img"]))
    _MACS[0] = 0
    logits, tokens = vit_forward(img, p, TINY)
    mt = analytic_macs(TINY)
    print(f"instrumented MACs={_MACS[0]:,}  analytic={mt['total']:,}  "
          f"diff={_MACS[0]-mt['total']}")
    assert _MACS[0] == mt["total"]
    print(f"tokens {tokens.shape}, logits[:4] = {logits[0, :4]}")

    # patchify == strided convolution
    if torch is not None:
        P, d, ch = TINY["patch"], TINY["d"], TINY["ch"]
        ref = torch.nn.functional.conv2d(
            torch.tensor(img)[None], torch.tensor(p["pe_w"].reshape(d, ch, P, P)),
            torch.tensor(p["pe_b"]), stride=P)[0].reshape(d, -1).T.numpy()
        ours = patchify(img, p["pe_w"], p["pe_b"], P)
        print(f"patchify vs torch conv2d(stride={P}): max|diff| = "
              f"{np.abs(ours-ref).max():.3e}")
        assert np.abs(ours - ref).max() < 1e-12
    else:
        print("[skipped: torch not installed] strided-conv cross-check")

    print("\n" + "=" * 74)
    print("PART 2  InfoNCE vs the sigmoid loss")
    print("=" * 74)
    B, D = 8, 6
    Xi = RNG.normal(size=(B, D)); Xt = RNG.normal(size=(B, D))
    tau, bias = 0.07, -10.0
    ls = 1.0 / tau
    f_nce = lambda a, b: infonce(a, b, ls)
    f_sig = lambda a, b: siglip(a, b, ls, bias)
    print(f"batch B={B}, dim={D}, tau={tau} (logit scale {ls:.3f}), sigmoid bias {bias}")
    print(f"  InfoNCE loss = {f_nce(Xi,Xt)[0]:.6f}   sigmoid loss = {f_sig(Xi,Xt)[0]:.6f}")
    e_nce = fd_check(f_nce, Xi, Xt, np.random.default_rng(1))
    e_sig = fd_check(f_sig, Xi, Xt, np.random.default_rng(1))
    print(f"  finite-difference max|error|: InfoNCE {e_nce:.3e}   sigmoid {e_sig:.3e}")
    assert e_nce < 1e-6 and e_sig < 1e-6

    # (a) chunked accumulation over blocks of pairs
    nc = 4
    Zi, Zt = l2n(Xi), l2n(Xt)
    S = Zi @ Zt.T
    A = ls * S + bias
    Zsign = 2.0 * np.eye(B) - 1.0
    part = 0.0
    for a in range(0, B, nc):
        for b in range(0, B, nc):
            sub, zs = A[a:a+nc, b:b+nc], Zsign[a:a+nc, b:b+nc]
            part += -np.log(expit(zs * sub)).sum() / B
    print(f"\n  sigmoid: full {f_sig(Xi,Xt)[0]:.12f}  chunked {part:.12f}  "
          f"|diff| = {abs(part-f_sig(Xi,Xt)[0]):.3e}")
    assert abs(part - f_sig(Xi, Xt)[0]) < 1e-12
    # the same block-local accumulation applied to InfoNCE renormalizes inside
    # each chunk, which is simply the wrong loss
    part_n = 0.0
    for a in range(0, B, nc):
        for b in range(0, B, nc):
            sub = ls * S[a:a+nc, b:b+nc]
            if a == b:
                Pr, Pc = softmax(sub, 1), softmax(sub, 0)
                part_n += -0.5*(np.log(np.diag(Pr)).sum()+np.log(np.diag(Pc)).sum())/B
    print(f"  InfoNCE: full {f_nce(Xi,Xt)[0]:.12f}  chunk-local {part_n:.12f}  "
          f"|diff| = {abs(part_n-f_nce(Xi,Xt)[0]):.3e}")

    # (b) cross-pair coupling: does moving text j change the gradient at text k?
    def cross_coupling(fn, j, k, h=1e-5):
        Xp = Xt.copy(); Xp[j] += h * np.eye(D)[0]
        Xm = Xt.copy(); Xm[j] -= h * np.eye(D)[0]
        gp = fn(Xi, Xp)[2][k]
        gm = fn(Xi, Xm)[2][k]
        return np.abs((gp - gm) / (2 * h)).max()
    c_nce = cross_coupling(f_nce, 3, 5)
    c_sig = cross_coupling(f_sig, 3, 5)
    print(f"\n  d/d(text_3) of grad w.r.t. text_5:  InfoNCE {c_nce:.6e}   "
          f"sigmoid {c_sig:.3e}")
    assert c_nce > 1e-3 and c_sig < 1e-9

    # (c) temperature controls how peaked the negative distribution is
    for t in (1.0, 0.2, 0.07, 0.01):
        Pr = softmax(S / t, axis=1)
        H = float(-(Pr * np.log(Pr + 1e-30)).sum(1).mean())
        print(f"  tau={t:<5} mean row entropy {H:.4f} nats "
              f"(uniform = {np.log(B):.4f}), max off-diag weight "
              f"{(Pr - np.eye(B)*Pr).max():.4f}")
    print("\nall assertions passed.")
