"""
ex08 — Initialization, normalization, and signal propagation.  (Book: Chapter 12)

Initialization and normalization solve the SAME problem: keeping the variance
of activations (forward) and gradients (backward) at order one through depth.
Solve it and a 50-layer network trains; fail and the signal is gone by layer 20
— not "harder to train", gone, below float precision.

The arithmetic, which you can derive on one line: a linear layer y = Wx with
W_ij ~ N(0, s^2), fan_in inputs, gives Var(y_i) = fan_in * s^2 * Var(x_i).
Repeat over L layers and the variance is (fan_in * s^2)^L: anything but
fan_in * s^2 = 1 is exponential in depth.

  * Xavier:  s^2 = 1/fan_in   (preserves variance through the LINEAR map)
  * He:      s^2 = 2/fan_in   (the extra 2 compensates ReLU zeroing half)

ReLU kills the mean too, which is why the factor is exactly 2: for x ~ N(0, v),
E[relu(x)^2] = v/2.

Normalization attacks the same problem at run time instead of init time:
LayerNorm re-centers and re-scales every layer, so the variance is pinned to 1
regardless of what the weights did. RMSNorm drops the re-centering and keeps
only the re-scaling — cheaper, and empirically just as good.

To learn: replace each function body with `pass` and reimplement.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check import check, summary  # noqa: E402


def init_weights(fan_in, fan_out, scheme, rng):
    """naive: N(0,1). xavier: N(0, 1/fan_in). he: N(0, 2/fan_in)."""
    # === YOUR CODE HERE ===
    scale = {"naive": 1.0, "xavier": 1.0 / fan_in, "he": 2.0 / fan_in}[scheme]
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(scale)


def layernorm(x, eps=1e-5):
    """(x - mean) / sqrt(var + eps), statistics over the LAST axis."""
    # === YOUR CODE HERE ===
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def rmsnorm(x, eps=1e-5):
    """x / sqrt(mean(x^2) + eps). No mean subtraction — that is the whole diff."""
    # === YOUR CODE HERE ===
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)


def forward_stats(depth, width, scheme, norm, rng, batch=256):
    """Push a batch through `depth` ReLU layers; record activation std per layer."""
    x = rng.standard_normal((batch, width))
    stds = []
    for _ in range(depth):
        W = init_weights(width, width, scheme, rng)
        x = x @ W
        if norm == "layernorm":
            x = layernorm(x)
        elif norm == "rmsnorm":
            x = rmsnorm(x)
        x = np.maximum(x, 0.0)
        stds.append(float(x.std()))
    return stds


def backward_gradient_norm(depth, width, scheme, rng, batch=64):
    """Gradient norm at the INPUT of a deep ReLU stack, by explicit backprop.

    Forward saves the masks; backward applies g <- (g W^T) * mask. The returned
    ratio ||g_input|| / ||g_output|| is the number the whole chapter is about.
    """
    Ws, masks = [], []
    x = rng.standard_normal((batch, width))
    for _ in range(depth):
        W = init_weights(width, width, scheme, rng)
        z = x @ W
        mask = (z > 0).astype(np.float64)
        x = z * mask
        Ws.append(W)
        masks.append(mask)
    g = rng.standard_normal((batch, width))
    g_out_norm = float(np.linalg.norm(g))
    # === YOUR CODE HERE ===
    for W, mask in zip(reversed(Ws), reversed(masks)):
        g = (g * mask) @ W.T
    return float(np.linalg.norm(g)) / g_out_norm


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("\n--- the one-layer arithmetic, verified ---")
    width = 512
    x = rng.standard_normal((4096, width))
    # Linear map with Xavier: output variance == input variance.
    W = init_weights(width, width, "xavier", rng)
    check("xavier preserves variance through a linear map",
          float((x @ W).var()), float(x.var()), tol=0.05)
    # ReLU halves the second moment of a centered Gaussian: E[relu(x)^2] = v/2.
    z = rng.standard_normal(1_000_000) * 3.0
    check("E[relu(x)^2] = Var(x)/2 for centered Gaussian x",
          float(np.maximum(z, 0) ** 2).__float__() if False else float((np.maximum(z, 0) ** 2).mean()),
          9.0 / 2, tol=0.05)
    # He restores it — but be precise about WHAT is preserved. The derivation
    # controls the SECOND MOMENT: E[relu(y)^2] = Var(y)/2 = E[x^2]. The
    # variance of the relu output is a different number, (1 - 1/pi) * E[x^2],
    # because relu output has nonzero mean. Asserting "variance is preserved"
    # here is a category error — and a first draft of this exercise made it.
    W_he = init_weights(width, width, "he", rng)
    relu_out = np.maximum(x @ W_he, 0)
    check("he init + relu preserves the SECOND MOMENT",
          float((relu_out ** 2).mean()), float((x ** 2).mean()), tol=0.08)
    check("while the relu-output VARIANCE is (1 - 1/pi) of it",
          float(relu_out.var()), (1 - 1 / np.pi) * float((x ** 2).mean()), tol=0.08)

    print("\n--- fifty layers forward ---")
    depth = 50
    for scheme, expect in (("naive", "explodes"), ("xavier", "decays"), ("he", "holds")):
        stds = forward_stats(depth, 256, scheme, None, rng)
        print(f"      {scheme:7s} layer-1 std {stds[0]:9.3e}   layer-50 std {stds[-1]:9.3e}")
    naive_stds = forward_stats(depth, 256, "naive", None, rng)
    xavier_stds = forward_stats(depth, 256, "xavier", None, rng)
    he_stds = forward_stats(depth, 256, "he", None, rng)
    # naive: fan_in * 1 = 256 per layer -> astronomically large by layer 50.
    check("naive init explodes by many orders of magnitude",
          naive_stds[-1] > 1e30 * naive_stds[0] or not np.isfinite(naive_stds[-1]))
    # xavier under ReLU: each layer multiplies the second moment by 1/2.
    check("xavier + relu decays (the missing factor of 2 compounds)",
          xavier_stds[-1] < 1e-3 * xavier_stds[0])
    # he: flat. Within a factor of ~3 over 50 layers.
    check("he init holds activation scale across 50 layers",
          0.3 < he_stds[-1] / he_stds[0] < 3.0)

    print("\n--- fifty layers backward ---")
    ratios = {}
    for scheme in ("naive", "xavier", "he"):
        with np.errstate(over="ignore", invalid="ignore"):
            ratios[scheme] = backward_gradient_norm(depth, 256, scheme, rng)
        print(f"      {scheme:7s} ||g_in||/||g_out|| = {ratios[scheme]:9.3e}")
    check("he keeps the gradient within two orders of magnitude over 50 layers",
          1e-2 < ratios["he"] < 1e2)
    check("xavier loses the gradient under relu",
          ratios["xavier"] < 1e-4)
    check("naive gradient is astronomically large or non-finite",
          (not np.isfinite(ratios["naive"])) or ratios["naive"] > 1e20)

    print("\n--- normalization rescues even bad init ---")
    # With LayerNorm after every linear layer, even naive N(0,1) init keeps
    # activations at scale 1 — the run-time fix for the init-time problem.
    for norm in ("layernorm", "rmsnorm"):
        stds = forward_stats(depth, 256, "naive", norm, rng)
        print(f"      naive init + {norm:9s}: layer-50 std = {stds[-1]:.4f}")
        check(f"{norm} pins the scale under naive init",
              0.2 < stds[-1] < 2.0)

    print("\n--- layernorm vs rmsnorm ---")
    x = rng.standard_normal((128, 64)) * 5 + 3          # scale AND shift
    ln, rn = layernorm(x), rmsnorm(x)
    check("layernorm output has mean ~0", float(np.abs(ln.mean(axis=-1)).max()), 0.0, tol=1e-10)
    check("layernorm output has std ~1", float(ln.std(axis=-1).mean()), 1.0, tol=1e-3)
    check("rmsnorm does NOT remove the mean", float(np.abs(rn.mean(axis=-1)).max()) > 0.1)
    check("rmsnorm output has unit rms", float(np.sqrt((rn ** 2).mean(axis=-1)).mean()), 1.0, tol=1e-3)
    # On CENTERED inputs they coincide — which is why dropping the mean is safe
    # deep in a residual network, where the stream is approximately centered.
    xc = x - x.mean(axis=-1, keepdims=True)
    check("on centered inputs layernorm == rmsnorm", layernorm(xc), rmsnorm(xc), tol=1e-6)

    try:
        import torch
        xt = torch.tensor(x)
        ref_ln = torch.nn.functional.layer_norm(xt, (64,)).numpy()
        ref_rn = torch.nn.functional.rms_norm(xt, (64,)).numpy()
        check("layernorm matches torch", layernorm(x), ref_ln, tol=1e-5)
        check("rmsnorm matches torch", rmsnorm(x), ref_rn, tol=1e-5)
    except (ImportError, AttributeError):
        print("  ....  [skipped] torch layer_norm/rms_norm cross-check")

    # -----------------------------------------------------------------------
    # BREAK IT
    # -----------------------------------------------------------------------
    print("\n--- break it ---")

    # (a) The eps placement bug: sqrt(var + eps) is not sqrt(var) + eps. For
    #     near-constant features (variance ~ 0) the wrong placement divides by
    #     eps^1 instead of eps^0.5 and the output explodes by 1/sqrt(eps).
    x_flat = np.full((4, 64), 7.0) + 1e-9 * rng.standard_normal((4, 64))
    good = (x_flat - x_flat.mean(-1, keepdims=True)) / np.sqrt(x_flat.var(-1, keepdims=True) + 1e-5)
    bad = (x_flat - x_flat.mean(-1, keepdims=True)) / (np.sqrt(x_flat.var(-1, keepdims=True)) + 1e-5)
    print(f"      near-constant input: |out| with eps inside sqrt {np.abs(good).max():.2e}, "
          f"outside {np.abs(bad).max():.2e}  (ratio {np.abs(bad).max()/np.abs(good).max():.0f}x)")
    check("eps inside the sqrt keeps the output small", float(np.abs(good).max()) < 1.0)
    check("eps outside the sqrt amplifies noise ~300x",
          float(np.abs(bad).max()) > 30 * float(np.abs(good).max()))

    # (b) fan_in versus fan_out is NOT a typo-level bug — they preserve two
    #     different things, and on rectangular layers you cannot have both:
    #       fan_in  keeps the FORWARD second moment  (x @ W scale)
    #       fan_out keeps the BACKWARD second moment (g @ W.T scale)
    #     On square layers they coincide, which is why the confusion survives.
    #     A first draft of this demo used ALTERNATING widths (512-256-512-...)
    #     and showed no drift at all — the fo/fi factors cancel in pairs. The
    #     drift needs a MONOTONE funnel, where every layer narrows and the
    #     factors compound instead of cancelling.
    widths = [1024, 512, 256, 128, 64, 32, 16, 8]
    x0 = rng.standard_normal((512, widths[0]))
    correct, wrong = x0.copy(), x0.copy()
    for fi, fo in zip(widths[:-1], widths[1:]):
        Wc = rng.standard_normal((fi, fo)) * np.sqrt(2.0 / fi)   # preserves forward
        Ww = rng.standard_normal((fi, fo)) * np.sqrt(2.0 / fo)   # preserves backward
        correct = np.maximum(correct @ Wc, 0)
        wrong = np.maximum(wrong @ Ww, 0)
    # Each funnel layer has fi/fo = 2, so fan_out inflates the forward second
    # moment by 2 per layer: 2^7 = 128 in variance, ~11x in std.
    print(f"      7-layer halving funnel: fan_in std {correct.std():.3f}, "
          f"fan_out std {wrong.std():.3f}  (theory predicts ~{2**3.5:.0f}x)")
    check("fan_in keeps forward scale down the funnel", 0.2 < float(correct.std()) < 5.0)
    check("fan_out inflates the forward signal ~10x down the same funnel",
          float(wrong.std()) > 5 * float(correct.std()))

    summary()
