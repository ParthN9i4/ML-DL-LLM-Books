"""Artifact 41.1 -- Encrypted-generation cost model (extends Artifact 38.1's op counter).

COUNTING MODEL ONLY. No FHE library is invoked; nothing here is a measurement.
We count homomorphic operations (rotations, ct-ct multiplies, pt-ct multiplies,
fresh ciphertexts, bootstraps) for autoregressive decoding with a growing
encrypted KV cache under one explicitly stated packing.  Wall-clock enters only
through a single [Verify]-tagged ratio R_BOOT = bootstrap cost / ct-ct-mult
cost; effective precision is a parameterized model, not a measurement.
"""
import math
import numpy as np

# ---------------- model geometry (GPT-2-small-like; all stated, not measured) --
L, d, H, dk, dff = 12, 768, 12, 64, 3072   # layers, width, heads, head dim, FF
N = 1 << 16                                 # CKKS ring degree
SLOTS = N // 2                              # 32768 complex slots
G = SLOTS // d                              # cache steps packed per ciphertext (compact packing)
LIMB_BYTES = 8                              # one RNS limb coefficient = 8 bytes
ELL_CACHE = 12                              # limbs at which cache ciphertexts are held (stated)

# unit costs, in "ct-ct multiply equivalents" (keyswitch-bearing ops ~ 1)
W_ROT, W_CTMULT, W_PTMULT = 1.0, 1.0, 0.1

# depth (levels consumed) per layer per generated token -- stated component model:
# qkv proj 1, cache append masks 1, scores 1, softmax poly 6, AV 1, out proj 1,
# LN approx 2+2, MLP in 1, activation poly 4, MLP out 1  => 21
D_LAYER = 21
D_TOK = L * D_LAYER          # levels consumed per generated token (residual stream)
B_LEVELS = 18                # usable multiplicative budget between bootstraps
R_BOOT = 100.0               # [Verify] bootstrap / ct-mult cost ratio; HEonGPU's README
                             # reports ~99 ms slim bootstrapping (N=2^16, RTX 4090) vs
                             # ~1 ms-scale GPU ct-mults => order 1e2. Re-check per system.


def per_token_cost(t):
    """Op counts to decode ONE token when the cache already holds t-1 steps
    (so it holds t after the append).  Explicit loops over cache ciphertexts --
    the closed forms in the chapter are checked AGAINST this, not derived from it."""
    rot = ctm = ptm = 0.0
    m = math.ceil(t / G)                     # live cache ciphertexts per layer per K (and V)
    for _ in range(L):
        # -- cache-independent core (Halevi-Shoup/BSGS matvecs on one token vector)
        rot += 4 * 2 * math.ceil(math.sqrt(d));   ptm += 4 * d        # Wq,Wk,Wv,Wo
        rot += 2 * 2 * math.ceil(math.sqrt(dff)); ptm += 2 * dff     # MLP up/down
        ctm += H * (6 + 8)                       # softmax: exp poly + inverse poly, per head
        ctm += 6                                 # activation polynomial on FF ciphertext
        ctm += 2 * 8; rot += 2 * math.ceil(math.log2(d))  # 2 LayerNorm inv-sqrt approx
        # -- appending this step's K,V into the packed cache: rotate into slot + mask
        rot += 2; ptm += 2                       # one rotation + one mask mult, for K and V
        # -- attending over the packed cache: per cache ciphertext,
        #    scores: 1 ct-mult + log2(dk) rotations; AV: 1 ct-mult + log2(G) rotations
        for _ in range(m):
            ctm += 1; rot += math.ceil(math.log2(dk))
            ctm += 1; rot += math.ceil(math.log2(G))
    total = W_ROT * rot + W_CTMULT * ctm + W_PTMULT * ptm
    attend = L * m * (2 * W_CTMULT + W_ROT * (math.ceil(math.log2(dk)) + math.ceil(math.log2(G))))
    return dict(rot=rot, ctmult=ctm, ptmul=ptm, total=total, attend=attend, m=m)


def simulate(T):
    """Decode T tokens from an empty cache; greedy level accounting on the
    residual stream: bootstrap exactly when the budget is exhausted."""
    per_tok = np.zeros(T); attend = np.zeros(T); boots = np.zeros(T, dtype=int)
    levels, nboot = B_LEVELS, 0
    for i in range(T):
        c = per_token_cost(i + 1)
        per_tok[i], attend[i] = c["total"], c["attend"]
        b0 = nboot
        for _ in range(D_TOK):                # consume one level per depth unit
            if levels == 0:
                nboot += 1; levels = B_LEVELS  # bootstrap refreshes the budget
            levels -= 1
        boots[i] = nboot - b0
    return per_tok, attend, boots, nboot


if __name__ == "__main__":
    T = 1024
    per_tok, attend, boots, nboot = simulate(T)
    base = per_tok - attend                    # cache-independent share

    print("=== Artifact 41.1: encrypted-generation cost model (counting model) ===")
    print(f"geometry: L={L} d={d} H={H} dk={dk} | slots={SLOTS} G={G} steps/ct "
          f"| depth/token={D_TOK} budget={B_LEVELS} R_BOOT={R_BOOT} [Verify]")

    # ---- (1) per-token cost grows with cache length under this packing --------
    print("\n[1] per-token cost curve (ct-mult equivalents):")
    for t in [1, 64, 128, 256, 512, 1024]:
        print(f"    t={t:5d}  total={per_tok[t-1]:9.1f}  attend={attend[t-1]:7.1f} "
              f"({100*attend[t-1]/per_tok[t-1]:4.1f}%)  cache cts m={math.ceil(t/G)}")
    diffs = np.diff(per_tok)
    assert (diffs >= -1e-9).all() and per_tok[-1] > per_tok[0], "cost must grow with t"
    # cross-check: attend cost is exactly linear in m; fit and print the residual
    ms = np.array([math.ceil(t / G) for t in range(1, T + 1)], float)
    slope, icept = np.polyfit(ms, attend, 1)
    resid = float(np.abs(np.polyval([slope, icept], ms) - attend).max())
    b_analytic = L * (2 + math.ceil(math.log2(dk)) + math.ceil(math.log2(G)))
    print(f"    fitted slope/cache-ct = {slope:.6f} vs analytic b = {b_analytic} "
          f"(max fit residual {resid:.2e})")
    assert abs(slope - b_analytic) < 1e-6 and resid < 1e-6

    # ---- (2) bootstraps match the analytic ceil accounting --------------------
    total_depth = T * D_TOK
    analytic = math.ceil(total_depth / B_LEVELS) - 1   # start fresh; none after last op
    nb_tok = D_TOK / B_LEVELS
    print(f"\n[2] bootstraps over T={T} tokens: simulated={nboot}  "
          f"analytic ceil({total_depth}/{B_LEVELS})-1={analytic}  "
          f"(={nb_tok:.1f}/token; tokens/bootstrap={1/nb_tok:.4f})")
    assert nboot == analytic, (nboot, analytic)
    # steady state is exactly D_TOK/B per token (B | D_TOK); token 1 gets the
    # fresh budget for free and needs one fewer
    assert (boots[1:] == int(nb_tok)).all() and boots[0] == int(nb_tok) - 1

    # ---- (3) crossover: growing attention cost overtakes bootstrap cost -------
    boot_cost = nb_tok * R_BOOT                 # per-token, constant in t (depth is)
    scan = next(t for t in range(1, T + 1) if attend[t - 1] > boot_cost)
    m_star = math.floor(boot_cost / b_analytic) + 1
    closed = (m_star - 1) * G + 1
    print(f"\n[3] per-token bootstrap cost = {boot_cost:.0f} mult-equivs; "
          f"attention overtakes it at t={scan} (closed form: m*={m_star}, t*={closed})")
    assert scan == closed, (scan, closed)
    print(f"    below t={closed}: bootstrapping outweighs cache attention; above: cache wins")
    print(f"    bootstrap share of total modeled time at t=1: "
          f"{100*boot_cost/(per_tok[0]+boot_cost):.1f}%  at t={T}: "
          f"{100*boot_cost/(per_tok[-1]+boot_cost):.1f}%")

    # ---- (4) encrypted KV-cache memory at t=1024 (both packings) --------------
    ct_mb = 2 * N * ELL_CACHE * LIMB_BYTES / 2**20
    m = math.ceil(T / G)
    compact_cts, naive_cts = 2 * L * m, 2 * L * T
    plain_mb = 2 * L * T * d * 2 / 2**20       # fp16 plaintext KV cache
    print(f"\n[4] KV cache at t={T} ({ELL_CACHE} limbs, {ct_mb:.1f} MB/ct):")
    print(f"    compact packing : {compact_cts:6d} cts = {compact_cts*ct_mb/1024:8.2f} GB "
          f"({compact_cts*ct_mb/plain_mb:7.0f}x fp16)")
    print(f"    one-ct-per-step : {naive_cts:6d} cts = {naive_cts*ct_mb/1024:8.2f} GB "
          f"({naive_cts*ct_mb/plain_mb:7.0f}x fp16)   [fp16 plaintext: {plain_mb:.1f} MB]")

    # ---- (5) CKKS effective precision vs fp16 (parameterized model, NOT measured)
    P_FRESH, LOSS, P_BOOT, FP16 = 30.0, 0.5, 20.0, 11  # bits; P_BOOT is [Verify]
    prec = P_BOOT                                       # steady state: post-bootstrap
    seg = [prec := prec - LOSS for _ in range(B_LEVELS)]  # one inter-bootstrap segment
    floor_sim, floor_closed = min(seg), P_BOOT - LOSS * B_LEVELS
    b_max = math.floor((P_BOOT - FP16) / LOSS)
    print(f"\n[5] precision model: fresh {P_FRESH} bits, {LOSS} bit/level loss, "
          f"bootstrap output {P_BOOT} bits [Verify]")
    print(f"    worst-case bits in a {B_LEVELS}-level segment: simulated={floor_sim:.1f} "
          f"closed form={floor_closed:.1f}  | fp16 significand = {FP16} bits")
    print(f"    largest budget keeping >= fp16 precision: {b_max} levels "
          f"(chosen B_LEVELS={B_LEVELS})")
    assert abs(floor_sim - floor_closed) < 1e-12
    assert floor_closed >= FP16 - 1e-9, "budget exceeds the fp16-parity envelope"
    print("\nAll assertions passed. Counting model only -- no FHE library was run.")
