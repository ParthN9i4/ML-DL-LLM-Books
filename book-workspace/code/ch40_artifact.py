"""Artifact 40.1 -- A system-comparison database for private transformer inference.

Data-as-code: each published system is a record whose every field is either sourced
(with a provenance string naming the paper section/table or abstract) or the literal
string "n/r" (not reported / not verified by this author).  The HARD ASSERTIONS at the
bottom enforce sourcing discipline, not numerics: no record may claim to be
non-interactive while also charging per-layer client rounds; no latency claim may
appear without an explicit exclusions note; every record must carry venue+provenance.
Latency strings are the papers' own claims, quoted -- they are NOT comparable across
rows, and the second printed table shows why (different models, sequence lengths,
hardware, threat models).  No FHE library is used or needed: this is a reading tool.
"""
from dataclasses import dataclass, asdict

NR = "n/r"  # the only permitted marker for an unsourced field

@dataclass(frozen=True)
class System:
    name: str
    family: str          # "2PC+HE (interactive)" | "FHE (non-interactive)" | "TEE+FHE hybrid"
    interactive: bool
    client_rounds: str   # "per-protocol rounds" vs "one round" vs TEE-mediated
    threat_model: str
    scheme: str
    softmax: str
    norm: str
    activation: str
    model: str
    seq_len: str
    hardware: str
    latency: str         # the paper's own claim, quoted; NR if none verified
    comm: str
    exclusions: str      # what the latency claim does not cover; NR only if latency is NR
    venue: str
    provenance: str      # where each nontrivial field above was read

DB = [
    System("Iron", "2PC+HE (interactive)", True, "per-protocol rounds throughout inference",
        "semi-honest client+server 2PC; client learns architecture",
        "RLWE HE (compact packing) for matmul + secret sharing/OT for non-linear",
        "custom 2PC softmax protocol", "custom 2PC LayerNorm protocol",
        "custom 2PC GELU protocol (~2x cheaper than prior general protocols)",
        "BERT-family (basis of later head-to-head comparisons, e.g. BOLT)", NR, "CPU",
        NR, NR, NR, "NeurIPS 2022",
        "Iron abstract (Hao,Li,Chen,Xing,Xu,Zhang); packing gives sqrt(m)-less comm"),
    System("BOLT", "2PC+HE (interactive)", True, "per-protocol rounds throughout inference",
        "semi-honest client+server 2PC",
        "RLWE HE for matmul + secret sharing/OT for non-linear",
        "piecewise-polynomial 2PC protocol", "2PC LayerNorm protocol",
        "piecewise-polynomial GELU; ML-side 'word elimination' prunes tokens",
        "BERT-base", "128", "CPU, LAN/WAN network settings",
        "paper claims 4.8-9.5x faster than Iron across network settings",
        "~10.91x less than Iron (own claim); ~61 GB/inference per NEXUS's cross-table",
        "relative to Iron only; absolute seconds vary with emulated network; "
        "cross-paper absolute GB comes from NEXUS's comparison, not BOLT's table",
        "IEEE S&P 2024 (ePrint 2023/1893)",
        "BOLT abstract (Pang,Zhu,Moellering,Zheng,Schneider); NEXUS NDSS'25 comparison"),
    System("BumbleBee", "2PC+HE (interactive)", True, "per-protocol rounds throughout inference",
        "semi-honest client+server 2PC (SPU/SecretFlow stack)",
        "RLWE HE with ciphertext compression for matmul + OT-based non-linear",
        "segmented low-degree polynomial protocol (exp/div)",
        "2PC normalization protocol",
        "GELU/SiLU protocols; 80-95% comm cut vs two prior methods (own claim)",
        "BERT-base/large, GPT2-base, ViT-base, LLaMA-7B", "128 (GPT2); 8 (LLaMA-7B)",
        "CPUs", "paper reports ~8 min to generate 1 token of LLaMA-7B on CPUs; "
        "3x faster than BOLT at 1/10 the communication; >10x faster than Iron",
        "matmul comm cut 80-90% vs prior (own claim)",
        "LLaMA-7B figure is for 8 input tokens generating 1 token; "
        "network setting and setup cost conventions per paper Table IV",
        "NDSS 2025 (ePrint 2023/1678)",
        "BumbleBee abstract + eval (Lu,Huang,Gu,Li,Liu,Hong,Ren,Wei,Chen); Table IV"),
    System("Nimbus", "2PC+HE (interactive)", True, "per-protocol rounds throughout inference",
        "semi-honest client+server 2PC",
        "RLWE HE, outer-product/row-wise encoding for linear layers + SS for non-linear",
        "low-degree polynomial approx + model adaptation",
        "handled within adapted-model protocols",
        "low-degree polynomial GELU + fine-tuning to recover accuracy",
        "BERT-base", NR, "LAN 3Gbps/1ms and WAN 400Mbps/10ms (as retrieved)",
        "paper reports 2.7x-4.7x end-to-end BERT-base speedup vs prior SOTA 2PC",
        NR, "relative speedups only; split of gains between linear-layer paradigm "
        "and polynomial approximations is per-component in the paper",
        "NeurIPS 2024 (arXiv 2411.15707)",
        "Nimbus abstract (Li et al.); NeurIPS 2024 proceedings page"),
    System("SHAFT", "2PC (secret sharing, interactive)", True,
        "constant-round protocols, but still interactive per operator",
        "semi-honest 2PC; CrypTen/PyTorch integration ('handy')",
        "additive secret sharing (CrypTen backend); no HE in the online path",
        "first constant-round softmax: input clipping + ODE-derived iteration",
        "CrypTen-style normalization protocols",
        "Fourier-series GELU characterization",
        "BERT-base/large, GPT-2, ViT-base", NR, NR,
        "paper reports 4.6-5.3x faster than BumbleBee (LAN), 2.9-4.4x (WAN); "
        "matches SIGMA runtime with 25-41% less communication",
        "25-41% less than SIGMA/BumbleBee (own claim)",
        "relative numbers; note SHAFT is NOT non-interactive despite NDSS'25 cohort",
        "NDSS 2025, Distinguished Artifact Award (ePrint 2025/2324)",
        "SHAFT abstract (Kei, Chow); NDSS'25 paper page; Zenodo artifact"),
    System("NEXUS", "FHE (non-interactive)", False, "one round: send ct, receive ct",
        "semi-honest server; client offline during inference; weights plaintext server-side",
        "RNS-CKKS with bootstrapping (SEAL+FHE-MP-CNN; HEXL CPU, Phantom GPU)",
        "polynomial/iterative approx under CKKS; secure Argmax via sign polynomials",
        "LayerNorm approximated under CKKS",
        "GELU approximated under CKKS; SIMD compression/slot folding",
        "BERT-based model", NR, "GPU (42.3x over its CPU version, own claim)",
        "paper reports 37.3 s per BERT-based inference (GPU) at 164 MB bandwidth",
        "164 MB; 372.5x less than BOLT, 53.6x less than BumbleBee (own claims)",
        "GPU figure; sequence length and amortization conventions live in the "
        "evaluation section -- verify before quoting as single-input CPU latency",
        "NDSS 2025 (ePrint 2024/136)",
        "NEXUS abstract (Zhang,Liu et al.); zju-abclab/NEXUS README"),
    System("THOR", "FHE (non-interactive)", False, "one round: send ct, receive ct",
        "semi-honest server; client offline during inference",
        "CKKS; diagonal-major encoding, compact packing; PC-MM + BSGS CC-MM",
        "approximation + adaptive iterative methods",
        "LayerNorm via approximation + adaptive iteration",
        "GELU/Tanh via approximation + adaptive iteration",
        "BERT-base", "128", "single GPU",
        "paper reports BERT-base/128-token secure inference in ~10 minutes on one GPU",
        NR, "single-input latency; PC-MM speedup 5.3x vs BOLT and CC-MM 9.7x vs "
        "Powerformer are kernel-level, not end-to-end",
        "ACM CCS 2025 (ePrint 2024/1881)",
        "THOR abstract (Moon,Yoo,Jiang,Kim); ACM DL page"),
    System("Powerformer", "FHE (non-interactive)", False, "one round: send ct, receive ct",
        "semi-honest server; client offline during inference",
        "CKKS; Transformer-optimized homomorphic matmul",
        "REPLACED: Batch Rectifier-Power max via distillation (no exp, no max)",
        "REPLACED: LayerNorm distilled into linear function",
        "pseudo-sign composite approximation for GELU/tanh",
        "BERT-base", NR, NR,
        "paper reports 45% computation-time reduction vs the SOTA HE-based "
        "private language model, at no accuracy loss",
        NR, "relative to a single HE baseline; hardware not stated in abstract",
        "ACL 2025 long paper, pp. 11090-11111 (ePrint 2024/1429)",
        "Powerformer abstract (Park,Lee,Lee); ACL Anthology 2025.acl-long.543"),
    System("ATLAS", "FHE (non-interactive) automation layer", False,
        "one round (inherits CKKS pipeline it configures)",
        "semi-honest server (inherited); ATLAS itself is an offline search",
        "CKKS (configures approximation hyperparameters per layer)",
        "per-layer approx budgets found by multi-objective search",
        "per-layer approx budgets found by multi-objective search",
        "per-layer approx budgets found by multi-objective search",
        "BERT, LLaMA3-8B, ViT (120 or 320 decision variables)", NR,
        "search runs in cleartext; 70-1000 s per candidate config (own claim)",
        NR, NR, NR,
        "arXiv 2607.23478 (preprint, 2026)",
        "ATLAS abstract/HTML (Xie,Tan,Boddeti,Lu); 'in one hour' = search budget"),
    System("Bifrost", "TEE+FHE hybrid", False,
        "client talks to attested CPU TEE; no per-layer client rounds",
        "trusts attested CPU TEE; GPU, device memory, driver/host all untrusted",
        "accelerator-backed CKKS for projection/FFN linear layers; secrets only in TEE",
        "computed in cleartext inside CPU TEE",
        "computed in cleartext inside CPU TEE",
        "computed in cleartext inside CPU TEE; TEE decrypt-recrypt refresh "
        "replaces bootstrapping",
        "GPT-2 (124M/1.5B), LLaMA-3 8B, Qwen3 0.6B", NR, "CPU TEE + GPU (untrusted)",
        "paper reports PROJECTED 9.25x (GPT-2 1.5B) / 9.91x (LLaMA-3 8B) latency "
        "cuts; Bifrost+ TTFT 14.6-45.8x (GPT-2 124M), 15.3-53.4x (Qwen3 0.6B) vs "
        "direct CKKS",
        NR, "PROJECTED from a cost model, not end-to-end wall clock; baselines are "
        "direct-CKKS deployments", "arXiv 2606.17421 (preprint, 2026)",
        "Bifrost abstract (SJTU); arXiv listing June 2026"),
]

# ---------------- HARD ASSERTIONS: sourcing discipline ----------------
def check(db):
    names = {s.name for s in db}
    assert len(db) >= 7 and {"BOLT", "BumbleBee", "NEXUS", "SHAFT"} <= names, \
        "coverage: need >=7 systems incl. BOLT/BumbleBee/NEXUS/SHAFT"
    for s in db:
        # 1. non-interactive claims must not carry per-layer client rounds
        assert not (not s.interactive and "per-protocol" in s.client_rounds), \
            f"{s.name}: non-interactive yet charges per-protocol client rounds"
        assert not (("non-interactive" in s.family) and s.interactive), \
            f"{s.name}: family/interactive flag contradiction"
        # 2. any latency claim must state exclusions (even 'none stated')
        if s.latency != NR:
            assert s.exclusions not in ("", NR), f"{s.name}: latency without exclusions"
        # 3. every record is sourced
        assert s.venue not in ("", NR) and s.provenance not in ("", NR), \
            f"{s.name}: missing venue/provenance"
        # 4. no empty fields: every field is a claim or an explicit n/r
        for k, v in asdict(s).items():
            assert v != "" and v is not None, f"{s.name}.{k}: empty field"
    return len(db)

def row(cols, widths):
    return " | ".join(str(c)[:w].ljust(w) for c, w in zip(cols, widths))

if __name__ == "__main__":
    n = check(DB)
    fams = []
    for s in DB:
        if s.family not in fams:
            fams.append(s.family)
    print(f"ALL ASSERTIONS PASSED: {n} systems, {len(fams)} protocol families, "
          f"{sum(1 for s in DB if s.latency != NR)} latency claims each with an "
          f"exclusions note, {sum(asdict(s)[k] == NR for s in DB for k in asdict(s))} "
          f"fields honestly marked n/r\n")
    W1 = (11, 34, 38, 40)
    print("TABLE 1 -- strategies, grouped by protocol family")
    print(row(("system", "softmax strategy", "activation strategy", "venue"), W1))
    print("-" * (sum(W1) + 9))
    for f in fams:
        print(f"[{f}]")
        for s in DB:
            if s.family == f:
                print(row((s.name, s.softmax, s.activation, s.venue), W1))
    W2 = (11, 26, 12, 24, 48)
    print("\nTABLE 2 -- why the latency column is NOT comparable")
    print(row(("system", "benchmark model", "seq len", "hardware", "reported latency"), W2))
    print("-" * (sum(W2) + 12))
    for s in DB:
        if s.latency != NR:
            print(row((s.name, s.model, s.seq_len, s.hardware, s.latency), W2))
    print("\nEvery row above measures a DIFFERENT model, sequence length, hardware "
          "platform,\nand threat model, and each excludes different setup costs; "
          "the column must never\nbe read as a ranking. Consistency spot-check from "
          "the papers' own claims:")
    bolt_over_bb_via_nexus = 372.5 / 53.6   # NEXUS: bandwidth vs BOLT / vs BumbleBee
    print(f"  NEXUS implies BOLT uses {bolt_over_bb_via_nexus:.2f}x BumbleBee's "
          f"bandwidth; BumbleBee itself claims 10x --")
    print("  both are 'true' because they measured different configurations. QED.")
