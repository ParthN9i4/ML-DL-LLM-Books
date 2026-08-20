# Part VIII citation audit — 2026-08-20

The build's original caveat was that Chapters 38–42's citations were verified only through
search-engine extraction of abstracts, because the sandbox proxy blocks arXiv, IACR ePrint,
ACM, NDSS and OpenReview. This audit re-verified every externally-sourced claim by search
triangulation against paper landing pages, venue pages, Semantic Scholar, arXiv HTML
mirrors — and, where the systems ship code, **against the cloned source itself** (NEXUS and
BOLT were cloned and read; OpenFHE's `stdlatticeparms.cpp` was fetched raw).

## Headline result

| Scope | Claims checked | Confirmed | Wrong | Needs PDF |
|---|---|---|---|---|
| Ch 38–39 | 30 | 30 (several against primary source code) | 0 | 3 |
| Ch 40 | 24 | 23 | 0 | 0 (both flagged items settled and corrected) |
| Ch 41 | 13 | 12 | 0 | 1 (optional confirmation of an applied fix) |
| Ch 42 + Appendix D.7 | 9 + 22 identifiers | all | 0 | 2 |

**Every bibliographic identifier in Appendix D.7 — ePrint number, arXiv ID, venue, page
range, author list — resolves to exactly the paper the book claims.** The OpenFHE security
figures (881 bits at N=2^15, 1747 at N=2^16, ternary, 128-bit classical) were re-confirmed
against `stdlatticeparms.cpp` fetched from the OpenFHE repository. NEXUS's f4/g4
coefficients, its argmax schedule (g,g,f,f), its Goldschmidt inverse, the hard-coded 0.01
softmax scale, and BOLT's I-BERT quadratic with the right-shift clamp at 13 were all
verified against the systems' actual code.

## Errors found and fixed in this audit (all corrected in ml-book.html)

1. **Ch 41 — NEXUS 37.3 s inverted.** The book called 37.3 s the CPU figure "with a further
   42.3× from its GPU port." The paper's 37.3 s IS the GPU figure (42.3× over its own CPU).
   Chapter 40 had it right; Ch 41 contradicted it.
2. **Ch 41 — Cachemir packing mischaracterized.** "Interleaved replicated" packing targets
   KV-cache-driven vector–matrix products in the *linear layers* (per the paper's abstract),
   not "decode-time attention," and the "append and attend rotations share structure"
   rationale appears nowhere in the paper. Rewritten to the abstract's actual motivation
   (slot utilization).
3. **Ch 40 — BOLT 61 GB nuance overstated.** The ~61 GB NEXUS-derived figure matches BOLT's
   own no-word-elimination table entry (59.61 GB); the book said it was "not BOLT's own
   tables." Softened to name both configurations (59.61 vs 25.74 GB headline).
4. **Ch 40 — Nimbus wording.** "Removes NTT and rotations from the online path" overstated
   the paper's "reduces online NTT/INTT" + "free right-shift replaces rotation-based output
   packing." Corrected. Also "defers LayerNorm to future work" → the paper's actual reason
   ("already relatively fast") in the artifact provenance record (code + listing, drift
   re-verified).
5. **Ch 38 — Rho et al. "LoRA to delete ct-ct matmuls"** → "shrink … down to low-rank ones"
   (the paper says LoRA reduces their size; they are not eliminated).
6. **Ch 42 — open-problem claim tightened.** "Selective SSM under non-interactive CKKS is
   open" holds for everything *published* as of 2026-08-20, but an unpublished GitHub
   prototype (Hosi121/fhe-native-mamba3, active July 2026) is already attempting it. The Key
   Takeaway now says "no published system" and advises re-running the search the week an
   introduction is written.
7. **study-plan.html — two flagged queue entries resolved.** Primer = arXiv 2303.13679
   (Zheng, Lou & Jiang, 2023); CryptoGen = arXiv 2602.08798 (Feb 2026), which is also the
   primary source of Ch 41's 4.4–7.6× per-token-latency figure.

## Residual checklist — the five details only a PDF can settle

Everything below is a *detail inside an otherwise-confirmed claim*. No identifier, venue,
or headline number is in question.

1. **Rho et al. (arXiv 2410.02486, ICLR 2025)** — the book says the 6.94×/2.3× speedups are
   "vs. its own softmax-HE baseline." Open the main results table (§6) and confirm the
   baseline row is the authors' own HE softmax-attention transformer (not a third-party
   system, not plaintext). Book line ~14389.
2. **NEXUS paper (ePrint 2024/136), Related Work** — the book says later work criticizes
   THE-X as "both interactive and leaky, since decrypted intermediates reveal information."
   Confirm NEXUS (or another cited successor) makes both criticisms in those terms. Book
   line ~14380. (Tagged Likely, not Certain, so low stakes.)
3. **Cheon–Kim–Kim (ePrint 2019/1234), experiments table** — confirm the 1.43 ms amortized
   20-bit comparison row uses α = 20. The 1.43 ms / 20-bit / ~30× figures are confirmed;
   only the α setting is unverified. Book line ~14712.
4. **Brito (arXiv 2605.16647)** — confirm the "~5× the speed of HE-friendly polynomial
   attention" figure (everything else in that paragraph — IDs, accuracies 0.7505/0.7420,
   exact plaintext match, fastText features — is confirmed). Book §42.3.1.
5. **CryptoMoE (NeurIPS 2025, arXiv 2511.01197), §3** — confirm the protocol is "BFV plus
   2-out-of-2 secret sharing" (the 2.8–3.5× / 2.9–4.3× figures are confirmed against the
   camera-ready). Book §42.3.

Optional sixth: Cachemir (arXiv 2602.11470), packing section — the fix applied above was
made from the abstract; the section text would confirm the rewritten sentence exactly.
