# Foundations of Modern Machine Learning

A rigorous, from-scratch book on machine learning — classical ML, deep learning,
Transformers, Vision Transformers, State Space Models, and large language models — ending
in a research-level treatment of machine learning under homomorphic encryption. Written as
a companion to an earlier book on FHE from first principles, and built the same way: every
chapter's central claim is backed by a runnable artifact that verifies itself.

## What's here

| Path | What it is |
|---|---|
| [`ml-book.html`](ml-book.html) | **The book.** One self-contained HTML file — open it in a browser. 42 chapters across 8 parts plus appendices. MathJax renders the math (needs network access to cdn.jsdelivr.net, or a local MathJax install; see Appendix B). |
| [`ml-foundations/`](ml-foundations/) | **The exercise ladder.** 30 standalone runnable Python files, tiers 0–6, from pure-numpy matrix arithmetic up to an encrypted linear layer. `python3 check.py` runs everything; the assertions in each file *are* the specification. |
| [`study-plan.html`](study-plan.html) | **A 42-week study schedule** pacing the book at 10–15 h/week, with a parallel weekly paper-analysis track on homomorphic encryption for LLMs — a fixed nine-point extraction template and a seeded 20-paper queue. |
| `book-workspace/` | Build scaffolding: the frozen chapter contract, the assembler/verifier, and every executed artifact. Not part of the book; kept so the build is reproducible. Deletable once the book is final. |

## The book, briefly

- **Part I — Mathematical and Statistical Foundations** (Ch. 1–5): linear algebra as computation, probability and information, generalization theory and its deep-learning failure, optimization, floating point.
- **Part II — Classical Machine Learning** (Ch. 6–9): linear/logistic regression, kernels and the NTK, trees and boosting, unsupervised learning.
- **Part III — Neural Networks and Optimization at Scale** (Ch. 10–15): the MLP, reverse-mode autodiff, initialization and normalization, optimizers through Muon, regularization, muP.
- **Part IV — Architectures** (Ch. 16–21): CNNs, RNNs and the gradient pathology, attention and the Transformer, positional encoding and long context, ViTs and multimodal encoders, state space models through the SSD duality.
- **Part V — Large Language Models** (Ch. 22–29): MoE, tokenization, scaling laws, the pretraining data pipeline, SFT and PEFT, preference optimization, reasoning and test-time compute, evaluation.
- **Part VI — Systems and Efficiency** (Ch. 30–34): the GPU memory wall, FlashAttention, distributed training, inference serving, quantization.
- **Part VII — The Frontier** (Ch. 35–37): diffusion and flow matching, agents, interpretability.
- **Part VIII — Machine Learning Under Encryption** (Ch. 38–42): what breaks when you encrypt a transformer, approximating the non-polynomial core, private-inference systems, encrypted serving at scale, and where a research contribution fits.

## Conventions worth knowing before reading

- **Every chapter ends in one Artifact**: a complete, self-verifying module (pure NumPy
  wherever possible) that was *executed before it was written into the book*, then
  independently re-run by a fact-check pass. The code is evidence, not illustration.
- **Claims carry confidence tags** — Certain / Likely / Verify / Contested. `Verify` marks
  numbers attached to moving targets (hardware, prices, model versions); `Contested` marks
  questions the field genuinely disagrees on, presented with both positions.
- **"Under Encryption" boxes** thread the homomorphic-encryption consequence of ordinary
  architectural choices through the whole book, pointing into Part VIII. Skippable without
  losing the main thread.

## Status

**Complete.** 42 chapters, 6 appendices, ~201,400 words, 1.98 MB.

Verified end to end:

| Check | Result |
|---|---|
| Structural (`book-workspace/assemble.py verify`) | all passed — every TOC anchor resolves, five objectives and five tagged takeaways per chapter, one artifact each, per-type box counters sequential, tags balanced |
| Artifacts re-executed | **42/42 run clean** |
| Embedded code vs executed files (`drift_check.py`) | **42/42 byte-identical, zero drift** |
| Exercise ladder (`ml-foundations/check.py`) | **30/30 pass**, and the 13 newest files were each run **10 times consecutively** after an adversarial review — a single lucky run is not evidence, which that review proved |
| Ladder adversarially reviewed | 13 new files, **69 confirmed defects found and fixed**: flaky timing assertions, tautological checks that no bug could fail, two false factual claims, a `nan` swallowed by Python's `max()` |
| Ladder with torch, scipy and scikit-learn blocked | tiers 0–3 **still pass on numpy alone**, cross-checks skipping cleanly |
| Rendered in Chromium against a local MathJax | **11,162 expressions typeset, 0 math errors, 0 leftover LaTeX, no horizontal scroll** |
| Confidence tags (book-wide, incl. appendices) | 363 Certain · 243 Likely · 332 Verify · 70 Contested |

Two things the automated checks cannot cover, both worth knowing before you cite anything:

- ~~MathJax could not be rendered during the build.~~ **Resolved.** The proxy blocks
  `cdn.jsdelivr.net`, but not the npm registry, so MathJax 3 was installed locally and the
  book rendered in Chromium against it. Result: **11,162 typeset expressions, zero MathJax
  errors, zero leftover raw LaTeX, no horizontal scroll**, all 42 chapters and 42 artifact
  boxes present. Typesetting the full 2 MB file takes 20–30 s on one core — slow, but it
  completes. Screenshots of the title page, Chapter 1, Chapter 11 and Chapter 38 were
  inspected.
- ~~Part VIII's citations are partly abstract-level.~~ **Audited** (2026-08-20, see
  `book-workspace/part8-citation-audit.md`): every externally-sourced claim in Chapters
  38–42 and every identifier in Appendix D.7 was re-verified by search triangulation
  against paper landing pages and venue pages — and, for NEXUS and BOLT, against their
  cloned source code. All 22 bibliographic identifiers resolve exactly; six wording-level
  errors were found and fixed (the worst: Chapter 41 had inverted NEXUS's 37.3 s GPU
  figure into a CPU one). A second pass then settled the five remaining details without
  PDF access — one via Iron's actual NeurIPS PDF recovered through a GitHub mirror, the
  rest by triangulated deep-search of the blocked pages — so **no citation detail in
  Part VIII now rests on an unverified source**. The audit file records every verdict,
  quote, and route.

The commit history records every factual error the verification passes caught and fixed —
a fabricated benchmark, an inverted estimator bias, superseded FlashAttention-3 numbers,
security-table figures that were wrong twice over, an algorithm named after the wrong
inventor. That record is the honest argument for why the verification passes exist.
