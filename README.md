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
| [`ml-foundations/`](ml-foundations/) | **The exercise ladder.** Standalone runnable Python files, tiers 0–6, from pure-numpy matrix arithmetic up to encrypted inference. `python3 check.py` runs everything; the assertions in each file *are* the specification. |
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

Being written part by part, one commit per part. Chapters 1–21 are complete and
fact-checked; Part V (22–29) is in verification; Parts VI–VIII and the appendices are in
progress. The commit history documents every factual error the verification passes caught
and fixed — which is itself an honest record of why the verification passes exist.
