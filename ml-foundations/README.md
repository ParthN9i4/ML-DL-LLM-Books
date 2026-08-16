# ML Foundations — Coding Exercise Ladder

A hands-on exercise sequence that builds from raw matrix arithmetic up to encrypted
transformer inference. Companion to the book (`ml-book.html`) and paced by `study-plan.html`.

## How to use

Each exercise is a standalone file. Run it directly:

```bash
python3 tier0_math/ex01_linalg.py
```

Every file contains:
- A short docstring explaining the concept (the book has the full treatment)
- Reference implementations that make all tests pass
- `check()` assertions at the bottom — **read these first, they ARE the specification**
- Most exercises include a "break it" section where wrong parameters show the failure mode

**To learn**: delete a function body (replace with `pass`), re-read the docstring and the
assertions, then reimplement. The tests tell you when you've got it right. Reading an
implementation and writing one are different skills, and only the second one transfers.

Run the whole ladder, or one tier:

```bash
python3 check.py              # every exercise, as a subprocess each
python3 check.py tier0_math   # just one tier
```

## Tier progression

| Tier | Directory | Dependencies | What you build |
|------|-----------|--------------|----------------|
| 0 | `tier0_math/` | numpy, scipy | Matrix calculus, SVD, gradient descent, MLE, entropy, floating point — by hand |
| 1 | `tier1_autodiff/` | numpy | A reverse-mode autodiff engine, an MLP, and a training loop from scratch |
| 2 | `tier2_architectures/` | numpy, torch | Attention, multi-head, a Transformer block, convolution, a ViT patch embedder |
| 3 | `tier3_sequence/` | numpy, torch | Linear attention, the S4 kernel, selective scan ≡ parallel scan, RoPE, MoE routing |
| 4 | `tier4_llm/` | torch | Tokenizer training, a small LM pretrain, LoRA, DPO, a GRPO loop |
| 5 | `tier5_systems/` | numpy, torch | Tiled attention, quantization, speculative decoding, ring all-reduce |
| 6 | `tier6_encrypted/` | numpy, (tenseal, OpenFHE) | Polynomial softmax and inverse-sqrt, an encrypted linear layer, depth budgeting |

Tiers 0–3 run on CPU in well under a minute each. Tier 4 targets a single small GPU and
degrades to a tiny model on CPU. Tier 6 is the join point with encrypted computation.

## Installation

```bash
# Tiers 0-1 — pure math and autodiff
pip install numpy scipy

# Tiers 2-5 — cross-checks against the reference implementations
pip install torch scikit-learn

# Tier 6 — homomorphic encryption
pip install tenseal
# OpenFHE (C++, build from source) for the depth-budgeting exercises:
git clone https://github.com/openfheorg/openfhe-development.git
cd openfhe-development && mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install
```

Every exercise guards its optional imports. If `torch` is missing, the cross-check prints
`[skipped: torch not installed]` and the from-scratch assertions still run — so tiers 0–3
are fully usable with numpy alone.

## Confidence markers

Same convention the book uses:

- Code using **numpy/scipy only** — fully standalone, runs as-is, assertions are exact.
- Code using **torch / scikit-learn** — verified against the current API, runs after `pip install`.
  These appear only as cross-checks, never as the implementation.
- Code using **TenSEAL / OpenFHE** — marked `[VERIFY]` where API signatures should be
  checked against the current examples before running. Homomorphic-encryption library
  surfaces move faster than any book.

## The one rule

An exercise is finished when you can delete it, wait a week, and write it again from the
docstring. Passing the assertions on the first read means you read carefully. Passing them
from an empty file means you understood.
