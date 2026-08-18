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

Thirty exercises, numbered globally. Every one is listed here, so a gap in the numbering
would be visible rather than silent.

| Tier | Directory | Dependencies | Exercises |
|------|-----------|--------------|-----------|
| 0 | `tier0_math/` | numpy, scipy | `ex01` blocked matmul + Jacobi SVD · `ex02` entropy, KL, arithmetic coding · `ex03` GD and heavy-ball convergence rates · `ex04` IEEE-754, bf16, Kahan summation · `ex05` OLS/ridge/logistic, IRLS |
| 1 | `tier1_autodiff/` | numpy | `ex06` scalar autodiff (micrograd) · `ex07` tensor reverse-mode engine · `ex08` initialization and normalization · `ex09` SGD → Adam → Muon · `ex10` an MLP and a training loop |
| 2 | `tier2_architectures/` | numpy, torch | `ex11` convolution as im2col · `ex12` the conv backward pass · `ex13` the ViT stem · `ex14` scaled dot-product and multi-head attention · `ex15` a full pre-norm block · `ex16` RoPE, two ways |
| 3 | `tier3_sequence/` | numpy, torch | `ex17` linear attention and the associativity trick · `ex18` selective scan ≡ parallel scan · `ex19` the S4 kernel: recurrence, convolution, FFT · `ex20` MoE routing and load balancing |
| 4 | `tier4_llm/` | numpy, torch | `ex21` training byte-level BPE · `ex22` pretraining a tiny LM · `ex23` LoRA and the rank question · `ex24` DPO and GRPO on a bandit |
| 5 | `tier5_systems/` | numpy, torch | `ex25` quantization and the outlier problem · `ex26` speculative decoding · `ex27` tiled attention and the online softmax · `ex28` ring all-reduce |
| 6 | `tier6_encrypted/` | numpy | `ex29` Chebyshev approximation and a depth ledger · `ex30` an encrypted linear layer: packing, rotations, depth |

The whole ladder runs in about three and a half minutes on one CPU core; no single exercise
takes more than about forty seconds.

**Tiers 0–3 are fully usable with numpy alone.** This is verified, not assumed: those files
are re-run with `torch`, `scipy` and `scikit-learn` blocked at the import hook, and they
pass — the cross-checks skip and every from-scratch assertion still runs. Tiers 4–6 use
torch where a reference implementation genuinely helps, still never as the implementation.

Tier 6 needs no homomorphic-encryption library. It models the slot algebra and the depth
ledger in numpy, which is the part that decides whether a circuit is feasible; the
cryptography itself is the companion FHE book's subject.

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
