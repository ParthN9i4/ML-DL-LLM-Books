# CONTRACT — Foundations of Modern Machine Learning

This is the frozen interface every chapter is written against. It is NOT shipped in the
book. Follow it exactly; deviations break the assembled file.

## Output format

You produce a **raw HTML fragment**, nothing else. No `<!DOCTYPE>`, no `<html>`, no
`<head>`, no `<body>`, no markdown fences around it, no commentary before or after.
The fragment is one `<div class="chapter" id="chN">…</div>` block, appended verbatim
into `ml-book.html`.

Math is LaTeX delimited `$…$` inline and `$$…$$` display. MathJax is configured to skip
`<pre>` and `<code>`, so never put `$` math inside a code block. Use HTML entities for
`&mdash;` `&ndash;` `&hellip;` `&times;`. Escape `<` and `>` inside prose as `&lt;` `&gt;`.

## Chapter skeleton — exact order, no omissions

```html
<div class="chapter" id="chN">
<div class="chapter-header"><div class="ch-label">Chapter N</div><h3>TITLE</h3></div>

<div class="learning-objectives">
<div class="lo-title">Learning Objectives</div>
<ul>
<li>…</li>   <!-- EXACTLY 5. Verb-led. Testable. Math inline where natural. -->
</ul>        <!-- The 5th is ALWAYS "Implement … in working code." -->
</div>

<div class="read-alongside"><strong>Read alongside</strong> &mdash; …</div>

<p>…</p>     <!-- ONE orienting paragraph, ~150 words, explicit forward/back refs -->

<h4>N.1 Section Title</h4>
<p>…</p>
<h5>N.1.1 Subsection Title</h5>
<p>…</p>

<!-- interleaved boxes, see vocabulary below -->

<div class="artifact-box">…</div>     <!-- EXACTLY ONE, near the end -->
<div class="encrypted-note">…</div>   <!-- where applicable, AFTER the artifact -->
<div class="key-takeaways">…</div>    <!-- EXACTLY 5 bullets, each tagged -->
</div>
```

## Box vocabulary

| Class | Title format | Use |
|---|---|---|
| `definition` | `Definition N.k (Name)` | Formal definitions |
| `theorem` | `Theorem N.k (Name)` | Theorems; follow with `<p><strong>Proof.</strong> … $\blacksquare$</p>` or `<strong>Proof sketch.</strong>` |
| `example` | `Example N.k (Name)` | A worked **numeric** example small enough to verify by hand. Show the actual arithmetic. |
| `warning-box` | `Common Pitfall` | A real implementation bug, named concretely. At least one per chapter. |
| `artifact-box` | `Artifact N.1 — Title` | The runnable module. Exactly one per chapter. |
| `encrypted-note` | `Under Encryption` | What this construct costs on ciphertext + pointer to Part VIII |
| `key-takeaways` | (uses `kt-title`) | 5 bullets, each ending in a confidence tag |

Every box is `<div class="X"><div class="box-title">TITLE</div><p>…</p></div>`.

Numbering (`Definition 3.1`, `Theorem 3.2`, …) is **sequential within the chapter across
all box types sharing a counter**? No — each type has its **own** counter: Definition 1.1,
1.2, 1.3…; Theorem 1.1, 1.2…; Example 1.1, 1.2…. Artifact is always `N.1`.

## Confidence tags

Attach to every Key Takeaway bullet, and inline on non-trivial claims in prose:

- `<span class="tag tag-certain">Certain</span>` — textbook-stable, would survive any review.
- `<span class="tag tag-likely">Likely</span>` — strong inference the literature supports but has not settled.
- `<span class="tag tag-verify">Verify</span>` — correct as of writing, attached to a moving API or a number that drifts. Anything hardware-, price-, or model-version-specific gets this.
- `<span class="tag tag-contested">Contested</span>` — the field genuinely disagrees. Present BOTH positions and state which evidence would settle it. Never hedge into mush; state a position, then say what would change it.

## The Artifact — hard requirements

1. **It must actually run.** Write it to a file, execute it with
   `/tmp/claude-0/-home-user-ParthN9i4/8320e92f-b068-5164-8e36-b20b4cf863b1/scratchpad/venv/bin/python`,
   and iterate until it runs clean. Available: `numpy` 2.4.6, `scipy` 1.17.1,
   `torch` 2.13.0 (CPU), `scikit-learn`. Paste only code you have executed.
2. **Pure NumPy for the core.** The chapter's idea is built from scratch. `torch`/`sklearn`
   appear only as *cross-checks*, and each cross-check is wrapped:
   ```python
   try:
       import torch
   except ImportError:
       torch = None
   ```
   with a printed `[skipped: torch not installed]` fallback, so the artifact runs anywhere.
3. **It verifies itself.** Assertions against finite differences, brute force, a closed form,
   or a reference implementation. Print the actual measured residual, not just "OK".
4. **Runs in under ~30 seconds on CPU.** No downloads, no datasets — synthesize data with a
   fixed `np.random.default_rng(0)`.
5. 100–250 lines, heavily commented, `if __name__ == "__main__":` demo at the end.
6. Wrap as `<pre><code>…</code></pre>` inside the artifact box, preceded by a one-paragraph
   description and **followed by a paragraph stating what running it actually prints** —
   quote real numbers from your run.

## Notation (fixed book-wide — do not invent alternatives)

| Symbol | Meaning |
|---|---|
| $N$ | parameter count (Part V onward); also matrix dimension in Ch. 1 where stated |
| $D$ | training tokens / dataset size |
| $C$ | compute in FLOPs |
| $d$, $d_{\text{model}}$ | model/residual width |
| $d_k$, $d_h$ | per-head key/head dimension |
| $L$ | number of layers |
| $n$, $T$ | sequence length |
| $B$ | batch size |
| $\eta$ | learning rate |
| $\theta$, $w$ | parameters |
| $\mathcal{L}$ | loss |
| $x$, $y$ | input, target |
| $\sigma(\cdot)$ | logistic sigmoid |
| $\mathbb{E}$, $\mathrm{Var}$ | expectation, variance |
| $\|\cdot\|_F$, $\|\cdot\|_2$ | Frobenius, spectral/Euclidean norm |
| $\odot$ | elementwise product |

Vectors are column vectors. Weight matrices act as $y = Wx$. Layer index superscript in
parentheses: $a^{(l)}$. Big-O is worst case unless stated.

## Cross-referencing

Prose only: "see Section 1.4", "covered in Chapter 18", "Part VIII returns to this".
**Never** write `<a href="#ch18">` inside a chapter — anchors live only in the TOC.
You may forward-reference any chapter; the full 42-chapter list is below.

## Read-alongside sources (for the `read-alongside` box)

Pick the 1–3 that genuinely cover this chapter. Name the source and what it is good for.
Do not invent chapter numbers for these books — refer to topics, not numbered chapters.

- **Géron**, *Hands-On Machine Learning with Scikit-Learn and PyTorch* (O'Reilly, Dec 2025) — practical, PyTorch-based, end-to-end project discipline. NOTE: this is the 2025 PyTorch edition, which replaced the TensorFlow/Keras editions.
- **Raschka**, *Build a Large Language Model (From Scratch)* (Manning) — line-by-line LLM implementation.
- **Trask**, *Grokking Deep Learning* (Manning) — backprop and autodiff by hand, no framework.
- **Serrano**, *Grokking Machine Learning* (Manning) — gentlest correct treatment of classical algorithms.
- **Chakraborty**, *Introduction to Large Language Models* + NPTEL course `106102576` (Prof. Tanmoy Chakraborty, IIT Delhi, with Prof. Soumen Chakrabarti, IIT Bombay; free YouTube lectures) — NLP-side framing of LLMs.
- **Goodfellow/Bengio/Courville**, *Deep Learning* (2016) — still canonical for Parts I–III.
- **Murphy**, *Probabilistic Machine Learning: Advanced Topics* (2023) — probabilistic lookup.
- **Prince**, *Understanding Deep Learning* (2023, free PDF) — best diagrams.
- **Bishop**, *PRML* / **Hastie et al.**, *ESL* — Part II canon.
- **Karpathy**, *Neural Networks: Zero to Hero*, `nanoGPT` — the from-scratch pedagogy standard.
- Stanford **CS336** (Language Modeling from Scratch) — closest course to Parts V–VI.
- HuggingFace **Ultra-Scale Playbook** — Part VI systems.

## The `encrypted-note` box

Include one wherever the chapter's content has a real consequence for computing on
ciphertext. Homomorphic encryption (CKKS in particular) evaluates **polynomials** cheaply
and everything else expensively; each multiplication consumes **multiplicative depth** from
a fixed budget, and refreshing that budget requires **bootstrapping**, which is expensive.

Be concrete and correct. Good triggers: softmax, any normalization involving a division or
inverse square root, non-polynomial activations, comparisons/max, argmax, data-dependent
control flow, quantization (which *helps* under TFHE-style schemes), matrix multiplication
(cheap — it is the one thing FHE does well), and anything sequential (depth) versus parallel.
End with a pointer such as "Chapter 39 takes up the approximation problem in detail."

Do NOT force one where there is nothing real to say. Better to omit than to invent.

## Tone

Precise, opinionated, quantitative. Real numbers everywhere — FLOPs, parameter counts,
bytes, wall-clock, measured error. Name contested claims as contested and take a position.
Name common misconceptions and correct them. Never write filler like "in this section we
will explore"; state the thing. Assume a mathematically strong reader who is new to this
specific material. No diagrams — structure comes from boxes and `<table>` elements.

Where a source and the book would disagree, the source wins: if you are unsure of a number,
either verify it or tag it `Verify` and say what to check it against.

## Full chapter list (for cross-references)

**Part I — Mathematical and Statistical Foundations**
1 Linear Algebra as Computation · 2 Probability, Information, and Why Cross-Entropy Is the
Only Loss · 3 Generalization: Classical Theory and Its Deep-Learning Failure · 4 Optimization:
Convergence Rates and the Non-Convex Reality · 5 Numerics: Floating Point, Precision, Stability

**Part II — Classical Machine Learning**
6 Linear and Logistic Regression: the Atom · 7 Kernels, Margins, and the Bridge to
Infinite-Width Networks · 8 Trees, Bagging, and Gradient Boosting · 9 Unsupervised Learning:
PCA, Clustering, and Latent Variables

**Part III — Neural Networks and Optimization at Scale**
10 The Multilayer Perceptron and What Depth Buys · 11 Automatic Differentiation and
Backpropagation · 12 Initialization, Normalization, and Signal Propagation · 13 Optimizers:
SGD to Muon · 14 Regularization and the Generalization Levers That Actually Work ·
15 Hyperparameter Transfer: muP, Schedules, and Batch Size

**Part IV — Architectures**
16 Convolutional Networks · 17 Recurrence: RNNs, LSTMs, and the Gradient Pathology ·
18 Attention and the Transformer · 19 Positional Information and Long Context · 20 Vision
Transformers and Multimodal Encoders · 21 State Space Models: S4, Mamba, and the SSD Duality

**Part V — Large Language Models**
22 Mixture of Experts and Conditional Computation · 23 Tokenization and the Language-Modeling
Objective · 24 Scaling Laws · 25 The Pretraining Pipeline: Data, Curricula, Mid-Training ·
26 Supervised Fine-Tuning and Parameter-Efficient Adaptation · 27 Preference Optimization:
RLHF, DPO, GRPO · 28 Reasoning Models and Test-Time Compute · 29 Evaluation and the
Measurement Crisis

**Part VI — Systems and Efficiency**
30 The GPU Execution Model and the Memory Wall · 31 FlashAttention and Writing Fast Kernels ·
32 Distributed Training · 33 Inference: Serving, Caching, and Speculative Decoding ·
34 Quantization, Distillation, and Compression

**Part VII — The Frontier**
35 Diffusion, Flow Matching, and Non-Autoregressive Generation · 36 Agents, Tool Use, and
Long-Horizon Training · 37 Interpretability, Safety, and Open Problems

**Part VIII — Machine Learning Under Encryption**
38 What Breaks When You Encrypt a Transformer · 39 Approximating the Non-Polynomial Core ·
40 Private Transformer Inference Systems · 41 Encrypted LLM Serving at Scale · 42 The
Research Frontier: Where a Contribution Fits

**Appendices** — A Notation · B Environment Setup · C Complete Code for All Artifacts ·
D Sources and Further Reading · E Reading Crosswalk · F Reference Numbers (dated)

## Length target — HARD CAP

**3,200–4,000 words of rendered prose per chapter, and never above 4,500.** This is a hard
constraint, not a suggestion. Part I overran by 45% and the book is on track for 230,000
words in a single HTML file, which is a real usability defect: MathJax typesetting time
scales with document size.

If you are over, the fix is **never** to cut a correct derivation. Cut in this order:

1. Restatement — anywhere you make the same point twice in different words.
2. The artifact's prose description and results paragraph. Quote two or three real numbers,
   not the whole table.
3. Motivational throat-clearing at the head of sections. The section title already says
   what the section is about.
4. Hedging clauses. "It is worth noting that X" is just "X".

If you are short, you have under-developed the derivations, not the prose. Add mathematics,
not sentences.
