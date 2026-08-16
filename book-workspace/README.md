# Book workspace — build state, not the book

Everything here is scaffolding for producing `ml-book.html`. None of it ships. Once the
book is finished and Appendix C is generated from `code/`, this whole directory can be
deleted.

It exists in the repo for one reason: the build runs in an ephemeral container, and the
drafts and artifact code below would otherwise be lost when that container is reclaimed.

## State as of the last commit

| Chapters | State | Where |
|---|---|---|
| 1–9 | **Finished** — drafted, artifact executed, adversarially fact-checked, assembled | in `ml-book.html` |
| 10, 11, 16, 17, 18 | **Drafted, artifact runs clean, NOT fact-checked** | `drafts/` |
| 12, 13 | Artifact code started, does not run; draft agent was killed mid-work | discarded |
| 14, 15, 19–42 | Not started | — |

Chapters 10–18 in `drafts/` are deliberately **not** in `ml-book.html`. Every one of the
nine finished chapters had real errors found by the fact-check pass — a false amplification
bound, an arithmetic slip in a perturbation ratio, a KKT theorem stated as a biconditional
its own proof did not support, artifact prose quoting numbers from no real run. Shipping a
chapter that has only been drafted means shipping content that is, on the observed base
rate, wrong somewhere. They wait in `drafts/` until verified.

## Resuming

1. **Verify the five drafts.** For each of 10, 11, 16, 17, 18: re-run
   `code/chNN_artifact.py`, confirm the code pasted in the fragment is byte-identical to
   the file that ran, check every theorem and numeric example by doing the arithmetic, and
   enforce the length cap. Then move the fragment into the scratchpad `parts/` directory
   the assembler reads.
2. **Redraft 12 and 13** from scratch (Chapter 13 needs live research — Muon is now
   production-standard and an optimizer chapter ending at AdamW would be wrong).
3. **Assemble**: `python3 assemble.py insert III` then `python3 assemble.py verify`.
4. Continue with Parts V–VIII, then the appendices.

## Files

- `CONTRACT.md` — the frozen interface every chapter is written against: HTML fragment
  format, box vocabulary, confidence tags, fixed notation, artifact requirements, length
  cap. Chapter drafting is only reproducible because this does not change underneath it.
- `assemble.py` — splices fragments into `ml-book.html` between the `PART_N_START/END`
  markers (idempotent, so re-running after a correction is safe), and runs the structural
  checks: TOC anchors resolve, five learning objectives per chapter, exactly one artifact
  box, five tagged Key Takeaways, per-type box counters sequential with no gaps, no
  in-chapter anchor links, no leaked markdown fences, no math stranded inside `<pre>`,
  balanced tags.
  - `python3 assemble.py insert I II` — splice those parts in
  - `python3 assemble.py verify` — check the whole book
- `code/` — every artifact that has been executed successfully, plus
  `autodiff_engine.py`, the Chapter 11 engine that later architecture chapters import.
  These are the source for Appendix C.
- `drafts/` — chapter fragments awaiting fact-check.

## Environment

The artifacts need `numpy`, `scipy`, `torch` (CPU), `scikit-learn`. Optional imports are
guarded, so anything not needing a cross-check runs on numpy alone. Not available in the
build container, and therefore never imported: `transformers`, `timm`, `xgboost`,
`lightgbm`, `pandas`, `matplotlib`.
