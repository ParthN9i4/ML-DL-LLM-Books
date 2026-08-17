# Resume state — blocked on usage credits

Work stopped on **2026-08-17 09:07 UTC**, not because of a resetting time limit but
because the account ran out of usage credits. Switching model does not help: a verify
agent relaunched on Fable failed instantly with the identical error. This needs credits
added at `claude.ai/settings/usage`; nothing on this side will unblock it.

Everything produced so far is committed and pushed. Nothing is lost.

## Where the book stands

| | |
|---|---|
| In `ml-book.html` | **37 of 42 chapters**, ~161,000 words, 1.6 MB — Parts I–VII, all fact-checked |
| Held in `drafts/` | **Chapters 38–42** (all of Part VIII), drafted, artifacts all run clean, **none fact-checked** |
| Appendices | Not written. The ground-truth index they generate from (`book_index.json`) exists. |
| Exercise ladder | 17 files across tiers 0–6, all passing |
| Study plan + weekly Routine | Complete and live (`trig_0158duggqYDeFVVBVrmumoSC`, Saturdays 09:00 IST) |

Chapters 38–42 are deliberately **not** spliced into the book. Every one of the 37
verified chapters had at least one real defect caught by its fact-check pass — a
fabricated benchmark, an inverted bias direction, superseded numbers presented as
current. On that base rate an unverified chapter is wrong somewhere, and Part VIII is the
research part where wrong attribution would be worst.

## To resume, in order

1. **Fact-check chapters 38–42.** Resume the existing workflow — the five drafts replay
   from cache, only the verifies run:
   ```
   Workflow({scriptPath: ".../workflows/scripts/ml-book-part8-wf_803a910f-ac4.js",
             resumeFromRunId: "wf_803a910f-ac4"})
   ```
   If the scratchpad has been recycled, restore `parts/` and `code/` from
   `book-workspace/drafts/` and `book-workspace/code/` first, plus `CONTRACT.md` and
   `assemble.py`, and rebuild the venv (numpy, scipy, scikit-learn, torch).

2. **Assemble Part VIII**: `python3 assemble.py insert VIII && python3 assemble.py verify`.
   Commit and push. The book is then 42/42.

3. **Regenerate `book_index.json`** — it currently covers only the 37 assembled chapters,
   and the appendices must be generated from the complete index, never from memory. This
   is the specific failure that left the predecessor FHE book's Appendix C listing
   part/chapter ranges that did not match its own table of contents.

4. **Write Appendices A–F**: resume
   ```
   Workflow({scriptPath: ".../workflows/scripts/ml-book-appendices-wf_c46b7a18-4f2.js",
             resumeFromRunId: "wf_c46b7a18-4f2"})
   ```
   The extract phase replays from cache; delete `book_index.json` first if you want it
   rebuilt against all 42 chapters (recommended — see step 3).

5. **Final verification pass**, not yet run at whole-book scale:
   - `assemble.py verify` — TOC anchors resolve, five objectives per chapter, one artifact
     box, five tagged takeaways, per-type box counters sequential, balanced tags.
   - Extract and execute every artifact from the HTML; each must run clean.
   - `python3 ml-foundations/check.py` for the ladder.
   - Render in Chromium via Playwright: confirm MathJax typesets without console errors
     and no horizontal scroll. **Note:** this sandbox's proxy blocks `cdn.jsdelivr.net`,
     so MathJax cannot load here and the math shows as LaTeX source. The render check
     needs an environment with CDN access, or a local MathJax install.
   - Word-count check against the ~140k target (currently ~161k at 37 chapters, so the
     finished book will land near 185k — worth a decision on whether to trim).

## Known open items

- **Size.** The book is heading past the 130k word estimate the plan agreed. Parts I–II
  averaged 5,500 prose words per chapter before a hard 4,500 cap was imposed; the later
  parts hold to ~3,900. Trimming Parts I–II is possible but risks reintroducing errors
  their fact-check passes removed, which is why it was not done during the build.
- **`ml-book.html` at 1.6 MB and growing.** MathJax typeset time scales with document
  size. If it proves slow to read, splitting into two volumes at the Part V boundary is
  the clean fix — the part markers make it mechanical.
- **Two paper-queue citations unverified**: CryptoGen (2602.08798) and Primer
  (2303.13679) could not be resolved against the bibliographic index. Flagged in
  `study-plan.html`.
