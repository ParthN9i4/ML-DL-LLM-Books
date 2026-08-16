#!/usr/bin/env python3
"""Assemble drafted chapter fragments into ml-book.html, and verify the result.

    python3 assemble.py insert I      # splice Part I's header + chapters in
    python3 assemble.py verify        # run all structural checks on the book

Insertion is idempotent: it replaces everything between the PART_<N>_START and
PART_<N>_END markers, so re-running after a chapter is corrected is safe.
"""

import os
import re
import sys

BOOK = "/home/user/ParthN9i4/ml-book.html"
PARTS = "/tmp/claude-0/-home-user-ParthN9i4/8320e92f-b068-5164-8e36-b20b4cf863b1/scratchpad/parts"

# Part number -> (label, title, chapter range, part-desc)
PART_META = {
    "I": ("Part I", "Mathematical and Statistical Foundations", (1, 5),
          "Everything downstream is a composition of three things: a linear map, a "
          "nonlinearity, and a gradient step. This part builds the linear algebra as "
          "computation rather than as abstract vector spaces, the probability that makes "
          "cross-entropy the only sensible loss, and the generalization theory that classical "
          "statistics gives us &mdash; along with the precise sense in which deep learning "
          "violates it. It ends in floating point, because at scale numerics is not a detail, "
          "it is the constraint."),
    "II": ("Part II", "Classical Machine Learning", (6, 9),
           "Deep learning did not replace classical machine learning; it absorbed some of it "
           "and lost to the rest. This part builds linear models &mdash; the atom every neural "
           "layer is made of &mdash; then kernels, whose theory turns out to predict the "
           "behaviour of infinitely wide networks, then gradient-boosted trees, which still win "
           "on tabular data and deserve an explanation rather than an excuse, and finally the "
           "classical unsupervised objectives that reappear almost verbatim as modern "
           "self-supervision."),
    "III": ("Part III", "Neural Networks and Optimization at Scale", (10, 15),
            "This part builds the machine: a differentiable computation graph, the reverse-mode "
            "engine that trains it, and the four disciplines &mdash; initialization, "
            "normalization, optimization, and hyperparameter transfer &mdash; that decide "
            "whether a large training run converges or diverges at step four thousand. The "
            "autodiff engine written in Chapter 11 is reused, unchanged, in every subsequent "
            "architecture chapter."),
    "IV": ("Part IV", "Architectures", (16, 21),
           "Architecture is the encoding of an inductive bias into a computational graph. This "
           "part builds convolutions (locality and translation equivariance), recurrence (state, "
           "and its gradient pathology), attention (content-based routing at quadratic cost), "
           "positional structure and long-context extension, vision and multimodal encoders, and "
           "state space models &mdash; recurrence rebuilt so that it parallelizes. Each is "
           "implemented from scratch and checked against the reference framework implementation."),
    "V": ("Part V", "Large Language Models", (22, 29),
          "A language model is a compression objective, a data pipeline, a scaling decision, and "
          "three stages of post-training. This part treats each as an engineering discipline with "
          "real numbers: what a token costs, what a scaling law predicts and where it breaks, why "
          "aligned models are trained with three different loss functions that all claim to solve "
          "the same problem, and what test-time compute actually buys. It is also where this book "
          "is most explicit about what remains contested."),
    "VI": ("Part VI", "Systems and Efficiency", (30, 34),
           "Above a billion parameters, machine learning is a systems discipline. This part builds "
           "the mental model that predicts runtime &mdash; arithmetic intensity and the memory wall "
           "&mdash; then derives FlashAttention as its consequence, then covers the four "
           "parallelisms, inference serving, and quantization. Every claim here is measured, "
           "because in systems the only acceptable evidence is a benchmark."),
    "VII": ("Part VII", "The Frontier", (35, 37),
            "This part covers what is genuinely unsettled: generative modelling beyond "
            "autoregression, agents that act over long horizons, and the interpretability work "
            "that is the field's only route to knowing what these systems compute. It is written "
            "to be replaced. Every claim carries a date, and the Key Takeaways lean hard on the "
            "Contested tag."),
    "VIII": ("Part VIII", "Machine Learning Under Encryption", (38, 42),
             "Everything to this point assumed the computer may look at the data. Remove that "
             "assumption and the architecture you have just spent seven parts learning becomes a "
             "cost model in a different currency: multiplication is cheap, comparison is "
             "ruinous, and depth is a budget you cannot overdraw. This part works out what "
             "survives, reads the systems that have tried, and ends where the open problems "
             "actually are."),
}

ORDER = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]


def part_header(pnum):
    label, title, _, desc = PART_META[pnum]
    return (
        f'<div class="part-header">\n'
        f'<div class="part-label">{label}</div>\n'
        f'<h2>{title}</h2>\n'
        f'<div class="part-desc">{desc}</div>\n'
        f'</div>\n'
    )


def insert(pnum):
    lo, hi = PART_META[pnum][2]
    chunks = [part_header(pnum)]
    missing = []
    for n in range(lo, hi + 1):
        path = os.path.join(PARTS, f"ch{n:02d}.html")
        if not os.path.exists(path):
            missing.append(n)
            continue
        with open(path) as f:
            chunks.append(f.read().strip() + "\n")
    if missing:
        print(f"  ! missing fragments for chapters: {missing}")

    with open(BOOK) as f:
        book = f.read()

    start = f"<!-- PART_{pnum}_START -->"
    end = f"<!-- PART_{pnum}_END -->"
    if start not in book or end not in book:
        sys.exit(f"markers for Part {pnum} not found in {BOOK}")

    body = "\n".join(chunks)
    # Splice by index, NOT re.sub: chapter text is full of LaTeX like \kappa,
    # which re.sub would try to interpret as escape sequences in the replacement.
    i = book.index(start) + len(start)
    j = book.index(end)
    new = book[:i] + "\n" + body + book[j:]
    with open(BOOK, "w") as f:
        f.write(new)
    inserted = (hi - lo + 1) - len(missing)
    print(f"  inserted Part {pnum}: {inserted} chapters, {len(body):,} bytes")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify():
    with open(BOOK) as f:
        book = f.read()

    problems = []

    # 1. every TOC anchor resolves to an id
    hrefs = set(re.findall(r'href="#([\w-]+)"', book))
    ids = set(re.findall(r'id="([\w-]+)"', book))
    dangling = sorted(hrefs - ids - {"top"})
    if dangling:
        problems.append(f"dangling TOC anchors (no matching id): {dangling}")

    # 2. no in-chapter cross-reference links (anchors belong only in the TOC)
    toc_match = re.search(r'<div class="toc"(.*?)</div>\s*<!-- PART_I_START', book, re.DOTALL)
    toc_span = toc_match.span() if toc_match else (0, 0)
    for m in re.finditer(r'href="#ch\d+"', book):
        if not (toc_span[0] <= m.start() <= toc_span[1]):
            problems.append(f"chapter-anchor link outside the TOC at offset {m.start()}")
            break

    # 3. per-chapter structural checks
    chapters = re.findall(r'<div class="chapter" id="ch(\d+)">(.*?)\n</div>\s*(?=<div class="chapter"|<div class="part-header"|<!-- PART_|<!-- APPENDICES_)', book, re.DOTALL)
    seen = []
    for num, body in chapters:
        n = int(num)
        seen.append(n)
        tag = f"ch{n}"

        los = re.findall(r'<div class="learning-objectives">(.*?)</ul>', body, re.DOTALL)
        if len(los) != 1:
            problems.append(f"{tag}: expected 1 learning-objectives block, found {len(los)}")
        else:
            n_li = len(re.findall(r"<li>", los[0]))
            if n_li != 5:
                problems.append(f"{tag}: expected 5 learning objectives, found {n_li}")

        n_art = body.count('class="artifact-box"')
        if n_art != 1:
            problems.append(f"{tag}: expected exactly 1 artifact box, found {n_art}")

        kt = re.search(r'<div class="key-takeaways">(.*?)</div>\s*$', body, re.DOTALL)
        if not kt:
            problems.append(f"{tag}: no key-takeaways block")
        else:
            items = re.findall(r"<li>(.*?)</li>", kt.group(1), re.DOTALL)
            if len(items) != 5:
                problems.append(f"{tag}: expected 5 key takeaways, found {len(items)}")
            untagged = [i for i, t in enumerate(items) if 'class="tag ' not in t]
            if untagged:
                problems.append(f"{tag}: key takeaways missing confidence tags at {untagged}")

        if not re.search(r'class="warning-box"', body):
            problems.append(f"{tag}: no Common Pitfall box")

        # box numbering: per-type counters, sequential from 1, no duplicates
        for kind in ("Definition", "Theorem", "Example", "Artifact"):
            nums = [
                tuple(int(x) for x in m)
                for m in re.findall(rf"{kind} (\d+)\.(\d+)", body)
            ]
            nums = [b for a, b in nums if a == n]
            if not nums:
                continue
            uniq = sorted(set(nums))
            if len(uniq) != len(set(nums)) or uniq != list(range(1, len(uniq) + 1)):
                problems.append(f"{tag}: {kind} numbering is {sorted(set(nums))}, expected 1..{len(uniq)}")

        # math must not appear inside code blocks (MathJax skips them)
        for pre in re.findall(r"<pre>(.*?)</pre>", body, re.DOTALL):
            if re.search(r"\$\$", pre):
                problems.append(f"{tag}: display math inside a <pre> block")
                break

        if "```" in body:
            problems.append(f"{tag}: markdown code fence leaked into the HTML")

    # 4. chapters present and in order
    expected = [n for p in ORDER for n in range(PART_META[p][2][0], PART_META[p][2][1] + 1)]
    written = [n for n in expected if n in seen]
    if seen != written:
        problems.append(f"chapters out of order: {seen}")

    # 5. crude tag balance for the div-heavy fragments
    opens, closes = book.count("<div"), book.count("</div>")
    if opens != closes:
        problems.append(f"unbalanced divs: {opens} open, {closes} close")

    words = len(re.sub(r"<[^>]+>", " ", re.sub(r"<(script|style|pre)\b.*?</\1>", " ", book, flags=re.DOTALL)).split())

    print(f"\nchapters present : {len(seen)}/42  {seen}")
    print(f"rendered words   : ~{words:,}")
    print(f"file size        : {os.path.getsize(BOOK):,} bytes")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nall structural checks passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "insert":
        for p in (sys.argv[2:] or ORDER):
            if p in PART_META:
                insert(p)
        sys.exit(0)
    elif cmd == "verify":
        sys.exit(verify())
    sys.exit(f"unknown command {cmd!r}")
