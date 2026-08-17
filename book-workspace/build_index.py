#!/usr/bin/env python3
"""Build ground-truth index from ml-book.html. Regex over HTML, no external deps."""
import re, json, html as htmlmod, collections, sys, os, glob

SRC = "/workspace/ml-dl-llm-books/ml-book.html"
DRAFTS = "/workspace/ml-dl-llm-books/book-workspace/drafts"
CODEDIR = "/workspace/ml-dl-llm-books/book-workspace/code"
OUT = "/tmp/claude-0/-home-user-ParthN9i4/8320e92f-b068-5164-8e36-b20b4cf863b1/scratchpad/book_index.json"

raw = open(SRC, encoding="utf-8").read()

def clean(s):
    s = re.sub(r"<[^>]+>", "", s)
    s = htmlmod.unescape(s)
    s = s.replace("\u2014", "\u2014")
    return re.sub(r"\s+", " ", s).strip()

anomalies = []

# ---------------- TOC (the promise) ----------------
toc_parts = []
toc_block = raw[raw.find('<div class="toc"'): raw.find('<!-- PART_I_START -->')]
for m in re.finditer(r'<div class="toc-part">(.*?)</div>\s*</div>', toc_block, re.S):
    pass
# simpler: split on toc-part-title
tp = re.split(r'<div class="toc-part-title">', toc_block)[1:]
for blk in tp:
    title = clean(blk.split("</div>")[0])
    chs = [(clean(a), clean(b)) for a, b in
           re.findall(r'<a class="toc-chapter"[^>]*><span class="ch-num">(.*?)</span>(.*?)</a>', blk, re.S)]
    toc_parts.append({"title": title, "entries": chs})

toc_chapter_nums = []
toc_titles = {}
toc_appendices = []
for p in toc_parts:
    for num, t in p["entries"]:
        m = re.match(r"Ch\.\s*(\d+)", num)
        if m:
            n = int(m.group(1)); toc_chapter_nums.append(n); toc_titles[n] = t
        else:
            toc_appendices.append({"letter": num, "title": t})

# ---------------- Parts actually rendered ----------------
part_headers = []
for m in re.finditer(r'<div class="part-header">\s*<div class="part-label">(.*?)</div>\s*<h2>(.*?)</h2>', raw, re.S):
    part_headers.append({"label": clean(m.group(1)), "title": clean(m.group(2)), "pos": m.start()})
if not part_headers:  # fallback if h2 differs
    for m in re.finditer(r'<div class="part-header">\s*<div class="part-label">(.*?)</div>(.*?)</div>\s*<div class="chapter"', raw, re.S):
        part_headers.append({"label": clean(m.group(1)), "title": clean(m.group(2))[:200], "pos": m.start()})

# ---------------- Chapters ----------------
ch_starts = [(int(m.group(1)), m.start()) for m in
             re.finditer(r'<div class="chapter" id="ch(\d+)">', raw)]
chapters = []
box_registry = collections.Counter()

for idx, (n, start) in enumerate(ch_starts):
    end = ch_starts[idx + 1][1] if idx + 1 < len(ch_starts) else len(raw)
    # don't run past a part boundary marker
    body = raw[start:end]

    hm = re.search(r'<div class="ch-label">Chapter (\d+)</div><h3>(.*?)</h3>', body, re.S)
    label_n = int(hm.group(1)) if hm else None
    title = clean(hm.group(2)) if hm else None
    if label_n != n:
        anomalies.append(f"Chapter div id=ch{n} carries header label 'Chapter {label_n}'")

    sections = [clean(x) for x in re.findall(r"<h4>(.*?)</h4>", body, re.S)]
    subsections = [clean(x) for x in re.findall(r"<h5>(.*?)</h5>", body, re.S)]

    def boxes(cls):
        out = []
        for m in re.finditer(r'<div class="%s"[^>]*>\s*<div class="box-title">(.*?)</div>' % cls, body, re.S):
            out.append(clean(m.group(1)))
        return out

    defs = boxes("definition")
    thms = boxes("theorem")
    exs = boxes("example")
    arts = boxes("artifact-box")
    warns = boxes("warning-box")

    for b in defs + thms + exs + arts:
        box_registry[b] += 1

    ra = re.search(r'<div class="read-alongside">(.*?)</div>', body, re.S)
    enc = 'class="encrypted-note"' in body

    lo = re.findall(r'<div class="learning-objectives">.*?</div>(.*?)</div>', body, re.S)
    lo_items = []
    lom = re.search(r'<div class="learning-objectives">(.*?)</div>\s*(?=<div|<p|<h)', body, re.S)
    lob = re.search(r'<div class="learning-objectives">(.*?)</ul>', body, re.S)
    if lob:
        lo_items = [clean(x) for x in re.findall(r"<li>(.*?)</li>", lob.group(1), re.S)]

    ktb = re.search(r'<div class="key-takeaways">(.*?)</ul>', body, re.S)
    kt_items = [clean(x) for x in re.findall(r"<li>(.*?)</li>", ktb.group(1), re.S)] if ktb else []

    tagc = {t: len(re.findall(r'class="tag tag-%s"' % t, body)) for t in
            ("certain", "likely", "verify", "contested")}

    # part membership
    part_label = None
    for ph in part_headers:
        if ph["pos"] < start:
            part_label = ph["label"]
    chapters.append({
        "n": n,
        "title": title,
        "sections": sections,
        "subsections": subsections,
        "artifact_title": arts[0] if arts else None,
        "artifact_count": len(arts),
        "definitions": defs,
        "theorems": thms,
        "examples": exs,
        "warning_boxes": warns,
        "has_encrypted_note": enc,
        "encrypted_note_count": body.count('class="encrypted-note"'),
        "read_alongside": clean(ra.group(1)) if ra else None,
        "learning_objectives": lo_items,
        "key_takeaways": kt_items,
        "tags": tagc,
        "part": part_label,
        "line_span": [raw[:start].count("\n") + 1, raw[:end].count("\n") + 1],
    })

# ---------------- parts list ----------------
parts = []
for i, ph in enumerate(part_headers):
    nxt = part_headers[i + 1]["pos"] if i + 1 < len(part_headers) else len(raw)
    mine = [c["n"] for c in chapters if ph["pos"] < raw.find('<div class="chapter" id="ch%d">' % c["n"]) < nxt]
    parts.append({"label": ph["label"], "title": ph["title"],
                  "chapters": [min(mine), max(mine)] if mine else [],
                  "chapter_list": mine})

# ---------------- tags (whole book) ----------------
tags = {t: len(re.findall(r'class="tag tag-%s"' % t, raw)) for t in
        ("certain", "likely", "verify", "contested")}

# ---------------- notation symbols ----------------
# Collect inline+display math, then harvest atomic defined quantities.
body_only = raw[raw.find('<!-- PART_I_START -->'):]
math_chunks = re.findall(r"\$\$(.+?)\$\$", body_only, re.S) + re.findall(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", body_only)
mathtext = "\n".join(math_chunks)
mathtext = htmlmod.unescape(mathtext)

sym = collections.Counter()

# Greek + named commands
for m in re.finditer(r"\\(alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|varphi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|nabla|ell|infty)\b", mathtext):
    sym[m.group(0)] += 1

# subscripted identifiers: d_{\text{model}}, d_{\mathrm{ff}}, W_Q, x_t, \sigma_1 ...
for m in re.finditer(r"(\\?[A-Za-z]+|\\[a-zA-Z]+)\s*_\s*\{?\s*(\\(?:text|mathrm|mathsf|mathtt)\{([A-Za-z0-9\-]+)\}|[A-Za-z0-9\\]+)\s*\}?", mathtext):
    base, sub, txt = m.group(1), m.group(2), m.group(3)
    if txt:
        s = "%s_{\\text{%s}}" % (base, txt)
    else:
        if len(sub) > 3 or sub.startswith("\\") and len(sub) > 8:
            continue
        s = "%s_{%s}" % (base, sub)
    if len(base) <= 8:
        sym[s] += 1

# blackboard/cal sets
for m in re.finditer(r"\\(mathbb|mathcal|mathfrak)\{([A-Za-z])\}", mathtext):
    sym["\\%s{%s}" % (m.group(1), m.group(2))] += 1

# bare single latin letters used as quantities
for m in re.finditer(r"(?<![A-Za-z\\_^{])([A-Za-z])(?![A-Za-z_}])", mathtext):
    sym[m.group(1)] += 1

# operator-ish named quantities
for kw in ["\\mathrm{d}", "\\top", "\\mathsf{T}"]:
    pass

MIN = 8
OPS = ("\\sum", "\\prod", "\\int", "\\log", "\\max", "\\min", "\\arg", "\\exp",
       "\\lim", "\\sup", "\\inf", "\\cup", "\\cap", "\\bigcup", "\\bigoplus")
notation = [s for s, c in sym.most_common()
            if c >= MIN and not s.startswith(OPS) and s != "\\text"]
notation_counts = {s: sym[s] for s in notation}

# ---------------- anomalies ----------------
present = sorted(c["n"] for c in chapters)
expected_toc = sorted(toc_chapter_nums)
missing = [n for n in expected_toc if n not in present]
extra = [n for n in present if n not in expected_toc]
if missing:
    anomalies.append("TOC promises chapters %s that have NO rendered <div class=\"chapter\">: %s"
                     % (len(missing), missing))
if extra:
    anomalies.append("Rendered chapters absent from TOC: %s" % extra)

# numbering gaps in what IS present
gaps = [present[i] + 1 for i in range(len(present) - 1) if present[i + 1] != present[i] + 1]
if gaps:
    anomalies.append("Gaps in rendered chapter numbering starting at: %s" % gaps)

# artifacts
no_art = [c["n"] for c in chapters if c["artifact_count"] == 0]
multi_art = [(c["n"], c["artifact_count"]) for c in chapters if c["artifact_count"] > 1]
if no_art:
    anomalies.append("Chapters with no artifact box: %s" % no_art)
if multi_art:
    anomalies.append("Chapters with >1 artifact box: %s" % multi_art)

# artifact numbering vs chapter
for c in chapters:
    if c["artifact_title"]:
        m = re.match(r"Artifact (\d+)\.(\d+)", c["artifact_title"])
        if m and int(m.group(1)) != c["n"]:
            anomalies.append("Chapter %d contains %s (number mismatch)" % (c["n"], c["artifact_title"]))

# duplicate box titles across book
dups = {k: v for k, v in box_registry.items() if v > 1}
if dups:
    anomalies.append("Duplicate box titles: %s" % dups)

# duplicate box NUMBERS (same kind + same N.k, different text)
numreg = collections.Counter()
for c in chapters:
    for t in c["definitions"] + c["theorems"] + c["examples"] + ([c["artifact_title"]] if c["artifact_title"] else []):
        m = re.match(r"(Definition|Theorem|Example|Artifact) (\d+)\.(\d+)", t)
        if m:
            numreg[(m.group(1), int(m.group(2)), int(m.group(3)))] += 1
dupnum = {"%s %d.%d" % k: v for k, v in numreg.items() if v > 1}
if dupnum:
    anomalies.append("Duplicate box NUMBERS: %s" % dupnum)

# >1 encrypted note in a chapter
multi_enc = [(c["n"], c["encrypted_note_count"]) for c in chapters if c["encrypted_note_count"] > 1]
if multi_enc:
    anomalies.append("Chapters with >1 'Under Encryption' note: %s" % multi_enc)

# chapters with zero definitions / theorems / examples
for kind in ("definitions", "theorems", "examples"):
    empt = [c["n"] for c in chapters if not c[kind]]
    if empt:
        anomalies.append("Chapters with zero %s: %s" % (kind, empt))

# TOC title vs rendered header title mismatch
for c in chapters:
    tt = toc_titles.get(c["n"])
    if tt and tt != c["title"]:
        anomalies.append("Ch%d title mismatch: TOC '%s' vs header '%s'" % (c["n"], tt, c["title"]))

# per-chapter internal numbering of Definition/Theorem/Example
for c in chapters:
    for kind, lst in (("Definition", c["definitions"]), ("Theorem", c["theorems"]), ("Example", c["examples"])):
        nums = []
        for t in lst:
            m = re.match(r"%s (\d+)\.(\d+)" % kind, t)
            if m:
                if int(m.group(1)) != c["n"]:
                    anomalies.append("Ch%d: '%s' has wrong chapter prefix" % (c["n"], t))
                nums.append(int(m.group(2)))
            else:
                anomalies.append("Ch%d: unnumbered %s box title '%s'" % (c["n"], kind, t))
        if nums != list(range(1, len(nums) + 1)):
            anomalies.append("Ch%d %s numbering not 1..n: %s" % (c["n"], kind, nums))

# section numbering
for c in chapters:
    snums = []
    for s in c["sections"]:
        m = re.match(r"(\d+)\.(\d+)\s", s)
        if m:
            if int(m.group(1)) != c["n"]:
                anomalies.append("Ch%d section '%s' has wrong prefix" % (c["n"], s[:40]))
            snums.append(int(m.group(2)))
        else:
            anomalies.append("Ch%d unnumbered h4: '%s'" % (c["n"], s[:60]))
    if snums != list(range(1, len(snums) + 1)):
        anomalies.append("Ch%d section numbering not contiguous: %s" % (c["n"], snums))

# read-alongside / structural boxes present everywhere?
no_ra = [c["n"] for c in chapters if not c["read_alongside"]]
if no_ra:
    anomalies.append("Chapters missing read-alongside box: %s" % no_ra)
no_lo = [c["n"] for c in chapters if not c["learning_objectives"]]
if no_lo:
    anomalies.append("Chapters missing learning objectives: %s" % no_lo)
no_kt = [c["n"] for c in chapters if not c["key_takeaways"]]
if no_kt:
    anomalies.append("Chapters missing key takeaways: %s" % no_kt)

# empty placeholder regions
for marker in re.findall(r"<!-- ([A-Z_]+)_START -->(.*?)<!-- \1_END -->", raw, re.S):
    if not clean(marker[1]):
        anomalies.append("Placeholder region <!-- %s_START/END --> is EMPTY (no content emitted)" % marker[0])

# appendix anchors referenced by TOC but absent
for a in toc_appendices:
    anchor = "appendix-%s" % a["letter"].lower()
    if 'id="%s"' % anchor not in raw:
        anomalies.append("TOC links #%s ('%s') but no element with that id exists" % (anchor, a["title"]))

# part headers vs toc parts
toc_part_titles = [p["title"] for p in toc_parts]
if len(part_headers) != len([t for t in toc_part_titles if t.startswith("Part")]):
    anomalies.append("TOC lists %d Parts but only %d part-header blocks rendered (%s)"
                     % (len([t for t in toc_part_titles if t.startswith("Part")]),
                        len(part_headers), [p["label"] for p in part_headers]))

# ---------------- unassembled draft chapters (Part VIII) ----------------
def parse_chapter_html(body, n):
    hm = re.search(r'<div class="ch-label">Chapter (\d+)</div><h3>(.*?)</h3>', body, re.S)

    def boxes(cls):
        return [clean(m.group(1)) for m in
                re.finditer(r'<div class="%s"[^>]*>\s*<div class="box-title">(.*?)</div>' % cls, body, re.S)]

    ra = re.search(r'<div class="read-alongside">(.*?)</div>', body, re.S)
    lob = re.search(r'<div class="learning-objectives">(.*?)</ul>', body, re.S)
    ktb = re.search(r'<div class="key-takeaways">(.*?)</ul>', body, re.S)
    arts = boxes("artifact-box")
    return {
        "n": n,
        "title": clean(hm.group(2)) if hm else None,
        "sections": [clean(x) for x in re.findall(r"<h4>(.*?)</h4>", body, re.S)],
        "subsections": [clean(x) for x in re.findall(r"<h5>(.*?)</h5>", body, re.S)],
        "artifact_title": arts[0] if arts else None,
        "artifact_count": len(arts),
        "definitions": boxes("definition"),
        "theorems": boxes("theorem"),
        "examples": boxes("example"),
        "warning_boxes": boxes("warning-box"),
        "has_encrypted_note": 'class="encrypted-note"' in body,
        "encrypted_note_count": body.count('class="encrypted-note"'),
        "read_alongside": clean(ra.group(1)) if ra else None,
        "learning_objectives": [clean(x) for x in re.findall(r"<li>(.*?)</li>", lob.group(1), re.S)] if lob else [],
        "key_takeaways": [clean(x) for x in re.findall(r"<li>(.*?)</li>", ktb.group(1), re.S)] if ktb else [],
        "tags": {t: len(re.findall(r'class="tag tag-%s"' % t, body)) for t in
                 ("certain", "likely", "verify", "contested")},
        "part": "Part VIII",
        "rendered_in_book": False,
        "source_file": None,
    }

unassembled = []
if os.path.isdir(DRAFTS):
    for f in sorted(glob.glob(os.path.join(DRAFTS, "ch*.html"))):
        n = int(re.search(r"ch(\d+)\.html", f).group(1))
        if n in present:
            continue
        rec = parse_chapter_html(open(f, encoding="utf-8").read(), n)
        rec["source_file"] = f
        unassembled.append(rec)

if unassembled:
    got = sorted(r["n"] for r in unassembled)
    anomalies.append(
        "Chapters %s exist as finished drafts in %s but were NEVER inserted into ml-book.html "
        "(the PART_VIII_START/END region is empty) -- this is an assembly gap, not missing writing" % (got, DRAFTS))
    still = [n for n in missing if n not in got]
    if still:
        anomalies.append("Chapter(s) %s promised by the TOC have NO draft anywhere: never written" % still)

# artifact code files on disk
code_files = sorted(int(re.search(r"ch(\d+)_artifact\.py", os.path.basename(p)).group(1))
                    for p in glob.glob(os.path.join(CODEDIR, "ch*_artifact.py")))
code_missing = [n for n in expected_toc if n not in code_files]
if code_missing:
    anomalies.append("No artifact code file ch%s_artifact.py in %s for chapter(s) %s"
                     % ("NN", CODEDIR, code_missing))

index = {
    "source": SRC,
    "book_title": clean(re.search(r"<title>(.*?)</title>", raw).group(1)),
    "chapters": chapters,
    "unassembled_chapters": unassembled,
    "artifact_code_files": {"dir": CODEDIR, "chapters_with_code": code_files,
                            "chapters_without_code": code_missing},
    "parts": parts,
    "tags": tags,
    "notation_symbols": notation,
    "notation_symbol_counts": notation_counts,
    "toc_promised": {
        "parts": toc_part_titles,
        "chapter_numbers": expected_toc,
        "chapter_titles": {str(k): v for k, v in toc_titles.items()},
        "appendices": toc_appendices,
    },
    "counts": {
        "chapters_rendered": len(chapters),
        "chapters_promised_by_toc": len(expected_toc),
        "chapters_drafted_but_unassembled": len(unassembled),
        "chapters_never_written": len([n for n in missing if n not in [r["n"] for r in unassembled]]),
        "artifacts_rendered": sum(c["artifact_count"] for c in chapters),
        "artifacts_in_unassembled_drafts": sum(c["artifact_count"] for c in unassembled),
        "artifacts_total_existing": sum(c["artifact_count"] for c in chapters) + sum(c["artifact_count"] for c in unassembled),
        "artifacts_promised_by_toc": len(expected_toc),
        "parts_rendered": len(part_headers),
        "parts_promised_by_toc": len([t for t in toc_part_titles if t.startswith("Part")]),
        "appendices_rendered": 0 + sum(1 for a in toc_appendices if 'id="appendix-%s"' % a["letter"].lower() in raw),
        "appendices_promised_by_toc": len(toc_appendices),
        "definitions": sum(len(c["definitions"]) for c in chapters),
        "theorems": sum(len(c["theorems"]) for c in chapters),
        "examples": sum(len(c["examples"]) for c in chapters),
        "warning_boxes": sum(len(c["warning_boxes"]) for c in chapters),
        "encrypted_notes": sum(c["encrypted_note_count"] for c in chapters),
        "read_alongside_boxes": sum(1 for c in chapters if c["read_alongside"]),
        "notation_symbols": len(notation),
    },
    "anomalies": anomalies,
}

json.dump(index, open(OUT, "w"), indent=1, ensure_ascii=False)
print("wrote", OUT)
print(json.dumps(index["counts"], indent=1))
print("\n--- ANOMALIES (%d) ---" % len(anomalies))
for a in anomalies:
    print(" *", a)
print("\n--- notation (%d) ---" % len(notation))
print(", ".join(notation[:120]))
