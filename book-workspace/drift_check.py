"""Extract every artifact listing from the book and diff it against the .py that ran."""
import html, re, os, difflib
BOOK = "/workspace/ml-dl-llm-books/ml-book.html"
CODE = "/tmp/claude-0/-home-user-ParthN9i4/8320e92f-b068-5164-8e36-b20b4cf863b1/scratchpad/code"
book = open(BOOK).read()
chapters = re.findall(r'<div class="chapter" id="ch(\d+)">(.*?)(?=<div class="chapter" id="|<div class="part-header">|<!-- APPENDICES_START)', book, re.S)
drift, missing, ok = [], [], 0
for num, body in chapters:
    n = int(num)
    pres = re.findall(r'<pre><code>(.*?)</code></pre>', body, re.S)
    if not pres:
        missing.append(n); continue
    embedded = html.unescape(max(pres, key=len)).strip()
    path = f"{CODE}/ch{n:02d}_artifact.py"
    if not os.path.exists(path):
        missing.append(n); continue
    actual = open(path).read().strip()
    # normalise trailing whitespace per line
    e = [l.rstrip() for l in embedded.splitlines()]
    a = [l.rstrip() for l in actual.splitlines()]
    if e == a:
        ok += 1
    else:
        sm = difflib.SequenceMatcher(None, e, a)
        drift.append((n, len(e), len(a), round(sm.ratio(), 4)))
print(f"byte-identical : {ok}/42")
print(f"no listing found: {missing}" if missing else "no listing found: none")
if drift:
    print(f"\nDRIFTED ({len(drift)}):")
    for n, le, la, r in drift:
        print(f"  ch{n:02d}: embedded {le} lines vs file {la} lines, similarity {r}")
else:
    print("drift          : none")
