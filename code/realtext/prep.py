# prep.py -- text cleaning, cl100k_base tokenisation and aligned C250/C750 chunking. Light, local.
# CORPUS=nobel (default): $RT_WORK/text/*.txt -> $RT_WORK/data/, with page header/footer cleaning (PDF-derived text;
# CORPUS=turing uses the same rule on $RT_WORK/turing/text).
# Other corpora (wiki, gutenberg, fields): $RT_WORK/<corpus>/text (or $TXT_DIR) -> $RT_WORK/<corpus>/data, no
# page-furniture cleaning, optional token cap CAP_TOKENS per document. Outputs: chunks_c250.jsonl, chunks_c750.jsonl
# (document, chunk idx, token span, text) and lecture_tokens.csv (per-document token counts, lines dropped as
# header/footer). The extraction-artefact filter is applied later, in features.py (CLEAN=1).
# Usage: [CORPUS=wiki CAP_TOKENS=15750] python prep.py
import os
import collections, csv, json, re, sys
from pathlib import Path
import tiktoken

ROOT = Path(os.environ.get("RT_WORK", str(Path(__file__).resolve().parent / "work")))  # see README
# CORPUS=nobel (default): text/ -> data/, with header/footer cleaning. Other corpora (wiki, gutenberg): <corpus>/text
# -> <corpus>/data, no page-furniture cleaning (plain text, no pages), optional token cap CAP_TOKENS per document.
import html, os
CORPUS = os.environ.get("CORPUS", "nobel")
BASE = ROOT if CORPUS == "nobel" else ROOT / CORPUS
TXT, DATA = Path(os.environ.get("TXT_DIR", BASE / "text")), BASE / "data"  # TXT_DIR: read texts in place
CAP = int(os.environ.get("CAP_TOKENS", "0"))
DATA.mkdir(exist_ok=True)
enc = tiktoken.get_encoding("cl100k_base")


def norm_key(line):  # header/footer key: drop digits, spaces, case
    return re.sub(r"[\d\s]", "", line).lower()


def clean(raw):
    """Drop trivially detectable page furniture. Only lines in the header/footer zone (first 2 and last 2
    non-empty lines of each page, pages split on form feed) are candidates; a candidate is dropped if it is
    digit-only (page number) or its digit/space-stripped form recurs >=3 times in that zone across the
    lecture (running heads like '1922 N.BOHR')."""
    pages = [[l.strip() for l in pg.split("\n")] for pg in raw.split("\f")]
    zone = set()
    for pi, pg in enumerate(pages):
        nz = [i for i, l in enumerate(pg) if l]
        zone |= {(pi, i) for i in nz[:2] + nz[-2:]}
    keys = collections.Counter(norm_key(pages[pi][i]) for pi, i in zone)
    out, dropped = [], 0
    for pi, pg in enumerate(pages):
        for i, s in enumerate(pg):
            if (pi, i) in zone and (re.fullmatch(r"[\d\s.\-–]+", s) or (keys[norm_key(s)] >= 3 and norm_key(s))):
                dropped += 1
                continue
            out.append(s)
    text = "\n".join(out)
    text = re.sub(r"-\n(?=[a-z])", "", text)        # re-join hyphenated line breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)     # unwrap lines inside paragraphs
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text, dropped


def dec(ids):  # bytes-level decode; drop partial multibyte chars at chunk edges (no U+FFFD)
    return enc.decode_bytes(ids).decode("utf-8", errors="ignore")


rows, c250, c750 = [], [], []
for f in sorted(TXT.glob("*.txt")):
    if f.name.endswith(".raw.txt"):
        continue
    lec = f.stem
    if CORPUS in ("nobel", "turing"):  # same pdftotext + header/footer rule as Nobel
        text, dropped = clean(f.read_text(errors="ignore"))
    else:
        text, dropped = re.sub(r"[ \t]+", " ", html.unescape(f.read_text(errors="ignore"))).strip(), 0
    ids = enc.encode(text, disallowed_special=())
    full_tokens = len(ids)
    if CAP:
        ids = ids[:CAP]
    n750 = len(ids) // 750
    n250 = len(ids) // 250
    rows.append(dict(lecture=lec, tokens=len(ids), tokens_uncapped=full_tokens, words=len(text.split()), dropped_lines=dropped,
                     n_c250=n250, n_c750=n750))
    for i in range(n250):
        a, b = 250 * i, 250 * (i + 1)
        c250.append(dict(id=f"{lec}#250-{i}", lecture=lec, idx=i, start=a, end=b, text=dec(ids[a:b])))
    for j in range(n750):
        a, b = 750 * j, 750 * (j + 1)
        c750.append(dict(id=f"{lec}#750-{j}", lecture=lec, idx=j, start=a, end=b, text=dec(ids[a:b])))

for name, rs in (("chunks_c250.jsonl", c250), ("chunks_c750.jsonl", c750)):
    with open(DATA / name, "w") as fh:
        for r in rs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(DATA / "lecture_tokens.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
tot = sum(r["tokens"] for r in rows)
print(f"lectures={len(rows)} tokens={tot} C250={len(c250)} C750={len(c750)}")
print(f"C250 tokens embedded={250*len(c250)} C750 tokens embedded={750*len(c750)}")
print(f"chars C250={sum(len(r['text']) for r in c250)} C750={sum(len(r['text']) for r in c750)}")
# alignment check: every C750 chunk spans exactly three consecutive C250 chunks of the same lecture
s250 = {(r["lecture"], r["start"], r["end"]) for r in c250}
assert all(all((r["lecture"], r["start"] + 250 * t, r["start"] + 250 * (t + 1)) in s250 for t in range(3)) for r in c750)
print("alignment ok")
