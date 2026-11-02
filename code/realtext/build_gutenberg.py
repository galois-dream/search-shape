# build_gutenberg.py -- build the Gutenberg real-text corpus ("Essays" in the paper): 40 long-form, US-public-domain,
# English science / philosophy non-fiction books from Project Gutenberg. Light, local, no models.
# Selection:
#   1. Official offline catalogue pg_catalog.csv (downloaded once; access time recorded).
#   2. Eligible = Type 'Text', Language exactly 'en', and an LoCC code starting with Q, or with B but not a religion
#      subclass BL-BX (books with no LoCC: a bookshelf containing 'science'/'philosoph', not 'Science Fiction'),
#      minus the metadata exclusions in EXCLUDE, minus books without a single named primary author.
#   3. Order: eligible list sorted by Text# ascending, then random.Random(0).shuffle().
#   4. Walk that order; skip an author already selected; confirm 'Public domain in the USA.' in the book's official
#      RDF record; download https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt (one request at a time, >=1 s apart;
#      cached in raw/); strip the PG header/footer (*** START OF / *** END OF markers); light cleaning; tokenise with
#      tiktoken cl100k_base; keep if body >= 6,000 tokens. Stop at 40.
#   5. Window: skip ceil(10%) of body tokens, keep C = floor(min(remainder, CAP) / 750) * 750 contiguous tokens.
#      CAP = Nobel lecture p90 (from the Nobel manifest's token counts) floored to a multiple of 750 (= 15,750).
# Outputs ($RT_WORK/gutenberg/): text/<id>.txt (input to prep.py with CORPUS=gutenberg), manifest.csv,
# selection_log.csv. Usage: python build_gutenberg.py
import csv, hashlib, math, random, re, sys, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import tiktoken

import os
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("RT_WORK", str(HERE / "work"))) / "gutenberg"
CAT, RAW, OUT = ROOT / "catalog", ROOT / "raw", ROOT / "text"
for d in (CAT, RAW, OUT):
    d.mkdir(parents=True, exist_ok=True)
NOBEL_MANIFEST = Path(os.environ.get("NOBEL_MANIFEST", str(HERE.parents[1] / "data" / "realtext" / "nobel" / "lecture_manifest.csv")))
CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
N_BOOKS, MIN_BODY, UNIT, SEED = 40, 6000, 750, 0
UA = "research-corpus-builder/0.1 (polite single-threaded)"
enc = tiktoken.get_encoding("cl100k_base")
now = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def nobel_cap():
    """90th percentile (linear interpolation, = numpy default) of Nobel lecture token lengths, floored to 750."""
    rows = [l for l in open(NOBEL_MANIFEST) if not l.startswith("#")]
    t = sorted(int(r.get("tokens_cl100k") or r["tokens"]) for r in csv.DictReader(rows))
    h = 0.9 * (len(t) - 1)
    p90 = t[int(h)] + (h - int(h)) * (t[int(h) + 1] - t[int(h)])
    return int(p90 // UNIT * UNIT), p90, len(t)


CAP, P90, N_NOBEL = nobel_cap()
print(f"CAP = {CAP} (Nobel p90 = {P90:.1f} tokens over {N_NOBEL} lectures)")

_last = [0.0]


def get(url):
    """One request at a time, >= 1 s after the previous one."""
    time.sleep(max(0.0, 1.0 - (time.time() - _last[0])))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()
    finally:
        _last[0] = time.time()


if not (CAT / "pg_catalog.csv").exists():
    (CAT / "pg_catalog.access_utc").write_text(now() + "\n")
    (CAT / "pg_catalog.csv").write_bytes(get(CATALOG_URL))
CATALOG_ACCESS = (CAT / "pg_catalog.access_utc").read_text().strip()

# ---- eligibility
RELIGION = re.compile(r"^B[L-X]")
EXCLUDE = [  # matched case-insensitively on subjects + bookshelves (+ title where noted)
    ("fiction", r"fiction|\bnovels?\b|short stories|fairy tales|fables"),
    ("poetry", r"poetry|\bpoems\b|\bverse\b|sonnets"),
    ("drama", r"drama|\bplays\b|tragedies|comedies"),
    ("periodical", r"periodical|magazine"),
    ("index", r"\bindex(es)?\b"),
    ("dictionary/reference", r"dictionar|encyclop|glossar|thesaur"),
    ("cookbook", r"cook(ery|ing|books?)\b|recipes"),
    ("juvenile/children's", r"juvenile|children"),
    ("scripture", r"bible|sacred books|koran|qur.?an|\bvedas\b|upanishad|bhagavad|scriptures|talmud|dhammapada|"
                  r"tripitaka|tao te ching"),
]
TITLE_CHECKED = {"index", "dictionary/reference", "scripture"}


def authors_of(field):
    """Primary authors = entries without a [role] tag; dates kept in the key so namesakes stay distinct."""
    return [a.strip() for a in field.split(";") if a.strip() and "[" not in a]


def eligibility(r):
    if r["Type"] != "Text" or r["Language"].strip() != "en":
        return "not English Text"
    locc = [c.strip() for c in r["LoCC"].split(";") if c.strip()]
    shelves = [s.strip().lower() for s in r["Bookshelves"].split(";") if s.strip()]
    # LoCC (curated) decides when present; PG's auto-generated 'Category:' bookshelves are only a fallback for books
    # without an LoCC (with bookshelves as an OR they admitted e.g. a Scots biography and a settlement-house memoir).
    if locc:
        sci_phil = any(c.startswith("Q") or (c.startswith("B") and not RELIGION.match(c)) for c in locc)
    else:
        sci_phil = any(("science" in s and "science fiction" not in s) or "philosoph" in s for s in shelves)
    if not sci_phil:
        return "not science/philosophy"
    fields = (r["Subjects"] + " || " + r["Bookshelves"]).lower()
    title = r["Title"].lower()
    for name, pat in EXCLUDE:
        if re.search(pat, fields) or (name in TITLE_CHECKED and re.search(pat, title)):
            return f"excluded: {name}"
    a = authors_of(r["Authors"])
    if not a:
        return "excluded: no named primary author"
    if any(re.match(r"(various|anonymous|unknown)\b", x.lower()) for x in a):
        return "excluded: collective/anonymous author"
    return None


rows = list(csv.DictReader(open(CAT / "pg_catalog.csv", encoding="utf-8")))
eligible = sorted((r for r in rows if eligibility(r) is None), key=lambda r: int(r["Text#"]))
print(f"catalogue rows {len(rows)}; eligible {len(eligible)}")
order = eligible[:]
random.Random(SEED).shuffle(order)

# ---- text handling
START = re.compile(r"^\*\*\* ?START OF.*$", re.M)
END = re.compile(r"^\*\*\* ?END OF", re.M)


def body_text(raw):
    s, e = START.search(raw), END.search(raw)
    if not s or not e or e.start() <= s.end():
        return None
    t = raw[s.end():e.start()].replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+\n", "\n", t)                 # trailing spaces
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)           # unwrap hard-wrapped lines inside paragraphs
    t = re.sub(r"\n{2,}", "\n\n", t)                 # one blank line between paragraphs
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    return t.strip()


def public_domain_usa(gid):
    rdf = get(f"https://www.gutenberg.org/ebooks/{gid}.rdf").decode("utf-8", errors="replace")
    m = re.search(r"<dcterms:rights>(.*?)</dcterms:rights>", rdf, re.S)
    return bool(m and m.group(1).strip().startswith("Public domain in the USA"))


log, chosen, used = [], [], set()
for k, r in enumerate(order, 1):
    if len(chosen) == N_BOOKS:
        break
    gid, authors = int(r["Text#"]), authors_of(r["Authors"])
    row = dict(shuffle_rank=k, gutenberg_id=gid, title=" ".join(r["Title"].split()), author="; ".join(authors))
    if used & set(authors):
        log.append({**row, "decision": "skipped: author already selected"})
        continue
    url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
    f = RAW / f"{gid}.txt"
    if not f.exists():
        if not public_domain_usa(gid):
            log.append({**row, "decision": "excluded: RDF rights not 'Public domain in the USA.'"})
            continue
        try:
            data = get(url)
        except urllib.error.HTTPError as e:  # e.g. TeX-sourced books have no pg<id>.txt
            log.append({**row, "decision": f"excluded: plain text unavailable (HTTP {e.code})"})
            continue
        (RAW / f"{gid}.access_utc").write_text(now() + "\n")
        f.write_bytes(data)
    body = body_text(f.read_bytes().decode("utf-8", errors="replace"))
    if body is None:
        log.append({**row, "decision": "excluded: START/END markers not found"})
        continue
    ids = enc.encode(body, disallowed_special=())
    n = len(ids)
    if n < MIN_BODY:
        log.append({**row, "decision": f"excluded: body {n} tokens < {MIN_BODY}"})
        continue
    start = math.ceil(0.1 * n)
    C = min(n - start, CAP) // UNIT * UNIT
    window = enc.decode_bytes(ids[start:start + C]).decode("utf-8", errors="ignore")  # no U+FFFD at the edges
    (OUT / f"{gid}.txt").write_text(window, encoding="utf-8")
    used |= set(authors)
    log.append({**row, "decision": "selected"})
    chosen.append(dict(gutenberg_id=gid, title=row["title"], author=row["author"],
                       subjects=r["Subjects"], locc=r["LoCC"], bookshelves=r["Bookshelves"], shuffle_rank=k,
                       access_utc=(RAW / f"{gid}.access_utc").read_text().strip(), catalog_access_utc=CATALOG_ACCESS,
                       body_tokens=n, window_start_token=start, window_tokens=C, url=url,
                       retok_tokens=len(enc.encode(window, disallowed_special=())),
                       text_sha256=hashlib.sha256(window.encode()).hexdigest()))
    print(f"[{len(chosen):2d}] #{k} {gid} {authors[0][:30]} -- {row['title'][:50]}  body={n} C={C}", flush=True)

for name, data in (("manifest.csv", chosen), ("selection_log.csv", log)):
    with open(ROOT / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
print(f"python {sys.version.split()[0]}; selected {len(chosen)}; total window tokens "
      f"{sum(c['window_tokens'] for c in chosen)}")
