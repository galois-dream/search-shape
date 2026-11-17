# Shipped copy of paper/trec_p2/build_index.py; only this line and the host name changed for the repo layout (PREREG.md deviation 7 = docs/analysis_plan_deviation7.md).
# build_index.py -- written 2026-11-14 09:34 CET. P2 (PREREG.md deviation 7), step 1 of the primary corpus B: a BM25
# index (SQLite FTS5, k1 = 1.2, b = 0.75) over the title and first paragraph of every page of the AmbigQA Wikipedia dump
# (2020-01-20, docs.db). Uses no TREC data and no labels, so it may run before the deviation is filed.
# Usage: python build_index.py DOCS_DB OUT_DB      (run at nice 19)
import sqlite3, sys, time

src, dst = sys.argv[1], sys.argv[2]
S = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
D = sqlite3.connect(dst)
D.execute("CREATE VIRTUAL TABLE fts USING fts5(field, tokenize = 'unicode61')")   # rowid = the dump's page id
D.execute("CREATE TABLE meta (id INTEGER PRIMARY KEY, title TEXT, disamb INTEGER)")
n, t0, batch_f, batch_m = 0, time.time(), [], []
for pid, text in S.execute("SELECT id, text FROM documents"):
    parts = text.split("\n\n", 2)
    title = parts[0].strip(); first = parts[1].strip() if len(parts) > 1 else ""
    dis = int(title.endswith("(disambiguation)") or "may refer to" in first)
    batch_f.append((int(pid), f"{title} {first}")); batch_m.append((int(pid), title, dis)); n += 1
    if len(batch_f) == 50_000:
        D.executemany("INSERT INTO fts(rowid, field) VALUES (?, ?)", batch_f); D.executemany("INSERT INTO meta VALUES (?, ?, ?)", batch_m)
        D.commit(); batch_f, batch_m = [], []
        print(f"{n} pages, {time.time() - t0:.0f} s", flush=True)
D.executemany("INSERT INTO fts(rowid, field) VALUES (?, ?)", batch_f); D.executemany("INSERT INTO meta VALUES (?, ?, ?)", batch_m)
D.commit(); D.execute("INSERT INTO fts(fts) VALUES ('optimize')"); D.commit()
print(f"done: {n} pages, {D.execute('SELECT sum(disamb) FROM meta').fetchone()[0]} disambiguation, {time.time() - t0:.0f} s", flush=True)
