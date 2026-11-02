# wiki_select_remote.py -- run wherever the AmbigQA Wikipedia docs.db (enwiki 2020-01-20) is available; opens it
# read-only. Walks a seed-0 random permutation of documents.rowid (numpy default_rng(0).permutation) and emits, in
# permutation order, pages that are not disambiguation/list pages and pass a conservative character prefilter
# (len(text) >= MIN_CHARS). Exact cl100k token eligibility and the final take-first-40 happen in
# wiki_select_local.py, so the result is a uniform random sample (seed 0) of eligible pages.
# Usage: python wiki_select_remote.py DB N_CANDIDATES MIN_CHARS > wiki_cand.jsonl  (published: 250, 12000)
import json, sqlite3, sys
import numpy as np

db_path, n_cand, min_chars = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
nmax = db.execute("SELECT max(rowid) FROM documents").fetchone()[0]
perm = np.random.default_rng(0).permutation(nmax) + 1
got = scanned = skipped_disamb = skipped_list = missing = 0
for rank, rid in enumerate(perm):
    row = db.execute("SELECT id, text FROM documents WHERE rowid = ?", (int(rid),)).fetchone()
    scanned += 1
    if row is None:
        missing += 1; continue
    pid, text = row
    title = text.split("\n", 1)[0].strip()
    if "(disambiguation)" in title.lower() or "may refer to" in text[:1000].lower():
        skipped_disamb += 1; continue
    if title.lower().startswith(("list of", "lists of")):
        skipped_list += 1; continue
    if len(text) < min_chars:
        continue
    print(json.dumps(dict(rank=rank, rowid=int(rid), id=pid, title=title, chars=len(text), text=text)), flush=True)
    got += 1
    if got >= n_cand:
        break
print(json.dumps(dict(_summary=True, scanned=scanned, nmax=int(nmax), missing=missing, skipped_disamb=skipped_disamb,
                      skipped_list=skipped_list, candidates=got, min_chars=min_chars)), flush=True)
