# wiki_select_local.py -- from the permutation-ordered candidates ($RT_WORK/wiki/raw/wiki_cand.jsonl, see
# wiki_select_remote.py) keep the first 40 pages whose cl100k_base token count (after html.unescape) is >= the median
# Nobel lecture (5346 tokens); write wiki/text/w<rowid>.txt and wiki/selection.csv (rank, rowid, doc_id, title,
# chars, tokens). Uniform random sample of eligible pages, seed 0.
import os
import csv, html, json, re
from pathlib import Path
import tiktoken

ROOT = Path(os.environ.get("RT_WORK", str(Path(__file__).resolve().parent / "work")))  # see README
W = ROOT / "wiki"; (W / "text").mkdir(parents=True, exist_ok=True)
MEDIAN_NOBEL = 5346
enc = tiktoken.get_encoding("cl100k_base")
cands = [json.loads(l) for l in open(W / "raw" / "wiki_cand.jsonl")]
summ = cands.pop()
sel, cpt = [], []
for c in cands:
    text = re.sub(r"[ \t]+", " ", html.unescape(c["text"])).strip()
    n = len(enc.encode(text, disallowed_special=()))
    cpt.append(len(text) / n)
    if n >= MEDIAN_NOBEL and len(sel) < 40:
        sel.append(dict(rank=c["rank"], rowid=c["rowid"], doc_id=c["id"], title=html.unescape(c["title"]),
                        chars=len(text), tokens=n))
        (W / "text" / f"w{c['rowid']}.txt").write_text(c["text"])
assert len(sel) == 40, f"only {len(sel)} eligible among candidates; fetch more"
with open(W / "selection.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(sel[0])); w.writeheader(); w.writerows(sel)
last = sel[-1]["rank"]
print(f"selected 40; last permutation rank {last}; candidates examined up to it: "
      f"{sum(c['rank'] <= last for c in cands)}; remote summary {summ}")
print(f"chars/token among candidates: min {min(cpt):.2f} median {sorted(cpt)[len(cpt)//2]:.2f} "
      f"(prefilter {summ['min_chars']} chars => safe if min chars/token > {summ['min_chars']/MEDIAN_NOBEL:.2f})")
