# kcut.py 2026-11-02 13:40 CET (run from nobel/out_clean): does the W1 signature hold at retrieval depths k=10, 20 (prefix features), mismatched cells?
import csv, gzip, glob, statistics as st
from collections import defaultdict
def auc(x, y):  # P(x > y) + .5 P(=), rank-based
    allv = sorted([(v, 0) for v in x] + [(v, 1) for v in y]); r = {}; i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]: j += 1
        for t in range(i, j): r.setdefault(allv[t][0], (i + j + 1) / 2)
        i = j
    sx = sum(r[v] for v in x); return (sx - len(x) * (len(x) + 1) / 2) / (len(x) * len(y))
res = defaultdict(list)
for root, name in [(".", "nobel"), ("fields", "fields"), ("turing", "turing"), ("gutenberg", "essays"), ("wiki", "wiki")]:
    for f in sorted(glob.glob(root + "/prefix_*.csv.gz")):
        g = defaultdict(list)
        for r in csv.DictReader(gzip.open(f, "rt")):
            if r["variant"] == "primary" and r["n"] in ("10", "20", "50"):
                g[r["cell"], r["n"]].append(float(r["w1"]))
        for k in ("10", "20", "50"):
            a, b = g.get(("Q250|C750", k)), g.get(("Q750|C250", k))
            if a and b: res[k].append((name, auc(b, a)))
for k in ("10", "20", "50"):
    v = [x for _, x in res[k]]; byc = defaultdict(list)
    for n, x in res[k]: byc[n].append(x)
    print(f"k={k}: n={len(v)} AUROC multi>poly median {st.median(v):.2f} range {min(v):.2f}-{max(v):.2f}; >0.5 in {sum(x>0.5 for x in v)}; corpus medians " + ", ".join(f"{c} {st.median(x):.2f}" for c, x in byc.items()))
