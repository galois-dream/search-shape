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

# 2026-11-05 CET (a co-author's note on N=50): neighbourhoods cut by distance instead of count. Keep the neighbours with
# ||c-q|| / min ||c-q|| - 1 <= eps (the epsilon rule of the preprint, as features.py's n_e04), at least 4 of them (persistence
# needs >3 points; queries with fewer are dropped and counted) and at most the 50 retrieved (never the corpus as a whole;
# queries whose ball holds all 50 are counted as capped). W1 is read from the prefix table at that n.
eps_list = (0.05, 0.1, 0.2, 0.4); res = defaultdict(list); ns = defaultdict(list); drop = defaultdict(int); cap = defaultdict(int); tot = defaultdict(int)
for root, name in [(".", "nobel"), ("fields", "fields"), ("turing", "turing"), ("gutenberg", "essays"), ("wiki", "wiki")]:
    for f in sorted(glob.glob(root + "/prefix_*.csv.gz")):
        q = defaultdict(list)
        for r in csv.DictReader(gzip.open(f, "rt")):
            if r["variant"] == "primary":
                q[r["cell"], r["query_id"]].append((int(r["n"]), float(r["ratio"]), float(r["w1"])))
        for e in eps_list:
            g = defaultdict(list)
            for (c, _), rows in q.items():
                rows.sort(); tot[e, c] += 1
                inside = [x for x in rows if x[1] <= e]
                if not inside or inside[-1][0] < 4: drop[e, c] += 1; continue
                n, _, w = inside[-1]; g[c].append(w); ns[e, c].append(n); cap[e, c] += n == 50
            a, b = g.get("Q250|C750"), g.get("Q750|C250")
            if a and b: res[e].append((name, auc(b, a)))
for e in eps_list:
    v = [x for _, x in res[e]]
    print(f"eps={e}: n={len(v)} AUROC multi>poly median {st.median(v):.2f} range {min(v):.2f}-{max(v):.2f}; >0.5 in {sum(x > 0.5 for x in v)}; "
          + "; ".join(f"{c}: median n {st.median(ns[e, c]):.0f}, dropped (<4) {drop[e, c] / tot[e, c]:.0%}, capped at 50 {cap[e, c] / max(len(ns[e, c]), 1):.0%}" for c in ("Q250|C750", "Q750|C250")))
