# samples_rule.py -- written on day 6 at 20:35. Data samples from the equal-length page test,
# by a fixed rule (no hand choice): bge-base, grouping g0, arm "all"; numpy default_rng(0), permutation of the 240
# queries of each cell, first two: polysemic-like Q1|C3 (a one-page text searched against three-page texts) and
# multi-factual-like Q3|C1 (a three-page text against one-page texts). Neighbours recomputed with feats.py's rule
# (top 50 by cosine, stable sort). For each query: its pages, the first 5 neighbours' pages, how many of the 50
# neighbours reach the overlap bound (polysemic: contain the query's page; multi-factual: pages all inside the query's),
# and the features from feats.csv. Usage: python samples_rule.py OUT_DIR  -> OUT_DIR/samples_rule.json
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(sys.argv[1]); T = json.load(open(OUT / "texts.json")); PAGE = np.array(T["page_of"]); NAMES = T["pages"]
PAGES = {k: [frozenset(int(PAGE[a]) for a in grp) for grp in v] for k, v in T["sets"].items()}
E = np.load(OUT / "emb_bge-base.npz"); F = pd.read_csv(OUT / "feats.csv")
F = F[(F.model == "bge-base") & (F.g == 0) & (F.arm == "all")]
rng = np.random.default_rng(0); rows = []; med = {}
for label, qk, ck, cellname in (("polysemic-like", "Q1", "C3_g0", "Q1|C3"), ("multi-factual-like", "Q3_g0", "C1", "Q3|C1")):
    Q, C = E[qk].astype(np.float64), E[ck].astype(np.float64); fc = F[F.cell == cellname].set_index("q")
    med[label] = {c: round(float(fc[c].median()), 3) for c in ("w1", "longbar", "mean_cos", "std_cos", "coherence")}
    for i in rng.permutation(len(Q))[:2]:
        top = np.argsort(-(C @ Q[i]), kind="stable")[:50]; qp = PAGES[qk][i]
        reach = sum((qp <= PAGES[ck][t]) if label == "polysemic-like" else (PAGES[ck][t] <= qp) for t in top)
        rows.append(dict(group=label, cell=cellname, q=int(i), query_pages=[NAMES[p] for p in sorted(qp)],
                         neighbours=[[NAMES[p] for p in sorted(PAGES[ck][t])] for t in top[:5]], reach_bound_of_50=int(reach),
                         **{c: round(float(fc.loc[i, c]), 3) for c in ("w1", "longbar", "mean_cos", "std_cos", "coherence")}))
json.dump(dict(rule="bge-base, g0, arm all; default_rng(0) permutation per cell, first two", rows=rows, medians=med), open(OUT / "samples_rule.json", "w"), indent=1)
print(json.dumps(dict(rows=rows, medians=med), indent=1))
