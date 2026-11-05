# swap_bearings.py -- written 2026-11-05 CET. Post hoc probe asked for by a co-author after reading the draft:
# keep every query's own 50 scores and give it the 50 real neighbour bearings of ANOTHER query, so that scores and
# arrangement are separated directly. Setting = the middle panel of Figure 7: Q250|C750 (polysemic-like, "primary",
# span-overlap exclusion) vs Q750|C250 with C250 size-matched to |C750| ("dens", c250_subsample_idx.npy). Bearing sources:
#   own     the query's real bearings (= its real W1, checked against features_<model>.csv.gz)
#   swap    a random query of the OTHER mismatched cell (polysemic-like <-> multi-factual-like)
#   shuf    a random other query of the SAME cell
#   eqA     a random query of the matched cell Q250|C250 (equal length ratio, full corpus)
#   eqB     a random query of the matched cell Q750|C750 (equal length ratio)
#   rand    uniform random bearings (the random-bearing control)
# The donor's bearings t_j (perpendicular to q_j) are carried to q_i by the reflection H = I - 2vv^T, v = (q_j-q_i)/|q_j-q_i|,
# which maps q_j to q_i and preserves every distance between bearings; rank r's score is paired with the donor's rank-r
# bearing. By Lemma 5.2 W1 depends only on the scores and the distances between bearings, so this is exact.
# One donor draw per query, rng default_rng(0) per corpus x model. Run: CORPUS=<c> CLEAN=1 OUTROOT=out_clean python swap_bearings.py
# Output: <out>/swap_bearings.csv.gz (corpus, model, cell, query_id, lecture, src, w1).
import os, sys, types, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.modules.setdefault("ripser", types.SimpleNamespace(ripser=None))
import features as F
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
MODELS = ["bge-base", "bge-m3", "gemini-001", "modernbert-embed", "openai-3-large", "openai-3-small", "openai-ada-002",
          "qwen3-0.6b", "vertex-005"]

def w1(q, a, V):  # W1(H0) = half the mean MST edge of the offset directions of c = a q + sqrt(1-a^2) t
    X = a[:, None] * q[None, :] + np.sqrt(1 - a ** 2)[:, None] * V
    return float(minimum_spanning_tree(squareform(pdist(F.offsets(q, X)))).data.sum() / (len(a) - 1) / 2)

def cell(meta, P, Q, qg, cg, rows=None):
    qm, cm, C = meta[qg], meta[cg], P[cg]
    if rows is not None: cm, C = cm.iloc[rows].reset_index(drop=True), C[rows]
    S = Q[qg] @ C.T
    same = qm.lecture.to_numpy(dtype=object)[:, None] == cm.lecture.to_numpy(dtype=object)[None, :]
    ov = (qm.start.values[:, None] < cm.end.values[None, :]) & (cm.start.values[None, :] < qm.end.values[:, None])
    S = np.where(same & ov, -np.inf, S)
    top = np.argsort(-S, axis=1, kind="stable")[:, :F.K]
    a = np.clip(np.take_along_axis(S, top, 1), -1, 1)
    assert np.isfinite(a).all()
    return dict(Q=Q[qg], C=C, top=top, a=a, meta=qm)

def bear(c, j):  # unit bearings of query j's 50 neighbours, perpendicular to q_j
    q = c["Q"][j]; X = c["C"][c["top"][j]]
    return F.unit(X - (X @ q)[:, None] * q[None, :])

def move(V, qf, qt):
    v = qf - qt; n = np.linalg.norm(v)
    if n < 1e-12: return V
    v = v / n
    return V - 2 * (V @ v)[:, None] * v[None, :]

corpus = F.CORPUS; out = []
sub = np.load(F.OUT / "c250_subsample_idx.npy")
for model in MODELS:
    meta, P, Q = F.load_model(model)
    cells = {"Q250|C750": cell(meta, P, Q, 250, 750), "Q750|C250": cell(meta, P, Q, 750, 250, sub),
             "eqA": cell(meta, P, Q, 250, 250), "eqB": cell(meta, P, Q, 750, 750)}
    other = {"Q250|C750": "Q750|C250", "Q750|C250": "Q250|C750"}
    rng = np.random.default_rng(0)
    for name in ("Q250|C750", "Q750|C250"):
        c = cells[name]; n = len(c["a"])
        for i in range(n):
            q, a = c["Q"][i], c["a"][i]
            def donor(src):
                d = cells[other[name]] if src == "swap" else (c if src == "shuf" else cells[src])
                j = rng.integers(len(d["a"]))
                while d is c and j == i: j = rng.integers(len(d["a"]))
                return move(bear(d, j), d["Q"][j], q)
            R = rng.standard_normal((F.K, len(q))); R = F.unit(R - (R @ q)[:, None] * q[None, :])
            res = {"own": w1(q, a, bear(c, i)), "swap": w1(q, a, donor("swap")), "shuf": w1(q, a, donor("shuf")),
                   "eqA": w1(q, a, donor("eqA")), "eqB": w1(q, a, donor("eqB")), "rand": w1(q, a, R)}
            for s, v in res.items():
                out.append((corpus, model, name, c["meta"].id[i], c["meta"].lecture[i], s, v))
    # check: own == the shipped real W1 (primary for Q250|C750, dens for Q750|C250)
    f = pd.read_csv(F.OUT / f"features_{model}.csv.gz", usecols=["variant", "cell", "query_id", "w1"])
    o = pd.DataFrame([r for r in out if r[1] == model and r[5] == "own"], columns=["corpus", "model", "cell", "query_id", "lecture", "src", "w1"])
    err = 0.0
    for name, var in (("Q250|C750", "primary"), ("Q750|C250", "dens")):
        m = o[o.cell == name].merge(f[(f.variant == var) & (f.cell == name)], on="query_id", suffixes=("", "_f"))
        assert len(m) == (o.cell == name).sum(), (model, name, len(m))
        err = max(err, float(np.abs(m.w1 - m.w1_f).max()))
    print(f"{corpus} {model}: n={len(o)} max|own - shipped W1| = {err:.2e}", flush=True)
pd.DataFrame(out, columns=["corpus", "model", "cell", "query_id", "lecture", "src", "w1"]).to_csv(F.OUT / "swap_bearings.csv.gz", index=False)
