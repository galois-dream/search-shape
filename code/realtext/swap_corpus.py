# swap_corpus.py -- (2026-11-04 23:40 CET, code review: the other-corpus pool now also excludes the query's top 50 in that
# corpus, symmetric with the own-corpus pool.) exploratory probe, 2026-11-04 ~19:10 CET (post hoc; run per corpus with CORPUS=<c> CLEAN=1 OUTROOT=out_clean). For each mismatched cell (Nobel,
# text-embedding-3-small, CLEAN=1, span-overlap exclusion as features.py), keep the query's 50 real scores and draw
# the 50 bearings (one draw) from (a) the cell's own corpus outside the top 50 [= corpus-bearing control],
# (b) the OTHER chunking's corpus, (c) uniform random. Reports median mean bar (2 W1) per cell and the gap.
import os, sys, types, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.modules.setdefault("ripser", types.SimpleNamespace(ripser=None))
import features as F
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform, pdist
meta, P, Q = F.load_model("openai-3-small")
def mean_bar(q, X):
    return float(minimum_spanning_tree(squareform(pdist(F.offsets(q, X)))).data.sum() / (len(X) - 1))
def run(qg, cg, other):
    qm, cm, C, Qe = meta[qg], meta[cg], P[cg], Q[qg]
    om, O = meta[other], P[other]
    S = Qe @ C.T
    same = qm.lecture.to_numpy(dtype=object)[:, None] == cm.lecture.to_numpy(dtype=object)[None, :]
    ov = (qm.start.values[:, None] < cm.end.values[None, :]) & (cm.start.values[None, :] < qm.end.values[:, None])
    excl = same & ov; S = np.where(excl, -np.inf, S)
    so = om.lecture.to_numpy(dtype=object)[None, :] == qm.lecture.to_numpy(dtype=object)[:, None]
    oo = (qm.start.values[:, None] < om.end.values[None, :]) & (om.start.values[None, :] < qm.end.values[:, None])
    oexcl = so & oo
    rng = np.random.default_rng(1); res = {k: [] for k in ("real", "own", "other", "random")}
    for i in range(len(qm)):
        top = np.argsort(-S[i], kind="stable")[:F.K]; a = np.clip(S[i, top], -1, 1); q = Qe[i]
        def ctrl(E):
            V = F.unit(E - (E @ q)[:, None] * q[None, :]); return mean_bar(q, a[:, None] * q[None, :] + np.sqrt(1 - a**2)[:, None] * V)
        res["real"].append(mean_bar(q, C[top]))
        pool = np.setdiff1d(np.flatnonzero(~excl[i]), top); res["own"].append(ctrl(C[rng.choice(pool, F.K, replace=False)]))
        So = np.where(oexcl[i], -np.inf, O @ q); otop = np.argsort(-So, kind="stable")[:F.K]   # the query's top 50 in the other corpus
        opool = np.setdiff1d(np.flatnonzero(~oexcl[i]), otop); res["other"].append(ctrl(O[rng.choice(opool, F.K, replace=False)]))
        res["random"].append(ctrl(rng.standard_normal((F.K, C.shape[1]))))
    return {k: np.median(v) for k, v in res.items()}, len(qm)
p, n1 = run(250, 750, 250); m, n2 = run(750, 250, 750)
print(f"Q250|C750 polysemic-like (n={n1}):", {k: round(v, 3) for k, v in p.items()})
print(f"Q750|C250 multi-factual-like (n={n2}):", {k: round(v, 3) for k, v in m.items()})
print("gap (multi - poly):", {k: round(m[k] - p[k], 3) for k in p})
print("swapped corpora gap (poly bearings from C250, multi from C750):", round(m["other"] - p["other"], 3))
