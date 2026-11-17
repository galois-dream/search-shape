# Shipped copy of paper/tois/explore_e7.py; only this line, the host name and the features.py path (nobel/src -> code/realtext) changed for the repo layout (EXPLORATORY.md = docs/review_checks_exploratory.md; explore_out/ = data/reference_outputs/review_checks/). nobel/src/ = code/realtext/.
# explore_e7.py -- written 2026-11-16 21:19 CET. EXPLORATORY.md E7: bearings from another corpus, same chunking (Thomas, p. 17).
# Extends nobel/src/swap_corpus.py: per query, real scores kept; bearings from (a) own corpus outside the top 50,
# (b) the next corpus in the cycle (same chunking, outside the query's top 50 there), (c) uniform random.
# Usage: NOBEL_ROOT=$RT_WORK CLEAN=1 python explore_e7.py <model> > explore_out/e7_<model>.txt
import os, sys, types
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "realtext"))
sys.modules.setdefault("ripser", types.SimpleNamespace(ripser=None))
import features as F  # noqa: E402
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform, pdist
ROOT = Path(os.environ["NOBEL_ROOT"]).expanduser()
CYCLE = [("Nobel", ROOT), ("Turing", ROOT / "turing"), ("Fields", ROOT / "fields"), ("Essays", ROOT / "gutenberg"), ("Wikipedia", ROOT / "wiki")]
model = sys.argv[1]

def load(base):
    F.BASE = base
    return F.load_model(model)

def mean_bar(q, X):
    return float(minimum_spanning_tree(squareform(pdist(F.offsets(q, X)))).data.sum() / (len(X) - 1))

def auc_smaller(poly, multi):  # P(polysemic-like < multi-factual-like) + 0.5 ties
    import scipy.stats as ss
    x, y = np.asarray(poly), np.asarray(multi); r = ss.rankdata(np.r_[x, y])   # average ranks for ties
    return (r[len(x):].sum() - len(y) * (len(y) + 1) / 2) / (len(x) * len(y))

def cell(meta, P, Q, Pother, qg, cg, rng):
    qm, cm, C, Qe, O = meta[qg], meta[cg], P[cg], Q[qg], Pother[cg]
    S = Qe @ C.T
    same = qm.lecture.to_numpy(dtype=object)[:, None] == cm.lecture.to_numpy(dtype=object)[None, :]
    ov = (qm.start.values[:, None] < cm.end.values[None, :]) & (cm.start.values[None, :] < qm.end.values[:, None])
    excl = same & ov; S = np.where(excl, -np.inf, S)
    res = {k: [] for k in ("real", "own", "cross", "random")}
    for i in range(len(qm)):
        top = np.argsort(-S[i], kind="stable")[:F.K]; a = np.clip(S[i, top], -1, 1); q = Qe[i]
        def ctrl(E):
            V = F.unit(E - (E @ q)[:, None] * q[None, :]); return mean_bar(q, a[:, None] * q[None, :] + np.sqrt(1 - a ** 2)[:, None] * V)
        res["real"].append(mean_bar(q, C[top]))
        pool = np.setdiff1d(np.flatnonzero(~excl[i]), top); res["own"].append(ctrl(C[rng.choice(pool, F.K, replace=False)]))
        otop = np.argsort(-(O @ q), kind="stable")[:F.K]
        opool = np.setdiff1d(np.arange(len(O)), otop); res["cross"].append(ctrl(O[rng.choice(opool, F.K, replace=False)]))
        res["random"].append(ctrl(rng.standard_normal((F.K, C.shape[1]))))
    return res, len(qm)

data = {name: load(base) for name, base in CYCLE}
print(f"# explore_e7 -- {model}, CLEAN={int(F.CLEAN)}; bearings: own corpus / next corpus in cycle (same chunking) / random; mean bar = 2 W1(H0); one draw per query, rng seed 1")
print("corpus (bearings from) | gap multi-poly: real own cross random | AUROC poly smaller: real own cross | n poly, multi")
for j, (name, _) in enumerate(CYCLE):
    other = CYCLE[(j + 1) % len(CYCLE)][0]
    meta, P, Q = data[name]; Pother = data[other][1]
    rng = np.random.default_rng(1)
    p, n1 = cell(meta, P, Q, Pother, 250, 750, rng); m, n2 = cell(meta, P, Q, Pother, 750, 250, rng)
    gap = {k: np.median(m[k]) - np.median(p[k]) for k in p}
    au = {k: auc_smaller(p[k], m[k]) for k in ("real", "own", "cross")}
    print(f"{name} ({other}) | " + " ".join(f"{gap[k]:+.3f}" for k in ("real", "own", "cross", "random")) + " | " + " ".join(f"{au[k]:.3f}" for k in ("real", "own", "cross")) + f" | {n1}, {n2}", flush=True)
