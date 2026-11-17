# Shipped copy of paper/tois/explore_e8.py; only this line, the host name and the features.py path (nobel/src -> code/realtext) changed for the repo layout (EXPLORATORY.md = docs/review_checks_exploratory.md; explore_out/ = data/reference_outputs/review_checks/). nobel/src/ = code/realtext/.
# explore_e8.py -- written 2026-11-17 20:03 CET. EXPLORATORY.md E8: Thomas's follow-ups to E7 (cross-corpus shell bearings;
# searching another corpus entirely). Extends explore_e7.py (same corpora, cycle, cells, seed, statistic).
# Usage: NOBEL_ROOT=$RT_WORK CLEAN=1 python explore_e8.py <model> > explore_out/e8_<model>.txt
import os, sys, types
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "realtext"))
sys.modules.setdefault("ripser", types.SimpleNamespace(ripser=None))
import features as F  # noqa: E402
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform, pdist
import scipy.stats as ss
# CYCLE, mean_bar and auc_smaller copied unchanged from explore_e7.py (which runs on import, so it is not imported).
ROOT = Path(os.environ["NOBEL_ROOT"]).expanduser()
CYCLE = [("Nobel", ROOT), ("Turing", ROOT / "turing"), ("Fields", ROOT / "fields"), ("Essays", ROOT / "gutenberg"), ("Wikipedia", ROOT / "wiki")]

def load(base, model):
    F.BASE = base
    return F.load_model(model)

def mean_bar(q, X):
    return float(minimum_spanning_tree(squareform(pdist(F.offsets(q, X)))).data.sum() / (len(X) - 1))

def auc_smaller(poly, multi):  # P(polysemic-like < multi-factual-like) + 0.5 ties
    x, y = np.asarray(poly), np.asarray(multi); r = ss.rankdata(np.r_[x, y])
    return (r[len(x):].sum() - len(y) * (len(y) + 1) / 2) / (len(x) * len(y))

def cell(meta, P, Q, Pother, qg, cg, rng):
    qm, cm, C, Qe, O = meta[qg], meta[cg], P[cg], Q[qg], Pother[cg]
    S = Qe @ C.T
    same = qm.lecture.to_numpy(dtype=object)[:, None] == cm.lecture.to_numpy(dtype=object)[None, :]
    ov = (qm.start.values[:, None] < cm.end.values[None, :]) & (cm.start.values[None, :] < qm.end.values[:, None])
    excl = same & ov; S = np.where(excl, -np.inf, S)
    keys = ("real", "own", "cross", "xshell", "else", "score_real", "score_else")
    res = {k: [] for k in keys}
    for i in range(len(qm)):
        top = np.argsort(-S[i], kind="stable")[:F.K]; a = np.clip(S[i, top], -1, 1); q = Qe[i]
        def ctrl(E):
            V = F.unit(E - (E @ q)[:, None] * q[None, :]); return mean_bar(q, a[:, None] * q[None, :] + np.sqrt(1 - a ** 2)[:, None] * V)
        res["real"].append(mean_bar(q, C[top])); res["score_real"].append(float(a.mean()))
        pool = np.setdiff1d(np.flatnonzero(~excl[i]), top); res["own"].append(ctrl(C[rng.choice(pool, F.K, replace=False)]))
        oorder = np.argsort(-(O @ q), kind="stable"); otop = oorder[:F.K]
        opool = np.setdiff1d(np.arange(len(O)), otop); res["cross"].append(ctrl(O[rng.choice(opool, F.K, replace=False)]))
        res["xshell"].append(ctrl(O[rng.choice(oorder[F.K:500], F.K, replace=False)]))   # ranks 51-500 in the other corpus
        res["else"].append(mean_bar(q, O[otop])); res["score_else"].append(float(np.clip(O[otop] @ q, -1, 1).mean()))
    return res, len(qm)

if __name__ == "__main__":
    model = sys.argv[1]
    data = {name: load(base, model) for name, base in CYCLE}
    print(f"# explore_e8 -- {model}, CLEAN={int(F.CLEAN)}; next corpus in the cycle, same chunking; one draw per query, rng seed 1; mean bar = 2 W1(H0)")
    print("# own = own corpus outside top 50; cross = next corpus outside top 50 (E7); xshell = next corpus ranks 51-500; else = query searches the next corpus (its top 50, real scores and bearings)")
    print("corpus (next) | gap multi-poly, mean bar: real own cross xshell else | AUROC poly smaller: real own cross xshell else | mean score poly/multi: real; else | n poly, multi")
    for j, (name, _) in enumerate(CYCLE):
        other = CYCLE[(j + 1) % len(CYCLE)][0]
        meta, P, Q = data[name]; Pother = data[other][1]
        rng = np.random.default_rng(1)
        p, n1 = cell(meta, P, Q, Pother, 250, 750, rng); m, n2 = cell(meta, P, Q, Pother, 750, 250, rng)
        bars = ("real", "own", "cross", "xshell", "else")
        gap = {k: np.median(m[k]) - np.median(p[k]) for k in bars}
        au = {k: auc_smaller(p[k], m[k]) for k in bars}
        sc = f"{np.median(p['score_real']):.3f}/{np.median(m['score_real']):.3f}; {np.median(p['score_else']):.3f}/{np.median(m['score_else']):.3f}"
        print(f"{name} ({other}) | " + " ".join(f"{gap[k]:+.3f}" for k in bars) + " | " + " ".join(f"{au[k]:.3f}" for k in bars) + f" | {sc} | {n1}, {n2}", flush=True)
