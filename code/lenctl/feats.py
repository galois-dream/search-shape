# feats.py -- written on day 4 at 10:09. Per-query features for the length-controlled test (docs/lenctl_plan.md): for each
# model, grouping seed, arm and cell, the top-50 neighbours of every query, the paper's persistence summaries of their
# unit offsets (W1 = half the mean finite H0 death, LT_max = half the longest H1 lifetime, L_max = longest H0 death;
# as in code/realtext/features.py), cosine and similarity statistics, and random- and corpus-bearing controls (5 draws
# each, averaged). Usage: python feats.py OUT_DIR  (reads texts.json, emb_*.npz; writes feats.csv)
import itertools, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from ripser import ripser

K, NDRAW, EPSM = 50, 5, np.finfo(float).eps
OUT = Path(sys.argv[1])
T = json.load(open(OUT / "texts.json"))
PAGE = np.array(T["page_of"])
PAGES = {k: [frozenset(PAGE[a] for a in grp) for grp in v] for k, v in T["sets"].items()}


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def offsets(q, C):
    return unit(C - q[None, :] + EPSM)


def ph(U):
    dg = ripser(U, maxdim=1)["dgms"]
    h0 = dg[0][:, 1]; h0 = h0[np.isfinite(h0)]
    lt = dg[1][:, 1] - dg[1][:, 0]
    return float(np.mean(h0 / 2.0)), (float(np.max(lt) / 2.0) if len(lt) else 0.0), float(np.max(h0))


def control(q, a, V):
    V = unit(V - (V @ q)[:, None] * q[None, :])
    return ph(offsets(q, a[:, None] * q[None, :] + np.sqrt(1 - a ** 2)[:, None] * V))


def cell(args):
    model, g, arm, qk, ck = args
    E = np.load(OUT / f"emb_{model}.npz")
    Q, C = E[qk].astype(np.float64), E[ck].astype(np.float64)
    rows = []
    for i, q in enumerate(Q):
        allowed = np.ones(len(C), bool)
        if arm == "ownout":
            allowed = np.array([not (PAGES[qk][i] & p) for p in PAGES[ck]])
        s = np.where(allowed, C @ q, -np.inf)
        top = np.argsort(-s, kind="stable")[:K]
        X, a = C[top], np.clip(s[top], -1, 1)
        r = dict(model=model, g=g, arm=arm, cell=f"{qk[:2]}|{ck[:2]}", q=i)
        r["w1"], r["ltmax"], r["longbar"] = ph(offsets(q, X))
        r.update(top1=a[0], mean_cos=a.mean(), std_cos=a.std(), slope=np.polyfit(np.arange(1, K + 1), a, 1)[0])
        G = X @ X.T; iu = np.triu_indices(K, 1); Gn = G.copy(); np.fill_diagonal(Gn, -np.inf)
        ev = np.sort(np.linalg.eigvalsh(G))[::-1]
        r.update(nn_med=float(np.median(Gn.max(1))), coherence=float(G[iu].mean()), eig2=float(ev[1] / ev.sum()))
        # amendment 1 (10:17): coverage gain = best cosine of q with a normalised sum of two neighbours, minus the
        # best single cosine; "diff" restricts the pair to neighbours that share no page
        Pn = [PAGES[ck][t] for t in top]
        S = (a[:, None] + a[None, :]) / np.sqrt(np.maximum(2 + 2 * G, 1e-12)); S[iu[1], iu[0]] = -np.inf; np.fill_diagonal(S, -np.inf)
        dis = np.array([[not (Pn[x] & Pn[y]) for y in range(K)] for x in range(K)])
        r.update(cov_gain=float(S.max() - a[0]), cov_gain_diff=float(np.where(dis, S, -np.inf).max() - a[0]))
        rng = np.random.default_rng((g, i, len(qk), len(ck)))
        pool = np.setdiff1d(np.flatnonzero(allowed), top)
        cr = np.array([control(q, a, rng.standard_normal((K, len(q)))) for _ in range(NDRAW)]).mean(0)
        cc = np.array([control(q, a, C[rng.choice(pool, K, replace=False)]) for _ in range(NDRAW)]).mean(0)
        for j, f in enumerate(("w1", "ltmax", "longbar")):
            r[f"{f}_rand"], r[f"{f}_corp"] = cr[j], cc[j]
        rows.append(r)
    return rows


if __name__ == "__main__":
    models = sorted(p.stem[4:] for p in OUT.glob("emb_*.npz"))
    tasks = []
    for model, arm in itertools.product(models, ("all", "ownout")):
        tasks.append((model, 0, arm, "Q1", "C1"))
        for g in range(3):
            tasks += [(model, g, arm, f"Q3_g{g}", "C1"), (model, g, arm, "Q1", f"C3_g{g}"), (model, g, arm, f"Q3_g{g}", f"C3_g{g}")]
    with Pool(8) as p:
        R = [r for rows in p.imap_unordered(cell, tasks) for r in rows]
    pd.DataFrame(R).to_csv(OUT / "feats.csv", index=False)
    print("rows", len(R), flush=True)
