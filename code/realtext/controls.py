# controls.py -- resampling-heavy controls (1 process, single-threaded BLAS recommended):
#   dens <model> : density control repeated over subsample seeds 0..19 (C250 subsampled to |C750| per document):
#                  W1 corpus-effect AUROC(Q250|C750 > Q250|C250sub) and the same for the empirical-bearing null
#                  (seed 1, as in features.py); point estimates per seed -> $OUTROOT[/corpus]/dens_seeds_<model>.csv
#   hgb <model> [R]: flexible non-topological rival A' (sorted 50 cosines + Gram summaries, features.py:gram_feats)
#                  vs B' = A' + W1, LT_max, longest H0 bar; sklearn HistGradientBoostingClassifier(random_state=0),
#                  GroupKFold(5) by document; out-of-fold AUROC; cluster bootstrap that REFITS the whole CV in each of
#                  R replicates (default 200; documents resampled with replacement, duplicates keep their document
#                  id so they share a fold) -> $OUTROOT[/corpus]/hgb_<model>.csv
#   hgbvmf <model> [R]: the same with B' = A' + vmf2_r5 (from vmf_<model>.csv.gz, joined on variant, cell, query_id;
#                  vMF plan, analysis (c), prediction P3) -> $OUTROOT[/corpus]/hgbvmf_<model>.csv
# Usage: [CORPUS=wiki] [OUTROOT=out_clean] python controls.py dens|hgb|hgbvmf <model> [R]
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import K, OUT, load_model, offsets, ph, unit  # noqa: E402
from stats import feat_path  # noqa: E402

A_COLS = [f"c{j:02d}" for j in range(1, K + 1)] + [f"g_eig{j}" for j in range(1, 6)] + \
         ["g_mean", "g_med", "g_q10", "g_q90", "nn_q10", "nn_med", "nn_q90"]
T_COLS = ["w1", "ltmax", "longbar"]


def auc(x, y):
    return roc_auc_score(np.r_[np.ones(len(x)), np.zeros(len(y))], np.r_[x, y])


def dens(model):
    meta, P, Q = load_model(model)
    F = pd.read_csv(feat_path(OUT, model))
    a = F[(F.variant == "primary") & (F.cell == "Q250|C750")]
    m250, n750 = meta[250], meta[750].lecture.value_counts()
    lec = m250.lecture.to_numpy(dtype=object)
    qm = meta[250]; ql = qm.lecture.to_numpy(dtype=object)
    rows = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        sub = np.sort(np.concatenate([rng.choice(np.flatnonzero(lec == L), n750.get(L, 0), replace=False)
                                      for L in sorted(m250.lecture.unique())]))
        cm = m250.iloc[sub].reset_index(drop=True); C = P[250][sub]
        S = Q[250] @ C.T
        excl = (ql[:, None] == cm.lecture.to_numpy(dtype=object)[None, :]) & \
               (qm.start.values[:, None] < cm.end.values[None, :]) & (cm.start.values[None, :] < qm.end.values[:, None])
        S = np.where(excl, -np.inf, S); top = np.argsort(-S, axis=1, kind="stable")[:, :K]
        rng1 = np.random.default_rng(1)
        w, we = [], []
        for i in range(len(qm)):
            idx = top[i]; q = Q[250][i]; cos = np.clip(S[i, idx], -1, 1)
            w.append(ph(offsets(q, C[idx]))[0])
            pool = np.setdiff1d(np.flatnonzero(~excl[i]), idx); E = C[rng1.choice(pool, K, replace=False)]
            V = unit(E - (E @ q)[:, None] * q[None, :])
            we.append(ph(offsets(q, cos[:, None] * q[None, :] + np.sqrt(1 - cos ** 2)[:, None] * V))[0])
        r = auc(a.w1.values, np.array(w)); n = auc(a.w1_enull.values, np.array(we))
        rows.append(dict(model=model, seed=seed, auc_w1=r, auc_w1_enull=n, real_minus_enull=r - n))
        print(f"[{time.strftime('%H:%M:%S')}] dens seed {seed}: AUROC {r:.3f} enull {n:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"dens_seeds_{model}.csv", index=False)


def oof(df, cols):
    X, y, g = df[cols].values, df._y.values, df.lecture.to_numpy(dtype=object)
    p = np.empty(len(df))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        p[te] = HistGradientBoostingClassifier(random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p)


def hgb(model, R, vmf=False):
    F = pd.read_csv(feat_path(OUT, model))
    if vmf:
        V = pd.read_csv(OUT / f"vmf_{model}.csv.gz", usecols=["variant", "cell", "query_id", "vmf2_r5"])
        F = F.merge(V, on=["variant", "cell", "query_id"], how="left")
    T = ["vmf2_r5"] if vmf else T_COLS
    sel = lambda v, c: F[(F.variant == v) & (F.cell == c)]
    # same-length Q250 contrast only, primary and leave-document-out arms
    comps = {"same length Q250: C750 vs C250 (subsampled)": (sel("primary", "Q250|C750"), sel("dens", "Q250|C250")),
             "LODO same length Q250: C750 vs C250 (subsampled)": (sel("lodo", "Q250|C750"), sel("lodo_dens", "Q250|C250"))}
    rows = []
    for name, (x, y) in comps.items():
        D = pd.concat([x.assign(_y=1), y.assign(_y=0)], ignore_index=True)
        assert D[T].notna().all().all(), (name, "missing vMF rows")
        docs = np.array(sorted(D.lecture.unique())); byd = {d: np.flatnonzero(D.lecture.values == d) for d in docs}
        rng = np.random.default_rng(0); t0 = time.time()
        for r in range(R + 1):
            if r == 0:
                Db = D
            else:
                pick = rng.choice(docs, len(docs), replace=True)
                Db = D.iloc[np.concatenate([byd[d] for d in pick])]
            if Db._y.nunique() < 2:
                continue
            a_, b_ = oof(Db, A_COLS), oof(Db, A_COLS + T)
            rows.append(dict(model=model, comparison=name, rep=r, auc_A=a_, auc_B=b_, B_minus_A=b_ - a_))
            if r in (0, 1, 10) or r % 50 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] hgb {name} rep {r}: A' {a_:.3f} B' {b_:.3f} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"hgb{'vmf' if vmf else ''}_{model}.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1] == "dens":
        dens(sys.argv[2])
    else:
        hgb(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 200, vmf=sys.argv[1] == "hgbvmf")
