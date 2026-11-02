# sufficiency.py -- a fit-free test of "persistence is redundant given the similarity structure".
# Comparison: fixed 250-token queries, own document left out, size-matched C250 (y = 1) vs C750 (y = 0).
#   rung 1: AUROC P(C250 > C750) of single statistics, no fitting: query cosines (top1, mean50, std50, slope50),
#           neighbour-neighbour similarity (coherence = nn_pair_cos, top Gram eigenvalue g_eig1, nearest-item cosine
#           quantiles nn_med / nn_q10), persistence.
#   rung 2: is Y independent of the persistence feature GIVEN coherence? Stratify on coherence deciles (pooled over
#           both chunkings); within each stratum the AUROC of the feature; pooled as a stratified Mann-Whitney AUROC
#           (pairs compared only within strata). 0.5 = redundant given coherence. Common support = share of rows in
#           strata holding >= 10 of each chunking; where it is small the test is empty and is reported as such.
#           Sensitivity: strata on coherence x mean cosine (5 x 5), and on each feature's nearest-neighbour "twin".
# Usage: [OUTROOT=out_clean] python sufficiency.py   (pandas/numpy only) -> $RT_WORK/$OUTROOT/sufficiency.csv
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("RT_WORK", str(Path(__file__).resolve().parent / "work"))) / os.environ.get("OUTROOT", "out_clean")
CORP = {"nobel": ROOT, "turing": ROOT / "turing", "fields": ROOT / "fields", "gutenberg": ROOT / "gutenberg", "wiki": ROOT / "wiki"}
COS, NN, PH = ["top1", "mean50", "std50", "slope50"], ["nn_pair_cos", "g_eig1", "nn_med", "nn_q10"], ["w1", "longbar", "ltmax"]
# Each feature is also conditioned on its nearest-neighbour "twin" (chunk-space nearest-item cosines, features.gram_feats):
# W1 (half mean MST edge) <-> nn_med; L_max (longest MST edge, = largest nearest-neighbour distance when it isolates one
# point) <-> nn_q10; deciles of the twin x quintiles of mean50 (the twin lives in chunk space, the features in offset space).
TWIN = {"w1": "nn_med", "longbar": "nn_q10", "ltmax": "nn_med"}


def auc(x, y):
    r = pd.Series(x).rank().to_numpy(); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def strat_auc(x, y, s):
    """Stratified AUROC: pairs (C250, C750) compared only within the same stratum."""
    num = den = 0.0; sup = 0
    for k in np.unique(s):
        m = s == k; n1 = y[m].sum(); n0 = m.sum() - n1
        if n1 and n0:
            num += auc(x[m], y[m]) * n1 * n0; den += n1 * n0
        sup += m.sum() if (n1 >= 10 and n0 >= 10) else 0
    return (num / den if den else np.nan), sup / len(y)


rows = []
for c, p in CORP.items():
    for f in sorted(p.glob("features_*.csv.[gx]z")):
        m = f.name[len("features_"):-len(".csv.gz")]
        F = pd.read_csv(f, usecols=["variant", "cell"] + COS + NN + PH)
        a = F[(F.variant == "lodo_dens") & (F.cell == "Q250|C250")]; b = F[(F.variant == "lodo") & (F.cell == "Q250|C750")]
        X = pd.concat([a, b]); y = np.r_[np.ones(len(a)), np.zeros(len(b))].astype(int)
        r = dict(corpus=c, model=m, n=len(y))
        for v in COS + NN + PH:
            r[f"auc_{v}"] = auc(X[v].to_numpy(), y)
        s1 = pd.qcut(X.nn_pair_cos.rank(method="first"), 10, labels=False).to_numpy()
        s2 = (pd.qcut(X.nn_pair_cos.rank(method="first"), 5, labels=False) * 5 + pd.qcut(X.mean50.rank(method="first"), 5, labels=False)).to_numpy()
        for v in PH:
            r[f"strat_{v}"], r["support_coh"] = strat_auc(X[v].to_numpy(), y, s1)
            st = (pd.qcut(X[TWIN[v]].rank(method="first"), 10, labels=False) * 5 + pd.qcut(X.mean50.rank(method="first"), 5, labels=False)).to_numpy()
            r[f"twin_{v}"], r[f"support_twin_{v}"] = strat_auc(X[v].to_numpy(), y, st)
            r[f"strat2_{v}"], r["support_coh_mean"] = strat_auc(X[v].to_numpy(), y, s2)
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(ROOT / "sufficiency.csv", index=False)
best = lambda d, cols: d[[f"auc_{v}" for v in cols]].sub(0.5).abs().max(axis=1) + 0.5
R["best_cos"], R["best_nn"], R["best_ph"] = best(R, COS), best(R, NN), best(R, PH)
pd.set_option("display.width", 220)
print(R[["corpus", "model", "best_cos", "best_nn", "best_ph", "auc_longbar", "strat_longbar", "strat2_longbar", "support_coh", "support_coh_mean",
         "strat_w1", "strat_ltmax", "twin_w1", "support_twin_w1", "twin_longbar", "support_twin_longbar"]].round(3).to_string(index=False))


# Rung-1 intervals. Per corpus x model: D = oriented AUROC of the best persistence
# feature (chosen on the full data) minus that of the median nearest-neighbour cosine (nn_med, the statistic that wins
# in 42 of 45 cells), at fixed length without the query's own document; 95% interval from a bootstrap over documents
# (1,000 draws, seed 0; a resampled document brings all its queries in both chunkings). Equivalence margin 0.05
# (post hoc): persistence is "not ahead" where the upper end of D is below +0.05.
def boot_rows(B=1000):
    out = []
    for c, p in CORP.items():
        for f in sorted(p.glob("features_*.csv.[gx]z")):
            m = f.name[len("features_"):-len(".csv.gz")]
            F = pd.read_csv(f, usecols=["variant", "cell", "lecture", "nn_med"] + PH)
            a = F[(F.variant == "lodo_dens") & (F.cell == "Q250|C250")]; b = F[(F.variant == "lodo") & (F.cell == "Q250|C750")]
            X = pd.concat([a, b], ignore_index=True); y = np.r_[np.ones(len(a)), np.zeros(len(b))].astype(int)
            ori = lambda v, idx: (lambda u: max(u, 1 - u))(auc(X[v].to_numpy()[idx], y[idx]))
            full = np.arange(len(y)); best = max(PH, key=lambda v: ori(v, full))
            docs = X.lecture.to_numpy(); ud = np.unique(docs); byd = {d: np.flatnonzero(docs == d) for d in ud}
            rng = np.random.default_rng(0); Ds = []
            for _ in range(B):
                idx = np.concatenate([byd[d] for d in rng.choice(ud, len(ud), replace=True)])
                Ds.append(ori(best, idx) - ori("nn_med", idx))
            out.append(dict(corpus=c, model=m, best_ph=best, D=ori(best, full) - ori("nn_med", full),
                            lo=np.percentile(Ds, 2.5), hi=np.percentile(Ds, 97.5)))
    T = pd.DataFrame(out); T.to_csv(ROOT / "sufficiency_boot.csv", index=False)
    print(f"cells {len(T)}; D <= 0 (nn_med at least as good): {(T.D <= 0).sum()}; upper end of D < +0.05: {(T.hi < 0.05).sum()}; "
          f"interval includes 0: {((T.lo <= 0) & (T.hi >= 0)).sum()}; persistence clearly ahead (lo > 0): {(T.lo > 0).sum()}")
    print(T[T.hi >= 0.05].round(3).to_string(index=False))


if __name__ == "__main__" and os.environ.get("SUFF_BOOT") == "1":
    boot_rows()
