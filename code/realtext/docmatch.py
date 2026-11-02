# docmatch.py -- exploratory DOCUMENT-MATCHED control, added after the main results. It keeps each neighbour's cosine
# to the query and its document, and takes the bearing from a random other chunk of that same document (outside the
# top 50 when the document has one; otherwise from any of its chunks, counted as a fallback). It asks whether the
# arrangement effect of a feature is explained by WHICH documents the 50 neighbours come from. Same comparison as the
# paper's fixed-length table: 250-token queries, own document left out, size-matched C250 vs C750, AUROC =
# P(C250 > C750), 20 draws per query, real minus mean control AUROC, paired Bayesian bootstrap over documents
# (vmf_stats.boot_d, 2,000 draws, seed 0). The corpus-bearing control is recomputed with features.multi_null's exact
# draws and checked against the feature table. Needs the embeddings.
# Usage: CORPUS=<c> CLEAN=1 OUTROOT=out_clean python docmatch.py <model>  -> $OUTROOT[/<c>]/docmatch_<model>.csv
import sys
import numpy as np
import pandas as pd
from features import load_model, offsets, ph, unit, OUT, K
from stats import feat_path
from vmf_stats import boot_d

model = sys.argv[1]
meta, P, Q = load_model(model)
FEATS = ("w1", "ltmax", "longbar")


def cell(cg, rows_c):
    """Real, corpus-bearing and document-matched features for every Q250 query against corpus rows rows_c of C<cg>."""
    qm, cm = meta[250], meta[cg].iloc[rows_c].reset_index(drop=True)
    E = P[cg][rows_c]; S = Q[250] @ E.T
    doc = cm.lecture.to_numpy(dtype=object)
    excl = qm.lecture.to_numpy(dtype=object)[:, None] == doc[None, :]
    S = np.where(excl, -np.inf, S)
    bydoc = {d: np.flatnonzero(doc == d) for d in np.unique(doc)}
    out, fall = [], 0
    for i in range(len(qm)):
        idx = np.argsort(-S[i], kind="stable")[:K]; cos = S[i, idx]; q = Q[250][i]; a = np.clip(cos, -1, 1)
        pool = np.setdiff1d(np.flatnonzero(~excl[i]), idx); inc = set(idx.tolist())
        r = dict(query_id=qm.id[i], lecture=qm.lecture[i], **dict(zip(FEATS, ph(offsets(q, E[idx])))))
        for k in range(20):
            V = E[np.random.default_rng((k, 1, i)).choice(pool, K, replace=len(pool) < K)]   # = features.multi_null enull
            V = unit(V - (V @ q)[:, None] * q[None, :])
            r.update({f"{f}_corp{k:02d}": v for f, v in zip(FEATS, ph(offsets(q, a[:, None] * q + np.sqrt(1 - a ** 2)[:, None] * V)))})
            rg = np.random.default_rng((k, 6, i)); pick = []
            for j in idx:
                cand = np.array([c for c in bydoc[doc[j]] if c not in inc])
                fall += len(cand) == 0
                pick.append(rg.choice(cand if len(cand) else bydoc[doc[j]]))
            V = E[np.array(pick)]; V = unit(V - (V @ q)[:, None] * q[None, :])
            r.update({f"{f}_doc{k:02d}": v for f, v in zip(FEATS, ph(offsets(q, a[:, None] * q + np.sqrt(1 - a ** 2)[:, None] * V)))})
        out.append(r)
    return pd.DataFrame(out), fall / (20 * K * len(qm))


A, fa = cell(750, np.arange(len(meta[750])))                       # Q250|C750, own document out
B, fb = cell(250, np.load(OUT / "c250_subsample_idx.npy"))           # Q250|C250 size-matched, own document out
F = pd.read_csv(feat_path(OUT, model))                   # reproducibility check against the pipeline
for X, var, c in ((A, "lodo", "Q250|C750"), (B, "lodo_dens", "Q250|C250")):
    ref = F[(F.variant == var) & (F.cell == c)].set_index("query_id").loc[X.query_id]
    for f in FEATS:
        assert np.allclose(ref[f].to_numpy(), X[f].to_numpy(), atol=1e-6), (var, f, "real feature differs from features.py")
    assert np.allclose(ref["longbar_enull00"].to_numpy(), X["longbar_corp00"].to_numpy(), atol=1e-6), (var, "control draw differs")
rows = []
for f in FEATS:
    res = dict(model=model, feature=f, fallback_share_c750=fa, fallback_share_c250=fb)
    for ctl in ("corp", "doc"):
        cn = [f"{f}_{ctl}{k:02d}" for k in range(20)]
        est, d = boot_d(B[f].to_numpy(float), A[f].to_numpy(float), B[cn].to_numpy(float), A[cn].to_numpy(float),
                        B.lecture.to_numpy(), A.lecture.to_numpy(), np.random.default_rng(0))
        res.update({f"d_{ctl}": est, f"lo_{ctl}": np.percentile(d, 2.5), f"hi_{ctl}": np.percentile(d, 97.5)})
    rows.append(res)
R = pd.DataFrame(rows); R.to_csv(OUT / f"docmatch_{model}.csv", index=False)
print(R.round(3).to_string(index=False))
