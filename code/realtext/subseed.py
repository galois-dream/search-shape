# subseed.py -- precision check: the size-matched C250 corpus is ONE random subsample (features.py, seed 0), and the
# document bootstrap does not include that randomness. For subsample seeds 0..9 (seed 0 reproduces features.py),
# recompute the lodo_dens cell (Q250 queries against the size-matched C250, own document out): real persistence
# features and the 20 corpus-bearing control draws (same construction as features.multi_null). The Q250|C750 side is
# read from the feature table (it has no subsample). Reports, per seed, real minus control AUROC P(C250 > C750); the
# spread over seeds is the variance the document bootstrap misses (folded in by bayes.py --subvar). Needs the embeddings.
# Usage: CORPUS=<c> CLEAN=1 OUTROOT=out_clean python subseed.py <model>  -> $OUTROOT[/<c>]/subseed_<model>.csv
import sys
import numpy as np
import pandas as pd
from features import load_model, offsets, ph, unit, OUT, K
from stats import feat_path
from bayes import wauc

model = sys.argv[1]
meta, P, Q = load_model(model)
FEATS = ("w1", "ltmax", "longbar")
F = pd.read_csv(feat_path(OUT, model))
A = F[(F.variant == "lodo") & (F.cell == "Q250|C750")]
one = lambda x, y: wauc(x, y, np.ones((1, len(x))), np.ones((1, len(y))))[0]
m250, n750 = meta[250], meta[750].lecture.value_counts()
rows = []
for seed in range(10):
    rng = np.random.default_rng(seed)   # = features.py's draw for seed 0
    sub = np.sort(np.concatenate([rng.choice(np.flatnonzero(m250.lecture.to_numpy(dtype=object) == L), n750.get(L, 0), replace=False)
                                  for L in sorted(m250.lecture.unique())]))
    cm = m250.iloc[sub].reset_index(drop=True); E = P[250][sub]; S = Q[250] @ E.T
    excl = m250.lecture.to_numpy(dtype=object)[:, None] == cm.lecture.to_numpy(dtype=object)[None, :]
    S = np.where(excl, -np.inf, S)
    B = []
    for i in range(len(m250)):
        idx = np.argsort(-S[i], kind="stable")[:K]; cos = S[i, idx]; q = Q[250][i]; a = np.clip(cos, -1, 1)
        pool = np.setdiff1d(np.flatnonzero(~excl[i]), idx)
        r = dict(zip(FEATS, ph(offsets(q, E[idx]))))
        for k in range(20):
            V = E[np.random.default_rng((k, 1, i)).choice(pool, K, replace=len(pool) < K)]
            V = unit(V - (V @ q)[:, None] * q[None, :])
            r.update({f"{f}_enull{k:02d}": v for f, v in zip(FEATS, ph(offsets(q, a[:, None] * q + np.sqrt(1 - a ** 2)[:, None] * V)))})
        B.append(r)
    B = pd.DataFrame(B)
    if seed == 0:   # reproduces the pipeline
        ref = F[(F.variant == "lodo_dens") & (F.cell == "Q250|C250")]
        assert np.allclose(ref.longbar.to_numpy(), B.longbar.to_numpy(), atol=1e-6), "seed 0 does not reproduce features.py"
    for f in FEATS:
        cn = [f"{f}_enull{k:02d}" for k in range(20)]
        real = one(B[f].to_numpy(float), A[f].to_numpy(float))
        ctrl = np.mean([one(B[c].to_numpy(float), A[c].to_numpy(float)) for c in cn])
        rows.append(dict(model=model, seed=seed, feature=f, auc_real=real, auc_ctrl=ctrl, d=real - ctrl))
R = pd.DataFrame(rows); R.to_csv(OUT / f"subseed_{model}.csv", index=False)
print(R.groupby("feature").d.agg(["mean", "std", "min", "max"]).round(3).to_string())
