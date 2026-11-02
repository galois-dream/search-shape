# densseed.py -- written 2026-11-02 14:03 CET. The size-matched contrast of the SIMODS paper (Q750 queries against
# C250 subsampled to |C750| per lecture) used ONE subsample (features.py, seed 0); red-team M2. This recomputes that
# cell for subsample seeds 0..9 with features.py's exact rule (uniform per lecture, n = the lecture's C750 count;
# overlap exclusion), and writes per-query mean and SD of the 50 scores and W1(H0). Seed 0 must reproduce the
# 'dens' rows of features_<model>.csv.gz. Not an extension of subseed.py, which covers the lodo_dens Q250 cell and
# the corpus-bearing controls only.
# Usage: CORPUS=<c> CLEAN=1 python densseed.py <model> <outdir>  -> <outdir>/densseed_<corpus>_<model>.csv
import os, sys
import numpy as np
import pandas as pd
from features import load_model, offsets, ph, K, CORPUS

model, outdir = sys.argv[1], sys.argv[2]
meta, P, Q = load_model(model)
m250, m750 = meta[250], meta[750]
n750 = m750.lecture.value_counts()
qm = meta[750]
rows = []
for seed in range(10):
    rng = np.random.default_rng(seed)  # seed 0 = features.py's draw
    sub = np.sort(np.concatenate([rng.choice(np.flatnonzero(m250.lecture.to_numpy(dtype=object) == L), n750.get(L, 0), replace=False)
                                  for L in sorted(m250.lecture.unique())]))
    cm = m250.iloc[sub].reset_index(drop=True); E = P[250][sub]
    S = Q[750] @ E.T
    same = qm.lecture.to_numpy(dtype=object)[:, None] == cm.lecture.to_numpy(dtype=object)[None, :]
    ov = (qm.start.values[:, None] < cm.end.values[None, :]) & (cm.start.values[None, :] < qm.end.values[:, None])
    S = np.where(same & ov, -np.inf, S)
    for i in range(len(qm)):
        idx = np.argsort(-S[i], kind="stable")[:K]; cos = S[i, idx]
        w1, _, _ = ph(offsets(Q[750][i], E[idx]))
        rows.append(dict(seed=seed, query_id=qm.id.iloc[i], lecture=qm.lecture.iloc[i], mean50=float(cos.mean()), std50=float(cos.std()), w1=w1))
os.makedirs(outdir, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(outdir, f"densseed_{CORPUS}_{model}.csv"), index=False)
print(model, CORPUS, len(rows))
