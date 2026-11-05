# n_check.py -- 2026-11-05 CET. Neighbourhood size in the simulation (a co-author's note on N=50): fig1 setting
# (D=256, 64 topics, sigma=0.1), five seeds, at N=50 (out/) and N=20 (out_n20/, SIM_NCORPUS=20 python run_sim.py --configs fig1
# --outdir out_n20). AUROC = P(multi-factual S1 > polysemic S2) for W1(H0), the longest bar and LT_max(H1) on all N
# neighbours, and W1(H0) under each epsilon cut of run_sim.py. Run: python n_check.py > out_n20/n_check.txt
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
def A(y, v):
    m = np.isfinite(v); return roc_auc_score(y[m], v[m])
for d, N in (("out", 50), ("out_n20", 20)):
    R = []
    for s in range(5):
        x = pd.read_csv(f"{d}/fig1_seed{s}.csv"); e = x[x.depsilon == 0.4]; y = (e.scenario == 1).astype(int).values
        r = {f: A(y, e[f].values) for f in ("all_w1_h0", "all_h0_maxbar", "all_ltmax_h1")}
        r.update({f"eps{eps}": A((g.scenario == 1).astype(int).values, g.w1_h0.values) for eps, g in x.groupby("depsilon")})
        R.append(r)
    R = pd.DataFrame(R)
    print(f"N={N} ({len(e)} queries per seed): " + "; ".join(f"{c} {R[c].min():.3f}-{R[c].max():.3f}" for c in R.columns))
