# Shipped copy of paper/tois/explore_e5.py; only this line, the host name and the SIM_DIR default (../sim -> ../simulation/matched) changed for the repo layout (EXPLORATORY.md = docs/review_checks_exploratory.md; explore_out/ = data/reference_outputs/review_checks/). paper/sim/ = code/simulation/matched/. Create explore_out/ before `run`.
# explore_e5.py -- written 2026-11-16 21:15 CET. EXPLORATORY.md E5 (amendment 1): one sense vs two at a realistic dimension.
# Construction, statistics and verdict rules from paper/sim/matched.py and matched_auroc.py (imported or copied, unchanged).
# Cells: p = 20, tau = 0.8 (noise s0.2), Levels A and B, phi in {10, 20, 30, 45}, D = 768; 5 seeds x 40 pairs.
# Usage (needs ripser, scipy, pandas; VMF_DIR = folder holding vmf.py):
#   python explore_e5.py run   -> explore_out/e5_cells.csv
#   python explore_e5.py auroc -> explore_out/e5_auroc.txt
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.environ.get("SIM_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation", "matched")))
import matched as M  # noqa: E402
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "explore_out")
CELLS = [(lev, phi) for lev in ("A", "B") for phi in (10, 20, 30, 45)]
P, TAU, D = 20, 0.8, 768

def run():
    rows = []
    for ci, (level, phi) in enumerate(CELLS):
        cid = 1000 + ci                                   # seeds disjoint from matched.py's cell indices
        for seed in M.SEEDS:
            for i in range(M.N_PER):
                base = np.random.default_rng((cid, seed, i))
                cos = np.sort(base.uniform(0.55, 0.75, M.K))[::-1]
                st_b = (cid, seed, i, 2)
                Cb = M.cloud(cos, M.bearings(np.random.default_rng(st_b), D, "two", P, M.S_BASE if level == "A" else 0.5 * M.S_BASE, TAU, phi))
                s_a = M.match_level_b((cid, seed, i, 1), D, "one", P, TAU, cos, M.coherence(Cb)) if level == "B" else M.S_BASE
                Ca = M.cloud(cos, M.bearings(np.random.default_rng((cid, seed, i, 1)), D, "one", P, s_a, TAU, phi))
                for lab, C in (("one", Ca), ("two", Cb)):
                    rows.append(dict(level=level, phi=phi, seed=seed, i=i, cls=lab, **M.features(C)))
        print(f"cell {ci} level={level} phi={phi}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "e5_cells.csv"), index=False)

def auc(x, y):
    r = pd.Series(np.r_[x, y]).rank().to_numpy(); n0, n1 = len(x), len(y)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)

def auroc():
    R = pd.read_csv(os.path.join(OUT, "e5_cells.csv")); rng = np.random.default_rng(0)
    RIV, PH, COS = ["coherence", "nn_med", "eig2", "max_nn"], ["w1", "lmax"], ["top1", "mean_cos", "std_cos", "slope_cos"]
    L = ["# e5_auroc -- one vs two senses, p = 20, tau = 0.8, D = 768; oriented AUROC; diff = feature - best rival, paired bootstrap 95% (2,000, seed 0)",
         "level phi | PR one/two | cos stats raw | coherence raw | best rival | W1 [lo,hi] diff [lo,hi] verdict | Lmax diff [lo,hi] verdict"]
    for (lev, phi), g in R.groupby(["level", "phi"]):
        A = g[g.cls == "one"].set_index(["seed", "i"]).sort_index(); B = g[g.cls == "two"].set_index(["seed", "i"]).sort_index()
        n = len(A); boots = rng.integers(0, n, (2000, n))
        raw = {c: auc(A[c].to_numpy(), B[c].to_numpy()) for c in COS + RIV + PH}
        ori = {c: max(raw[c], 1 - raw[c]) for c in raw}
        bt = {c: np.array([auc(A[c].to_numpy()[ix], B[c].to_numpy()[ix]) for ix in boots]) for c in RIV + PH}
        bto = {c: np.where(raw[c] >= 0.5, bt[c], 1 - bt[c]) for c in bt}
        best = max(RIV, key=lambda c: ori[c]); out = []
        for f in PH:
            lo, hi = np.percentile(bto[f], [2.5, 97.5]); d = bto[f] - bto[best]; dlo, dhi = np.percentile(d, [2.5, 97.5])
            v = "ceiling" if ori[best] >= 0.98 else "adds" if (dlo > 0 and ori[f] - ori[best] >= 0.05) else "redundant" if (dlo >= -0.02 and dhi <= 0.02) else "undecided"
            out.append(f"{ori[f]:.3f} [{lo:.3f},{hi:.3f}] {ori[f]-ori[best]:+.3f} [{dlo:+.3f},{dhi:+.3f}] {v}")
        cosr = " ".join(f"{raw[c]:.2f}" for c in COS)
        L.append(f"{lev} {phi:2d} | {A.part_ratio.median():.1f}/{B.part_ratio.median():.1f} | {cosr} | {raw['coherence']:.2f} | {best} {ori[best]:.3f} | " + " | ".join(out))
    open(os.path.join(OUT, "e5_auroc.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    {"run": run, "auroc": auroc}[sys.argv[1]]()
