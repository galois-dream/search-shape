"""Analysis of the matched-cosine simulation (matched.py), exactly as registered in docs/matched_plan.md part 1 and its
amendments 1 and 2 (Q1/Q4/Q6). Per cell and statistic: AUROC P(second class > first class), raw and oriented
(max(a, 1-a)); bootstrap over samples (2,000 draws, seed 0; a sample = one (seed, i) pair, which holds one
neighbourhood of each class with the same cosines).
  M1 check : cosine statistics (top1, mean_cos, std_cos, slope_cos) raw AUROC within 0.5 +- 0.05.
  wins     : persistence feature oriented AUROC >= 0.75, its interval clear of 0.5, cosine statistics at 0.5, and at
             Level B coherence within 0.05 of 0.5.
  adds     : paired interval of [feature - best rival] (oriented AUROCs; rival = the one with the highest point AUROC
             among coherence, nn_med, eig2, max_nn) above 0 and point difference >= 0.05; redundant: interval within
             +-0.02; ceiling: best rival >= 0.98; otherwise undecided.
  Q1 bands : stratified AUROC of each persistence feature within deciles of coherence, nn_med and eff_rank (pairs
             compared only within a band; orientation as overall).
  M7       : share of one-meaning neighbourhoods with vmf2 > 0, by p and tau.
Usage: python matched_auroc.py ../../../data/simulation/matched/matched_shard*of14.csv.xz
       -> $MATCHED_RESULTS (default ./results)/matched_auroc.csv, matched_summary.txt
"""
import os
import sys
import numpy as np
import pandas as pd

COS = ["top1", "mean_cos", "std_cos", "slope_cos"]
RIV = ["coherence", "nn_med", "eig2", "max_nn"]
PH = ["w1", "lmax", "ltmax"]
OTHER = ["vmf2", "eff_rank", "part_ratio", "top5_share"]
CLS = {"one_vs_two": ("one", "two"), "one_vs_three": ("one", "three"), "ring_vs_one": ("ring", "one"), "ring_vs_two": ("ring", "two")}
KEY = ["comparison", "level", "p", "tau", "phi", "D"]


def auc(x, y):  # P(y > x) + 0.5 P(tie)
    r = pd.Series(np.r_[x, y]).rank().to_numpy(); n0, n1 = len(x), len(y)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def strat(x, y, sx, sy, sign):
    s = np.r_[sx, sy]; edges = np.unique(np.quantile(s, np.linspace(0, 1, 11))); bx, by = np.digitize(sx, edges[1:-1]), np.digitize(sy, edges[1:-1])
    num = den = 0.0
    for b in np.unique(np.r_[bx, by]):
        xa, ya = x[bx == b], y[by == b]
        if len(xa) and len(ya):
            num += auc(sign * xa, sign * ya) * len(xa) * len(ya); den += len(xa) * len(ya)
    return num / den if den else np.nan


R = pd.concat([pd.read_csv(f) for f in sys.argv[1:]], ignore_index=True)
rows, rng = [], np.random.default_rng(0)
for key, g in R.groupby(KEY):
    a_k, b_k = CLS[key[0]]
    A = g[g.cls == a_k].set_index(["seed", "i"]).sort_index(); B = g[g.cls == b_k].set_index(["seed", "i"]).sort_index()
    assert A.index.equals(B.index), key
    n = len(A); boots = rng.integers(0, n, (2000, n))
    stats = [c for c in COS + RIV + PH + OTHER if c in g]
    raw = {c: auc(A[c].to_numpy(), B[c].to_numpy()) for c in stats}
    ori = {c: max(raw[c], 1 - raw[c]) for c in stats}
    bt = {c: np.array([auc(A[c].to_numpy()[ix], B[c].to_numpy()[ix]) for ix in boots]) for c in RIV + PH}
    bto = {c: np.where(raw[c] >= 0.5, bt[c], 1 - bt[c]) for c in bt}
    cos_ok = all(abs(raw[c] - 0.5) <= 0.05 for c in COS)
    coh_ok = key[1] == "A" or abs(raw["coherence"] - 0.5) <= 0.05
    best = max(RIV, key=lambda c: ori[c])
    r = dict(zip(KEY, key), n=n, cos_at_half=cos_ok, coherence_at_half=abs(raw["coherence"] - 0.5) <= 0.05, best_rival=best, best_rival_auc=ori[best],
             **{f"auc_{c}": raw[c] for c in stats})
    for f in PH:
        lo, hi = np.percentile(bto[f], [2.5, 97.5]); d = bto[f] - bto[best]; dlo, dhi = np.percentile(d, [2.5, 97.5])
        wins = ori[f] >= 0.75 and lo > 0.5 and cos_ok and coh_ok
        if ori[best] >= 0.98: verdict = "ceiling"
        elif dlo > 0 and ori[f] - ori[best] >= 0.05: verdict = "adds"
        elif dlo >= -0.02 and dhi <= 0.02: verdict = "redundant"
        else: verdict = "undecided"
        sign = 1 if raw[f] >= 0.5 else -1
        r.update({f"{f}_ori": ori[f], f"{f}_lo": lo, f"{f}_hi": hi, f"{f}_wins": wins, f"{f}_minus_best": ori[f] - ori[best],
                  f"{f}_diff_lo": dlo, f"{f}_diff_hi": dhi, f"{f}_verdict": verdict,
                  **{f"{f}_band_{s}": strat(A[f].to_numpy(), B[f].to_numpy(), A[s].to_numpy(), B[s].to_numpy(), sign) for s in ("coherence", "nn_med", "eff_rank")}})
    r["vmf2_pos_first"] = float((A.vmf2 > 0).mean()); r["vmf2_pos_second"] = float((B.vmf2 > 0).mean())
    r["eff_rank_first"] = float(A.eff_rank.median()); r["eff_rank_second"] = float(B.eff_rank.median())
    rows.append(r)
OUT = os.environ.get("MATCHED_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")); os.makedirs(OUT, exist_ok=True)
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "matched_auroc.csv"), index=False)
pd.set_option("display.width", 250)
cols = KEY[:4] + ["cos_at_half", "coherence_at_half", "auc_coherence", "auc_nn_med", "auc_eig2", "lmax_ori", "lmax_verdict", "ltmax_ori", "ltmax_verdict",
                  "w1_ori", "lmax_band_nn_med", "vmf2_pos_first", "eff_rank_first"]
txt = T[cols].round(3).to_string(index=False)
open(os.path.join(OUT, "matched_summary.txt"), "w").write("# matched_summary.txt -- generated by matched_auroc.py\n" + txt + "\n"); print(txt)
