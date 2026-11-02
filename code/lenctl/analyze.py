# analyze.py -- written on day 4 at 11:04. AUROCs for the length-controlled test, scored against docs/lenctl_plan.md (L2-L6).
# Main contrast M: multi-factual Q3|C1 vs polysemic Q1|C3, AUROC = P(statistic larger for Q3|C1). Ratio contrast R:
# matched Q3|C3 vs Q1|C1. 95% bootstrap over queries (B = 2,000, seed 0; the two groups resampled separately).
# Usage: python analyze.py feats.csv > results.txt
import sys
import numpy as np
import pandas as pd

F = pd.read_csv(sys.argv[1])
STATS = ["w1", "longbar", "ltmax", "w1_corp", "longbar_corp", "ltmax_corp", "top1", "mean_cos", "std_cos", "slope",
         "nn_med", "coherence", "eig2", "cov_gain", "cov_gain_diff"]
rng = np.random.default_rng(0)


def auc(x, y):  # P(y > x) + 0.5 P(tie)
    r = pd.Series(np.r_[x, y]).rank().to_numpy(); n0, n1 = len(x), len(y)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def boot(x, y, B=2000):
    return np.array([auc(x[rng.integers(0, len(x), len(x))], y[rng.integers(0, len(y), len(y))]) for _ in range(B)])


ori = lambda a: max(a, 1 - a)
for (model, arm), G in F.groupby(["model", "arm"]):
    print(f"\n=== {model}, arm {arm} ===")
    for g in sorted(G.g.unique()):
        c = lambda cell, gg: G[(G.cell == cell) & (G.g == gg)]
        m3, p1 = c("Q3|C1", g), c("Q1|C3", g)
        r1, r3 = c("Q1|C1", 0), c("Q3|C3", g)
        res = {}
        for s in STATS:
            res[s] = (auc(p1[s].to_numpy(), m3[s].to_numpy()), auc(r1[s].to_numpy(), r3[s].to_numpy()))
        line = "  ".join(f"{s} M {res[s][0]:.3f} R {res[s][1]:.3f}" for s in ("w1", "longbar", "ltmax", "nn_med", "cov_gain_diff"))
        print(f"g{g}: {line}")
        if g == 0:
            for s in STATS:
                bm = boot(p1[s].to_numpy(), m3[s].to_numpy()); br = boot(r1[s].to_numpy(), r3[s].to_numpy())
                print(f"   {s:14s} M {res[s][0]:.3f} [{np.percentile(bm, 2.5):.3f}, {np.percentile(bm, 97.5):.3f}]   "
                      f"R {res[s][1]:.3f} [{np.percentile(br, 2.5):.3f}, {np.percentile(br, 97.5):.3f}]")
            cos_best = max(("top1", "mean_cos", "std_cos", "slope"), key=lambda s: ori(res[s][0]))
            ph_best = max(("w1", "longbar", "ltmax"), key=lambda s: ori(res[s][0]))
            print(f"   L2 ratio: W1 R {res['w1'][1]:.3f}, L_max R {res['longbar'][1]:.3f} (holds if both within 0.45-0.55)")
            print(f"   L3 direction: W1 M {res['w1'][0]:.3f} (holds if > 0.5 with interval above)")
            print(f"   L4 cosines: best {cos_best} oriented {ori(res[cos_best][0]):.3f} (holds if >= 0.75)")
            print(f"   L5 sufficiency: best persistence {ph_best} {ori(res[ph_best][0]):.3f} vs nn_med {ori(res['nn_med'][0]):.3f}: "
                  f"{ori(res[ph_best][0]) - ori(res['nn_med'][0]):+.3f} (holds if <= 0.02)")
            print(f"   L6 coverage (different pages): M {res['cov_gain_diff'][0]:.3f} (holds if > 0.5 with interval above)")
            for f in ("w1", "longbar", "ltmax"):
                print(f"   gain over corpus-bearing control, {f}: oriented {ori(res[f][0]) - ori(res[f + '_corp'][0]):+.3f}")
