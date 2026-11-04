# probe_slot2.py -- written on day 6 at 10:20. Control for probe_slot.py: the topic-split stratum is chosen by where
# the readings retrieve, so it may just pick questions whose OWN neighbourhood spans many pages. Re-compare against clear
# questions within bins of the question's own page diversity (distinct titles among its top 10 / top 50 neighbours).
import sys, numpy as np, pandas as pd
out = sys.argv[1]
F = pd.read_csv(f"{out}/feats.csv").set_index("i"); D = pd.read_csv(f"{out}/probe_slot.csv").set_index("i")
TT = np.load(f"{out}/corpus_titles.npy", allow_pickle=True); IDX = np.load(f"{out}/retrieved.npz")["idx"]
F["pages10"] = [len(set(TT[r[:10]])) for r in IDX]; F["pages50"] = [len(set(TT[r])) for r in IDX]
def auc(a, b):
    r = pd.Series(np.r_[a, b]).rank().to_numpy(); return (r[:len(a)].sum() - len(a)*(len(a)+1)/2) / (len(a)*len(b))
clear = F[F.label == 0]; split = F.loc[D.index[D.jac < 0.2]]; same = F.loc[D.index[D.jac >= 0.5]]
print("distinct pages in own top-10: clear", clear.pages10.median(), " topic-split", split.pages10.median(), " same-pages", same.pages10.median())
print("distinct pages in own top-50: clear", clear.pages50.median(), " topic-split", split.pages50.median(), " same-pages", same.pages50.median())
for name, G in (("topic split", split), ("same pages", same)):
    for key in ("pages10", "pages50"):
        bins = pd.qcut(F[key], 5, duplicates="drop"); res = {}
        for c in ("w1", "mean_cos", "w1_corp"):
            num = den = 0
            for b in bins.cat.categories:
                a = G[c][bins.loc[G.index] == b].to_numpy(); z = clear[c][bins.loc[clear.index] == b].to_numpy()
                if len(a) >= 10 and len(z) >= 10: num += auc(a, z) * len(a); den += len(a)
            res[c] = num / den
        print(f"{name:12s} within {key} bins: " + "  ".join(f"{c} {v:.2f}" for c, v in res.items()))
