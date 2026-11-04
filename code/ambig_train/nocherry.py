# nocherry.py -- written on day 6 at 19:50. Removes hand choice from the paper's AmbigNQ section. (a) topic-split share and W1 AUROC (ambiguous stratum vs all clear) at every Jaccard threshold, not one;
# (b) Table 4 rows by a fixed rule: seed 0, the first two draws from jac < 0.2 and the first two from jac >= 0.5,
# each with its first two readings and their first answers. Usage: python nocherry.py <train.json> <ambig out dir>
import ast, json, sys, numpy as np, pandas as pd
T = json.load(open(sys.argv[1])); out = sys.argv[2]
F = pd.read_csv(f"{out}/feats.csv").set_index("i"); D = pd.read_csv(f"{out}/probe_slot.csv").set_index("i")
clear = F.w1[F.label == 0].to_numpy()
def auc(a, b):
    r = pd.Series(np.r_[a, b]).rank().to_numpy(); return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
print("(a) threshold | share of ambiguous below | n | W1 AUROC (below vs clear) | W1 AUROC (at/above vs clear)")
for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
    lo = D.index[D.jac < t]; hi = D.index[D.jac >= t]
    print(f"    {t:4.2f} | {len(lo)/len(D):5.1%} | {len(lo):5d} | {auc(F.w1[lo].to_numpy(), clear):.3f} | {auc(F.w1[hi].to_numpy(), clear):.3f}")
ann = lambda t: ast.literal_eval(t["annotations"]) if isinstance(t["annotations"], str) else t["annotations"]
rng = np.random.default_rng(0)
for name, ix in (("jac < 0.2", D.index[D.jac < 0.2]), ("jac >= 0.5", D.index[D.jac >= 0.5])):
    for k in rng.permutation(np.asarray(ix))[:2]:
        qa = [a for a in ann(T[k]) if a["type"] == "multipleQAs"][0]["qaPairs"]
        print(f"(b) {name} [{k}] jac={D.jac[k]:.2f} Q: {T[k]['question']}")
        for p in qa[:2]: print(f"      - {p['question']} / {p['answer'][0]}")
