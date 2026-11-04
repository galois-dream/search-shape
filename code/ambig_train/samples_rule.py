# samples_rule.py -- written on day 6 at 20:00. Data samples by a fixed rule (no hand choice):
# numpy default_rng(0), permutation of each group, first two: ambiguous with page overlap < 0.2, ambiguous with overlap
# >= 0.5, clear. Prints question, first two readings + first answers, top-5 distinct neighbour titles, and W1(H0),
# longest H0 bar, mean score, distinct pages among the 50 neighbours; then group medians. AmbigNQ train (already seen).
import ast, json, sys, numpy as np, pandas as pd
T = json.load(open(sys.argv[1])); out = sys.argv[2]
F = pd.read_csv(f"{out}/feats.csv").set_index("i"); D = pd.read_csv(f"{out}/probe_slot.csv").set_index("i")
TT = np.load(f"{out}/corpus_titles.npy", allow_pickle=True).astype(str); IDX = np.load(f"{out}/retrieved.npz")["idx"]
F["pages50"] = [len(set(TT[r])) for r in IDX]
ann = lambda t: ast.literal_eval(t["annotations"]) if isinstance(t["annotations"], str) else t["annotations"]
G = {"ambiguous, readings on different pages (overlap < 0.2)": D.index[D.jac < 0.2],
     "ambiguous, readings on the same pages (overlap >= 0.5)": D.index[D.jac >= 0.5],
     "clear": F.index[F.label == 0]}
rng = np.random.default_rng(0); rows = []
for g, ix in G.items():
    for k in rng.permutation(np.asarray(ix))[:2]:
        t = T[k]; r = dict(group=g, i=int(k), question=t["question"], readings=[], overlap=None)
        if F.label[k] == 1:
            qa = [a for a in ann(t) if a["type"] == "multipleQAs"][0]["qaPairs"]
            r["readings"] = [[p["question"], p["answer"][0]] for p in qa[:2]]; r["overlap"] = round(float(D.jac[k]), 2)
        else:
            r["answer"] = [a for a in ann(t) if a["type"] == "singleAnswer"][0]["answer"][0]
        r["neighbours"] = list(dict.fromkeys(TT[IDX[k]]))[:5]
        r.update(w1=round(float(F.w1[k]), 3), longbar=round(float(F.longbar[k]), 3), mean_score=round(float(F.mean_cos[k]), 3), pages50=int(F.pages50[k]))
        rows.append(r)
med = {g: dict(n=int(len(ix)), w1=round(float(F.w1[ix].median()), 3), longbar=round(float(F.longbar[ix].median()), 3),
               mean_score=round(float(F.mean_cos[ix].median()), 3), pages50=float(F.pages50[ix].median())) for g, ix in G.items()}
json.dump(dict(rule="numpy default_rng(0); permutation of each group; first two", rows=rows, medians=med), open(f"{out}/samples_rule.json", "w"), indent=1)
print(json.dumps(dict(rows=rows, medians=med), indent=1))
