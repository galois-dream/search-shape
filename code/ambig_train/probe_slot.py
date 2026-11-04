# probe_slot.py -- written on day 6 at 10:05. EXPLORATORY, post hoc (not registered). Thesis under test: AmbigNQ
# ambiguity is mostly an unbound slot (year, version, granularity) inside ONE topic, so its readings sit too close
# together for the retrieved neighbourhood to show them; where the readings split across topics, a signature should
# appear. For each ambiguous train question: embed its annotator-written readings (all-MiniLM-L6-v2, as ambig.py),
# (1) mean pairwise cosine of the readings vs the scale of the question's own 50 neighbours; (2) topic split =
# readings retrieve different pages (mean Jaccard of top-10 page titles). Then AUROC P(ambiguous > clear) per stratum.
# Usage: python probe_slot.py <ambignq train.json> <ambig.py out dir>
import ast, json, sys
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

T = json.load(open(sys.argv[1])); out = sys.argv[2]
F = pd.read_csv(f"{out}/feats.csv").set_index("i")
E = np.load(f"{out}/corpus_emb.npy").astype(np.float32); TT = np.load(f"{out}/corpus_titles.npy", allow_pickle=True)
R = np.load(f"{out}/retrieved.npz"); Q = R["Q"].astype(np.float32); IDX = R["idx"]
ann = lambda t: ast.literal_eval(t["annotations"]) if isinstance(t["annotations"], str) else t["annotations"]
reads = {}
for k, t in enumerate(T):
    A = [a for a in ann(t) if a["type"] == "multipleQAs"]
    if A and len(A[0]["qaPairs"]) >= 2: reads[k] = [p["question"] for p in A[0]["qaPairs"]]
flat = [(k, s) for k, v in reads.items() for s in v]
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
V = m.encode([s for _, s in flat], batch_size=64, normalize_embeddings=True).astype(np.float32)
top10 = np.concatenate([np.argsort(-(E @ V[i:i + 2000].T), axis=0)[:10].T for i in range(0, len(V), 2000)])
own = {}
for j, (k, _) in enumerate(flat): own.setdefault(k, []).append(j)
rows = []
for k, js in own.items():
    v = V[js]; G = v @ v.T; n = len(js); rr = (G.sum() - n) / (n * n - n)
    N = E[IDX[k]]; GN = N @ N.T; nn = (GN.sum() - len(N)) / (len(N) ** 2 - len(N))
    S = [set(TT[top10[j]]) for j in js]
    jac = np.mean([len(S[a] & S[b]) / len(S[a] | S[b]) for a in range(n) for b in range(a + 1, n)])
    rows.append(dict(i=k, n_read=n, rr=rr, qr=float((v @ Q[k]).mean()), nbr=nn, jac=jac))
D = pd.DataFrame(rows).set_index("i")
rng = np.random.default_rng(0); P = rng.integers(0, len(Q), (20000, 2)); P = P[P[:, 0] != P[:, 1]]
print(f"# probe_slot.py run {pd.Timestamp.now(tz='Europe/Paris'):%Y-%m-%d %H:%M %Z}; {len(D)} ambiguous questions, {len(flat)} readings")
q = lambda x: f"median {np.median(x):.2f} [IQR {np.percentile(x,25):.2f}, {np.percentile(x,75):.2f}]"
print("cosine between readings of one question    ", q(D.rr))
print("cosine question-to-its-readings            ", q(D.qr))
print("cosine between the question's 50 neighbours", q(D.nbr))
print("cosine between two random questions        ", q(np.sum(Q[P[:, 0]] * Q[P[:, 1]], 1)))
print("readings closer to each other than the neighbours are:", f"{np.mean(D.rr > D.nbr):.0%}")
print("Jaccard of readings' top-10 pages          ", q(D.jac), f"; share with jac < 0.2 (topic split): {np.mean(D.jac < 0.2):.0%}")
y = F.label.to_numpy(); clear = F.index[y == 0]


def auc(a, b):
    a, b = np.asarray(a), np.asarray(b); r = pd.Series(np.r_[a, b]).rank().to_numpy()
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def ci(a, b, B=400):
    s = [auc(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(B)]
    return np.percentile(s, [2.5, 97.5])


strata = {"all ambiguous": D.index, "topic split (jac<0.2)": D.index[D.jac < 0.2], "same pages (jac>=0.5)": D.index[D.jac >= 0.5]}
t3 = D.rr.quantile([1 / 3, 2 / 3]).to_numpy()
strata["readings far (rr low third)"] = D.index[D.rr <= t3[0]]; strata["readings near (rr top third)"] = D.index[D.rr > t3[1]]
print("\nAUROC P(ambiguous stratum > clear questions), 95% bootstrap; clear n =", len(clear))
cols = ["w1", "longbar", "ltmax", "mean_cos", "std_cos", "w1_corp", "nwords"]
print(f"{'stratum':30s} {'n':>5s} " + " ".join(f"{c:>17s}" for c in cols))
for name, ix in strata.items():
    cells = []
    for c in cols:
        a, b = F.loc[ix, c].to_numpy(), F.loc[clear, c].to_numpy(); lo, hi = ci(a, b)
        cells.append(f"{auc(a, b):.2f} [{lo:.2f},{hi:.2f}]")
    print(f"{name:30s} {len(ix):5d} " + " ".join(f"{x:>17s}" for x in cells))
D.to_csv(f"{out}/probe_slot.csv")
