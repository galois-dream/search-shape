# audit_ambig.py -- written on day 6 at 19:45. Read-only audit of the AmbigNQ data behind the paper's AmbigNQ section and
# the probes. Checks: label counts; corpus row alignment (embeddings, titles,
# text); gold pages in corpus; gold page retrieved in top 50; spot checks of question / readings / neighbours.
import ast, json, sys, numpy as np, pandas as pd
T = json.load(open(sys.argv[1])); B = pd.read_parquet(sys.argv[2]); out = sys.argv[3]
keep = B.in_corpus.to_numpy(); TXT = B.text[keep].to_numpy(); BT = B.title[keep].to_numpy().astype(str)
TT = np.load(f"{out}/corpus_titles.npy", allow_pickle=True).astype(str); R = np.load(f"{out}/retrieved.npz"); IDX = R["idx"]
ann = lambda t: ast.literal_eval(t["annotations"]) if isinstance(t["annotations"], str) else t["annotations"]
A = [ann(t) for t in T]; lab = np.array([int(any(a["type"] == "multipleQAs" for a in x)) for x in A])
print("1. questions", len(T), "| ambiguous", lab.sum(), f"({lab.mean():.3f})", "| clear", (lab == 0).sum(),
      "| label matches retrieved.npz:", bool((lab == R["label"]).all()))
na = np.array([len(x) for x in A]); print("   annotations per question:", {int(k): int(v) for k, v in zip(*np.unique(na, return_counts=True))})
print("   keys in a record:", sorted(T[0].keys()))
print("2. corpus rows", keep.sum(), "| titles in parquet == corpus_titles.npy:", bool((BT == TT).all()), "| retrieval rows", IDX.shape)
gold = [t.get("nq_doc_title") for t in T]; has = sum(g is not None for g in gold); S = set(TT)
inC = np.array([g in S for g in gold]); print("3. nq_doc_title present for", has, "| gold page in corpus:", f"{inC.mean():.3f}")
hit = np.array([gold[i] in set(TT[IDX[i]]) for i in range(len(T))]); top1 = np.array([TT[IDX[i][0]] == gold[i] for i in range(len(T))])
print("4. gold page among top-50 titles:", f"{hit.mean():.3f}", "(ambiguous", f"{hit[lab==1].mean():.3f}", "clear", f"{hit[lab==0].mean():.3f})", "| top-1:", f"{top1.mean():.3f}")
rng = np.random.default_rng(7)
for i in rng.choice(np.flatnonzero(lab == 1), 3, replace=False):
    qa = [a for a in A[i] if a["type"] == "multipleQAs"][0]["qaPairs"]
    print(f"5. [{i}] Q: {T[i]['question']} | gold: {gold[i]}")
    print("     readings:", " || ".join(p["question"] for p in qa[:2]))
    print("     top-5 titles:", list(dict.fromkeys(TT[IDX[i][:10]]))[:5])
    print("     top-1 text:", TXT[IDX[i][0]][:140].replace("\n", " "))
