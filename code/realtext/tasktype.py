# tasktype.py -- why text-embedding-005 differs: it is the only model whose queries are embedded with a different
# task type (RETRIEVAL_QUERY) from its passages (RETRIEVAL_DOCUMENT). Reruns the paper's retrieval and H0 features for
# vertex-005 twice per corpus -- queries as embedded for the paper, and the same query texts' passage embeddings -- with
# the retrieval, cleaning, exclusion and features of features.py (imported). No new embeddings are made.
#   M:  multi-factual-like Q750|C250 vs polysemic-like Q250|C750 (span-overlap exclusion), AUROC = P(larger for Q750|C250)
#   F:  fixed query length, own document out: Q250 over the size-matched C250 subsample (seed 0) vs Q250|C750,
#       AUROC = P(larger with the C250 chunking); one subsample, no control (the paper averages ten and subtracts a control)
#   gap: mean cosine among the 50 neighbours minus their mean cosine to the query (> 0: query outside its neighbours)
# Needs the embeddings. Usage: [CORPUS=<c>] CLEAN=1 OUTROOT=out_clean python tasktype.py  -> one line per query type
import os
import numpy as np
import features as F

MODEL = "vertex-005"


def auc(x, y):  # P(y > x) + 0.5 P(tie)
    r = np.argsort(np.argsort(np.r_[x, y])) + 1.0; n0, n1 = len(x), len(y)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def cell(meta, P, Qe, qg, cg, rows, exclude):
    qm, cm = meta[qg], meta[cg].iloc[rows].reset_index(drop=True)
    C = P[cg][rows]; S = Qe[qg] @ C.T
    same = qm.lecture.to_numpy(dtype=object)[:, None] == cm.lecture.to_numpy(dtype=object)[None, :]
    ov = (qm.start.values[:, None] < cm.end.values[None, :]) & (cm.start.values[None, :] < qm.end.values[:, None])
    S = np.where(same if exclude == "doc" else same & ov, -np.inf, S)
    top = np.argsort(-S, axis=1, kind="stable")[:, :F.K]; out = []
    for i in range(len(qm)):
        X = C[top[i]]; a = S[i, top[i]]; G = X @ X.T
        w1, _, lb = F.ph(F.offsets(Qe[qg][i], X))
        out.append((w1, lb, G[np.triu_indices(F.K, 1)].mean() - a.mean()))
    return np.array(out)


meta, P, Q = F.load_model(MODEL)
sub = np.load(F.OUT / "c250_subsample_idx.npy")
full = {g: np.arange(len(meta[g])) for g in (250, 750)}
corpus = os.environ.get("CORPUS", "nobel")
for name, Qe in (("as query (paper)", Q), ("as passage", P)):
    m1, m3 = cell(meta, P, Qe, 250, 750, full[750], True), cell(meta, P, Qe, 750, 250, full[250], True)
    f750, f250 = cell(meta, P, Qe, 250, 750, full[750], "doc"), cell(meta, P, Qe, 250, 250, sub, "doc")
    print(f"{corpus:9s} {name:17s} medians Q250|C750: W1 {np.median(m1[:, 0]):.3f} L_max {np.median(m1[:, 1]):.3f} "
          f"gap {np.median(m1[:, 2]):+.3f} | M AUROC W1 {auc(m1[:, 0], m3[:, 0]):.2f} L_max {auc(m1[:, 1], m3[:, 1]):.2f} "
          f"| F AUROC W1 {auc(f750[:, 0], f250[:, 0]):.2f} L_max {auc(f750[:, 1], f250[:, 1]):.2f}", flush=True)
