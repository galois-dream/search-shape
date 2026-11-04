# probe_answers.py -- written on day 6 at 17:25. EXPLORATORY, post hoc (not registered). Question: can the
# method see AmbigNQ's kind of ambiguity (an unbound slot inside one topic) at all? Best case for the method: the
# question's own 50 neighbours contain separate passages for two of its readings. Passage p "serves" reading r if its
# text contains one of r's annotated answers (case-insensitive, word boundaries). For each ambiguous question, take the
# two readings whose exclusive passage groups (serving one reading, not the other) are both non-empty. In the offset
# directions used for the features (ambig.py), H0 is the minimum spanning tree; the two groups join at the largest
# edge on the tree path between them (single linkage). We report where that join sits among the tree's 49 edges,
# against random groups of the same sizes drawn from the same neighbourhood.
# Usage: python probe_answers.py <ambignq train.json> <chunks.parquet> <ambig.py out dir>
import ast, json, os, re, sys
import numpy as np, pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree

T = json.load(open(sys.argv[1])); B = pd.read_parquet(sys.argv[2]); out = sys.argv[3]
keep = np.ones(len(B), bool) if os.environ.get("FULL") == "1" else B.in_corpus.to_numpy()  # FULL=1 as in ambig.py (added day 6)
TXT = B.text[keep].str.lower().to_numpy()
E = np.load(f"{out}/corpus_emb.npy").astype(np.float64); R = np.load(f"{out}/retrieved.npz"); Q = R["Q"].astype(np.float64); IDX = R["idx"]
assert len(TXT) == len(E)
P = pd.read_csv(f"{out}/probe_slot.csv").set_index("i")
ann = lambda t: ast.literal_eval(t["annotations"]) if isinstance(t["annotations"], str) else t["annotations"]
unit = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
rng = np.random.default_rng(0)


def minimax(U):
    """All-pairs single-linkage merge height = largest edge on the MST path; also the 49 MST edge lengths."""
    D = np.linalg.norm(U[:, None] - U[None], axis=2); M = minimum_spanning_tree(D).toarray(); M = M + M.T
    n = len(U); H = np.zeros((n, n))
    for s in range(n):  # tree walk from s, carrying the largest edge so far
        st, seen = [(s, 0.0)], {s}
        while st:
            v, h = st.pop(); H[s, v] = h
            for w in np.flatnonzero(M[v]):
                if w not in seen: seen.add(w); st.append((w, max(h, M[v, w])))
    return H, np.sort(M[np.triu_indices(n, 1)][M[np.triu_indices(n, 1)] > 0])[::-1]


rows = []
for k, t in enumerate(T):
    A = [a for a in ann(t) if a["type"] == "multipleQAs"]
    if not A or len(A[0]["qaPairs"]) < 2: continue
    top = IDX[k]; txt = TXT[top]
    serve = []
    for p in A[0]["qaPairs"]:
        pats = [re.compile(r"(?<!\w)" + re.escape(s.lower().strip()) + r"(?!\w)") for s in p["answer"] if len(s.strip()) >= 2]
        serve.append(np.array([any(x.search(d) for x in pats) for d in txt]))
    best = None
    for a in range(len(serve)):
        for b in range(a + 1, len(serve)):
            ga, gb = np.flatnonzero(serve[a] & ~serve[b]), np.flatnonzero(serve[b] & ~serve[a])
            if len(ga) and len(gb) and (best is None or len(ga) + len(gb) > len(best[0]) + len(best[1])): best = (ga, gb)
    r = dict(i=k, n_read=len(serve), any_served=int(sum(s.any() for s in serve)))
    if best:
        U = unit(E[top] - Q[k][None] + np.finfo(float).eps); H, edges = minimax(U)
        join = H[np.ix_(*best)].min()
        null = []
        for _ in range(200):
            perm = rng.permutation(50); na, nb = len(best[0]), len(best[1]); null.append(H[np.ix_(perm[:na], perm[na:na + nb])].min())
        r.update(na=len(best[0]), nb=len(best[1]), join=join, join_rank=int((edges > join + 1e-12).sum()), med_edge=float(np.median(edges)),
                 join_ratio=join / np.median(edges), pct_vs_null=float(np.mean(np.array(null) < join) + 0.5 * np.mean(np.array(null) == join)),
                 null_top=float(np.mean(np.array(null) >= edges[0] - 1e-12)))
    rows.append(r)
D = pd.DataFrame(rows).set_index("i").join(P[["jac"]])
S = D.dropna(subset=["join"])
print(f"# probe_answers.py run {pd.Timestamp.now(tz='Europe/Paris'):%Y-%m-%d %H:%M %Z}; {len(D)} ambiguous questions")
print(f"readings with an answer-bearing passage among the 50 neighbours: none {np.mean(D.any_served == 0):.0%}, one {np.mean(D.any_served == 1):.0%}, two or more {np.mean(D.any_served >= 2):.0%}")
print(f"two readings with separate passages (best case): {len(S)} questions ({len(S) / len(D):.0%})")
for name, s in [("all best-case", S), ("topic split (jac<0.2)", S[S.jac < 0.2]), ("same pages (jac>=0.5)", S[S.jac >= 0.5])]:
    print(f"  {name:24s} n={len(s):4d}  groups joined by the longest bar: {np.mean(s.join_rank == 0):.0%} (random groups: {s.null_top.mean():.0%}); "
          f"by one of the 3 longest: {np.mean(s.join_rank < 3):.0%}; join / median edge: median {s.join_ratio.median():.2f}; "
          f"percentile vs random groups: median {s.pct_vs_null.median():.2f}, share above 0.95: {np.mean(s.pct_vs_null > 0.95):.0%}")
D.to_csv(f"{out}/probe_answers.csv")
