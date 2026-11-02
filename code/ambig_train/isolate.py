# isolate.py -- written on day 4 at 11:06. How often does cutting the longest MST edge (the longest H0 bar) of a query's 50
# unit offsets isolate a small group (<= 3 chunks) rather than split the neighbours into two sizeable groups?
# And how close is the longest bar to the largest nearest-neighbour distance (the loneliest neighbour)?
# Usage: python isolate.py ambig OUT/retrieved.npz OUT/corpus_emb.npy   (OUT from ambig.py retrieve)
#        python isolate.py lenctl OUT/emb_<model>.npz                (OUT from code/lenctl/build.py)
import sys, numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial.distance import pdist, squareform
def split(q, X):
    U = X - q; U /= np.linalg.norm(U, axis=1, keepdims=True); D = squareform(pdist(U))
    T = minimum_spanning_tree(D).toarray(); i, j = np.unravel_index(np.argmax(T), T.shape); L = T[i, j]
    T[i, j] = 0; _, lab = connected_components(T + T.T, directed=False); small = np.bincount(lab).min()
    np.fill_diagonal(D, np.inf); return small, L, D.min(1).max()
mode = sys.argv[1]
if mode == "ambig":
    R = np.load(sys.argv[2]); E = np.load(sys.argv[3]); rows = [split(R["Q"][k].astype(float), E[R["idx"][k]].astype(float)) for k in range(len(R["Q"]))]
    lab = R["label"]
else:
    import json
    Z = np.load(sys.argv[2]); E = {k: Z[k].astype(float) for k in Z.files}; rows = []; lab = []
    for qk, ck in (("Q1", "C3_g0"), ("Q3_g0", "C1")):
        C = E[ck]
        for q in E[qk]:
            top = np.argsort(-(C @ q))[:50]; rows.append(split(q, C[top])); lab.append(qk)
    lab = np.array(lab)
S = np.array(rows)
for g in np.unique(lab):
    s = S[lab == g]
    print(f"{mode} group {g}: n {len(s)}; cut isolates <= 3 chunks in {np.mean(s[:, 0] <= 3):.2f} (1 chunk: {np.mean(s[:, 0] == 1):.2f}); "
          f"longest bar = largest NN distance in {np.mean(np.isclose(s[:, 1], s[:, 2])):.2f}; median smaller side {np.median(s[:, 0]):.0f}")
