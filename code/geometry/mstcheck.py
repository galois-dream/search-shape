# mstcheck.py -- written 2026-11-01 21:25 CET. Does the H0 barcode of a query's neighbourhood reduce to nearest-neighbour
# ratios? Main contrast cells Q250|C750 and Q750|C250, span-overlap exclusion (as tasktype.py / the paper).
# Per query: U = offset directions, C = the 50 passages, a = their cosines to the query.
#   ident : ripser H0 deaths == MST(U) edge lengths
#   nnsh  : share of MST(U) edges that are nearest-neighbour edges in U (theory: >= N/2 of N-1)
#   ovl   : share of MST(U) edges also in MST(C)
#   r_i   : dC_i / |c_i - q|, passage i's distance to its nearest other passage over its distance to the query
#   proxyW = mean_i r_i / 2 (W1 is computed from ripser deaths / 2);  proxyL = max_i r_i
#   lone  : longest bar == largest NN distance in U (the cut isolates one point)
# Prints per corpus x model: rho(W1, proxyW), rho(Lmax, proxyL) pooled over both cells, and within each cell (21:31 CET); M AUROC (P larger for Q750|C250)
# for W1, proxyW, Lmax, proxyL; medians of nnsh, ovl, lone.
# Usage: CORPUS=<c> CLEAN=1 OUTROOT=out_clean python mstcheck.py <model>   (one line on stdout)
# All 45 corpus x model lines: out_clean/mst_nn.txt (run on an Apple M1 machine, 2026-11-01, finished 21:31 CET)
import sys, numpy as np
import features as F
from ripser import ripser
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr

def auc(x, y):
    r = np.argsort(np.argsort(np.r_[x, y])) + 1.0; n0, n1 = len(x), len(y)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)

def cell(meta, P, Q, qg, cg):
    qm, cm, C, Qe = meta[qg], meta[cg], P[cg], Q[qg]
    S = Qe @ C.T
    same = qm.lecture.to_numpy(dtype=object)[:, None] == cm.lecture.to_numpy(dtype=object)[None, :]
    ov = (qm.start.values[:, None] < cm.end.values[None, :]) & (cm.start.values[None, :] < qm.end.values[:, None])
    S = np.where(same & ov, -np.inf, S); top = np.argsort(-S, axis=1, kind="stable")[:, :F.K]; out = []
    for i in range(len(qm)):
        X = C[top[i]]; U = F.offsets(Qe[i], X); a = S[i, top[i]]
        DU = squareform(pdist(U)); T = minimum_spanning_tree(DU).tocoo(); w = T.data
        EU = {tuple(sorted(e)) for e in zip(T.row, T.col)}
        DC = squareform(pdist(X)); TC = minimum_spanning_tree(DC).tocoo(); EC = {tuple(sorted(e)) for e in zip(TC.row, TC.col)}
        h0 = ripser(U, maxdim=0)["dgms"][0][:, 1]; h0 = np.sort(h0[np.isfinite(h0)])
        np.fill_diagonal(DU, np.inf); np.fill_diagonal(DC, np.inf); nn = DU.argmin(1)
        NN = {tuple(sorted((j, nn[j]))) for j in range(len(nn))}
        r = DC.min(1) / np.sqrt(2 - 2 * a)
        out.append((np.allclose(h0, np.sort(w), atol=1e-6), len(EU & NN) / len(EU), len(EU & EC) / len(EU),
                    w.mean() / 2, r.mean() / 2, w.max(), r.max(), np.isclose(w.max(), DU.min(1).max())))
    return np.array(out, dtype=float)

model = sys.argv[1]; meta, P, Q = F.load_model(model)
m1, m3 = cell(meta, P, Q, 250, 750), cell(meta, P, Q, 750, 250); b = np.r_[m1, m3]
print(f"{F.CORPUS:9s} {model:16s} ident {b[:,0].mean():.3f} nnsh {np.median(b[:,1]):.2f} (min {b[:,1].min():.2f}) "
      f"ovl {np.median(b[:,2]):.2f} lone {b[:,7].mean():.2f} | rho W1~proxyW {spearmanr(b[:,3], b[:,4])[0]:+.3f} "
      f"Lmax~proxyL {spearmanr(b[:,5], b[:,6])[0]:+.3f} | within W {spearmanr(m1[:,3], m1[:,4])[0]:+.3f} {spearmanr(m3[:,3], m3[:,4])[0]:+.3f} "
      f"L {spearmanr(m1[:,5], m1[:,6])[0]:+.3f} {spearmanr(m3[:,5], m3[:,6])[0]:+.3f} | M AUROC W1 {auc(m1[:,3], m3[:,3]):.2f} proxyW {auc(m1[:,4], m3[:,4]):.2f} "
      f"Lmax {auc(m1[:,5], m3[:,5]):.2f} proxyL {auc(m1[:,6], m3[:,6]):.2f}", flush=True)
