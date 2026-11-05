# mstpieces.py -- written 2026-11-02 08:03 CET. Why does W1 rank queries like the mean nearest-neighbour ratio, and what
# breaks the agreement where it is weakest (Wikipedia, Spearman 0.93)? Exact accounting from paper/theory/nn_bound.md
# (Prop. 2 / Theorem 2), measured on the same cells as mstcheck.py (main contrast Q250|C750 and Q750|C250, span-overlap
# exclusion). A separate script so that mstcheck.py and its cited output (out_clean/mst_nn.txt) stay as they are.
# Per query, N = 50 neighbours, a_i = |c_i - q|, d = distances between offset directions U, rho_ij = |c_i - c_j|/sqrt(a_i a_j):
#   W1 - N/(N-1) S_r = gap + radial + asym   (exact; residual printed)
#   gap    = (G - M) / (2(N-1))     G: MST(d) edges that are not NN edges; M: lengths of mutual-NN pairs   (arrangement)
#   radial = N/(N-1) (S_d - S_rho)  cosines distort only the radial term (d <= rho)                       (cosines)
#   asym   = N/(N-1) (S_rho - S_r)  dividing by a_i instead of sqrt(a_i a_j)                              (cosines)
#   S_m = sum_i delta_i(m) / (2N), delta_i = distance to nearest other point; S_r = mean_i r_i / 2 = proxyW of mstcheck.
# Prints per corpus x model, for each cell: Spearman W1 vs S_r and vs S_rho; vs S_r + gap (only the cosine pieces left
# out) and vs S_r + radial + asym (only the arrangement piece left out); median |piece| / within-cell SD(W1);
# median cosine range; max |residual|.
# 2026-11-02 08:33 CET (review): also SD(term)/SD(W1) (variance share, not level), and two nulls: Spearman of W1 with
#   the mean cosine to the query and with the median cosine of each passage to its nearest other passage; NN share of bars.
#   DUMP=<file.json>: for the Q250|C750 cell, write the sorted bars and nearest-neighbour flags of the query whose NN share
#   is the cell median (for the paper's figure of one real barcode).
# Usage: CORPUS=<c> CLEAN=1 OUTROOT=out_clean python mstpieces.py <model>   (one line on stdout)
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr


def query(q, X, a):
    N = len(X); A = np.sqrt(2 - 2 * a)
    DU = squareform(pdist(F.offsets(q, X))); DC = squareform(pdist(X)); R = DC / np.sqrt(A[:, None] * A[None, :])
    T = minimum_spanning_tree(DU).tocoo(); W1 = T.data.sum() / (2 * (N - 1))
    for D in (DU, DC, R): np.fill_diagonal(D, np.inf)
    nn = DU.argmin(1); dd = DU.min(1)
    NN = {tuple(sorted((i, nn[i]))) for i in range(N)}
    flags = [tuple(sorted((i, j))) in NN for i, j in zip(T.row, T.col)]
    G_cos = 1 - DC ** 2 / 2; np.fill_diagonal(G_cos, -np.inf); mednn = np.median(G_cos.max(1))
    M = sum(DU[i, nn[i]] for i in range(N) if nn[nn[i]] == i and i < nn[i])
    G = sum(w for i, j, w in zip(T.row, T.col, T.data) if tuple(sorted((i, j))) not in NN)
    Sd, Srho, Sr = dd.sum() / (2 * N), R.min(1).sum() / (2 * N), (DC.min(1) / A).sum() / (2 * N)
    k = N / (N - 1); gap, radial, asym = (G - M) / (2 * (N - 1)), k * (Sd - Srho), k * (Srho - Sr)
    bars = sorted(zip(T.data.tolist(), flags), reverse=True)
    rr = sorted((DC.min(1) / A).tolist(), reverse=True)  # 2026-11-05: the 50 nearest-neighbour ratios r_i, for supplement Figure SM6
    return (W1, k * Sr, k * Srho, gap, radial, asym, W1 - k * Sr - gap - radial - asym, a.max() - a.min(),
            a.mean(), mednn, np.mean(flags)), (bars, rr)


def cell(meta, P, Q, qg, cg):
    qm, cm, C, Qe = meta[qg], meta[cg], P[cg], Q[qg]
    S = Qe @ C.T
    same = qm.lecture.to_numpy(dtype=object)[:, None] == cm.lecture.to_numpy(dtype=object)[None, :]
    ov = (qm.start.values[:, None] < cm.end.values[None, :]) & (cm.start.values[None, :] < qm.end.values[:, None])
    S = np.where(same & ov, -np.inf, S); top = np.argsort(-S, axis=1, kind="stable")[:, :F.K]
    out = [query(Qe[i], C[top[i]], S[i, top[i]]) for i in range(len(qm))]
    return np.array([o[0] for o in out]), [o[1][0] for o in out], [o[1][1] for o in out]


def summary(m):
    W, Xr, Xrho, gap, rad, asy, res, rng, mcos, mednn, nnsh = m.T; sd = W.std(); rho = lambda y: spearmanr(W, y)[0]
    return (f"rS {rho(Xr):.3f} rSrho {rho(Xrho):.3f} r+gap {rho(Xr + gap):.3f} r+cos {rho(Xr + rad + asy):.3f} "
            f"|gap| {np.median(abs(gap)) / sd:.2f} |rad| {np.median(abs(rad)) / sd:.2f} |asym| {np.median(abs(asy)) / sd:.2f} "
            f"cosrange {np.median(rng):.3f} res {abs(res).max():.0e} "
            f"sdgap {gap.std() / sd:.2f} sdrad {rad.std() / sd:.2f} sdasym {asy.std() / sd:.2f} "
            f"nullcos {rho(mcos):+.3f} nullnn {rho(mednn):+.3f} nnsh {np.median(nnsh):.2f}")


model = sys.argv[1]; meta, P, Q = F.load_model(model)
(m1, b1, r1), (m3, _, _) = cell(meta, P, Q, 250, 750), cell(meta, P, Q, 750, 250)
print(f"{F.CORPUS:9s} {model:16s} Q250|C750: {summary(m1)} || Q750|C250: {summary(m3)}", flush=True)
if os.environ.get("DUMP"):
    nnsh = m1[:, 10]; i = int(np.argsort(np.abs(nnsh - np.median(nnsh)))[0])
    json.dump({"corpus": F.CORPUS, "model": model, "cell": "Q250|C750", "query_index": i, "nn_share": float(nnsh[i]),
               "W1": float(m1[i, 0]), "bars": [[float(l), bool(f)] for l, f in b1[i]]}, open(os.environ["DUMP"], "w"))
    if os.environ.get("DUMPTERMS"):  # 2026-11-05 CET: the same query's terms of (5.6)-(5.7), in units of W1, for supplement Figure SM5
        W, Xr, Xrho, gap, rad, asy, res = m1[i, :7]
        json.dump({"query_index": i, "W1": float(W), "ratios": float(Xr), "clump": float(gap), "radial": float(rad),
                   "scaling": float(asy), "residual": float(res),
                   "cell_W1": [round(float(x), 5) for x in m1[:, 0]], "cell_ratios": [round(float(x), 5) for x in m1[:, 1]],
                   "cell_clump": [round(float(x), 5) for x in m1[:, 3]], "ratios_r": [round(float(x), 5) for x in r1[i]]},
                  open(os.environ["DUMPTERMS"], "w"))
