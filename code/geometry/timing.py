# timing.py -- written 2026-11-02 08:37 CET. Cost per query of each statistic, for the SIMODS version's computation
# paragraph (a separate script: it measures cost, not a result of the paper). Same retrieval as mstpieces.py
# (Q250|C750 cell, span-overlap exclusion, K = 50). Per query, wall time of:
#   dist  : the 50 x 50 distance matrices (offsets and passages)       O(K^2 D)
#   nn    : nearest-neighbour ratios from the passage distances          O(K^2)
#   mst   : minimum spanning tree of the offsets (scipy)                 O(K^2) dense
#   h0    : ripser, H0 only                                              ~ MST
#   h1    : ripser, H0 and H1                                            reduction over edges and triangles
#   ctrl  : one random-bearing control (redraw bearings + ripser H0/H1)
# 2026-11-02 08:52 CET (scale review): ripser now gets the distance matrix (distance_matrix=True), so h0/h1 exclude
#   ripser recomputing distances; 'e2e_nn' / 'e2e_ph' = end-to-end per query (distances + ratios / distances + H0,H1).
# Prints medians in microseconds over the first QMAX queries, plus D. Usage: CORPUS=<c> CLEAN=1 OUTROOT=out_clean
#   QMAX=300 python timing.py <model>
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from ripser import ripser
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform, pdist

model = sys.argv[1]; meta, P, Q = F.load_model(model); QMAX = int(os.environ.get("QMAX", 300))
qm, cm, C, Qe = meta[250], meta[750], P[750], Q[250]
S = Qe @ C.T
same = qm.lecture.to_numpy(dtype=object)[:, None] == cm.lecture.to_numpy(dtype=object)[None, :]
ov = (qm.start.values[:, None] < cm.end.values[None, :]) & (cm.start.values[None, :] < qm.end.values[:, None])
S = np.where(same & ov, -np.inf, S); top = np.argsort(-S, axis=1, kind="stable")[:, :F.K]
rng = np.random.default_rng(0); T = {k: [] for k in ("dist", "nn", "mst", "h0", "h1", "ctrl", "e2e_nn", "e2e_ph")}
for i in range(min(QMAX, len(qm))):
    q, X, a = Qe[i], C[top[i]], S[i, top[i]]
    t0 = time.perf_counter(); U = F.offsets(q, X); DU = squareform(pdist(U)); DC = squareform(pdist(X)); t1 = time.perf_counter()
    A = np.sqrt(2 - 2 * a); np.fill_diagonal(DC, np.inf); r = DC.min(1) / A; t2 = time.perf_counter()
    minimum_spanning_tree(DU); t3 = time.perf_counter()
    ripser(DU, maxdim=0, distance_matrix=True); t4 = time.perf_counter()
    ripser(DU, maxdim=1, distance_matrix=True); t5 = time.perf_counter()
    R = rng.standard_normal(X.shape); R -= np.outer(R @ q, q); R /= np.linalg.norm(R, axis=1, keepdims=True)
    Xc = a[:, None] * q + np.sqrt(1 - a ** 2)[:, None] * R; ripser(F.offsets(q, Xc), maxdim=1); t6 = time.perf_counter()
    for k, v in zip(T, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t2 - t0, (t1 - t0) + (t5 - t4))): T[k].append(v * 1e6)
print(f"{F.CORPUS:9s} {model:16s} D={X.shape[1]} n={len(T['dist'])} median us: " +
      " ".join(f"{k} {np.median(v):.0f}" for k, v in T.items()), flush=True)
