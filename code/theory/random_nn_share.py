# random_nn_share.py -- written 2026-11-02 08:49 CET (first run 08:35 CET, inline). Yardstick for the paper's 76%:
# share of minimum-spanning-tree edges that are nearest-neighbour edges, for N = 50 i.i.d. standard Gaussian points in
# dimension k, 300 clouds per k, seed 0. numpy only (Prim on the dense distance matrix). Output: random_nn_share.txt
# 2026-11-02 08:53 CET: also the share of clouds whose longest MST edge cuts off a single point (yardstick for the 73%).
# Usage: python3 -I random_nn_share.py > random_nn_share.txt
import numpy as np
rng = np.random.default_rng(0)
def share(X):
    N = len(X); D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
    nn = D.argmin(1); NN = {tuple(sorted((i, nn[i]))) for i in range(N)}
    inT = np.zeros(N, bool); inT[0] = True; best = D[0].copy(); par = np.zeros(N, int); E = []
    for _ in range(N - 1):
        j = int(np.where(inT, np.inf, best).argmin()); E.append(tuple(sorted((par[j], j)))); inT[j] = True
        upd = D[j] < best; par[upd] = j; best = np.minimum(best, D[j])
    T = [(D[a, b], a, b) for a, b in E]; L, a, b = max(T)
    adj = {i: set() for i in range(N)}
    for _, u, v in T: adj[u].add(v); adj[v].add(u)
    lone = len(adj[a]) == 1 or len(adj[b]) == 1
    return np.mean([e in NN for e in E]), lone
print("# N=50 iid Gaussian clouds, 300 per dimension, seed 0: NN share of MST edges, median (5-95%)")
for k in (1, 2, 5, 20, 100, 768):
    r = [share(rng.standard_normal((50, k))) for _ in range(300)]; s = [x for x, _ in r]
    print(f"dim {k:4d}: {np.median(s):.2f} ({np.percentile(s, 5):.2f}-{np.percentile(s, 95):.2f})  "
          f"longest edge cuts off one point: {np.mean([l for _, l in r]):.2f}")
