# crane_topo.py -- written on day 4 at 10:13. Persistence features of the crane queries' neighbourhoods
# (crane.npz from ring_examples.py with EXAMPLE_SET=crane DUMP_NEIGHBOURS=1), each against 50 draws of the
# random-bearing and the corpus-bearing control (same cosines, bearings redrawn; corpus bearings from 250 random
# non-neighbour chunks). Also the median nearest-neighbour cosine. One query per sentence: an illustration, not a test.
# Usage (needs ripser): python crane_topo.py crane.npz crane.json [OUT.dat]
import json, sys
import numpy as np
from ripser import ripser

EPSM = np.finfo(float).eps


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def ph(U):
    dg = ripser(U, maxdim=1)["dgms"]
    h0 = dg[0][:, 1]; h0 = h0[np.isfinite(h0)]; lt = dg[1][:, 1] - dg[1][:, 0]
    return np.array([np.mean(h0 / 2), np.max(lt) / 2 if len(lt) else 0.0, np.max(h0)])


def feats(q, X, V=None):
    if V is not None:
        a = np.clip(X @ q, -1, 1)
        V = unit(V - (V @ q)[:, None] * q[None, :])
        X = a[:, None] * q[None, :] + np.sqrt(1 - a ** 2)[:, None] * V
    return ph(unit(X - q[None, :] + EPSM))


D = np.load(sys.argv[1]); Q = [r["query"] for r in json.load(open(sys.argv[2]))["queries"]]
rows = []  # 10:14: also written to crane_topo.dat (sentence index, feature, real, control means and s.d.)
print(f"{'query':60s} {'feature':8s} {'real':>6s} {'random-bearing':>16s} {'corpus-bearing':>16s}  nn_med")
for k, text in enumerate(Q):
    q, X, P = (D[f"{s}{k}"].astype(np.float64) for s in ("q", "n", "pool"))
    real = feats(q, X); rng = np.random.default_rng(k)
    rnd = np.array([feats(q, X, rng.standard_normal(X.shape)) for _ in range(50)])
    cor = np.array([feats(q, X, P[rng.choice(len(P), len(X), replace=False)]) for _ in range(50)])
    G = X @ X.T; np.fill_diagonal(G, -np.inf); nn = np.median(G.max(1))
    for j, f in enumerate(("W1", "LT_max", "L_max")):
        pr, pc = (rnd[:, j] < real[j]).mean(), (cor[:, j] < real[j]).mean()
        rows.append(f"{k} {f} {real[j]:.4f} {cor[:, j].mean():.4f} {cor[:, j].std():.4f} {rnd[:, j].mean():.4f} {rnd[:, j].std():.4f} {nn:.4f}")
        print(f"{text[:60]:60s} {f:8s} {real[j]:6.3f} {rnd[:, j].mean():6.3f} (pct {pr:.2f}) {cor[:, j].mean():6.3f} (pct {pc:.2f})  {nn:.3f}")
open(sys.argv[3] if len(sys.argv) > 3 else "crane_topo.dat", "w").write("k feature real corp corp_sd rand rand_sd nn_med\n" + "\n".join(rows) + "\n")
