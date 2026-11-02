# h2probe.py -- written 2026-11-02 13:32 CET. Did we lose anything by stopping at H1? Computes H2 (voids) for real
# query neighbourhoods (Nobel lectures; the two mismatched cells; same neighbour rule as nobel/src/features.py, i.e. the
# query's overlapping chunks excluded), for the random-bearing control at the same scores, and for a synthetic case in
# which the scatter fills p random dimensions (as in the loops experiment). Reports, per setting, the share of
# neighbourhoods with any H2 bar, the median and 95th percentile of the longest H2 half-persistence, and the same for H1.
# Usage (from a copy of nobel/ with src/, data/, emb/): CLEAN=1 python h2probe.py [n_queries_per_cell]
# CLEAN=1 is required: the paper's corpora are the cleaned ones (red-team B1, 2026-11-02 14:00 CET).
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import features as F
from ripser import ripser

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 200
K = F.K


def h12(U):
    dg = ripser(U, maxdim=2)["dgms"]
    l1 = (dg[1][:, 1] - dg[1][:, 0]) / 2 if len(dg[1]) else np.zeros(0)
    l2 = (dg[2][:, 1] - dg[2][:, 0]) / 2 if len(dg[2]) else np.zeros(0)
    return (l1.max() if len(l1) else 0.0), (l2.max() if len(l2) else 0.0), len(l2)


def summ(name, R):
    R = np.array(R)
    print(f"{name:44s} n={len(R):4d}  H1 max: median {np.median(R[:,0]):.4f} p95 {np.percentile(R[:,0],95):.4f} | "
          f"H2: share with a bar {np.mean(R[:,2] > 0):.2f}, max median {np.median(R[:,1]):.4f} p95 {np.percentile(R[:,1],95):.4f} max {R[:,1].max():.4f}", flush=True)


t0 = time.time()
KEEP = {}


def auc(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(((x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y)))


for model in ("openai-3-small", "bge-base"):
    meta, P, Q = F.load_model(model)
    for qg, cg in F.MAIN:
        qm, cm = meta[qg], meta[cg]
        S = Q[qg] @ P[cg].T
        same = qm.lecture.to_numpy(dtype=object)[:, None] == cm.lecture.to_numpy(dtype=object)[None, :]
        ov = (qm.start.values[:, None] < cm.end.values[None, :]) & (cm.start.values[None, :] < qm.end.values[:, None])
        S = np.where(same & ov, -np.inf, S)
        rng = np.random.default_rng(0)
        pick = rng.choice(len(qm), min(NQ, len(qm)), replace=False)
        real, ctrl = [], []
        for i in pick:
            idx = np.argsort(-S[i], kind="stable")[:K]; q = Q[qg][i]; C = P[cg][idx]; cos = np.clip(S[i, idx], -1, 1)
            real.append(h12(F.offsets(q, C)))
            G = rng.standard_normal(C.shape); V = G - (G @ q)[:, None] * q[None, :]; V /= np.linalg.norm(V, axis=1, keepdims=True)
            ctrl.append(h12(F.offsets(q, cos[:, None] * q[None, :] + np.sqrt(1 - cos ** 2)[:, None] * V)))
        summ(f"{model} Q{qg}|C{cg} real", real); KEEP[model, qg, "real"] = np.array(real); KEEP[model, qg, "ctrl"] = np.array(ctrl)
        summ(f"{model} Q{qg}|C{cg} random bearings", ctrl)

# 13:32 CET: does H2 separate the mismatched cells? AUROC that the multi-factual-like cell (Q750|C250) is larger
for model in ("openai-3-small", "bge-base"):
    for kind in ("real", "ctrl"):
        A, B = KEEP[model, 750, kind], KEEP[model, 250, kind]
        print(f"{model:15s} {kind:4s} AUROC multi>poly: H1 max {auc(A[:,0], B[:,0]):.2f}  H2 max {auc(A[:,1], B[:,1]):.2f}", flush=True)

# 13:48 CET: effective dimension of the bearings (participation ratio of the centred 50 x D bearing matrix,
# (sum lambda)^2 / sum lambda^2), real neighbourhoods vs random bearings vs bearings drawn from the corpus (same scores).
def pr(T):
    T = T - T.mean(0); lam = np.linalg.svd(T, compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def bearings(q, C):
    B = C - (C @ q)[:, None] * q[None, :]
    return B / np.linalg.norm(B, axis=1, keepdims=True)


for model in ("openai-3-small", "bge-base"):
    meta, P, Q = F.load_model(model)
    for qg, cg in F.MAIN:
        qm, cm = meta[qg], meta[cg]
        S = Q[qg] @ P[cg].T
        same = qm.lecture.to_numpy(dtype=object)[:, None] == cm.lecture.to_numpy(dtype=object)[None, :]
        ov = (qm.start.values[:, None] < cm.end.values[None, :]) & (cm.start.values[None, :] < qm.end.values[:, None])
        S = np.where(same & ov, -np.inf, S)
        rng = np.random.default_rng(0)
        out = {"real": [], "random": [], "corpus": []}
        for i in rng.choice(len(qm), min(NQ, len(qm)), replace=False):
            idx = np.argsort(-S[i], kind="stable")[:K]; q = Q[qg][i]
            out["real"].append(pr(bearings(q, P[cg][idx])))
            G = rng.standard_normal((K, P[cg].shape[1])); out["random"].append(pr(bearings(q, G)))
            pool = np.setdiff1d(np.flatnonzero(np.isfinite(S[i])), idx)
            out["corpus"].append(pr(bearings(q, P[cg][rng.choice(pool, K, replace=False)])))
        print(f"PR {model} Q{qg}|C{cg}: " + "  ".join(f"{k} median {np.median(v):.1f} [{np.percentile(v,5):.1f}, {np.percentile(v,95):.1f}]" for k, v in out.items()), flush=True)

# synthetic: 50 neighbours around one topic centre at cosine 0.55, scatter filling p random dimensions (D=384)
D = 384; q = np.zeros(D); q[0] = 1
for p in (3, 5, 20, 40, 60, 383):
    R = []; PRs = []
    rng = np.random.default_rng(p)
    for _ in range(100):
        v = rng.standard_normal(D); v[0] = 0; v /= np.linalg.norm(v)
        c0 = 0.55 * q + np.sqrt(1 - 0.55 ** 2) * v
        B = np.linalg.qr(rng.standard_normal((D, p)))[0]
        X = c0[None, :] + (rng.standard_normal((K, p)) @ B.T) * (0.52 / np.sqrt(p))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        R.append(h12(F.offsets(q, X))); PRs.append(pr(bearings(q, X)))
    summ(f"synthetic, scatter fills p={p}", R); print(f"  participation ratio of the synthetic bearings: median {np.median(PRs):.1f}", flush=True)
print(f"elapsed {time.time() - t0:.0f} s")
