# vmf.py -- implements the vMF plan (docs/vmf_plan.md, written before any real value): the two-meaning von Mises-Fisher
# statistic vmf2 = BIC(k=1) - BIC(k=2) on the 50 unit offset directions projected onto their top-r principal
# directions (r = 5; secondary r = 3, 8), the Gram second-eigenvalue share eig2, and the same on the corpus-bearing
# control clouds that features.py draws (identical seeds). Also records, for the real neighbours, the longest MST
# edge's endpoints and the chunks on its smaller side (artefact audit of L_max). Needs the embeddings.
# Usage: [CORPUS=wiki] [CLEAN=1 OUTROOT=out_clean] [VMF_K=100] python vmf.py <model>   -> $OUTROOT[/corpus]/vmf_<model>.csv.gz
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import ive
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

import os
from features import OUT, load_model, offsets, unit

K = int(os.environ.get("VMF_K", "50"))  # K = 100: sensitivity run (vMF plan, amendment 2)
TAG = "" if K == 50 else f"_k{K}"

NDRAW = 20


def log_iv(nu, x):
    """log I_nu(x) (vMF plan, amendment 1): exact via scipy's exponentially scaled ive for the small
    orders used here (r = 3, 5, 8 -> nu = 0.5, 1.5, 3); the uniform (Debye) expansion, accurate only for large nu,
    is kept for nu >= 50."""
    if nu < 50:
        return np.log(ive(nu, x)) + x
    z = x / nu; s = np.sqrt(1 + z * z)
    return nu * (s + np.log(z / (1 + s))) - 0.5 * np.log(2 * np.pi * nu) - 0.5 * np.log(s)


def vmf_ll(X, mus, w, kappa):
    p = X.shape[1]; nu = p / 2 - 1
    logc = nu * np.log(kappa) - (p / 2) * np.log(2 * np.pi) - log_iv(nu, kappa)
    L = np.log(w)[None, :] + logc + kappa * X @ mus.T
    m = L.max(1, keepdims=True)
    return float((m[:, 0] + np.log(np.exp(L - m).sum(1))).sum()), L


def kappa_hat(rbar, p):  # Banerjee et al. (2005)
    rbar = min(max(rbar, 1e-6), 0.999)
    return rbar * (p - rbar ** 2) / (1 - rbar ** 2)


def bic_gap(Y, seed, restarts=5, iters=50):
    n, p = Y.shape
    m = Y.mean(0); r = np.linalg.norm(m)
    ll1 = vmf_ll(Y, (m / r)[None], np.ones(1), kappa_hat(r, p))[0]
    rng = np.random.default_rng(seed); ll2 = -np.inf
    for _ in range(restarts):
        mus = Y[rng.choice(n, 2, replace=False)]; w = np.full(2, 0.5); kap = kappa_hat(0.5, p)
        for _ in range(iters):
            _, L = vmf_ll(Y, mus, w, kap)
            R = np.exp(L - L.max(1, keepdims=True)); R /= R.sum(1, keepdims=True)
            w = np.clip(R.mean(0), 1e-3, None); w /= w.sum()
            S = R.T @ Y; nr = np.linalg.norm(S, axis=1); mus = S / np.maximum(nr, 1e-12)[:, None]
            kap = kappa_hat(nr.sum() / n, p)
        ll2 = max(ll2, vmf_ll(Y, mus, w, kap)[0])
    k1, k2 = (p - 1) + 1, 2 * (p - 1) + 1 + 1
    return (-2 * ll1 + k1 * np.log(n)) - (-2 * ll2 + k2 * np.log(n))


def stats(U, seed, rs=(5, 3, 8)):
    _, _, Vt = np.linalg.svd(U - U.mean(0), full_matrices=False)
    out = {}
    for r in rs:
        Y = U @ Vt[:r].T; Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        out[f"vmf2_r{r}"] = bic_gap(Y, seed)
    ev = np.linalg.eigvalsh(U @ U.T)[::-1]; out["eig2"] = float(ev[1] / ev.sum())
    return out


def longest_edge(U, ids):
    D = squareform(pdist(U)); T = minimum_spanning_tree(D).toarray()
    i, j = np.unravel_index(np.argmax(T), T.shape); le = float(T[i, j]); T[i, j] = 0
    _, lab = connected_components(T + T.T, directed=False)
    small = np.flatnonzero(lab == lab[i]) if (lab == lab[i]).sum() <= (lab == lab[j]).sum() else np.flatnonzero(lab == lab[j])
    return dict(le_len=le, le_a=str(ids[i]), le_b=str(ids[j]), le_small_n=int(len(small)),
                le_small_ids=";".join(str(ids[k]) for k in small) if len(small) <= 5 else "")


def control_cloud(q, cos, E):
    a = np.clip(cos, -1, 1); V = unit(E - (E @ q)[:, None] * q[None, :])
    return offsets(q, a[:, None] * q[None, :] + np.sqrt(1 - a ** 2)[:, None] * V)


def run(model, variant, qg, cg, meta, P, Q, rows_idx, exclude, ndraw, log):
    qm, cm = meta[qg], meta[cg].iloc[rows_idx].reset_index(drop=True)
    Cemb = P[cg][rows_idx]; S = Q[qg] @ Cemb.T
    same = qm.lecture.to_numpy(dtype=object)[:, None] == cm.lecture.to_numpy(dtype=object)[None, :]
    ov = (qm.start.values[:, None] < cm.end.values[None, :]) & (cm.start.values[None, :] < qm.end.values[:, None])
    excl = same if exclude == "doc" else same & ov
    S = np.where(excl, -np.inf, S)
    order = np.argsort(-S, axis=1, kind="stable"); top = order[:, :K]
    rng1 = np.random.default_rng(1)  # single-draw corpus-bearing control, same sequence as features.run_cell
    out = []
    for i in range(len(qm)):
        idx = top[i]; cos = S[i, idx]; q = Q[qg][i]
        U = offsets(q, Cemb[idx])
        r = dict(model=model, variant=variant, cell=f"Q{qg}|C{cg}", query_id=qm.id[i], lecture=qm.lecture[i])
        r.update(stats(U, seed=i)); r.update(longest_edge(U, cm.id.to_numpy()[idx]))  # n = K points
        pool_i = np.setdiff1d(np.flatnonzero(~excl[i]), idx)
        emp = rng1.choice(pool_i, K, replace=False)
        if ndraw == 1:
            c = stats(control_cloud(q, cos, Cemb[emp]), seed=i, rs=(5,))
            r["vmf2_enull"], r["eig2_enull"] = c["vmf2_r5"], c["eig2"]
        for k in range(ndraw if ndraw > 1 else 0):
            g = np.random.default_rng((k, 1, i))
            E = Cemb[g.choice(pool_i, K, replace=len(pool_i) < K)]
            c = stats(control_cloud(q, cos, E), seed=i, rs=(5,))
            r[f"vmf2_enull{k:02d}"], r[f"eig2_enull{k:02d}"] = c["vmf2_r5"], c["eig2"]
        out.append(r)
    log(f"{model} {variant} Q{qg}|C{cg}: n={len(out)}")
    return out


def main(model):
    t0 = time.time()
    log = lambda m: print(f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {m}", flush=True)
    meta, P, Q = load_model(model)
    sub = np.load(OUT / "c250_subsample_idx.npy"); full = {250: np.arange(len(P[250])), 750: np.arange(len(P[750]))}
    rows = []
    for variant, qg, cg, idx, ex, nd in (("primary", 250, 750, full[750], True, 1), ("primary", 750, 250, full[250], True, 0),
                                         ("dens", 250, 250, sub, True, 1), ("lodo", 250, 750, full[750], "doc", NDRAW),
                                         ("lodo_dens", 250, 250, sub, "doc", NDRAW)):
        rows += run(model, variant, qg, cg, meta, P, Q, idx, ex, nd, log)
    pd.DataFrame(rows).to_csv(OUT / f"vmf{TAG}_{model}.csv.gz", index=False)
    log(f"done {model} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
