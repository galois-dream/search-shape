# features.py -- per embedding model: exact cosine top-K (K=50) retrieval for the four query x corpus granularity
# cells with the span-overlap exclusion, and per-query persistence features (same definitions as
# ../simulation/simcore.py):
#   offsets u_i = (c_i - q + eps_machine) / ||.||, Vietoris-Rips via ripser (maxdim 1, Z/2, euclidean)
#   W1(H0) = mean over finite H0 deaths of d/2;  LT_max(H1) = max (d-b)/2 (0 if no H1);  longest H0 bar = max d
# at (a) all K=50 neighbours, (b) the eps=0.4 rule ||c_i-q||/min||c-q|| - 1 <= 0.4 among the 50 (needs >=3), plus the
# prefix table for the eps sweep (main cells). Nulls (each keeps every neighbour's cosine a_i to the query and replaces
# its bearing): isotropic matched-cosine (c_i' = a_i q + sqrt(1-a_i^2) v_i, v_i uniform unit _|_ q, rng seed 0);
# corpus-bearing / empirical (bearing of a random chunk outside the query's top-50 and its own text, seed 1); local
# (bearing of a chunk from the query's own ranks 51-500, seed 2); with --nullseeds 20, 20 independent draws of the
# last two on the leave-document-out arms. Baselines: top-1, mean, std, OLS slope of cosine on rank; coherence
# (nn_pair_cos = mean pairwise cosine among the 50); connectivity (nn_ncomp = connected components at tau = the
# query's median pairwise cosine); own-document share of the 50; sorted cosines and Gram summaries for the
# gradient-boosting rival; mean_i sin(theta_i/2) and R = ||mean_i u_i||. Variants: primary (4 cells, overlap
# exclusion), noexcl (2 off-diagonal cells, no exclusion), dens (C250 subsampled to |C750| uniformly per document,
# seed 0), lodo (4 cells, the query's whole document excluded), lodo_dens (Q250 vs subsampled C250, lodo).
# Extraction-artefact filter (CLEAN=1; the paper's analysis, written to OUTROOT=out_clean): clean_mask drops
# letter-spaced chunks (share of 1-2-letter alphabetic tokens > CLEAN_T, default 0.5) from both chunkings, keeping them
# aligned; CLEAN_T and CLEAN_RAND (extra random 750-token chunks dropped) give the arms of the dose-response (dose.py).
# Usage: [CORPUS=wiki] [CLEAN=1 OUTROOT=out_clean] python features.py <model> [--refcheck] [--nullseeds 20 --nullvariants lodo,lodo_dens]
import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from ripser import ripser
from scipy.sparse.csgraph import connected_components

ROOT = Path(os.environ.get("RT_WORK", str(Path(__file__).resolve().parent / "work")))  # see README
CORPUS = os.environ.get("CORPUS", "nobel")  # nobel: data/ emb/ out/ ; wiki: wiki/data wiki/emb out/wiki
BASE = ROOT if CORPUS == "nobel" else ROOT / CORPUS
OUTROOT = os.environ.get("OUTROOT", "out")  # the paper: OUTROOT=out_clean with CLEAN=1
CLEAN = os.environ.get("CLEAN", "0") == "1"  # drop letter-spaced extraction artefacts (see clean_mask)
OUT = ROOT / OUTROOT if CORPUS == "nobel" else ROOT / OUTROOT / CORPUS
K, EPS_RULE, EPSM = 50, 0.4, np.finfo(float).eps
CELLS = [(250, 750), (750, 250), (250, 250), (750, 750)]  # (query gran, corpus gran)
MAIN = [(250, 750), (750, 250)]


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def offsets(q, C):
    U = C - q[None, :] + EPSM
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def ph(U):
    dg = ripser(U, maxdim=1)["dgms"]
    h0 = dg[0][:, 1]; h0 = h0[np.isfinite(h0)]
    lt = dg[1][:, 1] - dg[1][:, 0]
    return (float(np.mean(h0 / 2.0)) if len(h0) else np.nan, float(np.max(lt) / 2.0) if len(lt) else 0.0,
            float(np.max(h0)) if len(h0) else np.nan)


def one_query(args):
    q, C, cos, seed_vecs, emp_vecs, want_prefix, loc_vecs = args  # C: K x d sorted by cosine desc; seed_vecs: K x d
    # gaussian draws (isotropic null); emp_vecs: K x d random other corpus chunks (empirical-bearing, global);
    # loc_vecs: K x d chunks drawn from the query's own ranks 51-500 (local-bearing null)
    out = {}
    U = offsets(q, C)
    out["w1"], out["ltmax"], out["longbar"] = ph(U)
    out["R"] = float(np.linalg.norm(U.mean(0)))
    out["mean_sin_half"] = float(np.mean(np.sqrt(np.clip((1 - cos) / 2, 0, None))))
    # eps rule on euclidean distances (unit vectors: ||c-q|| = sqrt(2-2cos)); prefix since sorted
    dist = np.linalg.norm(C - q[None, :], axis=1)
    ratio = dist / dist.min() - 1
    n04 = int(np.sum(ratio <= EPS_RULE))
    out["n_e04"] = n04
    out["w1_e04"], out["ltmax_e04"], out["longbar_e04"] = ph(U[:n04]) if n04 >= 3 else (np.nan,) * 3
    # matched-cosine null
    V = seed_vecs - (seed_vecs @ q)[:, None] * q[None, :]
    V = unit(V)
    a = np.clip(cos, -1, 1)
    Cn = a[:, None] * q[None, :] + np.sqrt(1 - a ** 2)[:, None] * V
    Un = offsets(q, Cn)
    out["w1_null"], out["ltmax_null"], out["longbar_null"] = ph(Un)
    out["R_null"] = float(np.linalg.norm(Un.mean(0)))
    # empirical-bearing null: same a_i, bearing from a random corpus chunk's component _|_ q
    Ve = unit(emp_vecs - (emp_vecs @ q)[:, None] * q[None, :])
    Ue = offsets(q, a[:, None] * q[None, :] + np.sqrt(1 - a ** 2)[:, None] * Ve)
    out["w1_enull"], out["ltmax_enull"], out["longbar_enull"] = ph(Ue)
    if loc_vecs is not None:
        Vl = unit(loc_vecs - (loc_vecs @ q)[:, None] * q[None, :])
        Ul = offsets(q, a[:, None] * q[None, :] + np.sqrt(1 - a ** 2)[:, None] * Vl)
        out["w1_lnull"], out["ltmax_lnull"], out["longbar_lnull"] = ph(Ul)
    if want_prefix:  # features for every prefix length n=3..K, and the eps ratio at which each n enters
        out["prefix"] = [(n, float(ratio[n - 1]), *ph(U[:n])) for n in range(3, K + 1)]
    return out


def gram_feats(G, cos):
    """Non-topological summary for the flexible rival A': sorted cosines to the query, top-5 eigenvalues
    of the 50x50 Gram matrix over its trace, quantiles of pairwise cosines, quantiles of each item's nearest-item cosine."""
    iu = np.triu_indices(K, 1); pc = G[iu]
    ev = np.sort(np.linalg.eigvalsh(G))[::-1][:5] / np.trace(G)
    Gn = G.copy(); np.fill_diagonal(Gn, -np.inf); nn = Gn.max(1)
    f = {f"c{j + 1:02d}": float(c) for j, c in enumerate(cos)}
    f.update({f"g_eig{j + 1}": float(e) for j, e in enumerate(ev)})
    f.update(g_mean=float(pc.mean()), g_med=float(np.median(pc)), g_q10=float(np.quantile(pc, 0.1)),
             g_q90=float(np.quantile(pc, 0.9)), nn_q10=float(np.quantile(nn, 0.1)), nn_med=float(np.median(nn)),
             nn_q90=float(np.quantile(nn, 0.9)))
    return f


def clean_mask(meta):
    """Rows to KEEP per chunking. A chunk is a letter-spaced PDF-extraction artefact when more than
    half of its alphabetic tokens have 1-2 letters ("T h e st ati sti c al"). A 750-token chunk is dropped if it or any
    250-token chunk inside its span is; a 250-token chunk is dropped if it is, or if the 750-token chunk containing it is."""
    import re
    short = lambda t: (lambda toks: sum(len(x) <= 2 for x in toks) / max(len(toks), 1))(re.findall(r"[A-Za-z]+", t))
    T = float(os.environ.get("CLEAN_T", "0.5"))  # other values: the dose-response arms (dose.py)
    bad = {g: np.array([short(json.loads(l)["text"]) > T for l in open(BASE / "data" / f"chunks_c{g}.jsonl")]) for g in (250, 750)}
    m250, m750 = meta[250], meta[750]
    b250, b750 = bad[250].copy(), bad[750].copy()
    span = lambda L, s, e: (m250.lecture.values == L) & (m250.start.values >= s) & (m250.end.values <= e)
    for j, (L, s0, e0) in enumerate(zip(m750.lecture, m750.start, m750.end)):
        if bad[250][span(L, s0, e0)].any():
            b750[j] = True
    nr = int(os.environ.get("CLEAN_RAND", "0"))  # also drop nr random kept 750-token chunks (seed 0): size control (dose.py)
    if nr:
        b750[np.random.default_rng(0).choice(np.flatnonzero(~b750), nr, replace=False)] = True
    for j, (L, s0, e0) in enumerate(zip(m750.lecture, m750.start, m750.end)):
        if b750[j]:
            b250 |= span(L, s0, e0)
    return {250: ~b250, 750: ~b750}


def load_model(model):
    E = BASE / "emb" / model
    meta = {g: pd.read_json(BASE / "data" / f"chunks_c{g}.jsonl", lines=True)[["id", "lecture", "idx", "start", "end"]]
            for g in (250, 750)}
    P = {g: unit(np.load(E / f"passage_c{g}.npy").astype(np.float64)) for g in (250, 750)}
    Qf = {g: E / f"query_c{g}.npy" for g in (250, 750)}
    Q = {g: unit(np.load(Qf[g]).astype(np.float64)) if Qf[g].exists() else P[g] for g in (250, 750)}
    if model.startswith("vertex"):  # Vertex needs its RETRIEVAL_QUERY embeddings; never fall back silently
        assert all(Qf[g].exists() for g in (250, 750)), "missing Vertex query embeddings"
    for g in (250, 750):
        assert len(P[g]) == len(meta[g]) == len(Q[g])
    if CLEAN:
        keep = clean_mask(meta)
        meta = {g: meta[g][keep[g]].reset_index(drop=True) for g in (250, 750)}
        P = {g: P[g][keep[g]] for g in (250, 750)}; Q = {g: Q[g][keep[g]] for g in (250, 750)}
    return meta, P, Q


def multi_null(q, C, cos, pool_i, loc, nseeds, tag):
    """nseeds independent draws of the empirical (global) and local bearing nulls; draw k uses
    default_rng((k, 1, tag)) / default_rng((k, 2, tag)), tag = query index, so draws are independent across k."""
    a = np.clip(cos, -1, 1); out = {}
    for k in range(nseeds):
        for name, src, sid in (("enull", pool_i, 1), ("lnull", loc, 2)):
            r = np.random.default_rng((k, sid, tag))
            E = C[r.choice(src, K, replace=len(src) < K)]
            V = unit(E - (E @ q)[:, None] * q[None, :])
            w, l, b = ph(offsets(q, a[:, None] * q[None, :] + np.sqrt(1 - a ** 2)[:, None] * V))
            out[f"w1_{name}{k:02d}"], out[f"ltmax_{name}{k:02d}"], out[f"longbar_{name}{k:02d}"] = w, l, b
    return out


def run_cell(pool, model, variant, qg, cg, meta, P, Q, corpus_rows, exclude, want_prefix, log, nullseeds=0):
    qm, cm = meta[qg], meta[cg].iloc[corpus_rows].reset_index(drop=True)
    Cemb = P[cg][corpus_rows]
    S = Q[qg] @ Cemb.T
    same = qm.lecture.to_numpy(dtype=object)[:, None] == cm.lecture.to_numpy(dtype=object)[None, :]
    ov = (qm.start.values[:, None] < cm.end.values[None, :]) & (cm.start.values[None, :] < qm.end.values[:, None])
    # exclude: True = overlapping spans; "doc" = every chunk of the query's own document (leave-document-out)
    excl = same if exclude == "doc" else same & ov
    if exclude:
        S = np.where(excl, -np.inf, S)
    order = np.argsort(-S, axis=1, kind="stable")
    top = order[:, :K]
    rng = np.random.default_rng(0)
    rng1 = np.random.default_rng(1)  # empirical-bearing null draws
    rng2 = np.random.default_rng(2)  # local-bearing null draws (ranks 51-500)
    cl = cm.lecture.to_numpy(dtype=object)
    d = Cemb.shape[1]
    jobs, rows = [], []
    for i in range(len(qm)):
        idx = top[i]; cos = S[i, idx]
        assert np.all(np.isfinite(cos)), "excluded candidate entered top-K"
        pool_i = np.setdiff1d(np.flatnonzero(~excl[i]), idx)  # not overlapping the query, not in its top-50
        emp = Cemb[rng1.choice(pool_i, K, replace=False)]
        loc = order[i, K:500]; loc = loc[np.isfinite(S[i, loc])]
        locv = Cemb[rng2.choice(loc, K, replace=len(loc) < K)]
        o = one_query((Q[qg][i], Cemb[idx], cos, rng.standard_normal((K, d)), emp, want_prefix, locv))
        if nullseeds:
            o.update(multi_null(Q[qg][i], Cemb, cos, pool_i, loc, nullseeds, i))
        jobs.append(o)
        G = Cemb[idx] @ Cemb[idx].T; iu = np.triu_indices(K, 1)
        t = cm.iloc[idx[0]]
        nested = (t.lecture == qm.lecture[i]) and ((t.start <= qm.start[i] and qm.end[i] <= t.end) or
                                                  (qm.start[i] <= t.start and t.end <= qm.end[i]))
        rows.append(dict(model=model, variant=variant, cell=f"Q{qg}|C{cg}", qg=qg, cg=cg, query_id=qm.id[i],
                         lecture=qm.lecture[i], top1=float(cos[0]), mean50=float(cos.mean()),
                         std50=float(cos.std()), slope50=float(np.polyfit(np.arange(1, K + 1), cos, 1)[0]), nn_pair_cos=float(G[iu].mean()), nn_ncomp=int(connected_components(G >= np.median(G[iu]), directed=False)[0]),
                         qlen_tok=int(qm.end[i] - qm.start[i]), top1_nested=bool(nested), n_local=len(loc),
                         own_share=float(np.mean(cl[idx] == qm.lecture[i])), **gram_feats(G, cos),
                         top1_same_lecture=bool(t.lecture == qm.lecture[i])))
    res = jobs
    prefix = []
    for r, o in zip(rows, res):
        if "prefix" in o:
            for n, rat, w1, lt, lb in o.pop("prefix"):
                prefix.append(dict(model=model, variant=variant, cell=r["cell"], query_id=r["query_id"],
                                   lecture=r["lecture"], n=n, ratio=rat, w1=w1, ltmax=lt, longbar=lb))
        r.update(o)
    log(f"{model} {variant} Q{qg}|C{cg}: n={len(rows)} corpus={len(cm)}")
    return rows, prefix


def refcheck(meta, P, Q, log):
    """Equality with the simulation's independent implementation (simcore.persistence_features) on 20 queries."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation"))
    from simcore import persistence_features
    ref = lambda q, C, dist, ids, depsilon: persistence_features(q, C, depsilon)
    rng = np.random.default_rng(0)
    worst = 0.0
    for i in rng.choice(len(meta[250]), 20, replace=False):
        q = Q[250][i]; S = P[750] @ q
        idx = np.argsort(-S)[:K]; C = P[750][idx]
        dist = np.linalg.norm(C - q, axis=1)
        mine = one_query((q, C, S[idx], np.zeros((K, len(q))) + 1, np.eye(K, len(q)), False, None))
        for dep, suf in ((1e9, ""), (EPS_RULE, "_e04")):
            r = ref(q, C, dist, np.arange(K), depsilon=dep)
            refv = [r["w1_h0"], 0.0 if np.isnan(r["ltmax_h1"]) and r["nneighbours"] >= 3 else r["ltmax_h1"],
                    r["h0_maxbar"]]
            mv = [mine["w1" + suf], mine["ltmax" + suf], mine["longbar" + suf]]
            for a, b in zip(refv, mv):
                if not (np.isnan(a) and np.isnan(b)):
                    worst = max(worst, abs(a - b))
    log(f"refcheck: 20 queries (Q250|C750, no exclusion), K=50 and eps=0.4, max |ref - mine| = {worst:.3g}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--workers", type=int, default=1)  # ignored (single process)
    ap.add_argument("--refcheck", action="store_true")
    ap.add_argument("--nullseeds", type=int, default=0)  # 20 = independent emp/local null draws on --nullvariants
    ap.add_argument("--nullvariants", default="lodo,lodo_dens")
    a = ap.parse_args()
    out = OUT; out.mkdir(parents=True, exist_ok=True)
    logf = open(out / f"features_{a.model}.log", "a")

    def log(m):
        s = f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {m}"; print(s, flush=True); logf.write(s + "\n"); logf.flush()

    t0 = time.time()
    meta, P, Q = load_model(a.model)
    log(f"loaded {a.model}: dim={P[250].shape[1]} C250={len(P[250])} C750={len(P[750])} vertex_query={not np.shares_memory(P[250], Q[250])}")
    if a.refcheck:
        refcheck(meta, P, Q, log)
    # density control: C250 subsample, n per lecture = that lecture's C750 count, seed 0
    rng = np.random.default_rng(0)
    m250 = meta[250]; n750 = meta[750].lecture.value_counts()
    sub = np.sort(np.concatenate([rng.choice(np.flatnonzero(m250.lecture.to_numpy(dtype=object) == L), n750.get(L, 0), replace=False)
                                  for L in sorted(m250.lecture.unique())]))
    np.save(out / "c250_subsample_idx.npy", sub)
    full = {250: np.arange(len(P[250])), 750: np.arange(len(P[750]))}
    rows, prefix = [], []
    ns = lambda v: a.nullseeds if v in a.nullvariants.split(",") else 0
    pool = None
    if True:
        for qg, cg in CELLS:
            r, p = run_cell(pool, a.model, "primary", qg, cg, meta, P, Q, full[cg], True, (qg, cg) in MAIN, log, ns("primary"))
            rows += r; prefix += p
        for qg, cg in MAIN:
            r, _ = run_cell(pool, a.model, "noexcl", qg, cg, meta, P, Q, full[cg], False, False, log); rows += r
        for qg in (250, 750):
            r, _ = run_cell(pool, a.model, "dens", qg, 250, meta, P, Q, sub, True, False, log, ns("dens")); rows += r
        for qg, cg in CELLS:  # leave-document-out arm (all four cells, incl. the diagonal test)
            r, _ = run_cell(pool, a.model, "lodo", qg, cg, meta, P, Q, full[cg], "doc", False, log, ns("lodo")); rows += r
        r, _ = run_cell(pool, a.model, "lodo_dens", 250, 250, meta, P, Q, sub, "doc", False, log, ns("lodo_dens")); rows += r
    pd.DataFrame(rows).to_csv(out / f"features_{a.model}.csv.gz", index=False)
    pd.DataFrame(prefix).to_csv(out / f"prefix_{a.model}.csv.gz", index=False)
    log(f"done {a.model} in {time.time() - t0:.0f}s; rows={len(rows)} prefix_rows={len(prefix)}")


if __name__ == "__main__":
    main()
