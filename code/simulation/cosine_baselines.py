"""Score-only and coherence baselines for the simulation queries (all 50 retrieved passages).

Regenerates the exact queries of run_sim.py (same seeds and RNG re-seeding) and computes, from the cosines
a_i = c_i . q sorted in descending order: top-1, mean, std, OLS slope on rank; R = ||mean unit offset||;
mean sin(theta_i/2); pairwise coherence (mean cosine over all pairs of the 50); number of connected components of the
graph linking pairs with cosine >= the query's median pairwise cosine. No persistence is computed except W1(H0) on
every 50th query (check_w1), which proves row alignment with the run_sim.py table (column all_w1_h0).
Usage: python cosine_baselines.py --configs fig1 fig2 superpos superpos_s02 --seeds 0 1 2 3 4 [--workers 2] [--outdir out]
"""
import argparse, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import run_sim as R  # sets single-threaded BLAS
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from simcore import persistence_features


def score_feats(q, C):
    a = np.sort(C @ q)[::-1]                       # unit vectors (these configs are unprojected) -> cosines
    i = np.arange(len(a))
    U = C - q; U /= np.linalg.norm(U, axis=1, keepdims=True)
    th = np.arccos(np.clip(a, -1, 1))
    G = C @ C.T
    iu = np.triu_indices(len(C), 1)
    pc = G[iu]
    adj = csr_matrix(G >= np.median(pc))
    return dict(cos_top1=a[0], cos_mean=a.mean(), cos_std=a.std(), cos_slope=np.polyfit(i, a, 1)[0],
                off_R=np.linalg.norm(U.mean(axis=0)), mean_sin_half=np.mean(np.sin(th / 2)),
                pair_coherence=pc.mean(), n_components=connected_components(adj, directed=False)[0])


def unit(args):
    cfg_name, seed, outdir = args
    cfg = R.CONFIGS[cfg_name]
    P, Ch = R.generate(cfg["D"], cfg["N"], cfg["n_parent"], cfg["sigma"], seed, cfg.get("topics", "qr"))
    rows = []
    for scen in cfg["scenarios"]:
        np.random.seed(10_000 + seed)               # identical to run_sim.run_unit
        for qi, (g, qdp, C, cnf, extra) in enumerate(R.queries(scen, P, Ch, cfg["n_parent"])):
            q = np.asarray(qdp["X"], dtype=float)
            r = dict(config=cfg_name, sim_seed=seed, scenario=scen, query_idx=qi, parent_group=g,
                     corpus_nfeat=int(cnf), query_nfeat=int(qdp["n_features"]), **score_feats(q, C))
            r["check_w1"] = persistence_features(q, C, np.inf)["w1_h0"] if qi % 50 == 0 else np.nan
            rows.append(r)
    pd.DataFrame(rows).to_csv(Path(outdir) / f"cosine_{cfg_name}_seed{seed}.csv", index=False)
    return cfg_name, seed, len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["fig1", "fig2", "superpos", "superpos_s02"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--outdir", default=str(R.HERE / "out"))
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for c, s, n in ex.map(unit, [(c, s, a.outdir) for c in a.configs for s in a.seeds]):
            print(f"done {c} seed{s}: {n} queries", flush=True)
    print("wall", round(time.time() - t0, 1), "s")
