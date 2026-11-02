"""Distractor variant: retrieval from n = 2,000 instead of a designed corpus of exactly K = 50.

Each query's designed 50-passage corpus (identical to run_sim.py's) is embedded among 1,950 distractors and the top 50
by cosine are retrieved:
  S1 (multi-factual): + 1,950 children of OTHER parents (all sizes 1..n_parent-1, drawn without replacement)
  S2 (polysemic):     + 1,950 freshly generated parents (n_parent topics from all topics, same sigma)
  S3 mixed (code 4):  + 1,950 children of other parents (exact duplicates of the designed 50 removed)
Per query: share of the top 50 that are designed passages, whether the own parent is retrieved (S2), persistence
features on the retrieved 50 (real and matched-cosine null, no eps filter), cosine statistics and coherence.
Usage: python distractor.py --configs fig1 fig2 --seeds 0 1 2 3 4 [--workers 2] [--outdir out]
"""
import argparse, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import run_sim as R  # sets single-threaded BLAS
import numpy as np
import pandas as pd

from cosine_baselines import score_feats
from simcore import PassageGenerator, matched_cosine_null, persistence_features, topic_bank_qr, topic_bank_random

N_DISTR = 1950
K = 50


def unit(args):
    cfg_name, seed, outdir = args
    cfg = R.CONFIGS[cfg_name]
    D, N, npar, sig = cfg["D"], cfg["N"], cfg["n_parent"], cfg["sigma"]
    P, Ch = R.generate(D, N, npar, sig, seed, cfg.get("topics", "qr"))
    allX = np.array([c["X"] for grp in Ch for c in grp])
    allpar = np.repeat(np.arange(len(P)), [len(grp) for grp in Ch])
    # fresh parents for S2: same topic bank, independent topic choice (rng [seed, 777]) and noise (global seed 20000+seed)
    bank = (topic_bank_random if cfg.get("topics", "qr") == "random" else topic_bank_qr)(N, D, seed)
    gen = PassageGenerator(bank, seed)
    gen.rng = np.random.default_rng([seed, 777])
    np.random.seed(20_000 + seed)
    fresh = [p for _ in range(2) for p in gen.sample(npar, pool=np.arange(N), n_points=N_DISTR // 2, sigma=sig)]
    freshX = np.array([p["X"] for p in fresh])
    drng = np.random.default_rng([seed, 999])
    null_rng = np.random.default_rng([0, seed, 2])
    rows = []
    for scen in cfg["scenarios"]:
        np.random.seed(10_000 + seed)               # identical designed corpora to run_sim.run_unit
        for qi, (g, qdp, C, cnf, extra) in enumerate(R.queries(scen, P, Ch, npar)):
            q = np.asarray(qdp["X"], dtype=float)
            if scen == 2:
                Dx = freshX
            else:
                pool = np.flatnonzero(allpar != g)
                Dx = allX[drng.choice(pool, size=N_DISTR + (60 if scen == 4 else 0), replace=False)]
                if scen == 4:                       # drop exact duplicates of designed passages
                    seen = {r.tobytes() for r in C}
                    Dx = np.array([r for r in Dx if r.tobytes() not in seen])[:N_DISTR]
            corpus = np.vstack([C, Dx])             # rows 0..49 = designed passages
            top = np.argsort(-(corpus @ q), kind="stable")[:K]
            Ct = corpus[top]
            real = persistence_features(q, Ct, np.inf)
            qn, Cn = matched_cosine_null(q, Ct, null_rng)
            null = persistence_features(qn, Cn, np.inf)
            r = dict(config=cfg_name, sim_seed=seed, scenario=scen, query_idx=qi, parent_group=g,
                     corpus_nfeat=int(cnf), query_nfeat=int(qdp["n_features"]), n_corpus=len(corpus),
                     frac_designed=float(np.mean(top < len(C))),
                     own_parent_retrieved=(bool(np.isin(g, top)) if scen == 2 else np.nan),
                     **{f"all_{k}": v for k, v in real.items()}, **{f"null_all_{k}": v for k, v in null.items()},
                     **score_feats(q, Ct))
            rows.append(r)
    pd.DataFrame(rows).to_csv(Path(outdir) / f"distractor_{cfg_name}_seed{seed}.csv", index=False)
    return cfg_name, seed, len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["fig1", "fig2"])
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
