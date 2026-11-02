"""Synthetic simulation driver: one CSV per (configuration, seed), one row per query x eps [x projection].

Scenarios (code ids; paper names in brackets):
  1 [S1, multi-factual]: query = a parent passage; corpus = its 50 children with n_child topics (n_child = 1..n_parent-1).
  2 [S2, polysemic]:     query = the first child (n_child topics) of each parent; corpus = all 50 parents.
  4 [S3, mixed]:         query = a parent; corpus = 10-39 "direct" children (topics subset of the parent's) plus
                         partial-overlap children up to 50 in total.
Per query: features at each eps of the grid; at eps = 0.4 also the matched-cosine null (null_*), and on all 50
neighbours without the eps filter (all_*, null_all_*).

Usage:  python run_sim.py --configs fig1 fig2 appD64 appD128 proj superpos superpos_s02 --seeds 0 1 2 3 4 --workers 4
        [--outdir out] [--max-groups G]   (G truncates each scenario to the first G parent groups; for quick checks)
"""
import argparse, json, os, platform, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_OPEN_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from simcore import (PassageGenerator, matched_cosine_null, persistence_features, topic_bank_qr,
                     topic_bank_random)

HERE = Path(__file__).resolve().parent
EPS = [0.01, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0]
EPS_NULL = 0.4          # eps at which the null and the statistics are computed
N_CORPUS = 50           # passages per group = retrieved set size

# N = number of topics; n_parent = topics per parent passage.
CONFIGS = {
    "fig1":    dict(D=256, N=64, n_parent=32, sigma=0.1,  scenarios=[1, 2],    proj=[None]),
    "fig2":    dict(D=256, N=64, n_parent=32, sigma=0.2,  scenarios=[1, 2, 4], proj=[None]),
    "appD64":  dict(D=64,  N=16, n_parent=12, sigma=0.1,  scenarios=[1, 2],    proj=[None]),
    "appD128": dict(D=128, N=32, n_parent=16, sigma=0.1,  scenarios=[1, 2],    proj=[None]),
    # the D=256 robustness setting is fig1
    "proj":    dict(D=256, N=64, n_parent=32, sigma=0.01, scenarios=[1, 2],    proj=[None, 224, 192, 128, 64]),
    # superposition regime: N = 1024 > D random unit topics
    "superpos":     dict(D=256, N=1024, n_parent=32, sigma=0.1, scenarios=[1, 2],    proj=[None], topics="random", eps=[0.4]),
    "superpos_s02": dict(D=256, N=1024, n_parent=32, sigma=0.2, scenarios=[1, 2, 4], proj=[None], topics="random", eps=[0.4]),
}
SCEN_LABEL = {1: "parent_to_children", 2: "children_to_parent", 4: "parent_to_children_mixed"}


def generate(D, N, n_parent, sigma, seed, topics="qr"):
    """50 parent passages (n_parent topics from a root permutation of all N topics) and, for each parent and each
    n_child = 1..n_parent-1, 50 children drawing n_child topics from that parent's topics."""
    np.random.seed(seed)    # noise uses the global numpy RNG
    bank = (topic_bank_random if topics == "random" else topic_bank_qr)(N, D, seed)
    gen = PassageGenerator(bank, seed)
    root = gen.sample(N, n_points=1, sigma=0.0)
    parents = gen.sample(n_parent, pool=root[0]["features_ids"], n_points=N_CORPUS, sigma=sigma)
    children = []
    for p in parents:
        ch = []
        for nf in range(1, n_parent):
            ch += gen.sample(nf, pool=p["features_ids"], n_points=N_CORPUS, sigma=sigma)
        children.append(ch)
    return parents, children


def project(parents, children, k, seed):
    """Gaussian random projection to k dimensions (random_state = sim seed), without re-normalisation."""
    from sklearn.random_projection import GaussianRandomProjection
    P = np.array([p["X"] for p in parents])
    proj = GaussianRandomProjection(n_components=k, random_state=seed).fit(P)
    np_ = [dict(p, X=x) for p, x in zip(parents, proj.transform(P))]
    nc = []
    for grp in children:
        Xg = proj.transform(np.array([c["X"] for c in grp]))
        nc.append([dict(c, X=x) for c, x in zip(grp, Xg)])
    return np_, nc


def queries(scen, parents, children, n_parent, max_groups=None):
    """Yield (parent_group, query_dp, corpus_X, corpus_nfeat, extra) for one scenario."""
    G = len(parents) if max_groups is None else min(max_groups, len(parents))
    if scen == 1:
        for i, p in enumerate(parents[:G]):
            for nf in range(1, n_parent):
                C = np.array([c["X"] for c in children[i] if c["n_features"] == nf])
                yield i, p, C, nf, {}
    elif scen == 2:
        Cpar = np.array([p["X"] for p in parents])
        for i in range(G):
            for nf in range(1, n_parent):
                q = next(c for c in children[i] if c["n_features"] == nf)   # first child of that size
                yield i, q, Cpar, n_parent, {}
    elif scen == 4:
        allc = pd.DataFrame([c for grp in children for c in grp])
        groups = sorted(set(allc.n_features) | {n_parent})
        for i, p in enumerate(parents[:G]):
            pf = set(p["features_ids"])
            for nf in groups:
                cdf = allc[allc["n_features"] == nf]
                r = np.random.randint(10, 40)
                direct = cdf.loc[cdf["features_ids"].apply(lambda x: len(set(x).difference(pf)) == 0)]
                if len(direct) == 0:
                    continue
                direct = direct.sample(n=min(len(direct), r))
                part = cdf.loc[cdf["features_ids"].apply(lambda x: 0 < len(set(x).difference(pf)) < len(x))]
                if len(part) == 0:
                    continue
                nd = min(len(direct), r)
                if len(part) < N_CORPUS - nd:
                    continue
                part = part.sample(n=N_CORPUS - nd)
                C = np.array([list(x) for x in pd.concat([direct, part]).X])
                yield i, p, C, nf, {"direct_children_count": nd}


def run_unit(args):
    cfg_name, seed, outdir, max_groups = args
    cfg = CONFIGS[cfg_name]
    t0 = time.time()
    parents0, children0 = generate(cfg["D"], cfg["N"], cfg["n_parent"], cfg["sigma"], seed, cfg.get("topics", "qr"))
    t_gen = time.time() - t0
    rows = []
    null_rng = np.random.default_rng([0, seed])         # matched-cosine null at eps = 0.4
    null_rng_all = np.random.default_rng([0, seed, 1])  # separate stream for the all-50 null
    for k in cfg["proj"]:
        parents, children = (parents0, children0) if k is None else project(parents0, children0, k, seed)
        for scen in cfg["scenarios"]:
            np.random.seed(10_000 + seed)                # scenario-4 sampling uses the global numpy RNG
            for qi, (g, qdp, C, cnf, extra) in enumerate(queries(scen, parents, children, cfg["n_parent"], max_groups)):
                q = np.asarray(qdp["X"], dtype=float)
                d = np.linalg.norm(C - q, axis=1)
                rel = np.sort(d) / d.min() - 1
                cache = {}
                qnorm = float(np.linalg.norm(q)); cnorm = float(np.mean(np.linalg.norm(C, axis=1)))
                for eps in cfg.get("eps", EPS):
                    n_in = int((rel <= eps).sum())
                    if n_in not in cache:
                        cache[n_in] = persistence_features(q, C, eps)
                    row = dict(config=cfg_name, sim_seed=seed, D=cfg["D"], N=cfg["N"], n_parent=cfg["n_parent"],
                               sigma=cfg["sigma"], proj_dim=(k or cfg["D"]), projected=k is not None,
                               scenario=scen, scenario_label=SCEN_LABEL[scen], parent_group=g, query_idx=qi,
                               query_nfeat=int(qdp["n_features"]), corpus_nfeat=int(cnf), n_corpus=len(C),
                               depsilon=eps, query_norm=qnorm, corpus_mean_norm=cnorm, **extra, **cache[n_in])
                    if eps == EPS_NULL:
                        qn, Cnull = matched_cosine_null(q, C, null_rng)
                        row.update({f"null_{kk}": v for kk, v in persistence_features(qn, Cnull, eps).items()})
                        ha = persistence_features(q, C, np.inf)
                        qn2, Cnull2 = matched_cosine_null(q, C, null_rng_all)
                        na = persistence_features(qn2, Cnull2, np.inf)
                        row.update({f"all_{kk}": v for kk, v in ha.items()})
                        row.update({f"null_all_{kk}": v for kk, v in na.items()})
                    rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(Path(outdir) / f"{cfg_name}_seed{seed}.csv", index=False)
    return cfg_name, seed, len(df), t_gen, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--outdir", default=str(HERE / "out"))
    ap.add_argument("--max-groups", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    import ripser, sklearn
    man = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
               ripser=ripser.__version__, sklearn=sklearn.__version__, eps=EPS, eps_null=EPS_NULL, n_corpus=N_CORPUS,
               max_groups=a.max_groups, configs={c: CONFIGS[c] for c in a.configs}, seeds=a.seeds, units={})
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for c, s, n, tg, tt in ex.map(run_unit, [(c, s, str(out), a.max_groups) for c in a.configs for s in a.seeds]):
            print(f"done {c} seed{s}: {n} rows, gen {tg:.1f}s, total {tt:.1f}s", flush=True)
            man["units"][f"{c}_seed{s}"] = dict(rows=n, gen_s=round(tg, 1), total_s=round(tt, 1))
    man["wall_s"] = round(time.time() - t0, 1)
    (out / f"manifest_{'_'.join(a.configs)}.json").write_text(json.dumps(man, indent=1, default=str))
    print("wall", man["wall_s"], "s")


if __name__ == "__main__":
    main()
