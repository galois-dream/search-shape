"""Statistics for the synthetic simulation (per-query tables from run_sim.py; cosine and distractor tables for the
red-team part). Per configuration and feature (W1(H0), LT_max(H1), longest finite H0 bar), at eps = 0.4: n per arm,
mean/median per arm, difference of medians, AUROC (scenario 1 vs 2; 1 vs mixed), 95% CIs from a cluster bootstrap that
resamples parent groups (sim_seed x parent_group), B = 2000, fixed seed; the matched-cosine null and the paired
real-minus-null difference. Second table: all 50 neighbours (LT_max = 0 if no H1 point). Red-team and distractor
tables (redteam.py) are appended when their inputs exist.
Inputs: $SIM_DATA/{features,cosine,distractor}/ (default ../../data/simulation). Outputs: $SIM_RESULTS (default ./results).
Usage: python stats.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import os
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SIM_DATA", str(HERE.parents[1] / "data" / "simulation")))
FEAT, COSD, DISD = DATA / "features", DATA / "cosine", DATA / "distractor"   # run_sim / cosine_baselines / distractor
OUT = Path(os.environ.get("SIM_RESULTS", str(HERE / "results"))); OUT.mkdir(parents=True, exist_ok=True)
EPS = 0.4
B = 2000
BOOT_SEED = 20261029
FEATS = [("w1_h0", "W1(H0)"), ("ltmax_h1", "LT_max(H1)"), ("h0_maxbar", "max H0 bar")]
COMPS = [(1, 2, "S1 vs S2"), (1, 4, "S1 vs mixed(S4)")]


def wauc(x, y, wx, wy):
    """Weighted AUROC = P(X > Y) + 0.5 P(X = Y) with row weights."""
    vals = np.concatenate([x, y])
    u, inv = np.unique(vals, return_inverse=True)
    out = []
    for a, b in zip(np.atleast_2d(wx), np.atleast_2d(wy)):
        sa = np.bincount(inv[:len(x)], weights=a, minlength=len(u))
        sb = np.bincount(inv[len(x):], weights=b, minlength=len(u))
        below = np.cumsum(sb) - sb
        out.append(np.sum(sa * (below + 0.5 * sb)) / (sa.sum() * sb.sum()))
    return np.array(out)


def wmedian(x, w):
    o = np.argsort(x); xs = x[o]
    W = np.atleast_2d(w)[:, o]
    c = np.cumsum(W, axis=1)
    idx = (c >= c[:, -1:] / 2).argmax(axis=1)
    return xs[idx]


def ci(v):
    return np.nanpercentile(v, [2.5, 97.5])


def analyse(df, rng_seed=BOOT_SEED, prefix=""):
    rows = []
    for (cfg, pdim), g in df.groupby(["config", "proj_dim"], sort=False):
        g = g[np.isclose(g.depsilon, EPS)]
        cl = pd.factorize(g.sim_seed.astype(str) + "_" + g.parent_group.astype(str))[0]
        g = g.assign(cluster=cl)
        K = cl.max() + 1
        rng = np.random.default_rng(rng_seed)
        Wc = rng.multinomial(K, np.full(K, 1.0 / K), size=B).astype(float)   # B x K cluster counts
        for sa, sb, cname in COMPS:
            A, Bd = g[g.scenario == sa], g[g.scenario == sb]
            if len(A) == 0 or len(Bd) == 0:
                continue
            for f, flabel in FEATS:
                r = dict(config=cfg, D=int(g.D.iloc[0]), N=int(g.N.iloc[0]), n_parent=int(g.n_parent.iloc[0]),
                         sigma=float(g.sigma.iloc[0]), proj_dim=int(pdim), projected=bool(g.projected.iloc[0]),
                         seeds=",".join(map(str, sorted(g.sim_seed.unique()))), comparison=cname, feature=flabel)
                res = {}
                for kind, col in (("real", prefix + f), ("null", f"null_{prefix}{f}")):
                    a = A[[col, "cluster", "sim_seed"]].dropna(); b = Bd[[col, "cluster", "sim_seed"]].dropna()
                    x, y = a[col].to_numpy(), b[col].to_numpy()
                    ca, cb = a.cluster.to_numpy(), b.cluster.to_numpy()
                    wx, wy = Wc[:, ca], Wc[:, cb]
                    res[kind] = dict(x=x, y=y, auc=wauc(x, y, np.ones_like(x), np.ones_like(y))[0], boot=wauc(x, y, wx, wy),
                                     nA=len(x), nB=len(y), kA=len(np.unique(ca)), kB=len(np.unique(cb)),
                                     nanA=int(A[col].isna().sum()), nanB=int(Bd[col].isna().sum()))
                    if kind == "real":
                        dmed = np.median(x) - np.median(y)
                        dboot = wmedian(x, wx) - wmedian(y, wy)
                        res[kind].update(dmed=dmed, dmed_ci=ci(dboot))
                    # per-seed AUROC: seeds are the fully independent replicates (parent groups share corpora in S2/S4)
                    ps = [wauc(xs, ys, np.ones_like(xs), np.ones_like(ys))[0] for sd in sorted(a.sim_seed.unique())
                          for xs, ys in [(a[a.sim_seed == sd][col].to_numpy(), b[b.sim_seed == sd][col].to_numpy())] if len(xs) and len(ys)]
                    res[kind].update(seed_min=min(ps), seed_max=max(ps))
                re, nu = res["real"], res["null"]
                r.update(n_A=re["nA"], clusters_A=re["kA"], nan_A=re["nanA"], n_B=re["nB"], clusters_B=re["kB"], nan_B=re["nanB"],
                         mean_A=re["x"].mean(), mean_B=re["y"].mean(), median_A=np.median(re["x"]), median_B=np.median(re["y"]),
                         dmedian=re["dmed"], dmedian_lo=re["dmed_ci"][0], dmedian_hi=re["dmed_ci"][1],
                         auroc=re["auc"], auroc_lo=ci(re["boot"])[0], auroc_hi=ci(re["boot"])[1],
                         null_n_A=nu["nA"], null_n_B=nu["nB"], null_median_A=np.median(nu["x"]), null_median_B=np.median(nu["y"]),
                         null_auroc=nu["auc"], null_auroc_lo=ci(nu["boot"])[0], null_auroc_hi=ci(nu["boot"])[1],
                         auroc_seed_min=re["seed_min"], auroc_seed_max=re["seed_max"], null_auroc_seed_min=nu["seed_min"], null_auroc_seed_max=nu["seed_max"])
                d = re["boot"] - nu["boot"]
                r.update(auroc_minus_null=re["auc"] - nu["auc"], amn_lo=ci(d)[0], amn_hi=ci(d)[1])
                rows.append(r)
    return pd.DataFrame(rows)


def fmt(v, lo=None, hi=None, p=3):
    s = f"{v:.{p}f}"
    return s if lo is None else f"{s} [{lo:.{p}f}, {hi:.{p}f}]"


def render(st, suffix):
    L = []
    for cmp_name in [c[2] for c in COMPS]:
        s = st[st.comparison == cmp_name]
        if s.empty:
            continue
        L += [f"## {cmp_name}{suffix}", "",
              "| config | D (proj) | N | n_par | sigma | feature | n A / B (clusters) | nan A / B | mean A / B | median A / B | Δmedian [95% CI] | AUROC [95% CI] | null AUROC [95% CI] | AUROC − null [95% CI] | per-seed AUROC real / null (min–max) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in s.iterrows():
            dlab = f"{r.D}" + (f"→{r.proj_dim}" if r.projected else "")
            L.append(f"| {r.config} | {dlab} | {r.N} | {r.n_parent} | {r.sigma} | {r.feature} | {r.n_A} / {r.n_B} ({r.clusters_A}/{r.clusters_B}) | "
                     f"{r.nan_A} / {r.nan_B} | {fmt(r.mean_A)} / {fmt(r.mean_B)} | {fmt(r.median_A)} / {fmt(r.median_B)} | "
                     f"{fmt(r.dmedian, r.dmedian_lo, r.dmedian_hi)} | {fmt(r.auroc, r.auroc_lo, r.auroc_hi)} | "
                     f"{fmt(r.null_auroc, r.null_auroc_lo, r.null_auroc_hi)} | {fmt(r.auroc_minus_null, r.amn_lo, r.amn_hi)} | {r.auroc_seed_min:.3f}–{r.auroc_seed_max:.3f} / {r.null_auroc_seed_min:.3f}–{r.null_auroc_seed_max:.3f} |")
        L.append("")
    return L


def main():
    files = sorted(FEAT.glob("*_seed*.csv*"))
    if not files:
        sys.exit(f"no per-query tables in {FEAT}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    st = analyse(df)
    st.to_csv(OUT / "stats.csv", index=False)
    norms = df[np.isclose(df.depsilon, EPS)].groupby(["config", "proj_dim"]).agg(
        q_min=("query_norm", "min"), q_max=("query_norm", "max"), c_min=("corpus_mean_norm", "min"), c_max=("corpus_mean_norm", "max"))
    import time
    L = [f"# Synthetic simulation: statistics at eps = {EPS}",
         "", f"Generated by `stats.py` from {len(files)} per-query tables ({len(df)} rows).",
         f"Cluster bootstrap over parent groups (sim_seed x parent_group), B = {B}, numpy seed {BOOT_SEED}; 95% percentile CIs.",
         "AUROC = P(feature in arm A > feature in arm B) + 0.5 P(tie); A = scenario 1 (parent->children). >0.5 means S1 higher.",
         "Arms: S1 = code scenario 1 (parent query -> its 50 children with n_child topics); S2 = code scenario 2 (first child of",
         "each n_child group -> all 50 parents); mixed = code scenario 4 (parent -> 10-39 subset children + partial-overlap children, 50 total).",
         "Each arm pools all child sizes n_child = 1..n_parent-1 (one query per parent group per n_child). NaN rows (fewer than 3",
         "neighbours within eps, or no H1 bar for LT_max) are dropped; counts in `nan`.",
         "Null = matched-cosine null: each neighbour keeps its cosine to the query, direction randomised orthogonally (seed 0);",
         "same pipeline (simcore.persistence_features). AUROC-null = paired difference on the same bootstrap draws.", ""]
    L += render(st, "")
    # --- all 50 neighbours, no eps filter, LT_max = 0 when no H1 point ---
    if "all_w1_h0" in df.columns:
        z = df[np.isclose(df.depsilon, EPS)].copy()
        nz = {c: int(z[c].isna().sum()) for c in ("all_ltmax_h1", "null_all_ltmax_h1")}
        for c in ("all_ltmax_h1", "null_all_ltmax_h1"):
            z[c] = z[c].fillna(0.0)
        z["depsilon"] = EPS
        st2 = analyse(z, prefix="all_")
        st2.to_csv(OUT / "stats_all50.csv", index=False)
        L += ["# Second table: all 50 neighbours, no missingness", "",
              "Same queries, features recomputed on all 50 corpus points (eps = inf in simcore.persistence_features,",
              f"min all_nneighbours = {int(z.all_nneighbours.min())}). LT_max(H1) = 0 when the diagram has no H1 point (paper convention):",
              f"{nz['all_ltmax_h1']} real and {nz['null_all_ltmax_h1']} null queries zero-filled. The eps=4.0 column was NOT used: it misses",
              "neighbours in the projection config (sigma=0.01). Null = fresh matched-cosine draw (seed 0, separate stream), same pipeline.", ""]
        L += render(st2, " (all 50 neighbours)")
    L += ["## S2 queries with < 3 neighbours at eps = 0.4 (NaN in the first tables)", ""]
    for (c, p), g in df[np.isclose(df.depsilon, EPS) & (df.scenario == 2)].groupby(["config", "proj_dim"]):
        L.append(f"- {c} (proj_dim {p}): {(g.nneighbours < 3).mean():.3f} ({int((g.nneighbours < 3).sum())}/{len(g)})")
    L.append("")
    if list(COSD.glob("cosine_*_seed*.csv*")):
        import redteam
        L += redteam.tables(FEAT, COSD)
        L += redteam.distractor_tables(DISD)
    L += ["## Vector norms at retrieval (query_norm, mean corpus norm)", "",
          "| config | proj_dim | query norm min–max | corpus mean-norm min–max |", "|---|---|---|---|"]
    for (c, p), r in norms.iterrows():
        L.append(f"| {c} | {p} | {r.q_min:.3f}–{r.q_max:.3f} | {r.c_min:.3f}–{r.c_max:.3f} |")
    L += ["", "Projected configs are not re-normalised after GaussianRandomProjection; the real",
          "features use the raw projected vectors, the null uses the normalised ones.", ""]
    (OUT / "stats.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
