# bayes.py -- Bayesian layer for the key real-text claim, on existing feature tables (no features or nulls recomputed).
# (a) Per corpus x model, LODO arm, fixed 250-token queries (density-matched C250 = lodo_dens Q250|C250 vs C750 = lodo
#     Q250|C750): Bayesian bootstrap (Rubin 1981; Dirichlet(1,...,1) weights over documents, 4,000 draws, seed 0) of
#     d = AUROC_real - mean_k AUROC_null_k (k = the 20 empirical corpus-bearing null draws), both recomputed with the same
#     weights. ORIENTED: AUROC = P(feature in C250 > feature in C750), so d > 0 means the real feature separates the corpora
#     more than the null does. Ties ignored (continuous features; LT_max is never 0 with 50 neighbours). Features: longest H0 bar, W1(H0), LT_max(H1).
# (b) Hierarchical normal models over all corpus x model cells, with s_cm = posterior sd from (a), fitted by Gibbs sampling
#     (conjugate normal updates; scale parameters drawn exactly from their conditionals on a 4,000-point grid):
#     main  : d_cm ~ N(mu + a_c + b_m, s_cm^2 + sigma^2), a_c ~ N(0, tau_c^2), b_m ~ N(0, tau_m^2)
#     simple: d_cm ~ N(theta_c, s_cm^2), theta_c ~ N(mu, tau^2)   (a simpler specification, as a sensitivity check)
#     priors mu ~ N(0, 0.5^2); tau, tau_c, tau_m ~ HalfNormal(0.2); sigma ~ HalfNormal(0.2). 4 chains x 6,000 iterations,
#     1,000 burn-in, seeds 0-3. theta_c = mu + a_c (corpus effect averaged over models); theta_new = mu + a_new.
# Inputs: $RT_WORK/$OUTROOT[/<corpus>]/features_<model>.csv.{gz,xz} (+ the shipped ltmax_enull_<model>.csv.xz sidecars).
# Writes $RT_WORK/$OUTROOT/bayes/{bb_cells.csv, bb_draws.npz, hier_summary.csv, bayes.md}.
#   --subvar: adds the size-matched subsample's variance (subseed_<model>.csv from subseed.py) to every cell and refits
#             -> bb_cells_subvar.csv, hier_summary_subvar.csv (the paper's intervals and hierarchical summary).
#   --prior-sweep [--subvar]: refits the longest-bar model with scale priors HalfNormal(0.1 / 0.2 / 0.4) -> prior_sweep.csv
#                (prior_sweep_subvar.csv with --subvar, on the subsample-adjusted cells).
# Usage: [OUTROOT=out_clean] python bayes.py [--subvar | --prior-sweep]
import time
from pathlib import Path
import numpy as np
import pandas as pd

OUTROOT = __import__('os').environ.get('OUTROOT', 'out')  # the paper: OUTROOT=out_clean
ROOT = Path(__import__('os').environ.get("RT_WORK", str(Path(__file__).resolve().parent / "work")))  # see README
OUT = ROOT / OUTROOT / "bayes"; OUT.mkdir(parents=True, exist_ok=True)
CORPORA = {"nobel": ROOT / OUTROOT, "wiki": ROOT / OUTROOT / "wiki", "gutenberg": ROOT / OUTROOT / "gutenberg",
           "turing": ROOT / OUTROOT / "turing", "fields": ROOT / OUTROOT / "fields"}
MODELS = ["openai-3-small", "openai-3-large", "vertex-005", "modernbert-embed", "bge-base", "gemini-001", "openai-ada-002", "bge-m3", "qwen3-0.6b"]
FEATS = ["longbar", "w1", "ltmax"]
NDRAW, CHUNK = 4000, 250


def wauc(pos, neg, Wp, Wn):
    """Weighted AUROC P(pos > neg) for each row of weight matrices (draws x n); ties ignored."""
    v = np.concatenate([pos, neg]); o = np.argsort(v, kind="stable")
    isneg = np.r_[np.zeros(len(pos), bool), np.ones(len(neg), bool)][o]
    W = np.concatenate([Wp, Wn], axis=1)[:, o]
    wn = np.where(isneg, W, 0.0); wp = np.where(isneg, 0.0, W)
    below = np.cumsum(wn, axis=1) - wn
    return (wp * below).sum(1) / (wp.sum(1) * wn.sum(1))


def bb_cell(df):
    a = df[(df.variant == "lodo") & (df.cell == "Q250|C750")]       # C750 corpus
    b = df[(df.variant == "lodo_dens") & (df.cell == "Q250|C250")]  # density-matched C250 corpus
    if a.empty or b.empty or "longbar_enull00" not in df.columns or a["longbar_enull00"].isna().all():
        return None
    docs = sorted(set(a.lecture) | set(b.lecture)); ix = {d: i for i, d in enumerate(docs)}
    ia, ib = a.lecture.map(ix).values, b.lecture.map(ix).values
    rng = np.random.default_rng(0)
    D = rng.dirichlet(np.ones(len(docs)), size=NDRAW)
    out = {}
    for f in FEATS:
        d = np.empty(NDRAW); r_ = np.empty(NDRAW)
        for s in range(0, NDRAW, CHUNK):
            Wa, Wb = D[s:s + CHUNK][:, ia], D[s:s + CHUNK][:, ib]
            real = wauc(b[f].values, a[f].values, Wb, Wa)  # oriented: C250 > C750
            nul = np.mean([wauc(b[f"{f}_enull{k:02d}"].values, a[f"{f}_enull{k:02d}"].values, Wb, Wa) for k in range(20)], axis=0)
            d[s:s + CHUNK] = real - nul; r_[s:s + CHUNK] = real
        out[f] = (d, r_)
    return out, len(docs), len(a), len(b)


def gibbs(d, s, cidx, midx, simple, seed, iters=6000, burn=1000, ps=0.2):
    rng = np.random.default_rng(seed); C, M = cidx.max() + 1, midx.max() + 1
    grid = np.linspace(1e-4, 1.0, 4000)
    mu, a, b = 0.0, np.zeros(C), np.zeros(M)
    tc, tm, sg = 0.1, 0.1, 0.05
    keep = []
    for it in range(iters):
        if simple:
            # theta_c | mu, tau ; mu | theta ; tau | theta   (a = theta - mu, no model term, sigma = 0)
            v = s ** 2
            for c in range(C):
                m = cidx == c; prec = (1 / v[m]).sum() + 1 / tc ** 2
                a[c] = rng.normal(((d[m] - mu) / v[m]).sum() / prec, prec ** -0.5)
            # mu | theta_c = mu + a_c ~ N(mu, tau^2): use theta directly
            th = mu + a
            prec = C / tc ** 2 + 1 / 0.5 ** 2
            mu_new = rng.normal((th.sum() / tc ** 2) / prec, prec ** -0.5)
            a = th - mu_new; mu = mu_new
            lp = -0.5 * (a ** 2).sum() / grid ** 2 - C * np.log(grid) - 0.5 * grid ** 2 / ps ** 2
            p = np.exp(lp - lp.max()); tc = rng.choice(grid, p=p / p.sum())
            keep.append((mu, tc, 0.0, 0.0, *(mu + a)))
            continue
        v = s ** 2 + sg ** 2
        r = d - a[cidx] - b[midx]; prec = (1 / v).sum() + 1 / 0.5 ** 2
        mu = rng.normal((r / v).sum() / prec, prec ** -0.5)
        for c in range(C):
            m = cidx == c; rr = d[m] - mu - b[midx[m]]; prec = (1 / v[m]).sum() + 1 / tc ** 2
            a[c] = rng.normal((rr / v[m]).sum() / prec, prec ** -0.5)
        for k in range(M):
            m = midx == k; rr = d[m] - mu - a[cidx[m]]; prec = (1 / v[m]).sum() + 1 / tm ** 2
            b[k] = rng.normal((rr / v[m]).sum() / prec, prec ** -0.5)
        for name, eff, n in (("tc", a, C), ("tm", b, M)):
            lp = -0.5 * (eff ** 2).sum() / grid ** 2 - n * np.log(grid) - 0.5 * grid ** 2 / ps ** 2
            p = np.exp(lp - lp.max()); val = rng.choice(grid, p=p / p.sum())
            if name == "tc": tc = val
            else: tm = val
        res = d - mu - a[cidx] - b[midx]
        V = s[None, :] ** 2 + grid[:, None] ** 2
        lp = -0.5 * (np.log(V) + res[None, :] ** 2 / V).sum(1) - 0.5 * grid ** 2 / ps ** 2
        p = np.exp(lp - lp.max()); sg = rng.choice(grid, p=p / p.sum())
        keep.append((mu, tc, tm, sg, *(mu + a)))
    return np.array(keep[burn:])


def rhat(chains):
    ch = np.array(chains); n = ch.shape[1] // 2
    sp = np.concatenate([ch[:, :n], ch[:, n:2 * n]]); m = sp.mean(1); w = sp.var(1, ddof=1).mean()
    bvar = n * m.var(ddof=1); return float(np.sqrt(((n - 1) / n * w + bvar / n) / w))


def main():
    t0 = time.time(); rows, draws = [], {}
    for corp, p in CORPORA.items():
        for m in MODELS:
            f = p / f"features_{m}.csv.gz"
            f = f if f.exists() else f.with_suffix(".xz")   # shipped tables are .csv.xz
            if not f.exists():
                continue
            cols = ["variant", "cell", "lecture"] + FEATS + [f"{x}_enull{k:02d}" for x in FEATS for k in range(20)]
            hdr = pd.read_csv(f, nrows=0).columns
            if "longbar_enull00" not in hdr:
                continue
            df = pd.read_csv(f, usecols=[c for c in cols + ["query_id"] if c in hdr])
            sc = p / f"ltmax_enull_{m}.csv.xz"   # shipped tables: LT_max corpus-null draws in a sidecar
            if "ltmax_enull00" not in df.columns and sc.exists():
                df = df.merge(pd.read_csv(sc), on=["variant", "cell", "query_id"], how="left", validate="one_to_one")
            res = bb_cell(df)
            if res is None:
                continue
            out, ndoc, na, nb = res
            for feat, (d, r) in out.items():
                draws[f"{corp}|{m}|{feat}"] = d.astype(np.float32)
                rows.append(dict(corpus=corp, model=m, feature=feat, n_docs=ndoc, n_c750=na, n_c250=nb,
                                 auroc_real_mean=r.mean(), d_mean=d.mean(), d_sd=d.std(ddof=1),
                                 d_lo=np.percentile(d, 2.5), d_hi=np.percentile(d, 97.5), p_gt0=(d > 0).mean()))
            print(f"[{time.strftime('%H:%M:%S')}] {corp} {m} done", flush=True)
    B = pd.DataFrame(rows); B.to_csv(OUT / "bb_cells.csv", index=False); np.savez_compressed(OUT / "bb_draws.npz", **draws)
    H = []
    for feat in FEATS:
        x = B[B.feature == feat].reset_index(drop=True)
        corp = sorted(x.corpus.unique(), key=list(CORPORA).index); mods = [m for m in MODELS if m in set(x.model)]
        ci = x.corpus.map({c: i for i, c in enumerate(corp)}).values; mi = x.model.map({m: i for i, m in enumerate(mods)}).values
        for spec in ("main", "simple"):
            ch = [gibbs(x.d_mean.values, x.d_sd.values, ci, mi, spec == "simple", seed) for seed in range(4)]
            allc = np.concatenate(ch)
            mu, tc = allc[:, 0], allc[:, 1]
            rng = np.random.default_rng(99)
            thnew = mu + rng.normal(0, 1, len(mu)) * tc
            def add(name, v, rh):
                H.append(dict(feature=feat, spec=spec, param=name, mean=v.mean(), lo=np.percentile(v, 2.5),
                              hi=np.percentile(v, 97.5), p_gt0=(v > 0).mean(), rhat=rh))
            add("mu", mu, rhat([c[:, 0] for c in ch]))
            add("tau_corpus" if spec == "main" else "tau", tc, rhat([c[:, 1] for c in ch]))
            if spec == "main":
                add("tau_model", allc[:, 2], rhat([c[:, 2] for c in ch])); add("sigma_resid", allc[:, 3], rhat([c[:, 3] for c in ch]))
            for j, c in enumerate(corp):
                add(f"theta[{c}]", allc[:, 4 + j], rhat([cc[:, 4 + j] for cc in ch]))
            add("theta_new (predictive)", thnew, np.nan)
    Hd = pd.DataFrame(H); Hd.to_csv(OUT / "hier_summary.csv", index=False)
    write_md(B, Hd)
    print(f"done in {time.time() - t0:.0f}s")


def subvar():
    """Fold the size-matched subsample's randomness into every cell (subseed.py: 10 subsample
    seeds per corpus x model). Per cell: d = mean over the 10 seeds; s = sqrt(bootstrap s.d.^2 + between-seed s.d.^2)
    (conservative: a single subsample's variance is added in full); interval d +- 1.96 s. Refits the hierarchical model
    (both specs, 12,000 iterations per chain, 2,000 burn-in). Writes bb_cells_subvar.csv and hier_summary_subvar.csv."""
    B = pd.read_csv(OUT / "bb_cells.csv")
    base = OUT.parent
    S = []
    for corp, sub in CORPORA.items():
        d = base if corp == "nobel" else base / corp
        for f in sorted(d.glob("subseed_*.csv")):
            S.append(pd.read_csv(f).assign(corpus=corp))
    S = pd.concat(S).groupby(["corpus", "model", "feature"]).d.agg(seed_mean="mean", seed_sd="std", n_seeds="size").reset_index()
    B = B.merge(S, on=["corpus", "model", "feature"], how="left")
    missing = B[B.n_seeds.isna()][["corpus", "model"]].drop_duplicates()
    assert missing.empty, f"cells without subsample seeds: {missing.values.tolist()}"
    B["d_boot_sd"] = B.d_sd
    B["d_mean"], B["d_sd"] = B.seed_mean, np.sqrt(B.d_sd ** 2 + B.seed_sd ** 2)
    B["d_lo"], B["d_hi"] = B.d_mean - 1.96 * B.d_sd, B.d_mean + 1.96 * B.d_sd
    B.to_csv(OUT / "bb_cells_subvar.csv", index=False)
    H = []
    for feat in FEATS:
        x = B[B.feature == feat].reset_index(drop=True)
        corp = sorted(x.corpus.unique(), key=list(CORPORA).index); mods = [m for m in MODELS if m in set(x.model)]
        ci = x.corpus.map({c: i for i, c in enumerate(corp)}).values; mi = x.model.map({m: i for i, m in enumerate(mods)}).values
        for spec in ("main", "simple"):
            ch = [gibbs(x.d_mean.values, x.d_sd.values, ci, mi, spec == "simple", seed, iters=12000, burn=2000) for seed in range(4)]
            allc = np.concatenate(ch); mu, tc = allc[:, 0], allc[:, 1]
            thnew = mu + np.random.default_rng(99).normal(0, 1, len(mu)) * tc
            def add(name, v, rh):
                H.append(dict(feature=feat, spec=spec, param=name, mean=v.mean(), lo=np.percentile(v, 2.5),
                              hi=np.percentile(v, 97.5), p_gt0=(v > 0).mean(), rhat=rh))
            add("mu", mu, rhat([c[:, 0] for c in ch]))
            add("tau_corpus" if spec == "main" else "tau", tc, rhat([c[:, 1] for c in ch]))
            if spec == "main":
                add("tau_model", allc[:, 2], rhat([c[:, 2] for c in ch])); add("sigma_resid", allc[:, 3], rhat([c[:, 3] for c in ch]))
            for j, c in enumerate(corp):
                add(f"theta[{c}]", allc[:, 4 + j], rhat([cc[:, 4 + j] for cc in ch]))
            add("theta_new (predictive)", thnew, np.nan)
    pd.DataFrame(H).to_csv(OUT / "hier_summary_subvar.csv", index=False)
    print("subvar done:", len(B), "cells")


def write_md(B, H):
    L = [f"<!-- bayes.md -- generated {time.strftime('%Y-%m-%d %H:%M')} by bayes.py from the per-query feature tables -->",
         "# Bayesian layer: does the persistence feature beat the corpus-bearing null at fixed query length?\n",
         "**Quantity.** LODO arm (query's own document removed), fixed 250-token queries, density-matched C250 vs C750 corpus. "
         "d = AUROC_real - mean over the 20 empirical (corpus-bearing) null draws of AUROC_null, oriented so d > 0 means the real "
         "feature separates the two corpora more than the null does.\n",
         "**(a) Bayesian bootstrap** (Rubin 1981): Dirichlet(1,...,1) weights over documents (the cluster unit), 4,000 draws, "
         "numpy default_rng(0); real and null AUROCs recomputed with the same weights in each draw. Posterior mean, 95% "
         "credible interval, P(d > 0).\n",
         "**(b) Hierarchical models** on the cell means with s_cm = posterior sd from (a). Main: d_cm ~ N(mu + a_c + b_m, "
         "s_cm^2 + sigma^2), corpus effects a_c ~ N(0, tau_c^2), model effects b_m ~ N(0, tau_m^2), residual sigma. Simple "
         "(sensitivity): d_cm ~ N(theta_c, s_cm^2), theta_c ~ N(mu, tau^2). Priors mu ~ N(0, 0.5^2), all scales HalfNormal(0.2). "
         "Gibbs sampler written here (no numpyro/pymc): conjugate normal updates, scales drawn exactly from their conditionals "
         "on a 4,000-point grid on (0, 1]; 4 chains x 6,000 iterations, 1,000 burn-in, seeds 0-3; split R-hat reported. "
         "theta_c = mu + a_c; theta_new = mu + a_new, a_new ~ N(0, tau_c^2) (posterior predictive for a new corpus, averaged over models).\n",
         "**Caveats.** The five models within a corpus share documents, so their d_cm are not independent; the model term and the "
         "residual absorb some of that, the simple spec does not, and its intervals are therefore too narrow. s_cm treats the "
         "20-draw null mean as fixed apart from document weighting.\n"]
    for feat, name in (("longbar", "Longest H0 bar"), ("w1", "W1(H0)"), ("ltmax", "LT_max(H1)")):
        L += [f"\n## {name}\n", "### (a) Per corpus x model\n", "| corpus | model | docs | AUROC real (oriented) | d mean | 95% CrI | P(d>0) |",
              "|---|---|---|---|---|---|---|"]
        for _, r in B[B.feature == feat].iterrows():
            L.append(f"| {r.corpus} | {r.model} | {r.n_docs} | {r.auroc_real_mean:.3f} | {r.d_mean:+.3f} | [{r.d_lo:+.3f}, {r.d_hi:+.3f}] | {r.p_gt0:.3f} |")
        for spec in ("main", "simple"):
            L += [f"\n### (b) Hierarchical, {spec} spec\n", "| parameter | mean | 95% CrI | P(>0) | R-hat |", "|---|---|---|---|---|"]
            for _, r in H[(H.feature == feat) & (H.spec == spec)].iterrows():
                rh = "--" if pd.isna(r.rhat) else f"{r.rhat:.3f}"
                L.append(f"| {r.param} | {r['mean']:+.3f} | [{r.lo:+.3f}, {r.hi:+.3f}] | {r.p_gt0:.3f} | {rh} |")
    (OUT / "bayes.md").write_text("\n".join(L) + "\n")


def prior_sweep():
    """Longest bar, both specs: refit the hierarchical model from bb_cells.csv with
    every scale prior HalfNormal(ps), ps in 0.1/0.2/0.4. theta_new = mu + a_new (corpus effect only, as in main()).
    Writes out/bayes/prior_sweep.csv. Usage: python bayes.py --prior-sweep"""
    # --subvar refits the sweep on the subsample-adjusted cells (bb_cells_subvar.csv).
    sv = "--subvar" in sys.argv
    B = pd.read_csv(OUT / ("bb_cells_subvar.csv" if sv else "bb_cells.csv")); rows = []
    x = B[B.feature == "longbar"].reset_index(drop=True)
    corp = sorted(x.corpus.unique(), key=list(CORPORA).index); mods = [m for m in MODELS if m in set(x.model)]
    ci = x.corpus.map({c: i for i, c in enumerate(corp)}).values; mi = x.model.map({m: i for i, m in enumerate(mods)}).values
    for ps in (0.1, 0.2, 0.4):
        for spec in ("main", "simple"):
            allc = np.concatenate([gibbs(x.d_mean.values, x.d_sd.values, ci, mi, spec == "simple", seed, ps=ps) for seed in range(4)])
            mu, tc = allc[:, 0], allc[:, 1]
            th = mu + np.random.default_rng(99).normal(0, 1, len(mu)) * tc
            rows.append(dict(prior_scale=ps, spec=spec, mu=mu.mean(), mu_lo=np.percentile(mu, 2.5), mu_hi=np.percentile(mu, 97.5),
                             tau_corpus=tc.mean(), p_new_gt0=(th > 0).mean(), new_lo=np.percentile(th, 2.5), new_hi=np.percentile(th, 97.5)))
    pd.DataFrame(rows).to_csv(OUT / ("prior_sweep_subvar.csv" if sv else "prior_sweep.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    import sys
    prior_sweep() if "--prior-sweep" in sys.argv else (subvar() if "--subvar" in sys.argv else main())
