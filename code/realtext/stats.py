# stats.py -- statistics for the real-text experiment from $OUTROOT[/<corpus>]/features_<model>.csv.{gz,xz}. Units
# are queries clustered by document (lecture / page / book); every CI is a cluster bootstrap that resamples documents
# with replacement (B=2000, numpy default_rng(0)); one draw matrix is shared by all contrasts of a model, so differences
# (real - null, interactions) are paired. AUROC(A vs B) = P(feature in A > feature in B) + 0.5 P(=), document-
# weighted. Orientation: query effect = Q250 vs Q750 at fixed corpus; corpus effect = C750 vs C250 at fixed query;
# main contrast = Q250|C750 vs Q750|C250 (combine.py re-orients). Includes the three nulls, 20-draw bearing nulls,
# own-document share, leave-document-out arm, effect sizes, grouped-CV logistic regression, geometry check.
# Writes $OUTROOT[/<corpus>]/stats.md, stats_long.csv (every estimate) and cell_medians.csv (for figures).
# Usage: [CORPUS=wiki] [OUTROOT=out_clean] python stats.py
import os, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("RT_WORK", str(Path(__file__).resolve().parent / "work")))  # see README
CORPUS = os.environ.get("CORPUS", "nobel")
OUTROOT = os.environ.get("OUTROOT", "out")  # the paper: OUTROOT=out_clean (with CLEAN=1 in features.py)
OUT = ROOT / OUTROOT if CORPUS == "nobel" else ROOT / OUTROOT / CORPUS
B, SEED = 2000, 0
UNIT = {"nobel": "lecture", "wiki": "page", "gutenberg": "book", "turing": "lecture", "fields": "lecture"}.get(CORPUS, "document")
TOPO = ["w1", "ltmax", "longbar"]
NAMES = {"w1": "W1(H0)", "ltmax": "LT_max(H1)", "longbar": "longest H0 bar", "top1": "top-1 cos",
         "mean50": "mean top-50 cos", "std50": "std top-50 cos", "qlen_tok": "query length (tok)", "nn_pair_cos": "coherence: mean pairwise cos of top-50", "nn_ncomp": "graph connectivity: #components at tau=median pair cos", "slope50": "slope of cos on rank"}
BASEL = ["top1", "mean50", "std50", "slope50", "nn_pair_cos", "nn_ncomp"]  # last two: coherence QPP rivals
MODELS = ["openai-3-small", "openai-3-large", "vertex-005", "modernbert-embed", "bge-base", "gemini-001", "openai-ada-002", "bge-m3", "qwen3-0.6b"]


def feat_path(out, m):
    """features_<m>.csv.gz as written by features.py, else the shipped features_<m>.csv.xz."""
    p = out / f"features_{m}.csv.gz"
    return p if p.exists() else out / f"features_{m}.csv.xz"


def auc(x, y, wx, wy):
    """Weighted AUROC P(X>Y)+.5P(X=Y) for each row of weight matrices wx (R x nx), wy (R x ny)."""
    vals = np.concatenate([x, y]); u, inv = np.unique(vals, return_inverse=True)
    ix, iy = inv[:len(x)], inv[len(x):]
    out = np.empty(len(wx))
    for r in range(len(wx)):
        sa = np.bincount(ix, weights=wx[r], minlength=len(u)); sb = np.bincount(iy, weights=wy[r], minlength=len(u))
        below = np.cumsum(sb) - sb
        out[r] = np.sum(sa * (below + 0.5 * sb)) / (sa.sum() * sb.sum())
    return out


def wmed(x, w):
    o = np.argsort(x); xs = x[o]; c = np.cumsum(w[:, o], axis=1)
    return xs[(c >= c[:, -1:] / 2).argmax(axis=1)]


def wq(x, w, p):
    o = np.argsort(x); xs = x[o]; c = np.cumsum(w[:, o], axis=1)
    return xs[(c >= c[:, -1:] * p).argmax(axis=1)]


def ci(v):
    lo, hi = np.nanpercentile(v[1:], [2.5, 97.5]); return v[0], lo, hi


def fmt(t, d=3):
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


class Model:
    def __init__(self, df):
        self.df = df
        lec = sorted(df.lecture.unique()); self.L = {l: i for i, l in enumerate(lec)}
        rng = np.random.default_rng(SEED)
        W = rng.multinomial(len(lec), np.full(len(lec), 1 / len(lec)), size=B).astype(float)
        self.W = np.vstack([np.ones(len(lec)), W])  # row 0 = point estimate

    def sel(self, variant, cell, extra=None):
        d = self.df[(self.df.variant == variant) & (self.df.cell == cell)]
        return d if extra is None else d[extra(d)]

    def w(self, d):
        return self.W[:, d.lecture.map(self.L).values]

    def auc(self, f, a, b):  # a, b = dataframes; NaN rows dropped
        a = a[np.isfinite(a[f])]; b = b[np.isfinite(b[f])]
        return auc(a[f].values, b[f].values, self.w(a), self.w(b))

    def med(self, f, a):
        a = a[np.isfinite(a[f])]; return wmed(a[f].values, self.w(a))


def analyse(model, df, rows, md):
    M = Model(df); P = lambda c: M.sel("primary", c)
    cells = ["Q250|C750", "Q750|C250", "Q250|C250", "Q750|C750"]
    feats = TOPO + [f + "_null" for f in TOPO] + [f + "_e04" for f in TOPO] + BASEL + ["qlen_tok"]

    def rec(kind, feat, contrast, v):
        e = ci(v); rows.append(dict(corpus=CORPUS, model=model, kind=kind, feature=feat, contrast=contrast,
                                    est=e[0], lo=e[1], hi=e[2])); return e

    md.append(f"\n## {model}\n")
    # --- per-cell description
    md.append(f"### Cells (K=50 unless marked; medians, lecture-bootstrap 95% CI in cell_medians.csv)\n")
    md.append("| cell | n queries | n " + UNIT + "s | W1 | LT_max | longest bar | W1 null | LT_max null | top-1 | mean50 | std50 | eps0.4: frac <3 nbrs | eps0.4 median n |")
    md.append("|---|" + "---|" * 12)
    for c in cells:
        d = P(c)
        for f in feats + ["mean_sin_half", "R", "R_null"]:
            rec("median", f, c, M.med(f, d))
        m = {f: M.med(f, d)[0] for f in feats}
        md.append(f"| {c} | {len(d)} | {d.lecture.nunique()} | {m['w1']:.4f} | {m['ltmax']:.4f} | {m['longbar']:.4f} | "
                  f"{m['w1_null']:.4f} | {m['ltmax_null']:.4f} | {m['top1']:.3f} | {m['mean50']:.3f} | {m['std50']:.4f} | "
                  f"{np.mean(d.n_e04 < 3):.3f} | {np.median(d.n_e04):.0f} |")
    # --- main contrast + factorial
    md.append("\n### Main contrast AUROC(Q250|C750 > Q750|C250), null, real-null, interaction (95% CI)\n")
    md.append("| feature | AUROC real | AUROC null (isotropic) | real - null | AUROC emp-bearing null | real - emp null | query eff @C750 | query eff @C250 | interaction (AUROC) | interaction (medians) | diagonal Q250C250 vs Q750C750 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    a, b = P("Q250|C750"), P("Q750|C250")
    for f in TOPO + [t + "_e04" for t in TOPO] + BASEL + ["qlen_tok"]:
        r = M.auc(f, a, b); rec("auc_main", f, "Q250|C750 vs Q750|C250", r)
        qe750 = M.auc(f, P("Q250|C750"), P("Q750|C750")); qe250 = M.auc(f, P("Q250|C250"), P("Q750|C250"))
        rec("auc_query_eff", f, "@C750", qe750); rec("auc_query_eff", f, "@C250", qe250)
        ce250 = M.auc(f, P("Q250|C750"), P("Q250|C250")); ce750 = M.auc(f, P("Q750|C750"), P("Q750|C250"))
        rec("auc_corpus_eff", f, "@Q250", ce250); rec("auc_corpus_eff", f, "@Q750", ce750)
        ia = rec("interaction_auc", f, "qeff@C750 - qeff@C250", qe750 - qe250)
        mi = rec("interaction_median", f, "dd", (M.med(f, P("Q250|C750")) - M.med(f, P("Q750|C750")))
                 - (M.med(f, P("Q250|C250")) - M.med(f, P("Q750|C250"))))
        dg = rec("auc_diagonal", f, "Q250|C250 vs Q750|C750", M.auc(f, P("Q250|C250"), P("Q750|C750")))
        if f in TOPO:
            n = M.auc(f + "_null", a, b); rec("auc_main_null", f, "Q250|C750 vs Q750|C250", n)
            dd = rec("auc_main_real_minus_null", f, "", r - n)
            for nm, x, y in (("@C750", P("Q250|C750"), P("Q750|C750")), ("@C250", P("Q250|C250"), P("Q750|C250"))):
                rec("auc_query_eff_null", f, nm, M.auc(f + "_null", x, y))
            rec("interaction_auc_null", f, "", M.auc(f + "_null", P("Q250|C750"), P("Q750|C750")) - M.auc(f + "_null", P("Q250|C250"), P("Q750|C250")))
            ne = M.auc(f + "_enull", a, b); rec("auc_main_enull", f, "Q250|C750 vs Q750|C250", ne)
            de = rec("auc_main_real_minus_enull", f, "", r - ne)
            ncol, dcol, necol, decol = fmt(ci(n)), fmt(dd), fmt(ci(ne)), fmt(de)
        else:
            ncol = dcol = necol = decol = "--"
        md.append(f"| {NAMES.get(f.replace('_e04',''), f)}{' (eps0.4)' if f.endswith('_e04') else ''} | {fmt(ci(r))} | {ncol} | {dcol} | {necol} | {decol} | "
                  f"{fmt(ci(qe750))} | {fmt(ci(qe250))} | {fmt(ia)} | {fmt(mi, 4)} | {fmt(dg)} |")
    # --- same-length (de-confounded) comparisons
    md.append("\n### Same query length: corpus effect AUROC(C750 > C250) at fixed query granularity, real vs matched-cosine null\n")
    md.append("Density-controlled rows use C250 subsampled to |C750| chunks, per lecture, seed 0 (variant `dens`).\n")
    md.append("| feature | query | C250 corpus | AUROC real | AUROC null (isotropic) | real - null | AUROC emp-bearing null | real - emp null |")
    md.append("|---|---|---|---|---|---|---|---|")
    for f in TOPO + BASEL:
        for q in (250, 750):
            for var, lab in (("primary", "full"), ("dens", "subsampled")):
                x = P(f"Q{q}|C750"); y = M.sel(var, f"Q{q}|C250")
                r = M.auc(f, x, y); rec(f"auc_corpus_eff_{var}", f, f"@Q{q}", r)
                if f in TOPO:
                    n = M.auc(f + "_null", x, y); rec(f"auc_corpus_eff_{var}_null", f, f"@Q{q}", n)
                    d = rec(f"auc_corpus_eff_{var}_real_minus_null", f, f"@Q{q}", r - n)
                    ne = M.auc(f + "_enull", x, y); rec(f"auc_corpus_eff_{var}_enull", f, f"@Q{q}", ne)
                    de = rec(f"auc_corpus_eff_{var}_real_minus_enull", f, f"@Q{q}", r - ne)
                    md.append(f"| {NAMES[f]} | Q{q} | {lab} | {fmt(ci(r))} | {fmt(ci(n))} | {fmt(d)} | {fmt(ci(ne))} | {fmt(de)} |")
                else:
                    md.append(f"| {NAMES[f]} | Q{q} | {lab} | {fmt(ci(r))} | -- | -- | -- | -- |")
    # --- density control on the main contrast and the no-exclusion (paper-like) variant
    md.append("\n### Secondary analyses of the main contrast AUROC(Q250|C750 > Q750|C250)\n")
    md.append("| feature | primary (exclusion) | density control (Q750 vs subsampled C250) | no exclusion | no exclusion, paper filter (top-1 nested) |")
    md.append("|---|---|---|---|---|")
    ne_a, ne_b = M.sel("noexcl", "Q250|C750"), M.sel("noexcl", "Q750|C250")
    fa = M.sel("noexcl", "Q250|C750", lambda d: d.top1_nested); fb = M.sel("noexcl", "Q750|C250", lambda d: d.top1_nested)
    for f in TOPO + BASEL:
        cols = []
        for var, x, y in (("primary", a, b), ("dens", a, M.sel("dens", "Q750|C250")), ("noexcl", ne_a, ne_b),
                          ("noexcl_paperfilter", fa, fb)):
            r = ci(M.auc(f, x, y)); rows.append(dict(corpus=CORPUS, model=model, kind="auc_main_" + var, feature=f,
                                                     contrast="Q250|C750 vs Q750|C250", est=r[0], lo=r[1], hi=r[2]))
            cols.append(fmt(r))
        md.append(f"| {NAMES[f]} | " + " | ".join(cols) + " |")
    md.append(f"\nPaper filter keeps {fa.shape[0]}/{ne_a.shape[0]} Q250|C750 and {fb.shape[0]}/{ne_b.shape[0]} Q750|C250 queries "
              f"(top-1 hit contains / is contained in the query).\n")
    # --- incremental value of topology over scores: grouped 5-fold CV logistic regression
    md.append("\n### Incremental value of topology over scores: grouped 5-fold CV logistic regression, out-of-fold AUROC\n")
    md.append("A0 = scores only (top-1, mean, std of the 50 cosines, slope of cosine on rank); A = A0 + coherence (mean pairwise "
              "cosine of the 50) + graph connectivity (#components at tau = that query's median pairwise cosine); B0 = A0 + topology; "
              "B = A + topology (W1, LT_max, longest H0 bar, K=50). B - A = what persistence adds beyond scores and coherence. Standardised within fold; "
              f"folds = sklearn GroupKFold(5) by {UNIT}; CIs resample {UNIT}s over the fixed out-of-fold scores (no refit).\n")
    md.append("| comparison | positive class | n pos / n neg | AUROC A0 | AUROC B0 | B0 - A0 | AUROC A | AUROC B | B - A |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    comps = [("main: Q250|C750 vs Q750|C250", "Q250|C750", a, b)]
    for q in (250, 750):
        for var, lab in (("primary", "full"), ("dens", "subsampled")):
            comps.append((f"same length Q{q}: C750 vs C250 ({lab})", f"Q{q}|C750", P(f"Q{q}|C750"), M.sel(var, f"Q{q}|C250")))
    for name, pos, x, y in comps:
        o = cv_oof(pd.concat([x.assign(_y=1), y.assign(_y=0)], ignore_index=True))
        res = {}
        for k in ("A0", "B0", "A", "B"):
            res[k] = M.auc(k, o[o._y == 1], o[o._y == 0])
            rec("cv_auc_" + k, k, name, res[k])
        dAB = rec("cv_auc_B_minus_A", "B-A", name, res["B"] - res["A"])
        d0 = rec("cv_auc_B0_minus_A0", "B0-A0", name, res["B0"] - res["A0"])
        md.append(f"| {name} | {pos} | {len(x)} / {len(y)} | {fmt(ci(res['A0']))} | {fmt(ci(res['B0']))} | {fmt(d0)} | "
                  f"{fmt(ci(res['A']))} | {fmt(ci(res['B']))} | {fmt(dAB)} |")
    md.append(f"\nLength-only classifier for the main contrast: AUROC = 1.000 (query length is 250 vs 750 tokens by construction).\n")
    # --- controls: nulls side by side, own-document share, leave-document-out arm, effect sizes
    nulls(M, P, rec, md, model)
    for arm in ("lodo", "primary"):
        multiseed(M, rec, md, arm, rows, model)
    # --- geometry check
    md.append("\n### Geometry check: R = ||mean u_i|| vs mean sin(theta_i/2) (K=50, primary)\n")
    md.append("| cell | median mean sin(theta/2) | median R | median R (null) | frac R >= mean sin(theta/2) | violations (strict) | frac R_null >= mean sin |")
    md.append("|---|---|---|---|---|---|---|")
    for c in cells:
        d = P(c); ok = d.R >= d.mean_sin_half; okn = d.R_null >= d.mean_sin_half
        md.append(f"| {c} | {d.mean_sin_half.median():.4f} | {d.R.median():.4f} | {d.R_null.median():.4f} | {ok.mean():.4f} | "
                  f"{(~ok).sum()} (max shortfall {max(0, (d.mean_sin_half - d.R).max()):.2e}) | {okn.mean():.4f} |")
        rows.append(dict(corpus=CORPUS, model=model, kind="frac_R_ge_meansin", feature="R", contrast=c, est=ok.mean(), lo=np.nan, hi=np.nan))


def effsize(M, f, hi, lo):
    """(median(hi) - median(lo)) / mean of the two within-cell IQRs, per bootstrap row (oriented: hi = paper's
    multi-factual-like cell, or the finer corpus at fixed query length)."""
    hi = hi[np.isfinite(hi[f])]; lo = lo[np.isfinite(lo[f])]
    wh, wl = M.w(hi), M.w(lo); xh, xl = hi[f].values, lo[f].values
    iqr = ((wq(xh, wh, .75) - wq(xh, wh, .25)) + (wq(xl, wl, .75) - wq(xl, wl, .25))) / 2
    return (wq(xh, wh, .5) - wq(xl, wl, .5)) / iqr


def multiseed(M, rec, md, arm, rows, model, nseeds=20):
    """20 independent draws of the empirical (global) and local bearing nulls: per comparison, real AUROC with
    lecture-bootstrap CI, and the null AUROC and real - null as mean [2.5, 97.5 percentile] over the draws (point
    estimates per draw; the interval reflects null-draw randomness only, not lecture sampling)."""
    dens = "lodo_dens" if arm == "lodo" else "dens"
    comps = [("main Q250|C750 vs Q750|C250", M.sel(arm, "Q250|C750"), M.sel(arm, "Q750|C250")),
             ("diagonal Q250|C250 vs Q750|C750", M.sel(arm, "Q250|C250"), M.sel(arm, "Q750|C750")),
             ("same length Q250: C750 vs C250 (density-matched)", M.sel(arm, "Q250|C750"), M.sel(dens, "Q250|C250"))]
    if "w1_enull00" not in M.df.columns or any(len(x) == 0 or len(y) == 0 or x["w1_enull00"].isna().all() or
                                               y["w1_enull00"].isna().all() for _, x, y in comps):
        return
    one = lambda f, x, y: auc(x[f].values, y[f].values, np.ones((1, len(x))), np.ones((1, len(y))))[0]
    md.append(f"\n### {arm.upper()} arm: 20-draw empirical (global) and local bearing nulls\n")
    md.append("AUROC first-named > second. Null columns: mean [2.5, 97.5 percentile] over 20 independent draws.\n")
    md.append("| comparison | feature | AUROC real [lecture CI] | emp null AUROC | real - emp null | local null AUROC | real - local null |")
    md.append("|---|---|---|---|---|---|---|")
    for name, x, y in comps:
        for f in [t for t in TOPO if f"{t}_enull00" in M.df.columns]:
            r = rec(f"ms_{arm}_auc_real", f, name, M.auc(f, x, y))
            cols = []
            for nk in ("enull", "lnull"):
                na = np.array([one(f"{f}_{nk}{k:02d}", x, y) for k in range(nseeds)]); dd = r[0] - na
                for kind, v in ((f"ms_{arm}_{nk}_auc", na), (f"ms_{arm}_real_minus_{nk}", dd)):
                    rows.append(dict(corpus=CORPUS, model=model, kind=kind, feature=f, contrast=name, est=v.mean(),
                                     lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
                    cols.append(f"{v.mean():.3f} [{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]")
            md.append(f"| {name} | {NAMES[f]} | {fmt(r)} | " + " | ".join(cols) + " |")


def nulls(M, P, rec, md, model):
    have_l = "w1_lnull" in M.df.columns
    comps = [("main", P("Q250|C750"), P("Q750|C250")),
             ("same length Q250 (density-matched)", P("Q250|C750"), M.sel("dens", "Q250|C250"))]
    lodo = (M.df.variant == "lodo").any()
    if lodo:
        comps += [("LODO main", M.sel("lodo", "Q250|C750"), M.sel("lodo", "Q750|C250")),
                  ("LODO same length Q250 (density-matched)", M.sel("lodo", "Q250|C750"), M.sel("lodo_dens", "Q250|C250"))]
    md.append("\n### Three nulls side by side (AUROC first-named > second; K=50)\n")
    md.append("Isotropic = matched-cosine with uniform bearings (seed 0); empirical (global) = bearing from a random chunk of the corpus "
              "outside the query's top-50 (seed 1); local = bearing from a chunk drawn from the query's own ranks 51-500 (seed 2). "
              "LODO = all chunks of the query's own document removed from retrieval.\n")
    md.append("| comparison | feature | AUROC real | real - iso null | real - emp (global) null | real - local null | effect size (oriented dMedian/IQR) |")
    md.append("|---|---|---|---|---|---|---|")
    for name, x, y in comps:
        for f in TOPO:
            r = M.auc(f, x, y); rec("n3_auc", f, name, r)
            cols = []
            for nk in ("null", "enull", "lnull"):
                if nk == "lnull" and not have_l:
                    cols.append("--"); continue
                n = M.auc(f + "_" + nk, x, y); rec(f"n3_auc_{nk}", f, name, n)
                cols.append(fmt(rec(f"n3_real_minus_{nk}", f, name, r - n)))
            es = rec("effsize_oriented", f, name, effsize(M, f, y, x))
            md.append(f"| {name} | {NAMES[f]} | {fmt(ci(r))} | " + " | ".join(cols) + f" | {fmt(es)} |")
        for f in BASEL:
            r = M.auc(f, x, y); rec("n3_auc", f, name, r); es = rec("effsize_oriented", f, name, effsize(M, f, y, x))
            md.append(f"| {name} | {NAMES[f]} | {fmt(ci(r))} | -- | -- | -- | {fmt(es)} |")
    if "own_share" in M.df.columns:
        md.append("\n### Own-document share of the top-50 (after the overlap exclusion; mean over queries)\n")
        md.append("| cell | primary | no exclusion |")
        md.append("|---|---|---|")
        for c in ["Q250|C750", "Q750|C250", "Q250|C250", "Q750|C750"]:
            d = P(c); v = (d.own_share.values[None, :] * M.w(d)).sum(1) / M.w(d).sum(1); e = rec("own_share", "own_share", c, v)
            ne = M.sel("noexcl", c); nes = f"{ne.own_share.mean():.3f}" if len(ne) else "--"
            md.append(f"| {c} | {fmt(e)} | {nes} |")
    if lodo:
        md.append("\n### Leave-document-out: grouped 5-fold CV logistic regression (as above)\n")
        md.append("| comparison | AUROC A0 | B0 - A0 | AUROC A | AUROC B | B - A |")
        md.append("|---|---|---|---|---|---|")
        for name, x, y in comps[2:]:
            o = cv_oof(pd.concat([x.assign(_y=1), y.assign(_y=0)], ignore_index=True))
            res = {k: M.auc(k, o[o._y == 1], o[o._y == 0]) for k in ("A0", "B0", "A", "B")}
            for k, v in res.items():
                rec("cv_auc_" + k, k, name, v)
            d0 = rec("cv_auc_B0_minus_A0", "B0-A0", name, res["B0"] - res["A0"]); d = rec("cv_auc_B_minus_A", "B-A", name, res["B"] - res["A"])
            md.append(f"| {name} | {fmt(ci(res['A0']))} | {fmt(d0)} | {fmt(ci(res['A']))} | {fmt(ci(res['B']))} | {fmt(d)} |")


SC_A = ["top1", "mean50", "std50", "slope50"]
SC_T = ["w1", "ltmax", "longbar"]


def cv_oof(df):
    """Out-of-fold P(y=1) for feature sets A0, B0, A, B (GroupKFold(5) by lecture, StandardScaler + logistic)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    coh = ["nn_pair_cos", "nn_ncomp"]
    sets = {"A0": SC_A, "B0": SC_A + SC_T, "A": SC_A + coh, "B": SC_A + coh + SC_T}
    out = df[["lecture", "_y"]].copy()
    y = df._y.values; g = df.lecture.to_numpy(dtype=object)
    for k, cols in sets.items():
        X = df[cols].values; p = np.full(len(df), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        out[k] = p
    return out


def main():
    rows, md = [], []
    present = [m for m in MODELS if feat_path(OUT, m).exists()]
    md.append(f"# Real-text experiment ({CORPUS}): statistics\n")
    md.append(f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} by stats.py from the per-query feature tables. "
              f"Cluster bootstrap over {UNIT}s, B={B}, numpy default_rng({SEED}). K=50 primary; eps0.4 = paper's neighbour rule. "
              f"AUROC > 0.5 means the first-named group is higher. Query length in tokens is constant within a query granularity, so its "
              f"AUROC is degenerate (0 or 1) wherever query granularity differs; it is shown to make the length confound explicit.\n")
    md.append("Reading the factorial: a pure query-length effect moves only the query effect (diagonal AUROC != 0.5, corpus effect ~0.5); "
              "a pure corpus-density effect moves only the corpus effect, and should vanish under the density control; an effect of "
              "query-vs-corpus granularity *ratio* (the paper's polysemic/multi-factual framing) moves query and corpus effects in the same "
              "direction and leaves the diagonal at 0.5; a non-additive effect shows as an interaction.\n")
    meds = []
    for m in present:
        df = pd.read_csv(feat_path(OUT, m))
        analyse(m, df, rows, md)
    L = pd.DataFrame(rows); L.to_csv(OUT / "stats_long.csv", index=False)
    L[L.kind == "median"].to_csv(OUT / "cell_medians.csv", index=False)
    (OUT / "stats.md").write_text(re.sub(r"Q(250|750)\|C(250|750)", r"Q\1\\|C\2", "\n".join(md) + "\n"))  # escape | in cells
    print(f"wrote {OUT/'stats.md'} for {present}")


if __name__ == "__main__":
    main()
