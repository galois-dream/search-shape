# combine.py -- corpus x model summary tables from each corpus's stats_long.csv (+ dens_seeds_*.csv and hgb_*.csv
# from controls.py), written to $OUTROOT/stats_summary.md and prepended to $OUTROOT/stats.md. ORIENTATION: every AUROC
# here is oriented so > 0.5 means the multi-factual-like cell (Q750|C250) is higher in the main contrast, Q750|C750 is
# higher in the diagonal (ratio) test, or the finer corpus (C250) is higher at fixed query length; real-minus-null
# differences and the interaction are sign-flipped accordingly. CV classifier AUROCs are orientation-free. Effect size
# = oriented (median difference) / mean within-cell IQR. Usage: [OUTROOT=out_clean] python combine.py
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

OUTROOT = __import__('os').environ.get('OUTROOT', 'out')  # the paper: OUTROOT=out_clean
ROOT = Path(os.environ.get("RT_WORK", str(Path(__file__).resolve().parent / "work")))  # see README
SRC = {"nobel": ROOT / OUTROOT, "wiki": ROOT / OUTROOT / "wiki", "gutenberg": ROOT / OUTROOT / "gutenberg", "turing": ROOT / OUTROOT / "turing", "fields": ROOT / OUTROOT / "fields"}
MODELS = ["openai-3-small", "openai-3-large", "vertex-005", "modernbert-embed", "bge-base", "gemini-001", "openai-ada-002", "bge-m3", "qwen3-0.6b"]
SCORES = ["top1", "mean50", "std50", "slope50"]
L = pd.concat([pd.read_csv(p / "stats_long.csv") for p in SRC.values() if (p / "stats_long.csv").exists()])
MAIN, SL = "main: Q250|C750 vs Q750|C250", "same length Q250: C750 vs C250 (subsampled)"
N3M, N3S = "main", "same length Q250 (density-matched)"


def row(c, m, kind, feat, contrast=None):
    x = L[(L.corpus == c) & (L.model == m) & (L.kind == kind) & (L.feature == feat)]
    if contrast is not None:
        x = x[x.contrast == contrast]
    return None if x.empty else x.iloc[0]


def g(c, m, kind, feat, contrast=None, flip=None, d=3):
    """flip: None (as is), 'auc' (1 - x), 'neg' (-x)."""
    r = row(c, m, kind, feat, contrast)
    if r is None:
        return "--"
    e, lo, hi = r.est, r.lo, r.hi
    if flip == "auc":
        e, lo, hi = 1 - e, 1 - hi, 1 - lo
    elif flip == "neg":
        e, lo, hi = -e, -hi, -lo
    return f"{e:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]" if pd.notna(lo) else f"{e:.{d}f}"


def best_score(c, m):
    x = L[(L.corpus == c) & (L.model == m) & (L.kind == "auc_main") & L.feature.isin(SCORES)]
    if x.empty:
        return "--"
    r = x.loc[(x.est - 0.5).abs().idxmax()]
    return f"{r.feature} {1 - r.est:.3f} [{1 - r.hi:.3f}, {1 - r.lo:.3f}]"


def present():
    for c in SRC:
        for m in MODELS:
            if not L[(L.corpus == c) & (L.model == m)].empty:
                yield c, m


md = ["# Combined summary (corpus x model)\n",
      "Oriented AUROCs: > 0.5 means the paper's multi-factual-like cell Q750|C250 is higher than Q250|C750 (main contrast), "
      "or the finer corpus C250 is higher than C750 at fixed query length (same-length). Real - null differences use the same "
      "orientation. K=50, overlap exclusion; CIs = cluster bootstrap over lectures/pages/books (B=2000, seed 0). 'iso' = "
      "isotropic matched-cosine null, 'emp' = empirical-bearing (global) null, 'local' = local-bearing null (ranks 51-500). "
      "Best score = the cosine baseline (top-1, mean, std, slope) furthest from 0.5. Query length alone separates the main "
      "contrast perfectly by construction. Effect size = oriented median difference / mean within-cell IQR.\n",
      "@@S0@@",
      "\n## Table S1. Main contrast (Q750|C250 vs Q250|C750)\n",
      "| corpus | model | W1 AUROC | W1 effect size | W1 real-iso | W1 real-emp | LT_max AUROC | LT_max real-iso | longest bar AUROC | longest bar real-iso | W1 interaction (oriented) | best score | coherence AUROC |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for c, m in present():
    md.append(f"| {c} | {m} | {g(c, m, 'auc_main', 'w1', flip='auc')} | {g(c, m, 'effsize_oriented', 'w1', N3M, d=2)} | "
              f"{g(c, m, 'auc_main_real_minus_null', 'w1', flip='neg')} | {g(c, m, 'auc_main_real_minus_enull', 'w1', flip='neg')} | "
              f"{g(c, m, 'auc_main', 'ltmax', flip='auc')} | {g(c, m, 'auc_main_real_minus_null', 'ltmax', flip='neg')} | "
              f"{g(c, m, 'auc_main', 'longbar', flip='auc')} | {g(c, m, 'auc_main_real_minus_null', 'longbar', flip='neg')} | "
              f"{g(c, m, 'interaction_auc', 'w1', flip='neg')} | {best_score(c, m)} | {g(c, m, 'auc_main', 'nn_pair_cos', flip='auc')} |")
md += ["\n## Table S2. Same query length (Q250; C250 density-matched vs C750) and incremental value of topology (logistic, grouped 5-fold CV)\n",
       "| corpus | model | W1 AUROC | W1 effect size | W1 real-iso | W1 real-emp | coherence AUROC | CV main: A | CV main: B-A | CV same-length Q250: A | CV same-length Q250: B-A | CV same-length Q750: B-A |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for c, m in present():
    md.append(f"| {c} | {m} | {g(c, m, 'auc_corpus_eff_dens', 'w1', '@Q250', flip='auc')} | {g(c, m, 'effsize_oriented', 'w1', N3S, d=2)} | "
              f"{g(c, m, 'auc_corpus_eff_dens_real_minus_null', 'w1', '@Q250', flip='neg')} | "
              f"{g(c, m, 'auc_corpus_eff_dens_real_minus_enull', 'w1', '@Q250', flip='neg')} | "
              f"{g(c, m, 'auc_corpus_eff_dens', 'nn_pair_cos', '@Q250', flip='auc')} | {g(c, m, 'cv_auc_A', 'A', MAIN)} | "
              f"{g(c, m, 'cv_auc_B_minus_A', 'B-A', MAIN)} | {g(c, m, 'cv_auc_A', 'A', SL)} | "
              f"{g(c, m, 'cv_auc_B_minus_A', 'B-A', SL)} | {g(c, m, 'cv_auc_B_minus_A', 'B-A', SL.replace('Q250', 'Q750'))} |")


def seeds(c, m):
    f = SRC[c] / f"dens_seeds_{m}.csv"
    if not f.exists():
        return "--", "--"
    d = pd.read_csv(f); a = 1 - d.auc_w1; r = -d.real_minus_enull
    return (f"{a.mean():.3f} ({a.min():.3f}-{a.max():.3f})", f"{r.mean():+.3f} ({r.min():+.3f} to {r.max():+.3f})")


def hgbcol(c, m, comp):
    f = SRC[c] / f"hgb_{m}.csv"
    if not f.exists():
        return "--", "--"
    d = pd.read_csv(f); d = d[d.comparison == comp]
    p, b = d[d.rep == 0].iloc[0], d[d.rep > 0]
    lo, hi = np.percentile(b.B_minus_A, [2.5, 97.5])
    return f"{p.auc_A:.3f} / {p.auc_B:.3f}", f"{p.B_minus_A:+.3f} [{lo:+.3f}, {hi:+.3f}] (R={len(b)})"


md += ["\n## Table S3. Red-team controls (oriented as above)\n",
       "Own-doc share = mean fraction of the top-50 from the query's own document after overlap exclusion (Q250|C750 / Q750|C250). "
       "LODO = leave-document-out retrieval. Local null = bearings from the query's own ranks 51-500. A'/B' = HistGradientBoosting on "
       "sorted cosines + Gram summaries, without/with the three persistence features, grouped 5-fold CV; CI from a cluster bootstrap "
       "that refits the CV in every replicate (R stated). Density seeds = 20 C250 subsamples (seeds 0-19), mean (range), point estimates.\n",
       "| corpus | model | own-doc share | LODO W1 AUROC (main) | LODO W1 real-emp (main) | LODO CV B-A (main) | W1 real-local (main) | W1 real-local (same-length) | A'/B' same-length | B'-A' same-length | A'/B' LODO same-length | B'-A' LODO same-length | dens seeds: W1 AUROC | dens seeds: W1 real-emp |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for c, m in present():
    own = row(c, m, "own_share", "own_share", "Q250|C750"), row(c, m, "own_share", "own_share", "Q750|C250")
    owns = "--" if own[0] is None else f"{own[0].est:.3f} / {own[1].est:.3f}"
    hm, hs = hgbcol(c, m, SL), hgbcol(c, m, "LODO " + SL)
    sa, sr = seeds(c, m)
    md.append(f"| {c} | {m} | {owns} | {g(c, m, 'n3_auc', 'w1', 'LODO main', flip='auc')} | "
              f"{g(c, m, 'n3_real_minus_enull', 'w1', 'LODO main', flip='neg')} | {g(c, m, 'cv_auc_B_minus_A', 'B-A', 'LODO main')} | "
              f"{g(c, m, 'n3_real_minus_lnull', 'w1', N3M, flip='neg')} | {g(c, m, 'n3_real_minus_lnull', 'w1', N3S, flip='neg')} | "
              f"{hm[0]} | {hm[1]} | {hs[0]} | {hs[1]} | {sa} | {sr} |")
def s0():
    """Consolidated corpus x model x arm table: longest bar and W1 (source of the paper's real-text and ratio tables)."""
    out = ["\n## Table S0. Consolidated: primary vs leave-document-out (LODO), oriented AUROCs\n",
           "Main = Q750|C250 > Q250|C750; diag = Q750|C750 > Q250|C250 (the granularity-ratio reading predicts 0.5); SL = same "
           "query length Q250, C250 (density-matched) > C750. Real AUROCs with 95% document-bootstrap CI. Null differences: "
           "primary arm = one null draw with document-bootstrap CI; LODO arm = mean [2.5, 97.5 pct] over 20 draws (draw "
           "randomness only). emp = global empirical bearings; local = bearings from the query's own ranks 51-500. "
           "Own-doc = share of the top-50 from the query's own document (Q250|C750 / Q750|C250). GBM = HistGradientBoosting "
           "B'-A' on SL (refit bootstrap). '--' = not computed by the time of this table.\n",
           "| corpus | model | arm | own-doc | longbar main | longbar diag | longbar SL | longbar SL real-emp | longbar SL real-local | W1 main | W1 diag | W1 SL | W1 SL real-emp | GBM B'-A' (SL) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    SLn = "same length Q250 (density-matched)"
    MSm, MSd, MSs = COMPS[0][0], COMPS[1][0], COMPS[2][0]
    for c, m in present():
        o1, o2 = row(c, m, "own_share", "own_share", "Q250|C750"), row(c, m, "own_share", "own_share", "Q750|C250")
        own = "--" if o1 is None else f"{o1.est:.2f} / {o2.est:.2f}"
        cells = []
        for f in ("longbar", "w1"):
            cells += [g(c, m, "auc_main", f, flip="auc"), g(c, m, "auc_diagonal", f, flip="auc"),
                      g(c, m, "n3_auc", f, SLn, flip="auc"), g(c, m, "n3_real_minus_enull", f, SLn, flip="neg")]
            if f == "longbar":
                cells.append(g(c, m, "n3_real_minus_lnull", f, SLn, flip="neg"))
        md0 = f"| {c} | {m} | primary | {own} | " + " | ".join(cells) + f" | {hgbcol(c, m, SL)[1]} |"
        cells = []
        for f in ("longbar", "w1"):
            cells += [g(c, m, "ms_lodo_auc_real", f, MSm, flip="auc"), g(c, m, "ms_lodo_auc_real", f, MSd, flip="auc"),
                      g(c, m, "ms_lodo_auc_real", f, MSs, flip="auc"), g(c, m, "ms_lodo_real_minus_enull", f, MSs, flip="neg")]
            if f == "longbar":
                cells.append(g(c, m, "ms_lodo_real_minus_lnull", f, MSs, flip="neg"))
        out += [md0, f"| {c} | {m} | LODO | 0 | " + " | ".join(cells) + f" | {hgbcol(c, m, 'LODO ' + SL)[1]} |"]
    return "\n".join(out)


COMPS = [("main Q250|C750 vs Q750|C250", "main (Q750|C250 > Q250|C750)"),
         ("diagonal Q250|C250 vs Q750|C750", "diagonal (Q750|C750 > Q250|C250)"),
         ("same length Q250: C750 vs C250 (density-matched)", "same length Q250 (C250 > C750)")]
md += ["\n## Table S4. Leave-document-out (LODO) and primary arms with 20-draw bearing nulls (oriented)\n",
       "Real AUROC with 95% cluster-bootstrap CI. Null AUROC and real - null: mean [2.5, 97.5 percentile] over 20 independent "
       "draws (emp = global empirical bearings, seed (k,1,query); local = bearings from the query's own ranks 51-500, seed (k,2,query)); "
       "that interval reflects null-draw randomness only. Diagonal = matched-granularity cells; the granularity-ratio reading of the "
       "definition predicts 0.5 there. Rows appear only where the 20-draw nulls were computed.\n",
       "| corpus | model | arm | comparison | feature | AUROC real | emp null AUROC | real - emp | local null AUROC | real - local |",
       "|---|---|---|---|---|---|---|---|---|---|"]
for c, m in present():
    for arm in ("lodo", "primary"):
        for comp, lab in COMPS:
            for f in ("w1", "ltmax", "longbar"):
                if row(c, m, f"ms_{arm}_auc_real", f, comp) is None:
                    continue
                md.append(f"| {c} | {m} | {arm} | {lab} | {f} | {g(c, m, f'ms_{arm}_auc_real', f, comp, flip='auc')} | "
                          f"{g(c, m, f'ms_{arm}_enull_auc', f, comp, flip='auc')} | {g(c, m, f'ms_{arm}_real_minus_enull', f, comp, flip='neg')} | "
                          f"{g(c, m, f'ms_{arm}_lnull_auc', f, comp, flip='auc')} | {g(c, m, f'ms_{arm}_real_minus_lnull', f, comp, flip='neg')} |")
md += ["\n## Table S5. Own-document share of the top-50 (primary arm, after overlap exclusion; mean with cluster-bootstrap CI)\n",
       "| corpus | model | Q250|C750 | Q750|C250 | Q250|C250 | Q750|C750 |", "|---|---|---|---|---|---|"]
for c, m in present():
    if row(c, m, "own_share", "own_share", "Q250|C750") is None:
        continue
    md.append(f"| {c} | {m} | " + " | ".join(g(c, m, "own_share", "own_share", cell) for cell in
                                         ("Q250|C750", "Q750|C250", "Q250|C250", "Q750|C750")) + " |")
md = [x if x != "@@S0@@" else s0() for x in md]
text = re.sub(r"Q(250|750)\|C(250|750)", r"Q\1\\|C\2", "\n".join(md) + "\n")  # escape | inside table cells
(ROOT / OUTROOT / "stats_summary.md").write_text(text)
s = (ROOT / OUTROOT / "stats.md").read_text()
if s.startswith("# Combined summary"):
    s = s[s.index("\n# Real-text experiment"):].lstrip("\n")
(ROOT / OUTROOT / "stats.md").write_text(text + "\n---\n\n" + s)
print(text)
