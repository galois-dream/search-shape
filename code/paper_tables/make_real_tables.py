# make_real_tables.py -- regenerates the bodies of the paper's real-text tables (tab_real.tex: per-corpus summary,
# median and range over the embedding models; tab_real_full.tex: every model; tab_ratio.tex; tab_controls.tex: every
# control x every persistence feature), the appendix dot plot's data (realctl.dat, realctl_meta.tex) and the ledger of
# every number the real-text results section states (ledger_sec6.txt), from the unrounded estimates in each corpus's
# stats_long.csv (the cells of Table S0), the Bayesian cells (bayes/bb_cells_subvar.csv: subsample-adjusted, else
# bb_cells.csv), and data/realtext/<corpus>/{hgb,docmatch}_*.csv. Columns named *_enull / *_lnull are the paper's
# corpus-bearing / shell-bearing controls.
# Usage: python make_real_tables.py <reference_outputs dir> <dir holding tab_real*.tex, tab_ratio.tex>  (edited in place)
import pathlib, sys
import datetime
import pandas as pd

import os
REF, TABDIR = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])   # data/reference_outputs, output folder
OUT = REF   # holds bayes/
SRC = {c: REF.parent / "realtext" / {"wiki": "wikipedia"}.get(c, c) for c in ("nobel", "turing", "fields", "gutenberg", "wiki")}
L = pd.concat([pd.read_csv(REF / f"realtext_{c}_stats_long.csv") for c in SRC if (REF / f"realtext_{c}_stats_long.csv").exists()])
SL_P, SL_L = "same length Q250 (density-matched)", "same length Q250: C750 vs C250 (density-matched)"
DIAG_L = "diagonal Q250|C250 vs Q750|C750"


def get(c, m, kind, feat, contrast=None, flip=None):
    """(est, lo, hi) as in combine.py's Table S0: flip 'auc' -> 1 - x, 'neg' -> -x.
    the fixed-length, own-document-out real - corpus-bearing difference comes from the
    subsample-adjusted cells when they exist (mean over 10 size-matched subsamples; bootstrap + between-seed variance),
    so tables, figure and ledger agree with each other and with the Bayesian analysis."""
    if kind == "ms_lodo_real_minus_enull" and contrast == SL_L and flip == "neg" and BBNAME == "bb_cells_subvar.csv":
        r = BB[(BB.corpus == c) & (BB.model == m) & (BB.feature == feat)].iloc[0]
        return r.d_mean, r.d_lo, r.d_hi
    x = L[(L.corpus == c) & (L.model == m) & (L.kind == kind) & (L.feature == feat)]
    if contrast is not None:
        x = x[x.contrast == contrast]
    r = x.iloc[0]
    e, lo, hi = r.est, r.lo, r.hi
    if flip == "auc":
        e, lo, hi = 1 - e, 1 - hi, 1 - lo
    elif flip == "neg":
        e, lo, hi = -e, -hi, -lo
    return e, lo, hi


def f2(v): return f"{v:.2f}"
def sg(v): s = f"{v:+.2f}"; return "$" + ("+0.00" if s == "-0.00" else s) + "$"
def shade(v, centre, full, cap=45):
    """cell tint on a diverging scale, so relative values in large tables read at a glance.
    Vermillion above the centre, blue below; tint proportional to |v - centre| / full, saturating at cap % (text stays legible)."""
    x = max(-1.0, min(1.0, (v - centre) / full)); k = round(abs(x) * cap)
    return "" if k < 2 else r"\cellcolor{" + ("tabV" if x > 0 else "tabB") + f"!{k}" + "}"
def dg(t):
    e, lo, hi = t
    return shade(e, 0.5, 0.5) + (r"\textbf{" + f2(e) + "}" if (lo > 0.5 or hi < 0.5) else f2(e))


# ONE "clear" rule for tables and figure: paired document-bootstrap interval of
# real - corpus-bearing control above 0 (bayes.py bb_cells, the registered criterion) AND real AUROC > 0.5.
# prefer the subsample-adjusted cells (bayes.py --subvar: seed-mean estimate, bootstrap + between-seed variance)
BBNAME = "bb_cells_subvar.csv" if (OUT / "bayes" / "bb_cells_subvar.csv").exists() else "bb_cells.csv"
BB = pd.read_csv(OUT / "bayes" / BBNAME)
def is_clear(ck, mk, feat="longbar"):
    r = BB[(BB.corpus == ck) & (BB.model == mk) & (BB.feature == feat)].iloc[0]
    return bool(r.d_lo > 0 and r.auroc_real_mean > 0.5)


def clear_cell(g, ck, mk):
    """L_max LODO real - corpus-bearing control; bold when the real AUROC's 95% interval is clear of the control AUROC."""
    d = g('ms_lodo_real_minus_enull', 'longbar', SL_L, flip='neg')[0]
    r = g('ms_lodo_auc_real', 'longbar', SL_L, flip='auc')
    return r"\textbf{" + f"{d:+.2f}" + r"}\rlap{\,\textcolor[HTML]{D55E00}{$\bullet$}}" if is_clear(ck, mk) else sg(d)   # clear implies d > 0; and real AUROC > 0.5


corp = [(k, n) for k, n in [("nobel", "Nobel"), ("turing", "Turing"), ("fields", "Fields"), ("gutenberg", "Essays"), ("wiki", "Wikipedia")]
        if (L.corpus == k).any()]
# models = every model present in the stats, in this order (was the fixed five)
ALLM = [("openai-3-small", "3-small"), ("openai-3-large", "3-large"), ("vertex-005", "005"), ("gemini-001", "Gemini"), ("openai-ada-002", "ada-002"),
        ("modernbert-embed", "ModernBERT"), ("bge-base", "bge-base"), ("bge-m3", "bge-m3"), ("qwen3-0.6b", "Qwen3-0.6B")]
mods = [(k, n) for k, n in ALLM if all((L.corpus == c).any() and ((L.corpus == c) & (L.model == k)).any() for c, _ in corp)]
t3, tr = [], []
for j, (ck, cn) in enumerate(corp):
    for i, (mk, mn) in enumerate(mods):
        g = lambda *a, **k: get(ck, mk, *a, **k)
        t3.append(f"{cn if i == 0 else ''} & {mn} & {f2(g('auc_main', 'w1', flip='auc')[0])} & {f2(g('auc_main', 'longbar', flip='auc')[0])}"
                  f" & {sg(g('ms_lodo_real_minus_enull', 'w1', SL_L, flip='neg')[0])}"
                  f" & {sg(g('n3_real_minus_enull', 'longbar', SL_P, flip='neg')[0])}"
                  f" & {clear_cell(g, ck, mk)} \\\\")
        tr.append(f"{cn if i == 0 else ''} & {mn} & {dg(g('auc_diagonal', 'w1', flip='auc'))} & {dg(g('ms_lodo_auc_real', 'w1', DIAG_L, flip='auc'))}"
                  f" & {dg(g('auc_diagonal', 'longbar', flip='auc'))} & {dg(g('ms_lodo_auc_real', 'longbar', DIAG_L, flip='auc'))} \\\\")
    if j < len(corp) - 1: t3.append(r"\midrule"); tr.append(r"\midrule")
import statistics
# Table 3 redesigned for legibility (was 7 columns shrunk by resizebox to ~5 pt): the "L_max, all chunks"
# column moved to the text (Section 6.2); each gain sits next to its own "clear" count.
COLS = [("auc_main", "w1", None, "auc"), ("auc_main", "longbar", None, "auc"), ("ms_lodo_real_minus_enull", "w1", SL_L, "neg"),
        ("ms_lodo_real_minus_enull", "longbar", SL_L, "neg")]
def cell(ck, kind, feat, con, fl):
    v = [get(ck, mk, kind, feat, con, flip=fl)[0] for mk, _ in mods]
    m, lo, hi = statistics.median(v), min(v), max(v)
    if fl == "auc":
        return f2(m) + r" {\scriptsize(" + f2(lo) + "--" + f2(hi) + ")}"
    return sg(m) + r" {\scriptsize(" + sg(lo) + " to " + sg(hi) + ")}"
def nclear(ck):
    n = 0
    for mk, _ in mods:
        n += is_clear(ck, mk)
    return f"{n} of {len(mods)}"
def nclear_f(ck, feat):  # W1 "clear" column too (cold reader: W1 is clear as often as L_max)
    return str(sum(is_clear(ck, mk, feat) for mk, _ in mods))
ts = [f"{cn} & {cell(ck, *COLS[0])} & {cell(ck, *COLS[1])} & {cell(ck, *COLS[2])} & {nclear_f(ck, 'w1')} & {cell(ck, *COLS[3])} & {nclear_f(ck, 'longbar')}" + r" \\" for ck, cn in corp]
oldf = (TABDIR / "tab_real_full.tex").read_text()
old3 = (TABDIR / "tab_real.tex").read_text(); oldr = (TABDIR / "tab_ratio.tex").read_text()
def swap(tex, body):
    a = tex.index(r"\midrule") + len(r"\midrule") + 1; b = tex.index(r"\bottomrule")
    return tex[:a] + "\n".join(body) + "\n" + tex[b:]
(TABDIR / "tab_real_full.tex").write_text(swap(oldf, t3)); (TABDIR / "tab_real.tex").write_text(swap(old3, ts))
(TABDIR / "tab_ratio.tex").write_text(swap(oldr, tr))
print("corpora:", [c for _, c in corp], "rows:", len(t3), len(tr))

# Fig. 11 redesigned: one question per panel -- how much
# does the actual arrangement add? Per corpus x model: real - corpus-bearing control AUROC (d) with its paired 95%
# interval (bb_cells), for W1, L_max and LT_max; clear as is_clear. Rows: corpus bands, models stacked inside.
FN = {"w1": "w1", "lb": "longbar", "lt": "ltmax"}
rows = ["y " + " ".join(f"{t}_d {t}_lo {t}_hi {t}_clear" for t in FN)]
for j, (ck, cn) in enumerate(corp):
    for i, (mk, mn) in enumerate(mods):
        vals = []
        for t, feat in FN.items():
            r = BB[(BB.corpus == ck) & (BB.model == mk) & (BB.feature == feat)].iloc[0]
            vals += [f"{r.d_mean:.4f}", f"{r.d_lo:.4f}", f"{r.d_hi:.4f}", str(int(is_clear(ck, mk, feat)))]
        rows.append(f"{(len(mods) + 1) * j + i} " + " ".join(vals))  # same rows as Fig. 11 (effects_meta.tex)
(TABDIR / "realctl.dat").write_text("\n".join(rows) + "\n")
G = len(mods) + 2
(TABDIR / "realctl_meta.tex").write_text(
    f"% make_real_tables.py output {len(mods)} models\n"
    f"\\def\\rcYmax{{{G * (len(corp) - 1) + len(mods) - 1 + 0.8}}}\\def\\rcN{{{len(mods)}}}\n"
    "\\def\\rcGroups{" + ",".join(f"{G * j + (len(mods) - 1) / 2:g}/{cn}" for j, (_, cn) in enumerate(corp)) + "}\n")

# ledger of every number Section 6 states, recomputed from the same stats (and the classifier,
# Bayes and geometry outputs), written to ledger_sec6.txt.
import datetime, glob
LG = [f"# ledger_sec6.txt -- models={[k for k, _ in mods]}; paired cells from {BBNAME}"]
def rng_(v): return f"{min(v):.2f} to {max(v):.2f}"
CM = [(ck, mk) for ck, _ in corp for mk, _ in mods]; N = len(CM)
w = [get(c, m, "auc_main", "w1", flip="auc") for c, m in CM]
LG.append(f"L1 main W1 AUROC {rng_([x[0] for x in w])}; CI above 0.5 in {sum(x[1] > 0.5 for x in w)} of {N}")
lb = {(c, m): get(c, m, "auc_main", "longbar", flip="auc") for c, m in CM}
LG.append(f"L1 main longbar CI above 0.5 in {sum(x[1] > 0.5 for x in lb.values())} of {N}; others: "
          + ", ".join(f"{c}/{m} {x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}]" for (c, m), x in lb.items() if x[1] <= 0.5))
best = [max(abs(get(c, m, "auc_main", f, flip="auc")[0] - 0.5) for f in ("top1", "mean50", "std50", "slope50")) + 0.5 for c, m in CM]
LG.append(f"L1 best single cosine statistic (max |AUROC-0.5|+0.5 over top1/mean/std/slope) {rng_(best)}")
for feat in ("w1", "longbar"):
    pa = [get(c, m, "auc_diagonal", feat, flip="auc") for c, m in CM]; la = [get(c, m, "ms_lodo_auc_real", feat, DIAG_L, flip="auc") for c, m in CM]
    ex = lambda t: t[1] > 0.5 or t[2] < 0.5
    LG.append(f"L2 diagonal {feat}: CI excludes 0.5 in {sum(map(ex, pa))} of {N} (all chunks), {sum(map(ex, la))} of {N} (own document out)")
# query-length and corpus-granularity effects (all chunks), oriented as in the draft: AUROC > 0.5 means
# larger W1 for 750-token queries (query effect) / for the size-matched 250-token corpus (corpus effect)
for kind, con, lab in (("auc_query_eff", "@C750", "query effect at C750"), ("auc_query_eff", "@C250", "query effect at C250"),
                       ("auc_corpus_eff_dens", "@Q250", "corpus effect (size-matched) at Q250"), ("auc_corpus_eff_dens", "@Q750", "corpus effect (size-matched) at Q750")):
    for feat in ("w1", "longbar"):
        v = [get(c, m, kind, feat, con, flip="auc") for c, m in CM]
        LG.append(f"L2 {lab} {feat}: AUROC {rng_([x[0] for x in v])}; CI above 0.5 in {sum(x[1] > 0.5 for x in v)} of {N}, below in {sum(x[2] < 0.5 for x in v)}")
dw = {(c, m): get(c, m, "ms_lodo_real_minus_enull", "w1", SL_L, flip="neg")[0] for c, m in CM}
LG.append(f"L3 W1 real - corpus-bearing control (LODO, fixed Q250): |d| <= 0.08 in {sum(abs(v) <= 0.08 for v in dw.values())} of {N}; "
          f"range {rng_(list(dw.values()))}; outside: " + ", ".join(f"{c}/{m} {v:+.2f}" for (c, m), v in dw.items() if abs(v) > 0.08))
for ck, cn in corp:
    ds = [get(ck, mk, "ms_lodo_real_minus_enull", "longbar", SL_L, flip="neg")[0] for mk, _ in mods]
    BBF = OUT / "bayes" / BBNAME   # paired document-bootstrap interval of real - control (the registered criterion)
    bb = pd.read_csv(BBF) if BBF.exists() else None
    bbx = bb[(bb.corpus == ck) & (bb.feature == "longbar")] if bb is not None else None
    pair = f"paired above (real AUROC > 0.5) {((bbx.d_lo > 0) & (bbx.auroc_real_mean > 0.5)).sum()}, below {(bbx.d_hi < 0).sum()} of {len(bbx)} (bb_cells: {sorted(bbx.model)})" if bbx is not None else "no bb_cells"
    LG.append(f"L3 longbar {cn}: clear {nclear(ck)}; d {rng_(ds)}; median {statistics.median(ds):+.2f}; {pair}")
lt = [get(c, m, "ms_lodo_auc_real", "ltmax", SL_L, flip="auc")[0] for c, m in CM]; ld = [get(c, m, "ms_lodo_real_minus_enull", "ltmax", SL_L, flip="neg")[0] for c, m in CM]
LG.append(f"L3 LT_max fixed-length AUROC median {statistics.median(lt):.2f} ({rng_(lt)}); minus control median {statistics.median(ld):+.2f} ({rng_(ld)})")
# random-bearing control direction, shell vs corpus-bearing control, own-document share
LSL = "LODO same length Q250 (density-matched)"
for feat in ("w1", "longbar"):
    opp = 0
    for c, m in CM:
        r = get(c, m, "ms_lodo_auc_real", feat, SL_L, flip="auc")[0]; nl = 1 - get(c, m, "n3_auc_null", feat, LSL)[0]
        opp += (r - 0.5) * (nl - 0.5) < 0
    LG.append(f"L3 random-bearing control (LODO, fixed Q250) on the other side of 0.5 from the real {feat}: {opp} of {N}")
for ck, cn in corp:
    de = [get(ck, mk, "ms_lodo_real_minus_enull", "longbar", SL_L, flip="neg")[0] for mk, _ in mods]
    dl = [get(ck, mk, "ms_lodo_real_minus_lnull", "longbar", SL_L, flip="neg")[0] for mk, _ in mods]
    LG.append(f"L3 longbar {cn}: median real - corpus-bearing {statistics.median(de):+.3f}, real - shell-bearing {statistics.median(dl):+.3f}")
for cell in ("Q750|C250", "Q250|C750"):
    v = [get(c, m, "own_share", "own_share", cell)[0] for c, m in CM]
    LG.append(f"L0 own-document share of top 50, {cell} (all chunks): {min(v):.0%} to {max(v):.0%}")
H = OUT / "bayes" / ("hier_summary_subvar.csv" if BBNAME == "bb_cells_subvar.csv" else "hier_summary.csv")
if H.exists():
    h = pd.read_csv(H); h = h[h.spec == "main"] if "spec" in h else h
    for _, r in h.iterrows():
        LG.append(f"L4 bayes {r.feature} {r.param}: {r['mean']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] P>0 {r.p_gt0:.2f} rhat {r.rhat:.3f}")
    LG.append("L4 (check that hier_summary's models match the list above)")
hg = []
for ck, cn in corp:
    for mk, mn in mods:
        f = SRC[ck] / f"hgb_{mk}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        for comp, g_ in d.groupby("comparison"):
            q = g_.B_minus_A.quantile([0.025, 0.5, 0.975]).to_numpy()
            hg.append((cn, mn, comp, g_.auc_A.median(), *q))
if hg:
    LG.append(f"L5 flexible classifier: {len(hg)} comparisons ({len({(a, b) for a, b, *_ in hg})} corpus-model pairs; models "
              f"{sorted({b for _, b, *_ in hg})}); A' median {rng_([x[3] for x in hg])}; B'-A' median {rng_([x[5] for x in hg])}; "
              f"2.5-97.5 interval includes 0 in {sum(x[4] <= 0 <= x[6] for x in hg)}; widest " +
              "{} {} {}: {:+.3f} [{:+.3f}, {:+.3f}]".format(*max(hg, key=lambda x: x[6] - x[4])[:3], *max(hg, key=lambda x: x[6] - x[4])[5:6], *max(hg, key=lambda x: x[6] - x[4])[4:5], max(hg, key=lambda x: x[6] - x[4])[6]))
    for x in sorted(hg, key=lambda x: x[3])[:4]:
        LG.append(f"L5 lowest A': {x[0]}/{x[1]} {x[2]}: A' {x[3]:.3f}, B'-A' {x[5]:+.3f} [{x[4]:+.3f}, {x[6]:+.3f}]")
fr = L[L.kind == "frac_R_ge_meansin"]
LG.append(f"L6 bound R >= mean sin(theta/2): min share of queries satisfying it {fr.est.min():.4f} over {len(fr)} cells")
# every control x every persistence feature. Median over models of real - control AUROC at fixed length, own document out. Document-matched control:
# docmatch.py, text-embedding-3-small and ModernBERT only. Writes tab_controls.tex (appendix) and a ledger line per row.
import glob as _g, os as _os
LSL = "LODO same length Q250 (density-matched)"
CR = []
for feat, fl in (("w1", "$\\Wone$"), ("longbar", "$\\Lmax$"), ("ltmax", "$\\LTmax$")):
    for ctl, kind, con in (("random-bearing", "n3_real_minus_null", LSL), ("corpus-bearing", "ms_lodo_real_minus_enull", SL_L),
                           ("shell-bearing", "ms_lodo_real_minus_lnull", SL_L)):
        med = [statistics.median(get(ck, mk, kind, feat, con, flip="neg")[0] for mk, _ in mods) for ck, _ in corp]  # flip via get, so the subsample-adjusted cells are used
        CR.append((fl, ctl, med))
    dm = []
    for ck, _ in corp:
        v = [pd.read_csv(f).query("feature == @feat").d_doc.iloc[0] for f in sorted(_g.glob(str(SRC[ck] / "docmatch_*.csv")))]
        dm.append(statistics.median(v) if v else float("nan"))
    CR.append((fl, "document-matched$^\\ast$", dm))
for fl, ctl, med in CR:
    LG.append(f"L7 control x feature: {fl} vs {ctl}: " + ", ".join(f"{cn} {m:+.3f}" for (_, cn), m in zip(corp, med)))
body = []
for i, (fl, ctl, med) in enumerate(CR):
    body.append(f"{fl if i % 4 == 0 else ''} & {ctl} & " + " & ".join(shade(m, 0.0, 0.3) + sg(m) for m in med) + r" \\")
    if i % 4 == 3 and i < len(CR) - 1: body.append(r"\midrule")
(TABDIR / "tab_controls.tex").write_text(
    "% tab_controls.tex -- generated by make_real_tables.py\n"
    r"\begin{table}[htbp]" "\n" r"\centering" "\n" r"\small" "\n" r"\begin{tabular}{llrrrrr}" "\n" r"\toprule" "\n"
    "Feature & Control & " + " & ".join(cn for _, cn in corp) + r" \\" "\n" r"\midrule" "\n" + "\n".join(body) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n"
    r"\caption{Every control against every persistence feature: median over the embedding models of the real feature's AUROC minus the control's, at fixed query length without the query's own document (positive: the real arrangement separates the two chunkings more than the control). For $\Wone$ and the longest bar the random-bearing control is far too easy ($+0.3$ to $+0.5$, except the longest bar on Wikipedia, $+0.08$); for $\LTmax$ it is not. The corpus-bearing and shell-bearing controls leave small differences (at most about 0.05), except for the longest bar on the essays (about $+0.1$). Shading: vermillion where the real arrangement separates the chunkings better than the control, blue where worse; the tint grows with the difference and is full at $\pm0.3$, so a near-white cell means the control explains the feature. A difference of $+0.10$ means that, of every 100 pairs of neighbourhoods (one from each chunking), the real feature orders 10 more correctly than the control. The corpus-bearing row averages ten size-matched subsamples; the other rows use one. $^\ast$Two models (text-embedding-3-small, ModernBERT); not neutral for $\Wone$ (\cref{app:real}).}" "\n"
    r"\label{tab:controls}" "\n" r"\end{table}" "\n")
(TABDIR / "ledger_sec6.txt").write_text("\n".join(LG) + "\n"); print("\n".join(LG))
