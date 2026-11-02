# real_data.py -- export the small tables that fig_loops, fig_predictions, fig_effects and fig_nulls read, from
# existing outputs only (no recomputation):
#   data/reference_outputs/loops_check.txt                 -> data/loops_T*.dat (fraction LT_max>0.05, Wilson 95%, per T, p)
#   reference_outputs/realtext_stats_summary.md Table S0   -> data/ratio_forest.dat (matched-cell AUROC, both arms)
#   reference_outputs/realtext_*_stats_long.csv            -> data/effects.dat (query, corpus, matched-cell AUROCs)
#   data/realtext/{gutenberg,wikipedia}/features_openai-3-small -> data/nulls_{gutenberg,wiki}_{w1,longbar}.dat
#                                                             (Gaussian KDEs; log in data/nulls_summary.txt)
# All real-text inputs are the clean-text outputs (extraction-artefact filter applied).
# fig_nulls: Q250 queries, leave-document-out arm: C250 = variant lodo_dens (density-matched), C750 = lodo.
# Real feature, corpus-bearing null (*_enull, one draw per query) and isotropic matched-cosine null (*_null).
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, mannwhitneyu

HERE = Path(__file__).parent; OUT = HERE / "data"; OUT.mkdir(exist_ok=True)
SUPP = HERE.parents[1]
REF = SUPP / "data" / "reference_outputs"   # clean-text reference outputs
log = [f"# real_data.py run {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M %Z}"]

# ---- loops
rows = []
for line in (REF / "loops_check.txt").read_text().splitlines():
    m = re.match(r"\s*(\d)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+\[([\d.]+),\s*([\d.]+)\]", line)
    if m:
        T, d, k, f, lo, hi = m.groups(); rows.append((int(T), int(d), int(k), float(f), float(lo), float(hi)))
L = pd.DataFrame(rows, columns=["T", "m", "k", "frac", "lo", "hi"])
assert len(L) == 12
L["x"] = L["m"] * L["T"].map({1: 0.90, 2: 1.0, 3: 1.11})       # small horizontal dodge on the log axis
L["eplus"] = L["hi"] - L["frac"]; L["eminus"] = L["frac"] - L["lo"]
for T in (1, 2, 3):
    L[L["T"] == T][["x", "m", "frac", "eplus", "eminus"]].to_csv(OUT / f"loops_T{T}.dat", sep=" ", index=False)
log.append("loops:\n" + L.to_string(index=False))

# ---- ratio forest (replaces the observed-medians panel): reference_outputs/realtext_stats_summary.md Table S0,
# columns 'W1 diag' and 'longbar diag' = AUROC(Q750|C750 above Q250|C250), already oriented; 0.5 = ratio prediction.
# Corpus 'gutenberg' is labelled "Essays" (assumption: the essays corpus); arms primary (all chunks) and LODO.
# Rerun after S0 changes: python real_data.py (no .tex edits needed).
md = (REF / "realtext_stats_summary.md").read_text()
sec = md[md.index("## Table S0"):md.index("## Table S1")]
hdr = None; recs = []
num = r"([-\d.]+) \[([-\d.]+), ([-\d.]+)\]"
for line in sec.splitlines():
    if not line.startswith("|") or line.startswith("|---"):
        continue
    cells = [c.strip() for c in line.strip("|").split("|")]
    if cells[0] == "corpus":
        hdr = cells; continue
    r = dict(zip(hdr, cells))
    rec = dict(corpus=r["corpus"], model=r["model"], arm=r["arm"])
    for col, tag in (("W1 diag", "w1"), ("longbar diag", "lb")):
        m = re.fullmatch(num, r[col]); assert m, (col, r[col])
        rec[tag], rec[tag + "_lo"], rec[tag + "_hi"] = map(float, m.groups())
    recs.append(rec)
F = pd.DataFrame(recs)
# corpora are read from S0 (whatever it contains), in this order; unknown ones go last under their own name.
# Also writes data/ratio_forest_meta.tex (y ticks, model labels, corpus groups, axis height) for fig_predictions.tex.
ORDER = ["nobel", "turing", "fields", "gutenberg", "wiki"]
LABEL = {"nobel": "Nobel", "gutenberg": "Essays", "turing": "Turing", "fields": "Fields", "wiki": "Wikipedia"}
present = list(dict.fromkeys(F.corpus))
CORP = [c for c in ORDER if c in present] + [c for c in present if c not in ORDER]
# every model present in S0, in this order (was the fixed five)
MLAB = {"openai-3-small": "3-small", "openai-3-large": "3-large", "vertex-005": "005", "gemini-001": "Gemini",
        "openai-ada-002": "ada-002", "modernbert-embed": "ModernBERT", "bge-base": "bge-base", "bge-m3": "bge-m3",
        "qwen3-0.6b": "Qwen3-0.6B"}
MOD = [m for m in MLAB if m in set(F.model)]
for c in CORP:
    assert len(F[F.corpus == c]) == 2 * len(MOD), (c, len(F[F.corpus == c]))
GAP = len(MOD) + 1                # rows per corpus group: the models + 1 blank
W = []
for g, c in enumerate(CORP):
    for k, mname in enumerate(MOD):
        row = {"y": g * GAP + k}
        for arm, a_ in (("primary", "p"), ("LODO", "l")):
            x = F[(F.corpus == c) & (F.model == mname) & (F.arm == arm)].iloc[0]
            for t in ("w1", "lb"):
                row[f"{t}_{a_}"] = x[t]; row[f"{t}_{a_}_ep"] = x[t + "_hi"] - x[t]; row[f"{t}_{a_}_em"] = x[t] - x[t + "_lo"]
        W.append(row)
W = pd.DataFrame(W)
W["yp"] = W["y"] - 0.17; W["yl"] = W["y"] + 0.17
ymax = (len(CORP) - 1) * GAP + len(MOD) - 1 + 0.9
unit = min(0.29 if len(CORP) <= 4 else 0.235, 10.5 / (ymax + 0.9))  # cm per row; figure <= ~10.5 cm tall
meta = [f"% real_data.py output, generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M %Z}; corpora: {', '.join(CORP)}",
        f"\\def\\rfYmax{{{ymax:.1f}}}", f"\\def\\rfHeight{{{unit * (ymax + 0.9):.2f}cm}}",
        "\\def\\rfYticks{" + ",".join(str(int(v)) for v in W["y"]) + "}",
        "\\def\\rfYlabels{" + ",".join(MLAB[m] for _ in CORP for m in MOD) + "}",
        "\\def\\rfGroups{" + ", ".join(f"{g * GAP}/{LABEL.get(c, c)}" for g, c in enumerate(CORP)) + "}"]
(OUT / "ratio_forest_meta.tex").write_text("%\n".join(meta) + "%\n")
W.to_csv(OUT / "ratio_forest.dat", sep=" ", index=False, float_format="%.4f")
log.append("ratio forest (diag AUROC, est [lo, hi]):\n" + F.to_string(index=False))
near = lambda t, arm: F[(F.arm == arm) & (F[t + "_lo"] <= 0.5) & (F[t + "_hi"] >= 0.5)][["corpus", "model"]].values.tolist()
log.append(f"intervals containing 0.5 -- W1 primary: {near('w1','primary')}; W1 LODO: {near('w1','LODO')}; "
           f"longbar primary: {near('lb','primary')}; longbar LODO: {near('lb','LODO')}")

# ---- effects (fig_effects): what moves the features, primary arm (all chunks), from every
# reference_outputs/realtext_<corpus>_stats_long.csv present.
# stats.py records P(first > second); flipped here so > 0.5 means the named direction:
#   query  = auc_query_eff @C750:          P(Q750|C750 > Q250|C750)       (query-length effect at the coarser corpus)
#   corpus = auc_corpus_eff_dens @Q250:    P(Q250|C250dens > Q250|C750)   (finer, density-matched corpus above C750)
#   match  = auc_diagonal:                 P(Q750|C750 > Q250|C250)       (matched cells; 0.5 = ratio prediction)
import glob
SL = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(REF / "realtext_*_stats_long.csv")))])
EFF = (("q", "auc_query_eff", "@C750"), ("c", "auc_corpus_eff_dens", "@Q250"), ("m", "auc_diagonal", "Q250|C250 vs Q750|C750"))
presentE = list(dict.fromkeys(SL.corpus))
CORPE = [c for c in ORDER if c in presentE] + [c for c in presentE if c not in ORDER]
E = []
for g, c in enumerate(CORPE):
    for k, mname in enumerate(MOD):
        row = {"y": g * GAP + k}
        for tag, kind, con in EFF:
            for feat, ft in (("w1", "w1"), ("longbar", "lb")):
                x = SL[(SL.corpus == c) & (SL.model == mname) & (SL.kind == kind) & (SL.contrast == con) & (SL.feature == feat)]
                assert len(x) == 1, (c, mname, kind, feat, len(x))
                x = x.iloc[0]; est, lo, hi = 1 - x.est, 1 - x.hi, 1 - x.lo
                row[f"{ft}_{tag}"], row[f"{ft}_{tag}_ep"], row[f"{ft}_{tag}_em"] = est, hi - est, est - lo
                row[f"{ft}_{tag}_has05"] = int(lo <= 0.5 <= hi)
        E.append(row)
E = pd.DataFrame(E)
E["yq"], E["yc"], E["ym"] = E["y"] - 0.3, E["y"], E["y"] + 0.3
E.to_csv(OUT / "effects.dat", sep=" ", index=False, float_format="%.4f")
ymaxE = (len(CORPE) - 1) * GAP + len(MOD) - 1 + 0.9
unitE = 0.3  # 9 models: 0.3 cm per row so labels do not overlap (appendix figure, ~15 cm tall)
metaE = [f"% real_data.py output, generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M %Z}; corpora: {', '.join(CORPE)}",
         f"\\def\\efYmax{{{ymaxE:.1f}}}", f"\\def\\efHeight{{{unitE * (ymaxE + 0.9):.2f}cm}}",
         "\\def\\efYticks{" + ",".join(str(int(v)) for v in E["y"]) + "}",
         "\\def\\efYlabels{" + ",".join(MLAB[m] for _ in CORPE for m in MOD) + "}",
         "\\def\\efGroups{" + ", ".join(f"{g * GAP}/{LABEL.get(c, c)}" for g, c in enumerate(CORPE)) + "}",
         "\\def\\efN{" + str(len(MOD)) + "}"]  # models per corpus, for Figs 11 and 12 (shared layout)
(OUT / "effects_meta.tex").write_text("%\n".join(metaE) + "%\n")
summ = []
for ft in ("w1", "lb"):
    for tag in ("q", "c", "m"):
        v = E[f"{ft}_{tag}"]; h = E[f"{ft}_{tag}_has05"]
        lo_side = int(((v + E[f"{ft}_{tag}_ep"]) < 0.5).sum())
        summ.append(f"  {ft} {tag}: range {v.min():.3f}-{v.max():.3f}; CI contains 0.5 in {h.sum()}/{len(v)}; entirely below 0.5 in {lo_side}")
log.append(f"effects ({', '.join(CORPE)}; primary arm):\n" + "\n".join(summ))

# ---- nulls
def auc(a, b):  # P(a > b)
    return mannwhitneyu(a, b).statistic / (len(a) * len(b))


# essays (gutenberg) replace Nobel; longbar grid narrowed to the clean data's range (0.73-1.29)
GRID = {"w1": np.linspace(0.20, 0.70, 301), "longbar": np.linspace(0.70, 1.35, 301)}
nlog0 = len(log)
for corpus, f in (("gutenberg", "data/realtext/gutenberg/features_openai-3-small.csv.xz"),
                  ("wiki", "data/realtext/wikipedia/features_openai-3-small.csv.xz")):
    d = pd.read_csv(SUPP / f)
    a = d[(d.variant == "lodo_dens") & (d.cell == "Q250|C250")]
    b = d[(d.variant == "lodo") & (d.cell == "Q250|C750")]
    log.append(f"{corpus}: n C250 (lodo_dens) = {len(a)}, n C750 (lodo) = {len(b)}")
    for feat in ("w1", "longbar"):
        g = GRID[feat]; cols = {"x": g}
        for src, suf in (("real", ""), ("enull", "_enull"), ("iso", "_null")):
            for tag, df in (("c250", a), ("c750", b)):
                v = df[feat + suf].to_numpy()
                assert g[0] < v.min() and v.max() < g[-1], (corpus, feat, suf, v.min(), v.max())
                cols[f"{src}_{tag}"] = gaussian_kde(v)(g)
            log.append(f"  {feat:8s} {src:5s}: median C250 {a[feat+suf].median():.3f}  C750 {b[feat+suf].median():.3f}"
                       f"  AUROC(C250>C750) {auc(a[feat+suf], b[feat+suf]):.3f}")
        pd.DataFrame(cols).to_csv(OUT / f"nulls_{corpus}_{feat}.dat", sep=" ", index=False, float_format="%.5g")

(OUT / "nulls_summary.txt").write_text("\n".join([log[0], "clean text, text-embedding-3-small"] + log[nlog0:]) + "\n")
s = "\n".join(log) + "\n"; (OUT / "real_summary.txt").write_text(s); print(s)
