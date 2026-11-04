# supp_distributions.py -- written 2026-11-04 17:20 CET. Distribution figures for the supplement: what each feature
# looks like, query by query, in every scenario and stand-in of the paper (the paper shows only AUROC summaries).
# Reads only the archive's shipped data, so a reader can rerun it:  python supp_distributions.py <data dir> <out dir>
# (<data dir> = supplementary_material/data). Writes dist_*.pdf and dist_summary.txt (medians and AUROCs per panel).
# Colours as in the paper: polysemic(-like) vermillion, multi-factual(-like) blue, matched cells grey.
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.patches

DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
V, B, G1, G2, P, GR = "#D55E00", "#0072B2", "#7F7F7F", "#3F3F3F", "#CC79A7", "#009E73"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8, "axes.linewidth": 1.0,
                     "xtick.major.width": 1.0, "ytick.major.width": 1.0, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
LOG = []
W1, LB, LT = r"$W_1(H_0)$", r"longest bar $L_{\max}(H_0)$", r"$LT_{\max}(H_1)$"


def auc(a, b):
    """P(b > a) + 0.5 P(b = a), the paper's AUROC with b as the first-named group."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = pd.Series(np.r_[a, b]).rank().to_numpy()
    return (r[len(a):].sum() - len(b) * (len(b) + 1) / 2) / (len(a) * len(b))


def panel(ax, groups, xlabel, tag, rng=None):
    """groups: list of (label, colour, linestyle, values). Step histograms (density) on one shared range, median ticks."""
    allv = np.concatenate([np.asarray(g[3], float) for g in groups])
    allv = allv[np.isfinite(allv)]
    lo, hi = rng if rng is not None else np.percentile(allv, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1e-6
    bins = np.linspace(lo, hi, 36 if rng is None else 90)  # shared-range rows get finer bins
    top = 0
    for lab, col, ls, v in groups:
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        h, _ = np.histogram(v[(v >= lo) & (v <= hi)], bins=bins, density=True)  # the outer 1% is not drawn
        ax.stairs(h, bins, color=col, linestyle=ls, linewidth=1.3, label=lab)
        top = max(top, h.max())
    for lab, col, ls, v in groups:
        ax.plot([np.nanmedian(v)] * 2, [top * 1.04, top * 1.16], color=col, linewidth=1.6, solid_capstyle="butt")
    ax.set_ylim(0, top * 1.2); ax.set_yticks([]); ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3)); ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    med = ", ".join(f"{lab} {np.nanmedian(v):.3f}" for lab, _, _, v in groups)
    LOG.append(f"{tag} | {xlabel} | medians: {med} | AUROC(first > second) {auc(groups[1][3], groups[0][3]):.3f}")


def strips(ax, groups, title, tag, labels=True):
    """groups: list of (label, colour, values), drawn top to bottom. Each group: a thin 5-95% line, a box over the
    middle 50%, and a thick median tick. Easier to read than overlaid histograms when there are four groups."""
    n = len(groups)
    for k, (lab, col, v) in enumerate(groups):
        v = np.asarray(v, float); v = v[np.isfinite(v)]; y = n - 1 - k
        p5, p25, p50, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
        ax.plot([p5, p95], [y, y], color=col, linewidth=1.0, solid_capstyle="butt")
        ax.add_patch(matplotlib.patches.Rectangle((p25, y - 0.28), p75 - p25, 0.56, facecolor=col, alpha=0.30,
                                                  edgecolor=col, linewidth=1.0))
        ax.plot([p50, p50], [y - 0.36, y + 0.36], color=col, linewidth=2.2, solid_capstyle="butt")
    ax.set_ylim(-0.7, n - 0.3); ax.set_yticks(range(n))
    ax.set_yticklabels([g[0] for g in groups][::-1] if labels else [])
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3)); ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    if title: ax.set_title(title, fontsize=8.5)
    med = ", ".join(f"{g[0]} {np.nanmedian(g[2]):.3f}" for g in groups)
    LOG.append(f"{tag} | medians: {med} | AUROC(last-listed > first-listed) {auc(groups[0][2], groups[-1][2]):.3f}")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight"); plt.close(fig)


# 1. Simulation (configuration fig1, seed 0; all 50 neighbours): S1 multi-factual vs S2 polysemic
s = pd.read_csv(f"{DATA}/simulation/features/fig1_seed0.csv.xz"); s = s[s.depsilon == 0.4]
c = pd.read_csv(f"{DATA}/simulation/cosine/cosine_fig1_seed0.csv.xz")
m = s.merge(c, on=["scenario", "query_idx", "parent_group", "corpus_nfeat", "query_nfeat"])
mf, po = m[m.scenario == 1], m[m.scenario == 2]
with plt.rc_context({"font.size": 9.5}):
    fig, ax = plt.subplots(1, 5, figsize=(7.0, 2.1), constrained_layout=True)
    for a, (col, xl) in zip(ax, [("cos_mean", "mean score"), ("cos_std", "SD of the scores"), ("all_w1_h0", W1),
                                 ("all_h0_maxbar", "longest bar"), ("all_ltmax_h1", LT)]):
        panel(a, [("multi-factual (S1)", B, "-", mf[col]), ("polysemic (S2)", V, "-", po[col])], "", f"sim {col}")
        a.set_title(xl, fontsize=10)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=10)
    save(fig, "dist_sim.pdf")

# 2. Real text, the 2x2 cells (text-embedding-3-small, five corpora, primary variant)
CORP = [("nobel", "Nobel"), ("turing", "Turing"), ("fields", "Fields"), ("gutenberg", "Essays"), ("wikipedia", "Wikipedia")]
CELLS = [("Q250|C750 polysemic-like", "Q250|C750", V, "-"), ("Q750|C250 multi-factual-like", "Q750|C250", B, "-"),
         ("Q250|C250 matched", "Q250|C250", G1, "--"), ("Q750|C750 matched", "Q750|C750", G2, ":")]
R = {k: pd.read_csv(f"{DATA}/realtext/{k}/features_openai-3-small.csv.xz") for k, _ in CORP}
R = {k: d[d.variant == "primary"] for k, d in R.items()}
ORDER = [("Q250|C750 (1/3)", "Q250|C750", V), ("Q250|C250 (1)", "Q250|C250", G1), ("Q750|C750 (1)", "Q750|C750", G2),
         ("Q750|C250 (3)", "Q750|C250", B)]   # by query-to-chunk length ratio
fig, ax = plt.subplots(5, 5, figsize=(7.0, 6.6), constrained_layout=True)
for i, (k, name) in enumerate(CORP):
    d = R[k]
    for j, (col, xl) in enumerate([("mean50", "mean score"), ("std50", "SD of the scores"), ("w1", W1),
                                   ("longbar", "longest bar"), ("ltmax", LT)]):
        strips(ax[i, j], [(lab, cl, d[d.cell == cell][col]) for lab, cell, cl in ORDER], xl if i == 0 else "",
               f"2x2 {name} {col}", labels=(j == 0))
    ax[i, 0].set_ylabel(name, fontsize=9.5, labelpad=6)
save(fig, "dist_2x2.pdf")

# 3. W1 against its controls in the mismatched cells: real, corpus-bearing, random-bearing (same scores)
fig, ax = plt.subplots(5, 4, figsize=(7.0, 6.2), constrained_layout=True, sharex="row")
for i, (k, name) in enumerate(CORP):
    d = R[k]; pl, mfl = d[d.cell == "Q250|C750"], d[d.cell == "Q750|C250"]
    ser = [x.dropna().to_numpy() for x in (pl.w1, mfl.w1, pl.w1_enull, mfl.w1_enull, pl.w1_lnull, mfl.w1_lnull, pl.w1_null, mfl.w1_null)]
    rng_row = (min(np.percentile(x, 0.5) for x in ser), max(np.percentile(x, 99.5) for x in ser))  # every series fits
    for j, (col, ttl) in enumerate([("w1", "real bearings"), ("w1_enull", "corpus-bearing\ncontrol"),
                                    ("w1_lnull", "shell-bearing control\n(ranks 51-500)"), ("w1_null", "random-bearing\ncontrol")]):
        panel(ax[i, j], [("multi-factual-like Q750|C250", B, "-", mfl[col]), ("polysemic-like Q250|C750", V, "-", pl[col])],
              W1 if i == 4 else "", f"controls {name} {ttl.replace(chr(10), ' ')}", rng=rng_row)
        if i == 0:
            ax[i, j].set_title(ttl, fontsize=9)
    ax[i, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
h = [plt.Line2D([], [], color=cl, linewidth=1.3) for cl in (V, B)]
fig.legend(h, ["Q250|C750 polysemic-like", "Q750|C250 multi-factual-like"], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "dist_controls.pdf")

# 4. The equal-length page test (arm all, grouping seed 0): 249-token texts of one page (1) or three (3)
L = pd.read_csv(f"{DATA}/lenctl/feats.csv.xz"); L = L[(L.arm == "all") & (L.g == 0)]
PC = [("Q1|C3 polysemic", "Q1|C3", V, "-"), ("Q3|C1 multi-factual", "Q3|C1", B, "-"),
      ("Q1|C1 matched", "Q1|C1", G1, "--"), ("Q3|C3 matched", "Q3|C3", G2, ":")]
PORDER = [("Q1|C3 (1/3)", "Q1|C3", V), ("Q1|C1 (1)", "Q1|C1", G1), ("Q3|C3 (1)", "Q3|C3", G2), ("Q3|C1 (3)", "Q3|C1", B)]
fig, ax = plt.subplots(2, 5, figsize=(7.0, 2.9), constrained_layout=True)
for i, mod in enumerate(["bge-base", "bge-m3"]):
    d = L[L.model == mod]
    for j, (col, xl) in enumerate([("mean_cos", "mean score"), ("coherence", "coherence"), ("w1", W1),
                                   ("longbar", "longest bar"), ("w1_corp", W1 + "\ncorpus bearings")]):
        strips(ax[i, j], [(lab, cl, d[d.cell == cell][col]) for lab, cell, cl in PORDER], xl if i == 0 else "",
               f"page {mod} {col}", labels=(j == 0))
    ax[i, 0].set_ylabel(mod, fontsize=9.5, labelpad=6)
save(fig, "dist_pagetest.pdf")

# 5. Joined readings (AmbigNQ train, all-MiniLM-L6-v2): question, all readings joined, first reading alone
F = pd.read_csv(f"{DATA}/ambig_train/feats_flip.csv.xz")
JG = [("joined readings (multi-factual stand-in)", 1, B, "-"), ("original question", 0, V, "-"), ("first reading alone", 2, G1, "--")]
fig, ax = plt.subplots(1, 5, figsize=(7.0, 1.45), constrained_layout=True)
for a, (col, xl) in zip(ax, [("mean_cos", "mean score"), ("w1", W1), ("longbar", LB), ("w1_corp", W1 + ", corpus bearings"),
                             ("nwords", "words in the query")]):
    panel(a, [(lab, cl, ls, F[F.label == g][col]) for lab, g, cl, ls in JG], xl, "joined")
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
save(fig, "dist_joined.pdf")

# 6. Questions people label (AmbigNQ train, all-MiniLM-L6-v2): ambiguous vs clear
T = pd.read_csv(f"{DATA}/ambig_train/feats_train.csv.xz")
fig, ax = plt.subplots(1, 5, figsize=(7.0, 1.45), constrained_layout=True)
for a, (col, xl) in zip(ax, [("mean_cos", "mean score"), ("coherence", "coherence"), ("w1", W1), ("longbar", LB), ("ltmax", LT)]):
    panel(a, [("clear", GR, "-", T[T.label == 0][col]), ("ambiguous", P, "-", T[T.label == 1][col])], xl, "ambignq")
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "dist_ambignq.pdf")

# 7. Is the paper's one-query barcode (Figure 8: Nobel, text-embedding-3-small, Q250|C750) typical of its cell?
d = R["nobel"]; d = d[d.cell == "Q250|C750"]; mr, mc, mn = 2 * d.w1, 2 * d.w1_enull, 2 * d.w1_null
LOG.append(f"fig8 cell (Nobel, text-embedding-3-small, Q250|C750; mean bar = 2 W1) | queries {len(d)} | median mean bar: "
           f"real {mr.median():.3f}, corpus-bearing {mc.median():.3f}, random-bearing {mn.median():.3f} | "
           f"share with real < corpus-bearing {(mr < mc).mean():.2f} | median (corpus-bearing - real) {(mc - mr).median():.3f}")

open(os.path.join(OUT, "dist_summary.txt"), "w").write(
    "# supp_distributions.py: per panel, group medians and AUROC = P(second-listed group > first-listed group)\n"
    + "\n".join(LOG) + "\n")
print(len(LOG), "panels")
