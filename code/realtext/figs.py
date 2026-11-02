# figs.py -- figures for the real-text experiment, per corpus.
#  (i)   fig_kde       : per model, Gaussian KDEs (scipy, Scott bandwidth) of W1(H0) and LT_max(H1), K=50, 4 cells
#  (ii)  fig_eps       : median (solid) and IQR (dashed) vs eps for the main contrast, one model
#                        (ModernBERT-base-embed if present, else text-embedding-3-large), from the prefix table
#  (iii) fig_interaction: medians with document-bootstrap 95% CI per cell, per model (x = query granularity,
#                        line = corpus granularity), W1 / LT_max / longest H0 bar
# Each written as PDF + PNG with a .txt sidecar. Usage: [CORPUS=wiki] [OUTROOT=out_clean] python figs.py
# (fig_eps needs prefix_<model>.csv.gz, written by features.py; it is skipped when absent.)
import os, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT = Path(os.environ.get("RT_WORK", str(Path(__file__).resolve().parent / "work")))  # see README
CORPUS = os.environ.get("CORPUS", "nobel")
OUTROOT = os.environ.get("OUTROOT", "out")  # the paper: OUTROOT=out_clean (with CLEAN=1 in features.py)
OUT = ROOT / OUTROOT if CORPUS == "nobel" else ROOT / OUTROOT / CORPUS
FIG = OUT / "figs"; FIG.mkdir(parents=True, exist_ok=True)
UNIT = {"nobel": "lecture", "wiki": "page", "gutenberg": "book", "turing": "lecture", "fields": "lecture"}.get(CORPUS, "document")
MODELS = ["openai-3-small", "openai-3-large", "vertex-005", "modernbert-embed", "bge-base"]


def feat_path(out, m):
    """features_<m>.csv.gz as written by features.py, else the shipped features_<m>.csv.xz."""
    p = out / f"features_{m}.csv.gz"
    return p if p.exists() else out / f"features_{m}.csv.xz"
LABEL = {"openai-3-small": "text-embedding-3-small", "openai-3-large": "text-embedding-3-large",
         "vertex-005": "text-embedding-005", "modernbert-embed": "ModernBERT-base-embed", "bge-base": "bge-base-en-v1.5"}
# categorical colours for the four cells; line style is the secondary encoding
STY = {"Q250|C750": ("#2a78d6", "-"), "Q750|C250": ("#eb6834", "-"), "Q250|C250": ("#1baf7a", "--"),
       "Q750|C750": ("#c98500", "--")}
CELLDESC = {"Q250|C750": "Q250|C750 (query finer; polysemic-like)", "Q750|C250": "Q750|C250 (query coarser; multi-factual-like)",
            "Q250|C250": "Q250|C250 (matched)", "Q750|C750": "Q750|C750 (matched)"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e", "lines.linewidth": 1.4})
STAMP = time.strftime("%Y-%m-%d %H:%M %Z")
present = [m for m in MODELS if feat_path(OUT, m).exists()]
F = {m: pd.read_csv(feat_path(OUT, m)) for m in present}


def save(fig, name, side):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    (FIG / f"{name}.txt").write_text(f"# {name} -- generated {STAMP} by figs.py (corpus={CORPUS}).\n{side}\n")
    plt.close(fig)


# (i) KDEs
fig, ax = plt.subplots(len(present), 2, figsize=(6.6, 1.55 * len(present) + 0.6), squeeze=False)
side = ["Purpose: KDEs of W1(H0) and LT_max(H1) at K=50 (primary variant, overlap exclusion), four cells.",
        "KDE: scipy.stats.gaussian_kde, Scott's rule bandwidth; no randomness.", "n queries per cell:"]
for i, m in enumerate(present):
    d = F[m][F[m].variant == "primary"]
    for j, (f, lab) in enumerate((("w1", "$W_1(H_0)$ (distance on unit sphere)"), ("ltmax", "$LT_{max}(H_1)$ (half-lifetime)"))):
        a = ax[i, j]
        lo, hi = d[f].quantile(0.001), d[f].quantile(0.999); xs = np.linspace(lo, hi, 300)
        for c, (col, ls) in STY.items():
            v = d[d.cell == c][f].values
            if np.ptp(v) > 0:
                a.plot(xs, gaussian_kde(v)(xs), color=col, ls=ls, label=CELLDESC[c])
        a.set_yticks([]); a.set_xlabel(lab if i == len(present) - 1 else "")
        if j == 0:
            a.set_ylabel(LABEL[m], rotation=0, ha="right", va="center")
    side.append(f"  {m}: " + ", ".join(f"{c}={int((d.cell == c).sum())}" for c in STY))
ax[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.05), ncol=2, frameon=False, fontsize=7)
save(fig, "fig_kde", "\n".join(side))

# (ii) eps curve (needs the prefix table written by features.py)
em = "modernbert-embed" if "modernbert-embed" in present else "openai-3-large"
if (OUT / f"prefix_{em}.csv.gz").exists():
    P = pd.read_csv(OUT / f"prefix_{em}.csv.gz"); P = P[P.variant == "primary"]
    grid = np.round(np.arange(0.0, 1.0001, 0.01), 3)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
    side = [f"Purpose: median (solid) and interquartile range (dashed) of W1(H0), LT_max(H1) vs eps, main contrast, model {em}.",
            "eps rule (paper): keep top-50 neighbours with ||c_i-q||/min||c-q|| - 1 <= eps; queries with < 3 kept are excluded at that eps.",
            "Grid eps = 0.00..1.00 step 0.01. No randomness."]
    for c in ("Q250|C750", "Q750|C250"):
        p = P[P.cell == c].sort_values(["query_id", "n"])
        qids = p.query_id.unique(); nq = len(qids)
        R = p.ratio.values.reshape(nq, -1); W = p.w1.values.reshape(nq, -1); L = p.ltmax.values.reshape(nq, -1)
        for j, M in enumerate((W, L)):
            med, q1, q3, nval = [], [], [], []
            for e in grid:
                k = (R <= e + 1e-12).sum(1)  # number of prefix lengths n>=3 admitted; index k-1 -> n = k+2
                ok = k > 0
                v = M[np.flatnonzero(ok), k[ok] - 1]
                nval.append(ok.sum())
                med.append(np.median(v) if len(v) else np.nan); q1.append(np.percentile(v, 25) if len(v) else np.nan)
                q3.append(np.percentile(v, 75) if len(v) else np.nan)
            col = STY[c][0]
            ax[j].plot(grid, med, color=col, label=CELLDESC[c]); ax[j].plot(grid, q1, color=col, ls="--", lw=0.9)
            ax[j].plot(grid, q3, color=col, ls="--", lw=0.9)
        side.append(f"  {c}: n queries={nq}; queries with >=3 neighbours at eps=0.1/0.2/0.4: "
                    f"{[(int(((R <= e).sum(1) > 0).sum())) for e in (0.1, 0.2, 0.4)]}")
    for j, lab in enumerate(("$W_1(H_0)$", "$LT_{max}(H_1)$")):
        ax[j].axvline(0.4, color="#8a8984", lw=0.8, ls=":"); ax[j].set_xlabel(r"neighbourhood scale $\varepsilon$"); ax[j].set_ylabel(lab)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False, fontsize=7)
    ax[0].set_title(LABEL[em], fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "fig_eps", "\n".join(side))

# (iii) interaction plot
CM = pd.read_csv(OUT / "cell_medians.csv")
feats = [("w1", "$W_1(H_0)$"), ("ltmax", "$LT_{max}(H_1)$"), ("longbar", "longest $H_0$ bar")]
fig, ax = plt.subplots(len(feats), len(present), figsize=(1.8 * len(present) + 0.8, 4.8), squeeze=False)
fig.subplots_adjust(wspace=0.55, hspace=0.35)
for j, m in enumerate(present):
    for i, (f, lab) in enumerate(feats):
        a = ax[i, j]
        for cg, col, mk in ((750, "#2a78d6", "o"), (250, "#eb6834", "s")):
            r = [CM[(CM.model == m) & (CM.feature == f) & (CM.contrast == f"Q{qg}|C{cg}")].iloc[0] for qg in (250, 750)]
            y = np.array([x.est for x in r]); e = np.array([[x.est - x.lo for x in r], [x.hi - x.est for x in r]])
            a.errorbar([0, 1], y, yerr=e, color=col, marker=mk, ms=4, capsize=2, lw=1.2, label=f"corpus C{cg}")
        a.set_xticks([0, 1]); a.set_xticklabels(["Q250", "Q750"]); a.set_xlim(-0.35, 1.35)
        if i == 0:
            a.set_title(LABEL[m], fontsize=7)
        if j == 0:
            a.set_ylabel(lab)
ax[0, 0].legend(frameon=False, fontsize=7)
save(fig, "fig_interaction", "Purpose: per-cell medians (K=50, primary) with 95% CIs from a cluster bootstrap over "
     f"{UNIT}s (B=2000, numpy default_rng(0)); x = query granularity, line = corpus granularity. Parallel lines = no "
     "interaction. n per cell as in fig_kde.txt.")
print("figs written for", present, "eps model", em)
