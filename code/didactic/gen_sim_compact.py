# gen_sim_compact.py -- Purpose: write tab_sim_compact.tex, one row per simulation
# configuration (S1 vs S2, all 50 neighbours), replacing tab_sim_all.tex in the paper body (the long table stays for
# the supplement). Sources, read only:
#   reference_outputs/simulation_stats_all50.csv  -> D, N_top, n_parent, sigma, W1(H0) AUROC, random-bearing control AUROC
#   reference_outputs/simulation_stats.md, table R2b (grouped logistic, folds grouped by seed) -> A0 (cosines only) and B - A
# proj: only the unprojected run (sigma = 0.01) and the projection to D' = 64. Rerun: python gen_sim_compact.py
import re
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent; SIM = HERE.parents[1] / "data" / "reference_outputs"
S = pd.read_csv(SIM / "simulation_stats_all50.csv")
S = S[(S.comparison == "S1 vs S2") & (S.feature == "W1(H0)")]

md = (SIM / "simulation_stats.md").read_text()
sec = md[md.index("## R2b."):]
sec = sec[:sec.index("\n## ", 5)]
cv = {}
for line in sec.splitlines():
    c = [x.strip() for x in line.strip("|").split("|")]
    if len(c) == 9 and c[1] == "S1 vs S2":
        cv[c[0]] = dict(A0=float(c[3]), BmA=float(re.match(r"([+-−]?[\d.]+)", c[7].replace("−", "-")).group(1)))

ROWS = [("appD64", None, "dimension sweep"), ("appD128", None, "dimension sweep"), ("fig1", None, "main"), ("fig2", None, "main"),
        ("proj", 256, "projection: unprojected"), ("proj", 64, r"projection to $D'{=}64$"),
        ("superpos", None, "superposition"), ("superpos_s02", None, "superposition")]


def f3(x):
    return f"{x:.3f}"


body = []
for cfg, pdim, label in ROWS:
    r = S[(S.config == cfg) & ((S.proj_dim == pdim) if pdim else True)]
    assert len(r) == 1, (cfg, pdim, len(r))
    r = r.iloc[0]
    a0 = f3(cv[cfg]["A0"]) if cfg in cv else "---"
    bma = f"{abs(cv[cfg]['BmA']):.3f}" if cfg in cv else "---"   # all computed values are +-0.000
    sig = f"{r.sigma:g}"
    body.append(f"{label} & {r.D} & {r.N} & {r.n_parent} & {sig} & {f3(r.auroc)} & {f3(r.null_auroc)} & {a0} & {bma} \\\\")

stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M %Z")
tex = rf"""% tab_sim_compact.tex -- generated {stamp} by gen_sim_compact.py from data/reference_outputs/simulation_stats_all50.csv and simulation_stats.md (R2b).
\begin{{table}}[t]
\centering
\small
\begin{{tabular}}{{lrrrrcccc}}
\toprule
Configuration & $D$ & $N_{{\text{{top}}}}$ & $n_{{\text{{parent}}}}$ & $\sigma$ & $\Wone$ AUROC & control & cosines only & B $-$ A \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}
\caption{{Wherever the comparison was made, the cosines alone separate the two regimes about as well as the persistence features: a classifier on four cosine statistics reaches 0.984 to 1.000, and, with that classifier at or near 1.000, no gain from persistence can be measured. One row per simulation configuration, S1 against S2, all 50 neighbours.
$D$, $N_{{\text{{top}}}}$, $n_{{\text{{parent}}}}$, $\sigma$: dimension, number of topics, topics per parent passage, noise.
$\Wone$ AUROC: probability that an S1 query has a larger $\Wone$ than an S2 query.
Control: the same AUROC on the random-bearing control (\cref{{sec:geometry}}).
Cosines only: out-of-fold AUROC of a logistic classifier on the top-1, mean, standard deviation and rank slope of the 50 cosines.
B $-$ A: out-of-fold AUROC gained by adding $\Wone$, $\LTmax$ and $\Lmax$ to that classifier plus coherence and connectivity; folds grouped by seed.
---: not computed. Every configuration was run with 5 seeds; the supplementary material gives intervals, the other features and the remaining projections.}}
\label{{tab:simcompact}}
\end{{table}}
"""
(HERE / "tab_sim_compact.tex").write_text(tex)
print(tex)
