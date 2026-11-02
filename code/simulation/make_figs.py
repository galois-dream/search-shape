"""Data tables (.dat) for the paper's four simulation figures, drawn in pgfplots by the didactic figure code
(code/didactic/fig_sim.tex). Densities are Gaussian KDEs (Scott) of the all-50-neighbour features; a density of y means
y% of queries per 0.01 of the feature. Columns named null_* in the per-query tables are the paper's random-bearing
control. eps table: mean over parent-group cluster means of the queries with a value, mean neighbours kept, and the
share of queries with a value (>= 3 neighbours within the cut-off eps).
Inputs: $SIM_DATA/features (default ../../data/simulation/features). Output: $SIM_RESULTS/figs (default ./results/figs).
Usage: python make_figs.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import os
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SIM_DATA", str(HERE.parents[1] / "data" / "simulation")))
FEAT, COSD, DISD = DATA / "features", DATA / "cosine", DATA / "distractor"   # run_sim / cosine_baselines / distractor
OUT = Path(os.environ.get("SIM_RESULTS", str(HERE / "results"))); OUT.mkdir(parents=True, exist_ok=True)
FIGS = OUT / "figs"; FIGS.mkdir(parents=True, exist_ok=True)
EPS0 = 0.4


def load():
    df = pd.concat([pd.read_csv(f) for f in sorted(FEAT.glob("*_seed*.csv*"))
                    if not f.name.startswith(("cosine", "distractor"))], ignore_index=True)
    df["cluster"] = df.sim_seed.astype(str) + "_" + df.parent_group.astype(str)
    return df


def kde_table(name, cols):
    """cols: {column name: values}; one shared x grid covering every series."""
    v = {k: s.dropna().to_numpy() for k, s in cols.items()}
    lo, hi = min(x.min() for x in v.values()), max(x.max() for x in v.values())
    xs = np.linspace(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo), 241)
    T = pd.DataFrame({"x": xs, **{k: np.where((y := gaussian_kde(x)(xs)) < 1e-3, 0, y) for k, x in v.items()}})
    T.to_csv(FIGS / f"{name}.dat", sep=" ", index=False, float_format="%.5g")


def all50(d, sc, f, null=False):
    s = d[(d.scenario == sc) & np.isclose(d.depsilon, EPS0)]
    return s[f"null_all_{f}" if null else f"all_{f}"]


def main(df):
    F = {"w1": "w1_h0", "lt": "ltmax_h1"}
    m = df[df.config == "fig1"]                                   # main: D=256, sigma=0.1
    for k, f in F.items():
        kde_table(f"sim_main_{k}", {"s1": all50(m, 1, f), "s2": all50(m, 2, f),
                                    "n1": all50(m, 1, f, True), "n2": all50(m, 2, f, True)})
    rows = []
    for e, g in m[m.depsilon > 0.01].groupby("depsilon"):
        r = {"eps": e}
        for sc in (1, 2):
            s = g[g.scenario == sc]
            for k, f in F.items():
                r[f"s{sc}_{k}"] = s.groupby("cluster")[f].mean().mean()
            r[f"s{sc}_n"], r[f"s{sc}_ok"] = s.nneighbours.mean(), s.w1_h0.notna().mean()
        rows.append(r)
    pd.DataFrame(rows).to_csv(FIGS / "sim_main_eps.dat", sep=" ", index=False, float_format="%.5g")
    d = df[df.config.isin(["appD64", "appD128", "fig1"])]           # dimension sweep, sigma=0.1
    p = df[df.config == "proj"]                                   # random projection, sigma=0.01
    x = df[df.config == "fig2"]                                   # mixed scenario, sigma=0.2
    for k, f in F.items():
        kde_table(f"sim_dims_{k}", {f"s{sc}_{D}": all50(d[d.D == D], sc, f) for D in (64, 128, 256) for sc in (1, 2)})
        kde_table(f"sim_proj_{k}", {f"s{sc}_{P}": all50(p[p.proj_dim == P], sc, f) for P in (256, 64) for sc in (1, 2)})
        kde_table(f"sim_mixed_{k}", {f"s{sc}": all50(x, sc, f) for sc in (1, 4, 2)})


if __name__ == "__main__":
    main(load())
    print(sorted(p.name for p in FIGS.glob("sim_*.dat")), file=sys.stderr)
