"""Child-size band check behind the paper's paragraph "Within one child size, the random-bearing control matches W1":
AUROC [95% CI] of S1 against S2 within bands of child size for W1(H0), LT_max(H1) and the longest H0 bar, each beside
its random-bearing control; main configuration (fig1, sigma = 0.1), all 50 neighbours, cluster bootstrap over parent
groups (redteam.bands). Inputs: $SIM_DATA/features (default ../../data/simulation/features).
Output: $SIM_RESULTS/band_controls.md (default ./results; seeds 0-4: data/reference_outputs/simulation_band_controls.md).
Usage: python band_controls.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import redteam

import os
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SIM_DATA", str(HERE.parents[1] / "data" / "simulation")))
FEAT, COSD, DISD = DATA / "features", DATA / "cosine", DATA / "distractor"   # run_sim / cosine_baselines / distractor
OUT = Path(os.environ.get("SIM_RESULTS", str(HERE / "results"))); OUT.mkdir(parents=True, exist_ok=True)
m = pd.concat([pd.read_csv(f) for f in sorted(FEAT.glob("fig1_seed*.csv*"))], ignore_index=True)
m = m[np.isclose(m.depsilon, 0.4)].copy()
m["cluster"] = m.sim_seed.astype(str) + "_" + m.parent_group.astype(str)
m["n_child"] = np.where(m.scenario == 2, m.query_nfeat, m.corpus_nfeat)
m = m[m.scenario.isin([1, 2])]
m["all_ltmax_h1"] = m.all_ltmax_h1.fillna(0.0); m["null_all_ltmax_h1"] = m.null_all_ltmax_h1.fillna(0.0)
L = redteam.bands(m, [("W1", "all_w1_h0"), ("control W1", "null_all_w1_h0"), ("LT_max", "all_ltmax_h1"),
                      ("control LT_max", "null_all_ltmax_h1"), ("max H0 bar", "all_h0_maxbar"), ("control max bar", "null_all_h0_maxbar")],
                  "## AUROC [95% CI] by child size, S1 vs S2, main configuration, all 50 neighbours (control = random-bearing)")
(OUT / "band_controls.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
