# dose.py -- artefact dose-response: do the surviving longest-bar effects track residual extraction mess? For each
# cleaning level -- all text (OUTROOT=out, no filter), out_t07 (CLEAN_T=0.7), out_clean (0.5, the paper), out_t04
# (0.4) -- and the size control out_t04rand (0.5 plus CLEAN_RAND = as many random 750-token chunks dropped as 0.4 drops
# beyond 0.5), the fixed-length comparison with the own document left out: AUROC P(C250 > C750) of each persistence
# feature, real minus the mean of its 20 corpus-bearing control draws, paired Bayesian bootstrap over documents
# (vmf_stats.boot_d). An artefact-driven effect shrinks as the threshold tightens, beyond the shrinkage of the same-size
# random deletion. Needs features.py run in each arm. Usage: python dose.py -> $RT_WORK/out_clean/dose.csv
import os
from pathlib import Path
import numpy as np
import pandas as pd
from vmf_stats import boot_d

ROOT = Path(os.environ.get("RT_WORK", str(Path(__file__).resolve().parent / "work")))  # see README
ARMS = [("out", 1.0), ("out_t07", 0.7), ("out_clean", 0.5), ("out_t04", 0.4), ("out_t04rand", "0.5+rand")]
rows = []
for c in ("nobel", "turing", "fields", "gutenberg"):
    for m in ("openai-3-small", "modernbert-embed"):
        for arm, t in ARMS:
            f = (ROOT / arm if c == "nobel" else ROOT / arm / c) / f"features_{m}.csv.gz"
            f = f if f.exists() else f.with_suffix(".xz")   # shipped tables are .csv.xz
            if not f.exists():
                rows.append(dict(corpus=c, model=m, arm=arm, threshold=t, missing=True)); continue
            F = pd.read_csv(f)
            a = F[(F.variant == "lodo") & (F.cell == "Q250|C750")]; b = F[(F.variant == "lodo_dens") & (F.cell == "Q250|C250")]
            r = dict(corpus=c, model=m, arm=arm, threshold=t, n_queries=len(a))  # queries are the Q250 chunks (same set in both cells)
            for feat in ("longbar", "w1", "ltmax"):
                cn = [f"{feat}_enull{k:02d}" for k in range(20)]
                est, d = boot_d(b[feat].to_numpy(float), a[feat].to_numpy(float), b[cn].to_numpy(float), a[cn].to_numpy(float),
                                b.lecture.to_numpy(), a.lecture.to_numpy(), np.random.default_rng(0))
                r.update({f"{feat}_d": est, f"{feat}_lo": np.percentile(d, 2.5), f"{feat}_hi": np.percentile(d, 97.5)})
            rows.append(r)
R = pd.DataFrame(rows); R.to_csv(ROOT / "out_clean" / "dose.csv", index=False)
pd.set_option("display.width", 200)
print(R[["corpus", "model", "arm", "n_queries", "longbar_d", "longbar_lo", "longbar_hi", "ltmax_d", "w1_d"]].round(3).to_string(index=False))
