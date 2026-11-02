"""Check that run_sim.py regenerates the shipped per-query tables (data/simulation; stored to 10 significant digits) to <= 1e-9.

Checks (about one minute in total on a laptop CPU):
  full       appD64 seed 0 and superpos seed 0: every row and every column, including the matched-cosine nulls
             (covers the orthonormal and the random/superposition topic banks, scenarios 1 and 2).
  truncated  fig2 seed 0 and proj seed 0 restricted to the first G parent groups per scenario (--max-groups): all
             non-null columns of the matching rows (covers scenario 4 = mixed and the random projections). Null
             columns are skipped here only because truncation shifts the null RNG stream.
  cosine     cosine_baselines.py for fig1 seed 0: every row and column of data/simulation/cosine.
  distractor distractor.py for fig1 seed 0: every row and column of data/simulation/distractor.
Usage: python verify_reimplementation.py [--ref ../../data/simulation/features] [--ref-cos ../../data/simulation/cosine]
"""
import argparse, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KEY = ["proj_dim", "scenario", "query_idx", "depsilon"]


def ref_table(ref, cfg, seed, prefix=""):
    for suf in (".csv.xz", ".csv.gz", ".csv"):
        f = Path(ref) / f"{prefix}{cfg}_seed{seed}{suf}"
        if f.exists():
            return pd.read_csv(f)
    sys.exit(f"missing reference table for {cfg} seed {seed} in {ref}")


def compare(new, ref, cols):
    worst, nan_mismatch = 0.0, 0
    for c in cols:
        a, b = new[c], ref[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) and a.dtype != bool:
            x, y = a.to_numpy(float), b.to_numpy(float)
            nan_mismatch += int((np.isnan(x) != np.isnan(y)).sum())
            ok = ~np.isnan(x) & ~np.isnan(y)
            if ok.any():
                worst = max(worst, float(np.max(np.abs(x[ok] - y[ok]))))
        else:
            x, y = a.astype(object).where(a.notna(), "NA"), b.astype(object).where(b.notna(), "NA")
            nan_mismatch += int((x.astype(str).to_numpy() != y.astype(str).to_numpy()).sum())
    return worst, nan_mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(HERE.parents[1] / "data" / "simulation" / "features"))
    ap.add_argument("--ref-cos", default=str(HERE.parents[1] / "data" / "simulation" / "cosine"))
    ap.add_argument("--ref-dis", default=str(HERE.parents[1] / "data" / "simulation" / "distractor"))
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--groups", type=int, default=2)
    a = ap.parse_args()
    checks = [("appD64", 0, None), ("superpos", 0, None), ("fig2", 0, a.groups), ("proj", 0, a.groups)]
    fail = False
    with tempfile.TemporaryDirectory() as tmp:
        for cfg, seed, G in checks:
            cmd = [sys.executable, "-W", "ignore", str(HERE / "run_sim.py"), "--configs", cfg, "--seeds", str(seed),
                   "--outdir", tmp] + ([] if G is None else ["--max-groups", str(G)])
            subprocess.run(cmd, check=True, capture_output=True)
            new = pd.read_csv(Path(tmp) / f"{cfg}_seed{seed}.csv")
            ref = ref_table(a.ref, cfg, seed)
            if G is None:
                assert len(new) == len(ref), f"{cfg}: {len(new)} rows vs {len(ref)}"
                cols = [c for c in ref.columns if c in new.columns]
                assert set(cols) == set(ref.columns) == set(new.columns), "column sets differ"
                ref_m = ref
            else:
                ref_m = new[KEY].merge(ref, on=KEY, how="left", validate="one_to_one")
                cols = [c for c in ref.columns if not c.startswith("null_")]
            worst, mism = compare(new.reset_index(drop=True), ref_m.reset_index(drop=True), cols)
            ok = worst <= a.tol and mism == 0
            fail |= not ok
            print(f"{'PASS' if ok else 'FAIL'} {cfg} seed {seed} ({'all groups' if G is None else f'first {G} groups'}): "
                  f"rows {len(new)}, columns {len(cols)}, max |diff| = {worst:.3g}, NaN/label mismatches = {mism}")
        subprocess.run([sys.executable, "-W", "ignore", str(HERE / "cosine_baselines.py"), "--configs", "fig1",
                        "--seeds", "0", "--outdir", tmp], check=True, capture_output=True)
        subprocess.run([sys.executable, "-W", "ignore", str(HERE / "distractor.py"), "--configs", "fig1",
                        "--seeds", "0", "--outdir", tmp], check=True, capture_output=True)
        for name, pre, ref_dir in (("cosine baselines", "cosine_", a.ref_cos), ("distractor variant", "distractor_", a.ref_dis)):
            new, ref = pd.read_csv(Path(tmp) / f"{pre}fig1_seed0.csv"), ref_table(ref_dir, "fig1", 0, pre)
            assert len(new) == len(ref) and list(new.columns) == list(ref.columns)
            worst, mism = compare(new, ref, list(ref.columns))
            ok = worst <= a.tol and mism == 0
            fail |= not ok
            print(f"{'PASS' if ok else 'FAIL'} {name} fig1 seed 0: rows {len(new)}, columns {len(ref.columns)}, "
                  f"max |diff| = {worst:.3g}, NaN/label mismatches = {mism}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
