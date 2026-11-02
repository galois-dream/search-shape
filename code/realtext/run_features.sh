#!/bin/bash
# run_features.sh -- per-model feature pass (1 worker, single-threaded BLAS) for one corpus, then statistics and figures.
# Usage: CORPUS=nobel|wiki|gutenberg|turing|fields [CLEAN=1 OUTROOT=out_clean] bash run_features.sh model [model ...]
# (the paper's analysis is CLEAN=1 OUTROOT=out_clean)
set -uo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CORPUS=${CORPUS:-nobel} OUTROOT=${OUTROOT:-out}
W=${RT_WORK:-work}; O=$W/$OUTROOT; [ "$CORPUS" != nobel ] && O=$W/$OUTROOT/$CORPUS; mkdir -p "$O"
for m in "$@"; do
  python -W ignore features.py "$m" --workers 1 --refcheck --nullseeds 20 > "$O/run_features_$m.log" 2>&1 || echo "FAIL features $m" >> "$O/run.log"
done
python -W ignore stats.py >> "$O/run.log" 2>&1 || echo "FAIL stats" >> "$O/run.log"
python -W ignore figs.py >> "$O/run.log" 2>&1 || echo "FAIL figs" >> "$O/run.log"
