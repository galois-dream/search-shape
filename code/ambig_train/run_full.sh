#!/bin/bash
# run_full.sh -- written on day 6 at 20:25. The AmbigNQ checks on the FULL gold pages (every chunk, not the first 5 per
# page). Usage, from code/ambig_train: bash run_full.sh DATA_DIR REAL_DIR   (DATA_DIR/ambignq/train.json; REAL_DIR as
# for ambig.py retrieve). Needs sentence-transformers, ripser, pandas, scipy. Outputs: out_full/ and *_full.txt.
set -euo pipefail
D=$1; R=$2
export FULL=1
python ambig.py retrieve "$D" "$R" out_full
python ambig.py feats out_full
python ambig.py auroc out_full > out_full/auroc.txt
python probe_slot.py "$D/ambignq/train.json" out_full > probe_slot_full.txt
python probe_slot2.py out_full > probe_slot2_full.txt
python probe_answers.py "$D/ambignq/train.json" "$R/chunks.parquet" out_full > probe_answers_full.txt
python nocherry.py "$D/ambignq/train.json" out_full > nocherry_full.txt
python samples_rule.py "$D/ambignq/train.json" out_full > samples_rule_full.txt   # also writes out_full/samples_rule.json
