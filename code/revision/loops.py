# loops.py 2026-11-02 13:10 CET (run from nobel/out_clean, the cleaned corpora): share of real neighbourhoods with a loop (LTmax>0.05) vs the cosine-keeping controls
import csv, gzip, glob, statistics as st
for root, name in [(".", "nobel"), ("fields", "fields"), ("turing", "turing"), ("gutenberg", "essays"), ("wiki", "wiki")]:
    for f in sorted(glob.glob(root + "/features_*.csv.gz")):
        r_, n_, e_ = [], [], []
        for r in csv.DictReader(gzip.open(f, "rt")):
            if r["variant"] == "primary" and r["cell"] in ("Q250|C750", "Q750|C250"):
                r_.append(float(r["ltmax"])); n_.append(float(r["ltmax_null"])); e_.append(float(r["ltmax_enull"]))
        sh = lambda v: sum(x > 0.05 for x in v) / len(v)
        print(f"{name:7s} {f.split('_',1)[1][:-7]:17s} n={len(r_):5d} share>0.05 real {sh(r_):.2f} random {sh(n_):.2f} corpus {sh(e_):.2f} | median real {st.median(r_):.3f} random {st.median(n_):.3f} corpus {st.median(e_):.3f}")
