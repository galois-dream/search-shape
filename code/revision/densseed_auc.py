# densseed_auc.py -- written 2026-11-02 14:09 CET. Size-matched contrast over ten C250 subsamples (densseed.py):
# per combination and seed, point AUROC P(Q750 vs subsampled C250 > Q250|C750) for the mean and SD of the 50 scores and
# W1(H0); checks that seed 0 reproduces the 'dens' rows of features_<model>.csv.gz. Run from nobel/out_clean.
import csv, gzip, glob, statistics as st
from collections import defaultdict
def auc(x, y):
    allv = sorted([(v, 0) for v in x] + [(v, 1) for v in y]); rk = {}; i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]: j += 1
        rk[allv[i][0]] = (i + j + 1) / 2; i = j
    return (sum(rk[v] for v in x) - len(x) * (len(x) + 1) / 2) / (len(x) * len(y))
sub = {"nobel": ".", "fields": "fields", "turing": "turing", "gutenberg": "gutenberg", "wiki": "wiki"}
res = defaultdict(lambda: defaultdict(list)); check = []
for f in sorted(glob.glob("densseed/densseed_*.csv")):
    corpus, model = f.split("densseed_", 1)[1][:-4].split("_", 1)
    base = {"mean50": [], "std50": [], "w1": []}; dens0 = {}
    for r in csv.DictReader(gzip.open(f"{sub[corpus]}/features_{model}.csv.gz", "rt")):
        if r["variant"] == "primary" and r["cell"] == "Q250|C750":
            for k in base: base[k].append(float(r[k]))
        if r["variant"] == "dens" and r["cell"] == "Q750|C250": dens0[r["query_id"]] = float(r["w1"])
    by = defaultdict(lambda: defaultdict(list)); w0 = {}
    for r in csv.DictReader(open(f)):
        for k in base: by[int(r["seed"])][k].append(float(r[k]))
        if r["seed"] == "0": w0[r["query_id"]] = float(r["w1"])
    common = set(w0) & set(dens0)
    check.append(max(abs(w0[q] - dens0[q]) for q in common) if common else float("nan"))
    for seed in range(10):
        for k in base: res[(corpus, model)][k].append(auc(by[seed][k], base[k]))
print(f"seed-0 reproduction: max |W1 diff| over combinations {max(check):.2e} (n={len(check)})")
for k, lab in (("mean50", "mean score"), ("std50", "SD of scores"), ("w1", "W1")):
    allseeds = [min(v[k]) > 0.5 for v in res.values()]; nones = [max(v[k]) < 0.5 for v in res.values()]
    means = [st.mean(v[k]) for v in res.values()]; spread = [max(v[k]) - min(v[k]) for v in res.values()]
    print(f"{lab:13s}: AUROC (multi-like > poly-like), mean over seeds: median {st.median(means):.2f} range {min(means):.2f}-{max(means):.2f}; "
          f"> 0.5 in every seed: {sum(allseeds)}/45; < 0.5 in every seed: {sum(nones)}/45; mean > 0.5: {sum(m > 0.5 for m in means)}/45; "
          f"seed-to-seed range: median {st.median(spread):.3f}, max {max(spread):.3f}")
