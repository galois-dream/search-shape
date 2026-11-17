# Shipped copy of paper/tois/explore_e6.py; only this line, the ROOT line (paper/nobel -> $RT_WORK, default code/realtext/work, as features.py) changed for the repo layout (EXPLORATORY.md = docs/review_checks_exploratory.md; explore_out/ = data/reference_outputs/review_checks/)
# explore_e6.py -- written 2026-11-16 21:14 CET. EXPLORATORY.md E6: do 750-token chunks crowd together more than 250-token ones?
# For each corpus x model: 500 random chunks per chunking (seed 0), unit rows; (i) mean pairwise cosine, (ii) participation
# ratio of the centred embeddings, (iii) mean cosine to the chunking's mean direction. Descriptive; no test.
# Amendment 1 (see EXPLORATORY.md): m = min(500, n250, n750) chunks from each chunking.
# Usage: python3 explore_e6.py > explore_out/e6.txt   (numpy only; reads $RT_WORK/**/emb)
import os, sys
import numpy as np
ROOT = os.environ.get("RT_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "realtext", "work"))
CORPORA = {"Nobel": "emb", "Turing": "turing/emb", "Fields": "fields/emb", "Essays": "gutenberg/emb", "Wikipedia": "wiki/emb"}
def stats(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    G = X @ X.T; iu = np.triu_indices(len(X), 1)
    Xc = X - X.mean(0); ev = np.clip(np.linalg.eigvalsh(Xc @ Xc.T / len(X)), 0, None)
    m = X.mean(0); m /= np.linalg.norm(m)
    return G[iu].mean(), ev.sum() ** 2 / (ev ** 2).sum(), (X @ m).mean()
print("# e6 -- corpus model: C250 (i coherence, ii participation ratio, iii cos to mean) | C750 | differences C750-C250")
cnt = np.zeros(3, int); n = 0; rows = []
for cname, sub in CORPORA.items():
    for model in sorted(os.listdir(os.path.join(ROOT, sub))):
        d = os.path.join(ROOT, sub, model)
        out = {}
        Xs = {ch: np.load(os.path.join(d, f"passage_{ch}.npy")).astype(np.float64) for ch in ("c250", "c750")}
        m = min(500, *(len(X) for X in Xs.values()))   # amendment 1: equal sample sizes
        for ch, X in Xs.items():
            idx = np.random.default_rng(0).choice(len(X), m, replace=False)
            out[ch] = stats(X[idx])
        out["m"] = m
        a, b = np.array(out["c250"]), np.array(out["c750"]); dlt = b - a
        crowd = np.array([dlt[0] > 0, dlt[1] < 0, dlt[2] > 0]); cnt += crowd; n += 1
        rows.append(dlt)
        print(f"{cname:9s} {model:17s} m={out['m']:3d} " + " ".join(f"{v:7.3f}" for v in a) + " | " + " ".join(f"{v:7.3f}" for v in b) + " | " + " ".join(f"{v:+7.3f}" for v in dlt))
R = np.array(rows)
print(f"# combinations: {n}; C750 more crowded: (i) higher coherence {cnt[0]}, (ii) lower participation ratio {cnt[1]}, (iii) closer to mean {cnt[2]}")
print("# median difference C750-C250: " + " ".join(f"{v:+.3f}" for v in np.median(R, 0)))
