# realbarcode_ctrl.py -- written 2026-11-04 16:05 CET. Corpus-bearing control for the paper's one-query barcode figure
# (fig_realbarcode, from mstpieces.py DUMP -> out_clean/example_barcode.json). A separate script so that mstpieces.py and
# its cited output stay as they are. Rebuilds the Q250|C750 cell exactly as mstpieces.py does (span-overlap exclusion,
# CLEAN=1), checks that the dumped query's W1 is reproduced, then draws the corpus-bearing control 20 times as in
# features.multi_null (draw k: default_rng((k, 1, query_index)); same scores, each bearing from a random chunk outside
# the query's top 50 that does not overlap it). Writes out_clean/example_barcode_ctrl.json: real mean bar, the 20
# control mean bars and their median. H0 bars = minimum-spanning-tree edges of the offset directions (Theorem 5.6(a)).
# Usage: CORPUS=nobel CLEAN=1 OUTROOT=out_clean python code/realtext/realbarcode_ctrl.py
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import types
if 'ripser' not in sys.modules:  # features.py imports ripser at load; this script needs only the MST (no ripser on this machine)
    try: import ripser  # noqa: F401
    except ImportError: sys.modules['ripser'] = types.SimpleNamespace(ripser=None)
import features as F
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform, pdist

ex = json.load(open(F.OUT / "example_barcode.json"))
assert ex["corpus"] == F.CORPUS and ex["cell"] == "Q250|C750"
meta, P, Q = F.load_model(ex["model"])
qm, cm, C, Qe = meta[250], meta[750], P[750], Q[250]
i = ex["query_index"]
S = Qe[i] @ C.T
excl = (cm.lecture.to_numpy(dtype=object) == qm.lecture[i]) & (qm.start[i] < cm.end.values) & (cm.start.values < qm.end[i])
S = np.where(excl, -np.inf, S); top = np.argsort(-S, kind="stable")[:F.K]; cos = S[top]


def mean_bar(q, X):
    T = minimum_spanning_tree(squareform(pdist(F.offsets(q, X))))
    return float(T.data.sum() / (len(X) - 1))


real = mean_bar(Qe[i], C[top])
assert abs(real / 2 - ex["W1"]) < 1e-9, (real / 2, ex["W1"])
pool = np.setdiff1d(np.flatnonzero(~excl), top)
a = np.clip(cos, -1, 1); q = Qe[i]; ctrl = []
for k in range(20):
    E = C[np.random.default_rng((k, 1, i)).choice(pool, F.K, replace=len(pool) < F.K)]
    V = F.unit(E - (E @ q)[:, None] * q[None, :])
    ctrl.append(mean_bar(q, a[:, None] * q[None, :] + np.sqrt(1 - a ** 2)[:, None] * V))
out = {"query_index": i, "real_mean_bar": real, "corpus_bearing_mean_bars": ctrl, "corpus_bearing_median": float(np.median(ctrl))}
json.dump(out, open(F.OUT / "example_barcode_ctrl.json", "w"), indent=1)
print(f"real mean bar {real:.3f}; corpus-bearing median {np.median(ctrl):.3f} (range {min(ctrl):.3f}-{max(ctrl):.3f})")
