# toy_barcode.py -- Purpose: data for fig_toy_barcode ("the near miss").
# 2-D toy, query at the origin. Scene A: 25 points at 20 deg + 25 at 200 deg, radius 0.6 (radial sd 0.03),
# angular sd 6 deg. Scene B: 50 points around 20 deg, angular sd tuned by bisection so W1(H0) of B matches A
# within 2%. One rng, seed 11 (A drawn first, then B's standard normals, fixed during the bisection).
# Features as in the paper: u_i = c_i/|c_i| (q = 0), Vietoris-Rips via ripser, W1 = mean finite H0 death / 2,
# L_max = max finite H0 death. Writes data/toy_{A,B}_{pts,dirs,bars}.dat and data/toy_summary.txt.
from pathlib import Path
import numpy as np
from ripser import ripser
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform, pdist

OUT = Path(__file__).parent / "data"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(11)
deg = np.pi / 180


def h0(U):
    d = ripser(U, maxdim=0)["dgms"][0][:, 1]
    return np.sort(d[np.isfinite(d)])[::-1]


def feats(P):
    U = P / np.linalg.norm(P, axis=1, keepdims=True)
    d = h0(U)
    return U, d, d.mean() / 2, d.max()


# scene A
angA = np.r_[20 + 6 * rng.standard_normal(25), 200 + 6 * rng.standard_normal(25)] * deg
radA = 0.6 + 0.03 * rng.standard_normal(50)
PA = np.c_[radA * np.cos(angA), radA * np.sin(angA)]
# scene B: standard normals fixed, spread tuned
zB = rng.standard_normal(50); radB = 0.6 + 0.03 * rng.standard_normal(50)
UA, dA, w1A, lA = feats(PA)


def sceneB(s):
    a = (20 + s * zB) * deg
    return np.c_[radB * np.cos(a), radB * np.sin(a)]


lo, hi = 1.0, 80.0
for _ in range(60):
    mid = (lo + hi) / 2
    if feats(sceneB(mid))[2] < w1A: lo = mid
    else: hi = mid
sB = (lo + hi) / 2
PB = sceneB(sB); UB, dB, w1B, lB = feats(PB)
assert abs(w1B / w1A - 1) < 0.02


def longest_edge(U):
    """endpoints of the longest Euclidean-MST edge (= longest finite H0 bar for Rips)."""
    T = minimum_spanning_tree(squareform(pdist(U))).tocoo()
    k = np.argmax(T.data); return U[T.row[k]], U[T.col[k]], T.data[k]


for tag, P, U, d in (("A", PA, UA, dA), ("B", PB, UB, dB)):
    np.savetxt(OUT / f"toy_{tag}_pts.dat", P, fmt="%.5f", header="x y", comments="")
    np.savetxt(OUT / f"toy_{tag}_dirs.dat", U, fmt="%.5f", header="x y", comments="")
    np.savetxt(OUT / f"toy_{tag}_bars.dat", np.c_[np.arange(1, len(d) + 1), d], fmt="%d %.5f",
               header="rank death", comments="")
    a, b, L = longest_edge(U)
    assert abs(L - d.max()) < 1e-6
    np.savetxt(OUT / f"toy_{tag}_longedge.dat", np.r_[[a], [b]], fmt="%.5f", header="x y", comments="")

s = (f"# toy_barcode.py run {np.datetime64('now')} UTC\n"
     f"A: W1={w1A:.4f} Lmax={lA:.4f} mean death={2*w1A:.4f}\n"
     f"B: spread sd={sB:.2f} deg  W1={w1B:.4f} Lmax={lB:.4f} mean death={2*w1B:.4f}  W1 ratio B/A={w1B/w1A:.4f}\n")
(OUT / "toy_summary.txt").write_text(s); print(s)
