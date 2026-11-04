# thm_steps_points.py -- written 2026-11-04 10:20 CET. Checks the 9-point example drawn in fig_thmsteps.tex (Theorem
# "barcode and nearest neighbours"): nearest neighbours, mutual pairs, clumps, minimum spanning tree, H0 bars.
# Prints everything the figure states, so the drawing is checked by code, not by eye. Stdlib only.
import itertools, math
P = {1: (0.0, 0.0), 2: (0.55, 0.35), 3: (1.25, 0.15), 4: (0.3, 1.1),
     5: (3.0, 1.55), 6: (3.45, 1.15), 7: (3.78, 1.8),
     8: (4.3, -0.15), 9: (4.95, 0.12)}
d = lambda i, j: math.dist(P[i], P[j])
pairs = sorted(itertools.combinations(P, 2), key=lambda e: (d(*e), e))
assert len({round(d(*e), 9) for e in pairs}) == len(pairs), "distances must be distinct"
nn = {i: min((j for j in P if j != i), key=lambda j: (d(i, j), j)) for i in P}
mutual = sorted({tuple(sorted((i, nn[i]))) for i in P if nn[nn[i]] == i})
forest = sorted({tuple(sorted((i, nn[i]))) for i in P})
parent = {i: i for i in P}
def find(i):
    while parent[i] != i: i = parent[i]
    return i
tree = []
for e in pairs:
    a, b = find(e[0]), find(e[1])
    if a != b: parent[a] = b; tree.append(e)
bridges = [e for e in tree if e not in forest]
N, p = len(P), len(mutual)
print("nearest neighbour:", nn)
print("mutual pairs (p=%d):" % p, mutual)
print("forest edges (N-p=%d):" % len(forest), [(e, round(d(*e), 2)) for e in forest])
print("tree edges (N-1=%d):" % len(tree), [(e, round(d(*e), 2)) for e in tree])
print("bridges (p-1=%d):" % len(bridges), [(e, round(d(*e), 2)) for e in bridges])
assert set(forest) <= set(tree) and len(forest) == N - p and len(bridges) == p - 1
bars = sorted(round(d(*e), 2) for e in tree)
print("H0 bars:", bars, " longest:", max(tree, key=lambda e: d(*e)), " max delta_i:", round(max(d(i, nn[i]) for i in P), 2))
