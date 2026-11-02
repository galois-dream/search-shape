# overview.py -- Purpose: coordinates and barcode for fig_overview (hero strip).
# 3-D toy on the unit sphere: query q = north pole, 16 neighbours in two clumps of 8 (angle to q 35 +- 4 deg,
# bearings at azimuth 0 +- 9 deg and 180 +- 9 deg). Offsets u_i = (c_i - q)/|c_i - q| (paper eq. tilt);
# Vietoris-Rips persistence of {u_i} with ripser (maxdim 1); W1 = mean finite H0 death / 2,
# LT_max = max (d-b)/2 over H1, L_max = max finite H0 death. The seed is the first in 0..499 whose H1 diagram
# has exactly one point, dying below 0.25 and lasting >= 0.015 (a within-clump loop visible on the broken axis of
# panel (d)), so the H1 bar is computed, not drawn by hand. criterion tightened from "one H1 point".
# redesign (same toy, same seed): clumps exported separately (clump 1 = first 8, azimuth 0; clump 2 =
# last 8, azimuth 180); (b)/(c) use the true top view of the u_i (x, y components, no magnification); the Rips
# radius r for (c) is 0.90 x half the smallest cross-clump distance in that top view (just before the groups join).
# Orthographic view: camera elevation 28 deg; screen x = p.x, screen y = p.y sin(el) + p.z cos(el).
# second row, the near miss: ONE clump of 16 neighbours with the same spread as one clump above
# (angle 35 +- 4 deg, azimuth 0 +- 9 deg), rng seed 69; same features; macros prefixed \ovO.
# Writes data/overview_coords.tex (TikZ macros) and data/overview_summary.txt.
from pathlib import Path
import numpy as np
from ripser import ripser
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform, pdist

OUT = Path(__file__).parent / "data"; OUT.mkdir(exist_ok=True)
deg = np.pi / 180; EL = 28 * deg
q = np.array([0.0, 0.0, 1.0])


def scene(seed):
    r = np.random.default_rng(seed)
    th = (35 + 4 * r.standard_normal(16)) * deg
    az = np.r_[0 + 9 * r.standard_normal(8), 180 + 9 * r.standard_normal(8)] * deg
    t = np.c_[np.cos(az), np.sin(az), np.zeros(16)]
    c = np.cos(th)[:, None] * q + np.sin(th)[:, None] * t
    u = (c - q) / np.linalg.norm(c - q, axis=1, keepdims=True)
    return th, az, c, u


for seed in range(500):
    th, az, c, u = scene(seed)
    dg = ripser(u, maxdim=1)["dgms"]
    if len(dg[1]) == 1 and dg[1][0, 1] < 0.25 and dg[1][0, 1] - dg[1][0, 0] >= 0.015:
        break
h0 = np.sort(dg[0][np.isfinite(dg[0][:, 1]), 1])[::-1]
b1, d1 = dg[1][0]
W1, LT, LM = h0.mean() / 2, (d1 - b1) / 2, h0.max()


def proj(P):
    return np.c_[P[:, 0], P[:, 1] * np.sin(EL) + P[:, 2] * np.cos(EL)]


def fmt(A):
    return ", ".join(f"{x:.4f}/{y:.4f}" for x, y in A)


Pc = proj(c)                     # (a) neighbours on the sphere (all on the visible cap: check below)
assert np.all(c @ np.array([0, -np.cos(EL), np.sin(EL)]) > 0)
top = u[:, :2].copy()            # (b), (c) directions seen from q, top view
A, B = slice(0, 8), slice(8, 16)
D2 = squareform(pdist(top))[A, B]                    # cross-clump distances in the drawn top view
i2, j2 = np.unravel_index(np.argmin(D2), D2.shape); j2 += 8
r_draw = 0.90 * D2.min() / 2
T = minimum_spanning_tree(squareform(pdist(u))).tocoo()
k = np.argmax(T.data)
assert {T.row[k] < 8, T.col[k] < 8} == {True, False}   # the longest MST edge joins the two clumps
TH = 35 * deg                    # cap outline in (a): circle at 35 deg from q
lines = [f"% overview.py output, seed {seed}; generated {np.datetime64('now')} UTC",
         f"\\def\\ovSeed{{{seed}}}",
         f"\\def\\ovQ{{{0:.4f},{np.cos(EL):.4f}}}",
         f"\\def\\ovSinEl{{{np.sin(EL):.4f}}}",
         f"\\def\\ovCapY{{{np.cos(TH) * np.cos(EL):.4f}}}", f"\\def\\ovCapR{{{np.sin(TH):.4f}}}",
         f"\\def\\ovNbrsA{{{fmt(Pc[A])}}}", f"\\def\\ovNbrsB{{{fmt(Pc[B])}}}",
         f"\\def\\ovTopA{{{fmt(top[A])}}}", f"\\def\\ovTopB{{{fmt(top[B])}}}",
         f"\\def\\ovGapTop{{{top[i2,0]:.4f},{top[i2,1]:.4f}}}", f"\\def\\ovGapTopB{{{top[j2,0]:.4f},{top[j2,1]:.4f}}}",
         f"\\def\\ovRdraw{{{r_draw:.4f}}}",
         "\\def\\ovHzero{" + ", ".join(f"{i+1}/{d:.4f}" for i, d in enumerate(h0)) + "}",
         f"\\def\\ovNbars{{{len(h0)}}}",
         f"\\def\\ovHoneB{{{b1:.4f}}}", f"\\def\\ovHoneD{{{d1:.4f}}}",
         f"\\def\\ovMeanD{{{h0.mean():.4f}}}", f"\\def\\ovLmax{{{LM:.4f}}}",
         f"\\def\\ovWoneTxt{{{W1:.3f}}}", f"\\def\\ovLTTxt{{{LT:.3f}}}", f"\\def\\ovLmaxTxt{{{LM:.2f}}}"]
# ---- second row: one clump (the near miss)
r1 = np.random.default_rng(69)
th1 = (35 + 4 * r1.standard_normal(16)) * deg; az1 = (0 + 9 * r1.standard_normal(16)) * deg
t1 = np.c_[np.cos(az1), np.sin(az1), np.zeros(16)]
c1 = np.cos(th1)[:, None] * q + np.sin(th1)[:, None] * t1
u1 = (c1 - q) / np.linalg.norm(c1 - q, axis=1, keepdims=True)
assert np.all(c1 @ np.array([0, -np.cos(EL), np.sin(EL)]) > 0)
dg1 = ripser(u1, maxdim=1)["dgms"]
h01 = np.sort(dg1[0][np.isfinite(dg1[0][:, 1]), 1])[::-1]
LT1 = ((dg1[1][:, 1] - dg1[1][:, 0]).max() / 2) if len(dg1[1]) else 0.0
assert len(dg1[1]) == 0   # the tex draws "H1: none"
lines += [f"\\def\\ovONbrs{{{fmt(proj(c1))}}}", f"\\def\\ovOTop{{{fmt(u1[:, :2])}}}",
          "\\def\\ovOHzero{" + ", ".join(f"{i+1}/{d:.4f}" for i, d in enumerate(h01)) + "}",
          f"\\def\\ovONbars{{{len(h01)}}}", f"\\def\\ovOMeanD{{{h01.mean():.4f}}}",
          f"\\def\\ovOWoneTxt{{{h01.mean()/2:.3f}}}", f"\\def\\ovOLTTxt{{{LT1:.0f}}}", f"\\def\\ovOLmaxTxt{{{h01.max():.2f}}}"]
(OUT / "overview_coords.tex").write_text("%\n".join(lines) + "%\n")  # trailing % : no stray spaces
s = (f"# overview.py run {np.datetime64('now')} UTC\nseed {seed}; W1={W1:.4f} LT_max={LT:.4f} (H1 bar {b1:.4f}-{d1:.4f}) "
     f"L_max={LM:.4f}; max within-clump MST edge {np.max(np.delete(T.data, k)):.4f}\n"
     f"top view: closest cross-clump pair {i2},{j2} at {D2.min():.4f} (3-D longest MST edge {T.row[k]},{T.col[k]}); "
     f"drawn r = {r_draw:.4f}\n"
     f"one clump (seed 69): W1={h01.mean()/2:.4f} LT_max={LT1:.4f} (H1 points: {len(dg1[1])}) L_max={h01.max():.4f}\n")
(OUT / "overview_summary.txt").write_text(s); print(s)
