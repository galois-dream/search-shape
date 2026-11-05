# companion_numbers.py -- written 2026-11-05 CET. Checks every number drawn or stated in the figures of the supplement's
# reader's companion (supp_part0.tex). Run with any Python that has numpy and ripser: python companion_numbers.py
import numpy as np
from ripser import ripser
from scipy.stats import norm
ok = lambda name, got, want, tol=5e-3: print(f"{'ok ' if abs(got - want) < tol else 'BAD'} {name}: {got:.4f} (stated {want})")

# Fig "crowding": side view, q = (0,1); two passages with opposite bearings t = (+-1, 0) at angle theta to q.
# Offset directions u = -sin(theta/2) q + cos(theta/2) t (Lemma 5.1); distance between them = 2 cos(theta/2).
for th, want in ((40, 1.879), (120, 1.0)):
    t = np.radians(th)
    c1, c2 = np.array([np.sin(t), np.cos(t)]), np.array([-np.sin(t), np.cos(t)]); q = np.array([0, 1.0])
    u1, u2 = (c1 - q) / np.linalg.norm(c1 - q), (c2 - q) / np.linalg.norm(c2 - q)
    ok(f"theta={th}: |u1-u2|", np.linalg.norm(u1 - u2), want)
    ok(f"theta={th}: u1 x", u1[0], np.cos(t / 2)); ok(f"theta={th}: u1 y", u1[1], -np.sin(t / 2))
# score factor of Corollary 5.4 at scores 0.9 and 0.6
for s, want in ((0.9, 0.975), (0.6, 0.894)):
    ok(f"cos(theta/2) at score {s}", np.sqrt((1 + s) / 2), want)

# Fig "rescaled distances": q = (0,0); c_i, c_j at distances a_i, a_j and angle gamma at q (Proposition 5.3).
ai, aj, g = 0.7, 1.6, np.radians(50)
ci, cj = ai * np.array([np.cos(np.radians(115)), np.sin(np.radians(115))]), aj * np.array([np.cos(np.radians(65)), np.sin(np.radians(65))])
cc = np.linalg.norm(ci - cj); rho = cc / np.sqrt(ai * aj); d = 2 * np.sin(g / 2); E = (ai - aj) ** 2 / (ai * aj)
ok("|ci-cj|", cc, 1.269); ok("sqrt(ai aj)", np.sqrt(ai * aj), 1.058); ok("rho", rho, 1.199); ok("E", E, 0.723); ok("d", d, 0.845); ok("d^2 = rho^2 - E", d ** 2, rho ** 2 - E, 1e-9)

# Fig "loops": 12 points on a unit circle; centre plus hexagon
n = 12; t = 2 * np.pi * np.arange(n) / n
h1 = ripser(np.c_[np.cos(t), np.sin(t)], maxdim=1)["dgms"][1]
ok("ring H1 birth", h1[0, 0], 0.518); ok("ring H1 death", h1[0, 1], 1.732); ok("ring LT_max", (h1[0, 1] - h1[0, 0]) / 2, 0.607)
hx = np.r_[[[0, 0]], np.c_[np.cos(np.pi * np.arange(6) / 3), np.sin(np.pi * np.arange(6) / 3)]]
print("ok " if len(ripser(hx, maxdim=1)["dgms"][1]) == 0 else "BAD", "hexagon with centre: no H1 bar")

# Fig "AUROC": two unit-variance normals shifted by delta have AUROC Phi(delta / sqrt 2)
for A, want in ((0.84, 1.41), (0.54, 0.14), (0.94, 2.20)):
    ok(f"shift for AUROC {A}", np.sqrt(2) * norm.ppf(A), want)

# Fig "model": Remark 4.1's example, nu = 0: 2-topic query, 4-topic passages sharing 2, 1, 1, 1 topics
for k, want in ((2, 0.71), (1, 0.35)):
    ok(f"score sharing {k} of 2", k / np.sqrt(2 * 4), want)
# noise weight of the simulation: sigma^2 D with sigma = 0.1, D = 256
ok("sigma^2 D", 0.1 ** 2 * 256, 2.56)

# Fig "Gram" (supp_part0.tex, Figure SM5): Example 5.5 of the paper, q = (0,0,1), c1 = (0.6,0,0.8), c2 = (0,0.6,0.8)
q, c1, c2 = np.array([0, 0, 1.0]), np.array([0.6, 0, 0.8]), np.array([0, 0.6, 0.8])
G = np.array([q, c1, c2]) @ np.array([q, c1, c2]).T
ok("Gram g1", G[0, 1], 0.8); ok("Gram C12", G[1, 2], 0.64)
s = np.sqrt(1 - 0.8 ** 2); ok("sin theta", s, 0.6)
ok("bearings t1.t2", (G[1, 2] - G[0, 1] * G[0, 2]) / (s * s), 0.0)
a = np.sqrt(2 - 2 * 0.8); ok("a_i", a, 0.632); ok("a1 a2", a * a, 0.4)
uu = (G[1, 2] - G[0, 1] - G[0, 2] + 1) / (a * a); ok("u1.u2", uu, 0.1); ok("d", np.sqrt(2 - 2 * uu), 1.342)

# SM8 "From an AUROC to a belief": two unit-variance normals shifted by delta (AUROC = Phi(delta/sqrt 2)); likelihood ratio
# LR(x) = exp(delta x - delta^2/2), x measured from the first group's mean, over x in [-1.645, delta + 1.645] (from the
# 5th percentile of one group to the 95th of the other); prior 0.47 = share of ambiguous AmbigNQ questions (main.tex Sec. 8).
post = lambda lr, p=0.47: p / (1 - p) * lr / (1 + p / (1 - p) * lr)
for A, want in ((0.54, (0.78, 1.28, 0.41, 0.53)), (0.84, (0.04, 27.2, 0.03, 0.96))):
    dl = np.sqrt(2) * norm.ppf(A); lo, hi = np.exp(-1.645 * dl - dl * dl / 2), np.exp(dl * (dl + 1.645) - dl * dl / 2)
    ok(f"AUROC {A}: LR low", lo, want[0], 0.01); ok(f"AUROC {A}: LR high", hi, want[1], 0.1)
    ok(f"AUROC {A}: posterior low", post(lo), want[2], 0.006); ok(f"AUROC {A}: posterior high", post(hi), want[3], 0.006)

# SM8 belief paragraph, expanded: expected evidence per observation under ambiguity = KL(N(delta,1) || N(0,1)) = delta^2/2 nats;
# in Jaynes's decibels, 10 log10(e) * delta^2/2.
for A, nats, db in ((0.54, 0.0101, 0.044), (0.84, 0.989, 4.30)):
    dl = np.sqrt(2) * norm.ppf(A)
    ok(f"AUROC {A}: KL nats", dl * dl / 2, nats, 0.0005 if A < 0.6 else 0.005); ok(f"AUROC {A}: KL dB", 10 * np.log10(np.e) * dl * dl / 2, db, 0.002 if A < 0.6 else 0.01)
# the prior odds of AmbigNQ in decibels: 10 log10(0.47/0.53)
ok("prior evidence dB", 10 * np.log10(0.47 / 0.53), -0.52, 0.01)
