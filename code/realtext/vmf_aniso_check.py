# vmf_aniso_check.py -- synthetic check with the pipeline's own statistic (vmf.stats): does vmf2_r5 > 0 on ONE-group
# clouds whose spread is concentrated in few directions (spectrum decaying as k^-alpha), against two groups 30 degrees
# apart? D = 768, n = 50, 30 clouds per cell. Prints P(vmf2 > 0) and the median effective rank per cell.
# Usage: python vmf_aniso_check.py   (numpy/scipy only)
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vmf import stats
D, n = 768, 50
def cloud(rng, alpha, phi, rho=0.6):
    a = np.zeros(D); a[0] = 1; b = np.zeros(D); b[1] = 1
    m2 = np.cos(np.radians(phi)) * a + np.sin(np.radians(phi)) * b
    sd = np.r_[0, 0, np.arange(1, D - 1) ** (-alpha / 2)]; G = rng.standard_normal((n, D)) * sd
    G /= np.linalg.norm(G, axis=1, keepdims=True); s = np.sqrt((1 - rho ** 2) / rho ** 2)
    Y = np.vstack([np.tile(a, (n // 2, 1)), np.tile(m2, (n // 2, 1))]) + s * G
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)
def er(U):
    ev = np.linalg.svd(U - U.mean(0), compute_uv=False) ** 2; return ev.sum() / ev.max()
for alpha in (0.0, 1.0, 1.5, 2.0):
    for phi in (0, 30):
        rng = np.random.default_rng((int(alpha * 10), phi)); v, e = [], []
        for i in range(30):
            U = cloud(rng, alpha, phi); v.append(stats(U, seed=i, rs=(5,))["vmf2_r5"]); e.append(er(U))
        print(f"alpha={alpha} phi={phi:2d} effrank={np.median(e):5.1f} P(vmf2>0)={np.mean(np.array(v) > 0):.2f}", flush=True)
