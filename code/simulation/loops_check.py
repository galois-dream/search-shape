# loops_check.py -- reproducible check behind the paper's "loops need not count topics" paragraph.
# Usage: python loops_check.py [samples per cell, default 20]
import numpy as np, sys
from ripser import ripser
def sample(rng, D, T, d, k=50, c=0.55, noise=0.52):
    q = np.zeros(D); q[0] = 1
    pts = []
    for t in range(T):
        v = rng.standard_normal(D); v[0] = 0; v /= np.linalg.norm(v)
        centre = c * q + np.sqrt(1 - c * c) * v
        B = np.linalg.qr(rng.standard_normal((D, d)))[0]          # d random directions
        for _ in range(k // T + (t < k % T)):
            x = centre + noise * B @ (rng.standard_normal(d) / np.sqrt(d)); pts.append(x / np.linalg.norm(x))
    U = np.array(pts) - q; U /= np.linalg.norm(U, axis=1, keepdims=True)
    h1 = ripser(U, maxdim=1)["dgms"][1]
    return 0.0 if len(h1) == 0 else float(np.max(h1[:, 1] - h1[:, 0]) / 2)
def wilson(k, n, z=1.96):
    c = (k + z * z / 2) / (n + z * z); h = z / (n + z * z) * np.sqrt(k * (n - k) / n + z * z / 4)
    return c - h, c + h
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
rng = np.random.default_rng(0)
print(f"T  d    loops/{n}  fraction  Wilson95%        median LT_max")
for T in (1, 2, 3):
    for d in (2, 5, 20, 383):
        v = [sample(rng, 384, T, d) for _ in range(n)]
        k = sum(x > 0.05 for x in v); lo, hi = wilson(k, n)
        print(f"{T}  {d:<4} {k:>5}     {k / n:.2f}     [{lo:.2f}, {hi:.2f}]     {np.median(v):.3f}")
