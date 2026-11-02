# vmf_highdim.py -- can a 1- vs 2-component von Mises-Fisher (vMF) mixture tell "one meaning" from "two" for n = 50
# neighbour bearings in high dimension p? Compares, on synthetic bearings:
#   A: vMF mixture (k = 1 vs 2) fitted in the ambient dimension, BIC;
#   B: the same after projecting onto the top-r principal subspace of the 50 points (r = 5), BIC;
#   C: spectral: share of the 2nd eigenvalue of the 50x50 Gram matrix, threshold = 95th pct under the one-cluster case.
# Two groups of 25 bearings around unit means at angle phi; within-group spread set so that each bearing has cosine
# ~rho to its group mean (rho = 0.6, like real bearing resultants). Usage: python vmf_highdim.py   (numpy only)
import numpy as np

import math
def log_iv(nu, x):  # exact power series for nu < 50 (Debye is off at small orders); Debye otherwise
    if nu < 50:
        x = float(np.asarray(x)); k = np.arange(2000)
        lt = (2 * k + nu) * math.log(x / 2) - np.array([math.lgamma(i + 1) + math.lgamma(i + nu + 1) for i in k])
        m = lt.max(); return m + math.log(np.exp(lt - m).sum())
    z = x / nu; s = np.sqrt(1 + z * z)
    return nu * (s + np.log(z / (1 + s))) - 0.5 * np.log(2 * np.pi * nu) - 0.5 * np.log(s)

def vmf_ll(X, mus, w, kappa):  # log-likelihood of a k-component vMF mixture with shared kappa on S^{p-1}
    p = X.shape[1]; nu = p / 2 - 1
    logc = nu * np.log(kappa) - (p / 2) * np.log(2 * np.pi) - log_iv(nu, kappa)
    L = np.log(w)[None, :] + logc + kappa * X @ mus.T
    m = L.max(1, keepdims=True)
    return float((m[:, 0] + np.log(np.exp(L - m).sum(1))).sum()), L

def kappa_hat(rbar, p):  # Banerjee et al. (2005) approximation
    return rbar * (p - rbar ** 2) / (1 - rbar ** 2)

def fit(X, k, rng, iters=50):
    n, p = X.shape
    if k == 1:
        m = X.mean(0); r = np.linalg.norm(m); kap = kappa_hat(r, p)
        return vmf_ll(X, (m / r)[None], np.ones(1), kap)[0]
    best = -np.inf
    for _ in range(5):  # restarts
        mus = X[rng.choice(n, 2, replace=False)]; w = np.ones(2) / 2; kap = kappa_hat(0.5, p)
        for _ in range(iters):
            _, L = vmf_ll(X, mus, w, kap)
            R = np.exp(L - L.max(1, keepdims=True)); R /= R.sum(1, keepdims=True)
            w = np.clip(R.mean(0), 1e-3, None); w /= w.sum()
            S = R.T @ X; norms = np.linalg.norm(S, axis=1); mus = S / norms[:, None]
            kap = kappa_hat(min(norms.sum() / n, 0.999), p)
        best = max(best, vmf_ll(X, mus, w, kap)[0])
    return best

def bic(ll, k, p, n):
    return -2 * ll + (k * (p - 1) + 1 + (k - 1)) * np.log(n)

def sample(p, phi, rho, rng, n=50):
    a = rng.standard_normal(p); a /= np.linalg.norm(a)
    b = rng.standard_normal(p); b -= (b @ a) * a; b /= np.linalg.norm(b)
    m2 = np.cos(np.radians(phi)) * a + np.sin(np.radians(phi)) * b
    s = np.sqrt((1 - rho ** 2) / rho ** 2)            # noise norm so that cos(point, mean) ~ rho
    out = []
    for m in (a, m2):
        G = rng.standard_normal((n // 2, p)); G /= np.linalg.norm(G, axis=1, keepdims=True)
        Y = m[None] + s * G; out.append(Y / np.linalg.norm(Y, axis=1, keepdims=True))
    return np.vstack(out)

def project(X, r=5):
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    Y = X @ Vt[:r].T; return Y / np.linalg.norm(Y, axis=1, keepdims=True)

def eig2(X):
    ev = np.linalg.eigvalsh(X @ X.T)[::-1]; return ev[1] / ev.sum()

rng = np.random.default_rng(0); REPS = 40
print("p     phi | A: vMF ambient BIC picks 2 | B: vMF in top-5 PCs picks 2 | C: spectral 2nd-eig share > null95")
for p in (64, 768, 1536, 3072):
    null = [eig2(sample(p, 0, 0.6, rng)) for _ in range(200)]; thr = np.percentile(null, 95)
    for phi in (0, 30, 60, 90):
        a = b = c = 0
        for _ in range(REPS):
            X = sample(p, phi, 0.6, rng)
            a += bic(fit(X, 2, rng), 2, p, 50) < bic(fit(X, 1, rng), 1, p, 50)
            Y = project(X); b += bic(fit(Y, 2, rng), 2, 5, 50) < bic(fit(Y, 1, rng), 1, 5, 50)
            c += eig2(X) > thr
        print(f"{p:5d} {phi:4d} | {a / REPS:26.2f} | {b / REPS:27.2f} | {c / REPS:.2f}", flush=True)
