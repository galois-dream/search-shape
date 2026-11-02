# check_bound.py -- written 2026-11-02 07:57 CET, revised 07:58 CET (error split into three pieces). Numerical check of paper/theory/nn_bound.md.
# numpy only (scipy is not installed for /opt/homebrew python3; Prim's MST and Spearman are written out below).
# Usage: python3 -I paper/theory/check_bound.py [n_queries_per_family=300]   (~3 s, < 200 MB)
# Checks, per neighbourhood (N=50 unit passages around q=e_1 in D=256):
#   L1  identity  d_u^2 = rho^2 - E          (rho_ij = |c_i-c_j|/sqrt(a_i a_j), a_i = |c_i-q|)
#   L2  e_k(u) <= e_k(rho) for every sorted MST edge k; W1(rho)-W1(u) <= mean_{T_u}(rho-d_u)/2 <= sqrt(Emax)/2
#   L3  2(N-1) W1 = sum_i delta_i - M + G   (NN decomposition; #NN-forest components == #mutual pairs)
#   L4  sqrt(a_i/a_hi) r_i <= delta^rho_i <= sqrt(a_i/a_lo) r_i
# and reports tightness ratios (actual / bound) and within-family Spearman of W1 against the NN statistics.
import sys, numpy as np

N, D = 50, 256
rng = np.random.default_rng(20261102)

def prim(W):
    """Edges (i, j, w) of the minimum spanning tree of a dense symmetric matrix W."""
    n = len(W); seen = np.zeros(n, bool); seen[0] = True
    best = W[0].copy(); par = np.zeros(n, int); out = []
    for _ in range(n - 1):
        b = np.where(seen, np.inf, best); j = int(b.argmin()); out.append((par[j], j, W[par[j], j])); seen[j] = True
        upd = (W[j] < best) & ~seen; best[upd] = W[j][upd]; par[upd] = j
    return out

def rankdata(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r

def spearman(x, y):
    return np.corrcoef(rankdata(x), rankdata(y))[0, 1]

def unit(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)

def assemble(cosines, bearings):
    """c_i = cos(theta_i) q + sin(theta_i) t_i with q = e_1 and t_i unit vectors in coordinates 2..D."""
    c = np.clip(cosines, -1, 1); C = np.zeros((len(c), D)); C[:, 0] = c
    C[:, 1:] = np.sqrt(1 - c ** 2)[:, None] * unit(bearings); return C

def pdist(X):
    G = X @ X.T; sq = np.diag(G); return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0))

def analyse(C):
    q = np.zeros(D); q[0] = 1.0
    cos = C @ q; th = np.arccos(np.clip(cos, -1, 1)); a = np.linalg.norm(C - q, axis=1); s = np.sin(th / 2)
    U = (C - q) / a[:, None]
    du, dc = pdist(U), pdist(C)
    rho = dc / np.sqrt(np.outer(a, a))
    Dl = th[:, None] - th[None, :]
    E = 4 * np.sin(Dl / 4) ** 2 * (np.cos(Dl / 4) ** 2 / np.outer(s, s) - 1)
    ident = np.abs(du ** 2 - (rho ** 2 - E)).max()
    Tu, Tr = prim(du), prim(rho)
    eu = np.sort([w for *_, w in Tu]); er = np.sort([w for *_, w in Tr])
    W1u, W1r = eu.mean() / 2, er.mean() / 2
    gap_tree = np.mean([rho[i, j] - du[i, j] for i, j, _ in Tu]) / 2          # refined bound on W1r - W1u
    th_lo, th_hi = th.min(), th.max()
    gg = np.linspace(th_lo, th_hi, 101); A, B = np.meshgrid(gg, gg); dd = A - B
    Emax = (4 * np.sin(dd / 4) ** 2 * (np.cos(dd / 4) ** 2 / np.outer(np.sin(gg / 2), np.sin(gg / 2)) - 1)).max()
    # NN decomposition for the offset directions
    M = du.copy(); np.fill_diagonal(M, np.inf); nn = M.argmin(1); delta = M.min(1)
    mutual = [(i, int(nn[i])) for i in range(N) if nn[nn[i]] == i and i < nn[i]]
    nnE = {tuple(sorted((i, int(nn[i])))) for i in range(N)}
    treeE = {tuple(sorted((int(i), int(j)))): w for i, j, w in Tu}
    G = sum(w for e, w in treeE.items() if e not in nnE); Msum = sum(du[i, j] for i, j in mutual)
    decomp = abs(2 * (N - 1) * W1u - (delta.sum() - Msum + G))
    nn_in_tree = nnE <= set(treeE)
    # NN statistics of passages
    Mc = dc.copy(); np.fill_diagonal(Mc, np.inf); r = Mc.min(1) / a                 # the paper's r_i
    Mr = rho.copy(); np.fill_diagonal(Mr, np.inf); drho = Mr.min(1)
    j = nn; radb = np.sum(rho[np.arange(N), j] - du[np.arange(N), j]) / (2 * N)   # bound on Srho - Su (Lemma 4)
    L4 = np.all(drho >= np.sqrt(a / a.max()) * r - 1e-12) and np.all(drho <= np.sqrt(a / a.min()) * r + 1e-12)
    Lu, Lr = eu.max(), er.max()
    Lbound = max(rho[i, j] - du[i, j] for i, j, _ in Tu)
    return dict(ident=ident, mono=bool(np.all(eu <= er + 1e-12)), W1u=W1u, W1r=W1r, dW=W1r - W1u, btree=gap_tree,
                bunif=np.sqrt(Emax) / 2, decomp=decomp, nn_in_tree=nn_in_tree, p=len(mutual), L4=bool(L4),
                Gterm=(G - Msum) / (2 * (N - 1)), Sr=r.mean() / 2, Srho=drho.mean() / 2, Su=delta.mean() / 2,
                radb=radb, Lu=Lu, Lr=Lr, dL=Lr - Lu, Lb=Lbound, maxr=r.max(), maxdu=delta.max(),
                coslo=cos.min(), coshi=cos.max())

# ---- families of neighbourhoods; each query draws its own parameters so queries differ ----
def tangent(n, k, sig, p):
    """n bearings around k random centres, scatter sig in p random tangent dimensions."""
    cen = unit(rng.standard_normal((k, D - 1))); lab = rng.integers(0, k, n)
    P = unit(rng.standard_normal((p, D - 1))) if p < D - 1 else np.eye(D - 1)
    noise = rng.standard_normal((n, min(p, D - 1))) @ P / np.sqrt(min(p, D - 1))
    return cen[lab] + sig * noise, lab

def fam_random(lo, hi):
    return assemble(rng.uniform(lo, hi, N), rng.standard_normal((N, D - 1)))

def fam_clusters(lo, hi):
    T, _ = tangent(N, rng.integers(1, 4), rng.uniform(0.3, 1.5), int(rng.choice([2, 5, 20, 255])))
    return assemble(rng.uniform(lo, hi, N), T)

def fam_clustercos(lo, hi):              # each cluster sits at its own cosine level: radial gaps between groups
    k = rng.integers(2, 4); T, lab = tangent(N, k, rng.uniform(0.3, 1.5), int(rng.choice([2, 5, 20, 255])))
    lev = rng.uniform(lo, hi, k); w = (hi - lo) * 0.1
    return assemble(np.clip(lev[lab] + rng.uniform(-w, w, N), lo, hi), T)

def fam_radial(lo, hi):                  # adversarial: tight bearings, cosines spread -> tree edges are radial
    T, _ = tangent(N, 1, rng.uniform(0.02, 0.3), int(rng.choice([2, 20, 255])))
    return assemble(rng.uniform(lo, hi, N), T)

def fam_retrieval(lo=None, hi=None):     # top-50 of a small corpus; cosines come out of the geometry
    k = rng.integers(1, 4); c0 = rng.uniform(0.45, 0.8, k); p = int(rng.choice([5, 20, 255])); sig = rng.uniform(0.3, 0.9)
    cen = assemble(c0, rng.standard_normal((k, D - 1)))
    P = unit(rng.standard_normal((p, D))); X = []
    for j in range(k):
        X.append(unit(cen[j] + sig * rng.standard_normal((200, p)) @ P / np.sqrt(p)))
    X.append(unit(rng.standard_normal((400, D)))); X = np.vstack(X)
    return X[np.argsort(-X[:, 0])[:N]]

def run(fam, nq, lo=0.5, hi=0.8):
    return [analyse(fam(lo, hi)) for _ in range(nq)]

def summary(name, R):
    g = lambda k: np.array([x[k] for x in R], float)
    dW, bt, bu, W1u = g("dW"), g("btree"), g("bunif"), g("W1u")
    ok = dict(identity_err=g("ident").max(), monotone=g("mono").all(), decomp_err=g("decomp").max(),
              nn_in_tree=g("nn_in_tree").all(), L4=g("L4").all(), dW_le_btree=bool(np.all(dW <= bt + 1e-12)),
              dL_le_Lb=bool(np.all(g("dL") <= g("Lb") + 1e-12)))
    print(f"\n== {name}  ({len(R)} queries; cos range per query median [{np.median(g('coslo')):.2f}, {np.median(g('coshi')):.2f}])")
    print("   checks:", " ".join(f"{k}={v:.1e}" if isinstance(v, float) else f"{k}={v}" for k, v in ok.items()))
    t1 = dW / np.maximum(bt, 1e-15); t2 = dW / bu
    print(f"   W1(rho)-W1(u): median {np.median(dW):.4f}, max {dW.max():.4f}; relative to W1(u): median {np.median(dW/W1u):.3f}, max {(dW/W1u).max():.3f}")
    print(f"   tightness actual/tree-bound: median {np.median(t1):.2f}, min {t1.min():.2f};  actual/uniform sqrt(Emax)/2: median {np.median(t2):.3f}, max {t2.max():.3f}")
    print(f"   uniform bound sqrt(Emax)/2: median {np.median(bu):.3f}  vs W1(u) median {np.median(W1u):.3f}")
    gt = g("Gterm")
    print(f"   NN decomposition: mutual pairs p median {np.median(g('p')):.0f} (NN bars = N-p); (G-M)/(2(N-1)) median {np.median(gt):+.4f}, "
          f"range [{gt.min():+.4f}, {gt.max():+.4f}]; share of W1: median {np.median(gt/W1u):+.3f}")
    print(f"   Spearman W1(u) vs mean r/2 {spearman(W1u, g('Sr')):+.3f} | vs mean delta_rho/2 {spearman(W1u, g('Srho')):+.3f} "
          f"| vs mean delta_u/2 {spearman(W1u, g('Su')):+.3f} | W1(rho) vs W1(u) {spearman(W1u, g('W1r')):+.3f}")
    print(f"   Spearman Lmax(u) vs max r {spearman(g('Lu'), g('maxr')):+.3f} | vs max delta_u {spearman(g('Lu'), g('maxdu')):+.3f}; "
          f"Lmax(rho)-Lmax(u) median {np.median(g('dL')):.4f} max {g('dL').max():.4f}")
    # Error of W1(u) against the paper's NN statistic, split exactly into three named pieces (nn_bound.md, Theorem 2):
    #   W1u - N/(N-1) Sr = (G-M)/(2(N-1))  +  N/(N-1)(Su - Srho)  +  N/(N-1)(Srho - Sr)
    #                      gap minus mutual    radial distortion      asymmetric scaling a_i vs sqrt(a_i a_j)
    k = N / (N - 1); Y = k * g("Sr")
    pg, pr, pa = gt, k * (g("Su") - g("Srho")), k * (g("Srho") - g("Sr"))
    assert np.allclose(W1u - Y, pg + pr + pa)
    sd = W1u.std()
    print(f"   error pieces, median |piece| / SD_across_queries(W1u) [SD {sd:.4f}]: gap-mutual {np.median(np.abs(pg))/sd:.3f}, "
          f"radial {np.median(np.abs(pr))/sd:.3f} (bound {np.median(k*g('radb'))/sd:.3f}), asym {np.median(np.abs(pa))/sd:.3f}")
    # certificate: two queries are guaranteed concordant if their Y differ by more than the sum of their certified errors,
    # using the proved bound for the radial piece and the exact values of the other two (both computable from the data).
    eta = np.abs(pg) + k * g("radb") + np.abs(pa); n = len(Y); iu = np.triu_indices(n, 1)
    sep = np.abs(Y[:, None] - Y[None, :])[iu] > (eta[:, None] + eta[None, :])[iu]
    conc = (np.sign(W1u[:, None] - W1u[None, :]) == np.sign(Y[:, None] - Y[None, :]))[iu]
    print(f"   rank certificate: pairs guaranteed concordant {sep.mean():.3f}; observed concordant {conc.mean():.3f}")

def spread_sweep(nq):
    print("\n== spread sweep (clusters family, cosines in [0.65-w/2, 0.65+w/2])")
    for w in [0.0, 0.1, 0.2, 0.3, 0.5]:
        R = run(fam_clusters, nq, 0.65 - w / 2, 0.65 + w / 2)
        g = lambda k: np.array([x[k] for x in R], float)
        print(f"   w={w:.1f}: (W1rho-W1u)/W1u median {np.median(g('dW')/g('W1u')):.3f} max {(g('dW')/g('W1u')).max():.3f}; "
              f"sqrt(Emax)/2 {np.median(g('bunif')):.3f}; Spearman W1u~mean r {spearman(g('W1u'), g('Sr')):+.3f}, ~mean delta_rho {spearman(g('W1u'), g('Srho')):+.3f}")

if __name__ == "__main__":
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    for name, fam in [("random bearings", fam_random), ("clustered bearings", fam_clusters),
                      ("clusters at own cosine levels", fam_clustercos), ("radial (adversarial)", fam_radial)]:
        summary(name + ", cos in [0.5, 0.8]", run(fam, nq))
    summary("retrieval top-50 (cosines from geometry)", run(fam_retrieval, nq))
    spread_sweep(max(nq // 2, 50))
