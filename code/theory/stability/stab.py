# Copied 2026-11-02 08:59 CET from the numerical-stability review (written 08:53-08:55 CET); unchanged below this line.
# Numerical-stability probes on real neighbourhoods (numpy only). Cell Q250|C750, span-overlap exclusion, K=50,
# NO letter-spacing clean filter (features.py CLEAN=1 not replicated). Usage: python3 -I stab.py <corpusdir> <model> [qmax]
import sys, json, numpy as np
base, model = sys.argv[1], sys.argv[2]; QMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 400
K = 50; EPSM = np.finfo(float).eps
def meta(g):
    rows = [json.loads(l) for l in open(f"{base}/data/chunks_c{g}.jsonl")]
    return (np.array([r["lecture"] for r in rows], dtype=object), np.array([r["start"] for r in rows]), np.array([r["end"] for r in rows]))
L2, s2, e2 = meta(250); L7, s7, e7 = meta(750)
P250 = np.load(f"{base}/emb/{model}/passage_c250.npy"); P750 = np.load(f"{base}/emb/{model}/passage_c750.npy")
assert P250.dtype == np.float32
unit = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
Q64, C64 = unit(P250.astype(np.float64)), unit(P750.astype(np.float64))
Q32, C32 = unit(P250), unit(P750)
S = Q64 @ C64.T
excl = (L2[:, None] == L7[None, :]) & (s2[:, None] < e7[None, :]) & (s7[None, :] < e2[:, None])
S = np.where(excl, -np.inf, S); order = np.argsort(-S, axis=1, kind="stable")
# corpus duplicates
DC_all = np.sqrt(np.clip(2 - 2 * C64 @ C64.T, 0, None)); np.fill_diagonal(DC_all, np.inf)
print(f"{model}: C750 n={len(C64)} D={C64.shape[1]}  min pairwise |ci-cj| = {DC_all.min():.2e}, pairs < 1e-3: {int((DC_all<1e-3).sum()//2)}, < 0.05: {int((DC_all<0.05).sum()//2)}")
def offsets(q, C): U = C - q + EPSM; return U / np.linalg.norm(U, axis=1, keepdims=True)
def pd_(X):
    G = X @ X.T; n = np.diag(G); D2 = n[:, None] + n[None, :] - 2 * G; return np.sqrt(np.clip(D2, 0, None))
def pd_direct(X):  # numerically careful (like pdist): differences
    return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
def prim(D):
    N = len(D); intree = np.zeros(N, bool); intree[0] = True; best = D[0].copy(); par = np.zeros(N, int); E = []
    best[0] = np.inf
    for _ in range(N - 1):
        b = np.where(intree, np.inf, best); j = int(np.argmin(b)); E.append((min(j, par[j]), max(j, par[j]), b[j])); intree[j] = True
        upd = (~intree) & (D[j] < best); best[upd] = D[j][upd]; par[upd] = j
    return E
def analyse(D):
    N = len(D); E = prim(D); Dn = D.copy(); np.fill_diagonal(Dn, np.inf); nn = Dn.argmin(1); srt = np.sort(Dn, 1)
    NN = {(min(i, nn[i]), max(i, nn[i])) for i in range(N)}; T = {(i, j) for i, j, _ in E}
    w = np.array([x for *_, x in E])
    # MST margin: min over non-tree edges of w(e) - max edge on tree path
    adj = {i: [] for i in range(N)}
    for i, j, x in E: adj[i].append((j, x)); adj[j].append((i, x))
    pm = np.zeros((N, N))
    for s in range(N):
        st = [(s, -1, 0.0)]
        while st:
            v, p, m = st.pop(); pm[s, v] = m
            for u, x in adj[v]:
                if u != p: st.append((u, v, max(m, x)))
    iu = np.triu_indices(N, 1); nt = np.array([(i, j) not in T for i, j in zip(*iu)])
    mst_margin = (D[iu] - pm[iu])[nt].min()
    nn_margin = (srt[:, 1] - srt[:, 0]).min()
    lone = np.isclose(w.max(), srt[:, 0].max())
    return dict(T=T, NN=NN, nnsh=len(T & NN) / len(T), W1=w.sum() / (2 * (N - 1)), Lmax=w.max(), mst_margin=mst_margin,
                nn_margin=nn_margin, lone=lone, sdmin=srt[:, 0], lgap=w.max()-srt[:, 0].max())
rng = np.random.default_rng(0)
nq = min(QMAX, len(Q64)); res = []
for i in range(nq):
    idx = order[i, :K]; a = S[i, idx]; q = Q64[i]; X = C64[idx]
    U = offsets(q, X); DU = pd_direct(U); r = analyse(DU)
    A = np.sqrt(2 - 2 * a); DC = pd_direct(X); np.fill_diagonal(DC, np.inf); ratio = (DC.min(1) / A).mean()
    # float32 pipeline end to end
    U32 = (C32[idx] - Q32[i]); U32 = U32 / np.linalg.norm(U32, axis=1, keepdims=True); r32 = analyse(pd_direct(U32).astype(np.float64))
    # Gram-formula distances (cancellation) in float64
    rg = analyse(pd_(U))
    out = dict(cmax=a.max(), cmin=a.min(), mst_margin=r["mst_margin"], nn_margin=r["nn_margin"], nnsh=r["nnsh"], W1=r["W1"], Lmax=r["Lmax"],
               ratio=ratio, lone=r["lone"], lgap=r["lgap"], f32_T=r32["T"] != r["T"], f32_NN=r32["NN"] != r["NN"], f32_dnnsh=r32["nnsh"] - r["nnsh"],
               f32_dW1=abs(r32["W1"] - r["W1"]), gram_T=rg["T"] != r["T"], gram_dW1=abs(rg["W1"] - r["W1"]))
    for eta in (1e-4, 1e-3, 1e-2):  # relative embedding noise, renormalised
        Xp = unit(X + eta * rng.standard_normal(X.shape) / np.sqrt(X.shape[1])); qp = unit(q + eta * rng.standard_normal(q.shape) / np.sqrt(len(q)))
        rp = analyse(pd_direct(offsets(qp, Xp)))
        out[f"T{eta}"] = rp["T"] != r["T"]; out[f"dnnsh{eta}"] = abs(rp["nnsh"] - r["nnsh"]); out[f"dW1{eta}"] = abs(rp["W1"] - r["W1"])
        out[f"dL{eta}"] = abs(rp["Lmax"] - r["Lmax"]); out[f"lone{eta}"] = rp["lone"] != r["lone"]
    # retrieval discontinuity: swap rank 50 for rank 51 (one passage leaves, one enters)
    idx2 = np.r_[order[i, :K - 1], order[i, K]]; a2 = S[i, idx2]; X2 = C64[idx2]; r2 = analyse(pd_direct(offsets(q, X2)))
    DC2 = pd_direct(X2); np.fill_diagonal(DC2, np.inf); ratio2 = (DC2.min(1) / np.sqrt(2 - 2 * a2)).mean()
    out.update(gap5051=S[i, order[i, K - 1]] - S[i, order[i, K]], sw_dW1=abs(r2["W1"] - r["W1"]), sw_dL=abs(r2["Lmax"] - r["Lmax"]),
               sw_dratio=abs(ratio2 - ratio), sw_dnnsh=abs(r2["nnsh"] - r["nnsh"]))
    res.append(out)
g = lambda k: np.array([o[k] for o in res], dtype=float)
sdW, sdL, sdR = g("W1").std(), g("Lmax").std(), g("ratio").std()
print(f"  queries {nq}; cos range in top-50: min {g('cmin').min():.3f} median-max {np.median(g('cmax')):.3f} max {g('cmax').max():.3f}; a_min={np.sqrt(2-2*g('cmax').max()):.3f}")
print(f"  MST margin (min over non-tree edges, per query): min {g('mst_margin').min():.1e} 1% {np.quantile(g('mst_margin'),.01):.1e} median {np.median(g('mst_margin')):.1e}")
print(f"  NN margin (min over points 2nd-1st NN): min {g('nn_margin').min():.1e} 1% {np.quantile(g('nn_margin'),.01):.1e} median {np.median(g('nn_margin')):.1e}")
print(f"  float32 pipeline: MST edge set differs {g('f32_T').mean():.3f}, NN set differs {g('f32_NN').mean():.3f}, max|d nnsh| {np.abs(g('f32_dnnsh')).max():.3f}, max|dW1| {g('f32_dW1').max():.1e}")
print(f"  Gram-formula distances (f64): MST set differs {g('gram_T').mean():.3f}, max|dW1| {g('gram_dW1').max():.1e}")
print(f"  SD across queries: W1 {sdW:.4f} Lmax {sdL:.4f} ratio {sdR:.4f}; median nnsh {np.median(g('nnsh')):.3f}; lone share {g('lone').mean():.3f}")
for eta in (1e-4, 1e-3, 1e-2):
    print(f"  noise {eta:g}: MST set changes {g(f'T{eta}').mean():.3f}; |d nnsh| median {np.median(g(f'dnnsh{eta}')):.3f} max {g(f'dnnsh{eta}').max():.3f}; "
          f"|dW1|/SD max {g(f'dW1{eta}').max()/sdW:.4f}; |dLmax|/SD max {g(f'dL{eta}').max()/sdL:.4f}; lone flips {g(f'lone{eta}').mean():.3f}")
print(f"  rank-50/51 swap (cos gap median {np.median(g('gap5051')):.1e}): |dW1|/SD median {np.median(g('sw_dW1'))/sdW:.3f} 95% {np.quantile(g('sw_dW1'),.95)/sdW:.3f}; "
      f"|dLmax|/SD median {np.median(g('sw_dL'))/sdL:.3f} 95% {np.quantile(g('sw_dL'),.95)/sdL:.3f} max {g('sw_dL').max()/sdL:.2f}; share Lmax changes >0: {(g('sw_dL')>1e-12).mean():.3f}; "
      f"|dratio|/SD median {np.median(g('sw_dratio'))/sdR:.3f} 95% {np.quantile(g('sw_dratio'),.95)/sdR:.3f}; |dnnsh| median {np.median(g('sw_dnnsh')):.3f}")
lg = g("lgap"); nl = lg[lg > 0]
print(f"  Lmax - max NN dist on non-lone queries: min {nl.min() if len(nl) else np.nan:.1e}, count < 1e-5 (isclose would mislabel): {(nl<1e-5).sum()}; exact-zero (lone) {np.mean(lg==0):.3f}")
