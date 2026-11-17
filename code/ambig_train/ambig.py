# Shipped copy of paper/ambig_train/ambig.py; only plan paths (PLAN.md -> docs/ambig_train_plan.md), file paths and environment names in comments changed for the repo layout.
# ambig.py -- written 2026-11-01 10:21 CET. Human-labelled check on AmbigNQ train (docs/ambig_train_plan.md). Three steps:
#   retrieve  (needs sentence-transformers):  python ambig.py retrieve DATA_DIR REAL_DIR OUT_DIR
#             embeds the 10,036 train questions (all-MiniLM-L6-v2), retrieves the top 50 of the registered corpus.
#   feats     (needs ripser):                 python ambig.py feats OUT_DIR
#             per question: persistence features and controls, cosine and similarity statistics, coverage gain.
#   auroc     (needs pandas):                 python ambig.py auroc OUT_DIR
# Feature definitions follow code/realtext/features.py and code/lenctl/feats.py.
import ast, json, os, sys
from pathlib import Path
import numpy as np

K, NDRAW, EPSM = 50, 5, np.finfo(float).eps


def retrieve(data, real, out):
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    T = json.load(open(Path(data) / "ambignq" / "train.json"))
    lab = np.array([int(any(a["type"] == "multipleQAs" for a in (ast.literal_eval(t["annotations"]) if isinstance(t["annotations"], str) else t["annotations"]))) for t in T])
    qs = [t["question"] for t in T]
    B = pd.read_parquet(Path(real) / "chunks.parquet"); keep = np.ones(len(B), bool) if os.environ.get("FULL") == "1" else B.in_corpus.to_numpy()  # 2026-11-03 20:20 CET: FULL=1 = every chunk of the gold pages, not the first 5
    E = np.load(Path(real) / "chunk_emb.npy")[keep].astype(np.float32); titles = B.title[keep].to_numpy()
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    Q = m.encode(qs, batch_size=64, normalize_embeddings=True).astype(np.float32)
    idx = np.stack([np.argsort(-(E @ q), kind="stable")[:K] for q in Q])
    np.savez_compressed(out / "retrieved.npz", Q=Q, idx=idx, label=lab, nwords=np.array([len(q.split()) for q in qs]))
    np.save(out / "corpus_emb.npy", E); np.save(out / "corpus_titles.npy", titles.astype(str))
    print("questions", len(qs), "ambiguous", lab.mean().round(3), "corpus", E.shape, flush=True)


def flip_retrieve(data, real, out):
    """Amendment 1 (11:15 CET): for each ambiguous train question with >= 2 disambiguated questions, three queries:
    0 = the original (polysemic), 1 = all disambiguations joined (multi-factual), 2 = the first disambiguation (single).
    Writes OUT/retrieved.npz in the same layout, with label = query type and 'group' = source question."""
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    T = json.load(open(Path(data) / "ambignq" / "train.json"))
    qs, lab, grp = [], [], []
    for k, t in enumerate(T):
        ann = ast.literal_eval(t["annotations"]) if isinstance(t["annotations"], str) else t["annotations"]
        dq = [p["question"] for a in ann if a["type"] == "multipleQAs" for p in a["qaPairs"]]
        if len(dq) >= 2:
            for typ, q in enumerate((t["question"], " ".join(dq), dq[0])):
                qs.append(q); lab.append(typ); grp.append(k)
    B = pd.read_parquet(Path(real) / "chunks.parquet"); keep = np.ones(len(B), bool) if os.environ.get("FULL") == "1" else B.in_corpus.to_numpy()  # 2026-11-03 20:20 CET: FULL=1 = every chunk of the gold pages, not the first 5
    E = np.load(Path(real) / "chunk_emb.npy")[keep].astype(np.float32); titles = B.title[keep].to_numpy()
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    Q = m.encode(qs, batch_size=64, normalize_embeddings=True).astype(np.float32)
    idx = np.stack([np.argsort(-(E @ q), kind="stable")[:K] for q in Q])
    np.savez_compressed(out / "retrieved.npz", Q=Q, idx=idx, label=np.array(lab), group=np.array(grp),
                        nwords=np.array([len(q.split()) for q in qs]))
    np.save(out / "corpus_emb.npy", E); np.save(out / "corpus_titles.npy", titles.astype(str))
    print("source questions", len(set(grp)), "queries", len(qs), flush=True)


def flip_auroc(out):
    import pandas as pd
    F = pd.read_csv(Path(out) / "feats.csv"); rng = np.random.default_rng(0)
    cols = ["w1", "longbar", "ltmax", "w1_corp", "longbar_corp", "top1", "mean_cos", "nn_med", "coherence", "eig2", "cov_gain_diff", "nwords"]
    def auc(x, y):
        r = pd.Series(np.r_[x, y]).rank().to_numpy(); n0, n1 = len(x), len(y)
        return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)
    L = [f"# flip_auroc: AmbigNQ train, {F.label.eq(0).sum()} source questions; AUROC = P(first type > second type), 95% bootstrap over source questions"]
    for a_, b_, name in ((0, 1, "multi-factual vs polysemic"), (2, 1, "multi-factual vs single"), (2, 0, "polysemic vs single")):
        A, Bq = F[F.label == a_].sort_values("i"), F[F.label == b_].sort_values("i")
        L.append(f"## {name}")
        for c in cols:
            x, y = A[c].to_numpy(), Bq[c].to_numpy(); v = auc(x, y)
            bs = [auc(x[ix], y[ix]) for ix in (rng.integers(0, len(x), len(x)) for _ in range(1000))]
            L.append(f"  {c:14s} {v:.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
    (Path(out) / "auroc.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def ph(U):
    from ripser import ripser
    dg = ripser(U, maxdim=1)["dgms"]
    h0 = dg[0][:, 1]; h0 = h0[np.isfinite(h0)]; lt = dg[1][:, 1] - dg[1][:, 0]
    return float(np.mean(h0 / 2.0)), (float(np.max(lt) / 2.0) if len(lt) else 0.0), float(np.max(h0))


def control(q, a, V):
    V = unit(V - (V @ q)[:, None] * q[None, :])
    return ph(unit(a[:, None] * q[None, :] + np.sqrt(1 - a ** 2)[:, None] * V - q[None, :] + EPSM))


def one(i):
    q = R["Q"][i].astype(np.float64); top = R["idx"][i]; X = E[top].astype(np.float64); a = np.clip(X @ q, -1, 1)
    r = dict(i=i, label=int(R["label"][i]), nwords=int(R["nwords"][i]))
    r["w1"], r["ltmax"], r["longbar"] = ph(unit(X - q[None, :] + EPSM))
    r.update(top1=a[0], mean_cos=a.mean(), std_cos=a.std(), slope=np.polyfit(np.arange(1, K + 1), a, 1)[0])
    G = X @ X.T; iu = np.triu_indices(K, 1); Gn = G.copy(); np.fill_diagonal(Gn, -np.inf)
    ev = np.sort(np.linalg.eigvalsh(G))[::-1]
    r.update(nn_med=float(np.median(Gn.max(1))), coherence=float(G[iu].mean()), eig2=float(ev[1] / ev.sum()))
    S = (a[:, None] + a[None, :]) / np.sqrt(np.maximum(2 + 2 * G, 1e-12)); np.fill_diagonal(S, -np.inf)
    tt = TITLES[top]; diff = tt[:, None] != tt[None, :]
    r.update(cov_gain=float(S.max() - a.max()), cov_gain_diff=float(np.where(diff, S, -np.inf).max() - a.max()))
    rng = np.random.default_rng((i, 3)); pool = rng.choice(len(E), 400, replace=False); pool = np.setdiff1d(pool, top)[:K * NDRAW]
    cr = np.array([control(q, a, rng.standard_normal((K, len(q)))) for _ in range(NDRAW)]).mean(0)
    cc = np.array([control(q, a, E[pool[d * K:(d + 1) * K]].astype(np.float64)) for d in range(NDRAW)]).mean(0)
    for j, f in enumerate(("w1", "ltmax", "longbar")):
        r[f"{f}_rand"], r[f"{f}_corp"] = cr[j], cc[j]
    return r


def load(out):  # 11:02 CET: worker initializer (macOS spawns workers, so globals are not inherited)
    global R, E, TITLES
    out = Path(out)
    R = dict(np.load(out / "retrieved.npz")); E = np.load(out / "corpus_emb.npy"); TITLES = np.load(out / "corpus_titles.npy")


def feats(out):
    import pandas as pd
    from multiprocessing import Pool
    load(out); out = Path(out)
    with Pool(8, initializer=load, initargs=(str(out),)) as p:
        rows = p.map(one, range(len(R["Q"])), chunksize=64)
    pd.DataFrame(rows).to_csv(out / "feats.csv", index=False); print("rows", len(rows), flush=True)


def auroc(out):
    import pandas as pd
    F = pd.read_csv(Path(out) / "feats.csv"); y = F.label.to_numpy()
    def auc(x):
        r = pd.Series(x).rank().to_numpy(); n1 = y.sum(); n0 = len(y) - n1
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)
    rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(2000)]
    cols = ["w1", "longbar", "ltmax", "w1_corp", "longbar_corp", "ltmax_corp", "w1_rand", "longbar_rand", "top1", "mean_cos",
            "std_cos", "slope", "nn_med", "coherence", "eig2", "cov_gain", "cov_gain_diff", "nwords"]
    L = [f"# ambig.py auroc -- AmbigNQ train, n = {len(y)}, ambiguous {y.mean():.3f}; AUROC = P(ambiguous > clear), 95% bootstrap over questions"]
    res = {}
    for c in cols:
        x = F[c].to_numpy(); a = auc(x)
        bs = []
        for b in B:
            yb, xb = y[b], x[b]; r = pd.Series(xb).rank().to_numpy(); n1 = yb.sum(); n0 = len(yb) - n1
            bs.append((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))
        lo, hi = np.percentile(bs, [2.5, 97.5]); res[c] = (a, lo, hi, np.array(bs))
        L.append(f"{c:15s} {a:.3f} [{lo:.3f}, {hi:.3f}]  oriented {max(a, 1 - a):.3f}")
    for f in ("w1", "longbar", "ltmax"):
        d = res[f][3] - res[f + "_corp"][3]
        L.append(f"{f} real - corpus-bearing control: {res[f][0] - res[f + '_corp'][0]:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
    ori = lambda c: max(res[c][0], 1 - res[c][0])
    bp = max(("w1", "longbar", "ltmax"), key=ori); bs_ = max(("nn_med", "coherence", "eig2"), key=ori)
    L.append(f"A1: best persistence {bp} {ori(bp):.3f} vs best similarity {bs_} {ori(bs_):.3f}: difference {ori(bp) - ori(bs_):+.3f}")
    L.append(f"A4: question length oriented {ori('nwords'):.3f}; best persistence {ori(bp):.3f}")
    (Path(out) / "auroc.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))


def _toks(t):
    import re
    return re.findall(r"\w+", t.lower())


def _dqpp(i):  # Dense-QPP for question i: mean average overlap, per (noise std, depth K), over the shared noises
    q = R["Q"][i].astype(np.float32); D = DQ["K"][-1]
    def top(v):
        s = E @ v; p = np.argpartition(-s, D)[:D]; return p[np.argsort(-s[p], kind="stable")]
    base = top(q); posA = np.full(len(E), D, np.int32); posA[base] = np.arange(D)
    out = np.zeros((len(DQ["s"]), len(DQ["K"])))
    for a, s in enumerate(DQ["s"]):
        S = E @ (q[:, None] + s * DQ["Z"].T)                                   # all 30 noisy queries at once
        P = np.argpartition(-S, D, axis=0)[:D]
        for c in range(S.shape[1]):
            b = P[np.argsort(-S[P[:, c], c], kind="stable"), c]; posB = np.full(len(E), D, np.int32); posB[b] = np.arange(D)
            u = np.union1d(base, b); m = np.maximum(posA[u], posB[u])           # item counted in both prefixes from depth m+1
            ov = np.cumsum(np.bincount(m[m < D], minlength=D))                  # |A_d ∩ B_d| for d = 1..D
            ao = np.cumsum(ov / np.arange(1, D + 1))                            # running sum of overlap ratios
            out[a] += np.array([ao[k - 1] / k for k in DQ["K"]]) / len(DQ["Z"])
    return out


def _dq_init(out):
    global DQ
    load(out); rng = np.random.default_rng(0)
    DQ = dict(s=[0.01 * j for j in range(1, 10)], K=list(range(100, 1001, 100)), Z=rng.standard_normal((30, R["Q"].shape[1])).astype(np.float32))


def qpp(out, real):
    """Amendment 2 of docs/ambig_train_plan.md (2026-11-14 09:32 CET): query-performance predictors on the same retrieval.
    Usage: [FULL=1] python ambig.py qpp OUT_DIR REAL_DIR  ->  OUT_DIR/qpp.csv (one row per question; same i as feats.csv)"""
    import pandas as pd
    from collections import Counter
    from multiprocessing import Pool
    load(out); out = Path(out)
    B = pd.read_parquet(Path(real) / "chunks.parquet"); keep = np.ones(len(B), bool) if os.environ.get("FULL") == "1" else B.in_corpus.to_numpy()
    T = B.text[keep].tolist(); assert len(T) == len(E), (len(T), len(E))
    toks = [_toks(t) for t in T]; cf = Counter(); df = Counter()
    for tk in toks: cf.update(tk); df.update(set(tk))
    V, NC, ND = len(cf), sum(cf.values()), len(T)
    vocab = {w: j for j, w in enumerate(cf)}; logPC = np.log((np.array(list(cf.values())) + 1.0) / (NC + V)); SlogPC = logPC.sum()
    qs = None
    if Path(real, "..", "data", "ambignq", "train.json").exists():
        qs = [t["question"] for t in json.load(open(Path(real, "..", "data", "ambignq", "train.json")))]
    assert qs is not None and len(qs) == len(R["Q"])
    cen = E.astype(np.float64).mean(0); rows = []
    for i in range(len(R["Q"])):
        q = R["Q"][i].astype(np.float64); top = R["idx"][i]; X = E[top].astype(np.float64); s = X @ q; sC = float(q @ cen)
        U = unit(X - q[None, :] + EPSM)
        # amendment 2a: provenance of each formula is in docs/ambig_train_plan.md
        W = X @ X.T; off = W[np.triu_indices(K, 1)]; A = (W > off.mean()) & ~np.eye(K, dtype=bool); deg = A.sum(1)
        from scipy.sparse.csgraph import connected_components
        r = dict(i=i, label=int(R["label"][i]), nqc=s.std() / np.abs(s).mean(), nqc_cen=s.std() / abs(sC), wig_dense=s.mean() - sC,
                 lsmv=s[0] * s.std(), sigma_max=max(s[:k].std() for k in range(2, K + 1)), gap=s[0] - s[1],
                 spread_R=float(np.linalg.norm(U.mean(0))), ac_embs=float(np.corrcoef(s, W @ s)[0, 1]),
                 wand=float(np.mean([A[j] @ deg / deg[j] if deg[j] else 0.0 for j in range(K)])),
                 density=float(A.sum() / (K * (K - 1))), components=int(connected_components(A, directed=False)[0]))
        c = Counter(w for j in top for w in toks[j]); nt = sum(c.values()); p0 = 1.0 / (nt + V)  # Clarity, add-one smoothed, over the corpus vocabulary
        ids = np.array([vocab[w] for w in c]); pt = (np.array(list(c.values())) + 1.0) / (nt + V)
        r["clarity"] = float((pt * (np.log(pt) - logPC[ids])).sum() + p0 * ((V - len(ids)) * np.log(p0) - (SlogPC - logPC[ids].sum())))
        qt = [w for w in _toks(qs[i]) if w in df]
        idf = [np.log(ND / df[w]) for w in qt]; scq = [(1 + np.log(cf[w])) * np.log(1 + ND / df[w]) for w in qt]
        r.update(avg_idf=np.mean(idf) if idf else 0.0, max_idf=max(idf) if idf else 0.0, avg_scq=np.mean(scq) if scq else 0.0, max_scq=max(scq) if scq else 0.0)
        rows.append(r)
    F = pd.DataFrame(rows)
    with Pool(8, initializer=_dq_init, initargs=(str(out),)) as p:
        dq = np.stack(p.map(_dqpp, range(len(R["Q"])), chunksize=16))       # (n, 9 noise levels, 10 depths)
    _dq_init(str(out))
    for a, s in enumerate(DQ["s"]):
        for b, k in enumerate(DQ["K"]):
            F[f"dqpp_s{s:.2f}_K{k}"] = dq[:, a, b]
    F.to_csv(out / "qpp.csv", index=False); print("rows", len(F), "vocab", V, "chunks", ND, flush=True)


def qpp_report(out):
    """Amendment 2/2a: AUROC (2,000 bootstrap draws over questions, seed 0) and Mann-Whitney U for every predictor in
    qpp.csv; Dense-QPP's (s, K) chosen on a random half (seed 0) by |AUROC - 0.5| and reported on the other half only.
    Usage: python ambig.py qpp_report OUT_DIR  ->  OUT_DIR/qpp.txt"""
    import pandas as pd
    from scipy.stats import mannwhitneyu, norm
    F = pd.read_csv(Path(out) / "qpp.csv"); y = F.label.to_numpy().astype(bool)
    def row(name, x, yy, rng):
        u = mannwhitneyu(x[yy], x[~yy], alternative="two-sided", method="asymptotic"); n1, n0 = yy.sum(), (~yy).sum(); a = u.statistic / (n1 * n0)
        bs = []
        for _ in range(2000):
            b = rng.integers(0, len(yy), len(yy)); xb, yb = x[b], yy[b]; r = pd.Series(xb).rank().to_numpy(); m1 = yb.sum()
            bs.append((r[yb].sum() - m1 * (m1 + 1) / 2) / (m1 * (len(yb) - m1)))
        lo, hi = np.percentile(bs, [2.5, 97.5]); z = norm.isf(u.pvalue / 2) * np.sign(a - 0.5)
        return f"{name:22s} {a:.3f} [{lo:.3f}, {hi:.3f}]  oriented {max(a, 1 - a):.3f}  U = {u.statistic:.0f}, z = {z:+.2f}, p = {u.pvalue:.2g}"
    L = [f"# ambig.py qpp_report -- AmbigNQ train, n1 = {y.sum()}, n2 = {(~y).sum()}; AUROC = P(ambiguous > clear) [95% bootstrap]"]
    for c in [c for c in F.columns if c not in ("i", "label") and not c.startswith("dqpp_")]:
        L.append(row(c, F[c].to_numpy(float), y, np.random.default_rng(0)))
    half = np.random.default_rng(0).permutation(len(y)) < len(y) // 2               # True = tuning half
    dq = [c for c in F.columns if c.startswith("dqpp_")]
    def auc(x, yy):
        r = pd.Series(x).rank().to_numpy(); m1 = yy.sum(); return (r[yy].sum() - m1 * (m1 + 1) / 2) / (m1 * (len(yy) - m1))
    best = max(dq, key=lambda c: abs(auc(F[c].to_numpy()[half], y[half]) - 0.5))
    L.append(f"Dense-QPP: (s, K) chosen on the tuning half ({half.sum()} questions) = {best}; held-out half ({(~half).sum()}):")
    L.append(row("  " + best, F[best].to_numpy(float)[~half], y[~half], np.random.default_rng(0)))
    L.append(f"  range over all {len(dq)} variants on the held-out half: "
             f"{min(auc(F[c].to_numpy()[~half], y[~half]) for c in dq):.3f} to {max(auc(F[c].to_numpy()[~half], y[~half]) for c in dq):.3f}")
    (Path(out) / "qpp.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))


def tests(out):
    """2026-11-14 09:29 CET (the journal asks for a named test, its df and p). Adds nothing new: the Mann-Whitney U of each
    statistic, with U/(n1*n2) = the AUROC printed by `auroc` (asserted), and DeLong's paired test for real - control.
    Usage: python ambig.py tests OUT_DIR  ->  OUT_DIR/tests.txt"""
    import pandas as pd
    from scipy.stats import mannwhitneyu, norm
    F = pd.read_csv(Path(out) / "feats.csv"); y = F.label.to_numpy().astype(bool); n1, n0 = y.sum(), (~y).sum()
    def placements(x):  # DeLong structural components: V10 per ambiguous, V01 per clear question
        xp, xn = x[y], x[~y]; r = pd.Series(np.r_[xp, xn]).rank().to_numpy()
        rp, rn = pd.Series(xp).rank().to_numpy(), pd.Series(xn).rank().to_numpy()
        v10 = (r[:n1] - rp) / n0; v01 = 1 - (r[n1:] - rn) / n1
        return v10.mean(), v10, v01
    L = [f"# ambig.py tests -- AmbigNQ, n1 = {n1} ambiguous, n2 = {n0} clear; two-sided p; AUROC = U/(n1*n2)"]
    for c in ["w1", "longbar", "ltmax", "top1", "mean_cos", "std_cos", "slope", "nn_med", "coherence", "eig2",
              "cov_gain", "cov_gain_diff", "nwords"]:
        x = F[c].to_numpy(); r = mannwhitneyu(x[y], x[~y], alternative="two-sided", method="asymptotic")
        a = r.statistic / (n1 * n0); assert abs(a - placements(x)[0]) < 1e-12
        z = norm.isf(r.pvalue / 2) * np.sign(a - 0.5)
        L.append(f"{c:14s} AUROC {a:.3f}  U = {r.statistic:.0f}, z = {z:+.2f}, p = {r.pvalue:.2g}")
    for f in ("w1", "longbar", "ltmax"):
        a1, v10a, v01a = placements(F[f].to_numpy()); a2, v10b, v01b = placements(F[f + "_corp"].to_numpy())
        var = np.cov(v10a, v10b)[0, 0] / n1 + np.cov(v01a, v01b)[0, 0] / n0 + np.cov(v10a, v10b)[1, 1] / n1 \
            + np.cov(v01a, v01b)[1, 1] / n0 - 2 * (np.cov(v10a, v10b)[0, 1] / n1 + np.cov(v01a, v01b)[0, 1] / n0)
        z = (a1 - a2) / np.sqrt(var)
        L.append(f"{f} real - corpus-bearing control: {a1 - a2:+.3f}, DeLong z = {z:+.2f}, p = {2 * norm.sf(abs(z)):.2g}")
    (Path(out) / "tests.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    {"retrieve": retrieve, "feats": feats, "auroc": auroc, "tests": tests, "qpp": qpp, "qpp_report": qpp_report, "flip_retrieve": flip_retrieve,
     "flip_auroc": flip_auroc}[sys.argv[1]](*sys.argv[2:])
