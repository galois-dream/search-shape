# Shipped copy of paper/trec_p2/p2.py; only the ambigqa import (evidence/ambigqa.py -> ./ambigqa.py) and environment names in comments changed for the repo layout (PREREG.md deviation 7 = docs/analysis_plan_deviation7.md; simulation env = numpy, scipy, scikit-learn, pandas, ripser).
# p2.py -- written 2026-11-14 09:38 CET. P2 (PREREG.md deviation 7): TREC Web track 2009-2012 diversity topics, NIST type
# ambiguous (1) vs faceted (0). Steps (labels are read by `fit` only; every earlier step reads topic number and query):
#   bm25  TOPICS_TSV INDEX_DB DOCS_DB OUT   env: sentence-transformers + Blowfish. Top 20 pages per topic by BM25 (FTS5), disambiguation pages
#                                           excluded (OUT/primary) and included (OUT/withdis); pooled pages.parquet.
#   corpus_a REAL_DIR DOCS_DB INDEX_DB OUT   env: sentence-transformers + Blowfish. Check corpus A (registered corpus + deviation 6(c)'s 50,000 random
#                                           chunks) in OUT/A; reads no TREC data.
#   prep  SUB TOPICS_TSV MODEL               env: sentence-transformers + Blowfish. Chunks and embeds SUB's pages as the registered corpus
#                                           (ambigqa.py), embeds the queries, retrieves the top 50 chunks.
#                                           Writes SUB/MODEL/{retrieved.npz, corpus_emb.npy}.
#   feats SUB/MODEL                         simulation env. Per topic: M0 and M1 covariates, corpus-bearing control
#                                           (20 draws, averaged) and bearing swap (seed 0). Writes feats.csv.
#   fit   SUB/MODEL TOPICS_TSV              simulation env. M0 vs M1, 10-fold CV x 20, bootstrap CIs, permutation test,
#                                           Mann-Whitney U of the longest bar. Writes result.txt.
# TOPICS_TSV columns: number, query, type (type is read only by `fit`).
import os, sys, json
from pathlib import Path
import numpy as np

K, NPAGES, NCORP = 50, 20, 20
HERE = Path(__file__).resolve().parent
MODELS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "mpnet": "sentence-transformers/all-mpnet-base-v2"}


def _topics(tsv, with_type=False):
    import pandas as pd
    T = pd.read_csv(tsv, sep="\t", dtype={"number": int, "query": str, "type": str})
    T = T[["number", "query"] + (["type"] if with_type else [])].sort_values("number")
    return T.drop_duplicates("query", keep="first").reset_index(drop=True)


def bm25(tsv, index_db, docs_db, out):
    import html, sqlite3, re, pandas as pd
    out = Path(out); T = _topics(tsv); I = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
    S = sqlite3.connect(f"file:{docs_db}?mode=ro", uri=True); rows = []
    for num, q in zip(T.number, T["query"]):
        m = " OR ".join(f'"{w}"' for w in re.findall(r"\w+", q.lower()))
        ids = [r[0] for r in I.execute("SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT 200", (m,))]  # rank = bm25
        meta = {i: (t, d) for i, t, d in I.execute(f"SELECT id, title, disamb FROM meta WHERE id IN ({','.join('?' * len(ids))})", ids)}
        hits = [(i, *meta[i]) for i in ids]
        for arm, keep in (("primary", lambda d: d == 0), ("withdis", lambda d: True)):
            for r, (pid, title, dis) in enumerate([h for h in hits if keep(h[2])][:NPAGES]):
                rows.append(dict(arm=arm, number=num, rank=r, page_id=pid, title=html.unescape(title), disamb=dis))
    B = pd.DataFrame(rows); B.to_csv(out / "bm25.csv", index=False)
    for arm in ("primary", "withdis"):
        d = out / arm; d.mkdir(parents=True, exist_ok=True); P = B[B.arm == arm].drop_duplicates("page_id")
        txt = {pid: S.execute("SELECT text FROM documents WHERE id = ?", (str(pid),)).fetchone()[0] for pid in P.page_id}
        pd.DataFrame(dict(title=P.title, page_id=P.page_id, text=[txt[p] for p in P.page_id])).to_parquet(d / "pages.parquet")
        print(arm, "topics", B[B.arm == arm].number.nunique(), "pages", len(P), flush=True)


def corpus_a(real, docs_db, index_db, out, n_chunks="50000"):
    """Check corpus A: the registered corpus (REAL/chunks.parquet, in_corpus) plus deviation 6(c)'s chunks: pages drawn
    uniformly without replacement (seed 0) from dump pages that are neither gold nor disambiguation pages, cut as the
    registered corpus, first 5 chunks per page, in draw order until n_chunks. Reads no TREC data. Writes OUT/A/."""
    import html, sqlite3, json, pandas as pd
    sys.path.insert(0, str(HERE)); import ambigqa
    from blowfish.datasets.chunkers import TokenSlidingWindowChunker
    from blowfish.datasets.types import Document
    out = Path(out) / "A"; out.mkdir(parents=True, exist_ok=True)
    G = pd.read_parquet(Path(real) / "pages.parquet"); gold_ids = set(int(p) for p in G.page_id)
    gold_titles = set(G.title) | set(json.load(open(Path(real) / "pages_missing.json")))
    I = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True); S = sqlite3.connect(f"file:{docs_db}?mode=ro", uri=True)
    pool = np.array(sorted(pid for pid, t, d in I.execute("SELECT id, title, disamb FROM meta")
                           if d == 0 and pid not in gold_ids and html.unescape(t) not in gold_titles))
    order = np.random.default_rng(0).permutation(pool); chunker = TokenSlidingWindowChunker(); rows, n = [], int(n_chunks)
    for pid in order:
        text = S.execute("SELECT text FROM documents WHERE id = ?", (str(pid),)).fetchone()[0]
        t = html.unescape(text.split("\n", 1)[0]).strip()
        cs = list(chunker.chunk(Document(doc_id=t, title=t, text=text)))[:ambigqa.CHUNKS_PER_PAGE]
        rows += [dict(title=t, idx=i, text=c.text, in_corpus=True) for i, c in enumerate(cs)][: n - len(rows)]
        if len(rows) >= n:
            break
    R = pd.read_parquet(Path(real) / "chunks.parquet"); R = R[R.in_corpus]
    C = pd.concat([R[["title", "idx", "text", "in_corpus"]], pd.DataFrame(rows)], ignore_index=True)
    C.to_parquet(out / "chunks.parquet")
    print(f"A: registered {len(R)} + random {len(rows)} chunks from {len(set(r['title'] for r in rows))} pages "
          f"(pool {len(pool)} pages)", flush=True)


def prep(sub, tsv, model):
    import pandas as pd
    sys.path.insert(0, str(HERE)); import ambigqa
    sub = Path(sub); out = sub / model; out.mkdir(parents=True, exist_ok=True)
    if not (sub / "chunks.parquet").exists():
        ambigqa.chunks(None, sub)
    C = pd.read_parquet(sub / "chunks.parquet"); keep = C.in_corpus.to_numpy(); titles = C.title[keep].to_numpy().astype(str)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODELS[model], device="cpu")
    if not (out / "corpus_emb.npy").exists():   # as ambigqa.embed (same model, batch 64, unit-normalised), corpus chunks only
        np.save(out / "corpus_emb.npy", m.encode(C.text[keep].tolist(), batch_size=64, normalize_embeddings=True).astype(np.float32))
    E = np.load(out / "corpus_emb.npy")
    T = _topics(tsv); Q = m.encode(T["query"].tolist(), normalize_embeddings=True).astype(np.float32)
    idx = np.stack([np.argsort(-(E @ q), kind="stable")[:K] for q in Q])
    B = pd.read_csv(sub.parent / "bm25.csv"); B = B[B.arm == (sub.name if sub.name in ("primary", "withdis") else "primary")]
    own = np.array([np.isin(titles[ix], B[B.number == n].title.to_numpy()).mean() for ix, n in zip(idx, T.number)])
    np.savez_compressed(out / "retrieved.npz", Q=Q, idx=idx, number=T.number.to_numpy(), coverage=own,
                        nwords=np.array([len(q.split()) for q in T["query"]]), titles=titles)
    print(sub.name, model, "topics", len(T), "corpus", E.shape, flush=True)


def _ph(U):
    from ripser import ripser
    dg = ripser(U, maxdim=1)["dgms"]; h0 = dg[0][:, 1]; h0 = h0[np.isfinite(h0)]; lt = dg[1][:, 1] - dg[1][:, 0]
    return float(np.mean(h0 / 2)), (float(np.max(lt) / 2) if len(lt) else 0.0), float(np.max(h0))


def _unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _from_bearings(q, a, T):  # neighbours with cosines a and unit bearings T (rows orthogonal to q) -> M1's topology
    X = a[:, None] * q[None, :] + np.sqrt(1 - a ** 2)[:, None] * T
    G = X @ X.T; ev = np.sort(np.linalg.eigvalsh(G))[::-1]
    w1, lt, lb = _ph(_unit(X - q[None, :] + np.finfo(float).eps))
    return dict(longbar=lb, w1=w1, ltmax=lt, eig1=float(ev[0] / ev.sum()))


def _bearings(q, X):
    T = X - (X @ q)[:, None] * q[None, :]; return _unit(T)


def feats(out):
    import pandas as pd
    out = Path(out); R = dict(np.load(out / "retrieved.npz")); E = np.load(out / "corpus_emb.npy").astype(np.float64)
    n = len(R["Q"]); rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    while np.any(perm == np.arange(n)):                                        # bearing swap: a derangement
        perm = rng.permutation(n)
    rows = []
    for i in range(n):
        q = R["Q"][i].astype(np.float64); top = R["idx"][i]; X = E[top]; a = np.clip(X @ q, -1, 1)
        r = dict(number=int(R["number"][i]), nwords=int(R["nwords"][i]), pages=len(set(R["titles"][top])), coverage=float(R["coverage"][i]))
        r.update({f"cos{j:02d}": float(v) for j, v in enumerate(np.sort(a)[::-1])})
        r.update(_from_bearings(q, a, _bearings(q, X)))
        rr = np.random.default_rng((i, 7)); pool = np.setdiff1d(rr.choice(len(E), 3000, replace=False), top)
        cc = [_from_bearings(q, a, _bearings(q, E[pool[d * K:(d + 1) * K]])) for d in range(NCORP)]
        r.update({f"{k}_corp": float(np.mean([c[k] for c in cc])) for k in cc[0]})
        j = perm[i]; qo = R["Q"][j].astype(np.float64); To = _bearings(qo, E[R["idx"][j]])
        v = qo - q; v /= np.linalg.norm(v); H = np.eye(len(q)) - 2 * np.outer(v, v)   # Householder: qo -> q, rank-matched
        r.update({f"{k}_swap": val for k, val in _from_bearings(q, a, To @ H.T).items()})
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "feats.csv", index=False); print("rows", len(rows), flush=True)


def _ns_basis(x, knots):  # natural cubic spline basis without intercept (ESL eqs. 5.4-5.5): x and K - 2 nonlinear terms
    d = lambda k: (np.clip(x - knots[k], 0, None) ** 3 - np.clip(x - knots[-1], 0, None) ** 3) / (knots[-1] - knots[k])
    return np.column_stack([x] + [d(k) - d(len(knots) - 2) for k in range(len(knots) - 2)])


def _design(Ftr, Fte, cols, lin):
    A, B = [], []
    for c in cols:
        x, z = Ftr[c].to_numpy(float), Fte[c].to_numpy(float); m, s = x.mean(), x.std() or 1.0; x, z = (x - m) / s, (z - m) / s
        kn = np.unique(np.percentile(x, [5, 35, 65, 95]))
        if c in lin or len(kn) < 4:
            A.append(x[:, None]); B.append(z[:, None])
        else:
            A.append(_ns_basis(x, kn)); B.append(_ns_basis(z, kn))
    return np.hstack(A), np.hstack(B)


def _oof(F, y, cols, lin, seed):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=seed).split(F, y):
        Xa, Xb = _design(F.iloc[tr], F.iloc[te], cols, lin)
        m = LogisticRegressionCV(Cs=np.logspace(-3, 3, 13), cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                 scoring="neg_log_loss", penalty="l2", max_iter=10000).fit(Xa, y[tr])
        p[te] = m.predict_proba(Xb)[:, 1]
    return np.clip(p, 1e-12, 1 - 1e-12)


def _gain(F, yy, m0, topo, cos):  # one repetition (seed 0): per-topic out-of-fold log-likelihood, M1 minus M0
    p1, p0 = _oof(F, yy, m0 + topo, cos, 0), _oof(F, yy, m0, cos, 0)
    return np.mean(np.log(np.where(yy, p1, 1 - p1))) - np.mean(np.log(np.where(yy, p0, 1 - p0)))


def _auc(p, y):
    from scipy.stats import rankdata
    r = rankdata(p); n1 = y.sum(); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def fit(out, tsv, nperm="1000"):
    import pandas as pd
    from scipy.stats import mannwhitneyu
    out = Path(out); F = pd.read_csv(out / "feats.csv"); T = _topics(tsv, with_type=True)
    lab = dict(zip(T.number, T.type)); keep = F.number.map(lab).isin(["ambiguous", "faceted"]).to_numpy()
    L = [f"# p2.py fit -- {out}; topics {len(F)}, kept (ambiguous or faceted) {keep.sum()}, excluded {len(F) - keep.sum()}"]
    F = F[keep].reset_index(drop=True); y = (F.number.map(lab) == "ambiguous").to_numpy().astype(int)
    cos = [c for c in F.columns if c.startswith("cos")]
    m0 = ["nwords", "pages", "coverage"] + cos; topo = ["longbar", "w1", "ltmax", "eig1"]
    L.append(f"ambiguous {y.sum()}, faceted {len(y) - y.sum()}; coverage by type (median): ambiguous "
             f"{F.coverage[y == 1].median():.3f}, faceted {F.coverage[y == 0].median():.3f}")
    def run(Fx, tag, reps=20):
        P0 = np.stack([_oof(Fx, y, m0, cos, r) for r in range(reps)]); P1 = np.stack([_oof(Fx, y, m0 + topo, cos, r) for r in range(reps)])
        ll = lambda P, ix: np.mean([np.mean(y[ix] * np.log(p[ix]) + (1 - y[ix]) * np.log(1 - p[ix])) for p in P])
        au = lambda P, ix: np.mean([_auc(p[ix], y[ix]) for p in P])
        ix0 = np.arange(len(y)); dll, dau = ll(P1, ix0) - ll(P0, ix0), au(P1, ix0) - au(P0, ix0)
        rng = np.random.default_rng(0); bl, ba = [], []
        for _ in range(2000):
            ix = rng.integers(0, len(y), len(y))
            if 0 < y[ix].sum() < len(ix):
                bl.append(ll(P1, ix) - ll(P0, ix)); ba.append(au(P1, ix) - au(P0, ix))
        ql, qa = np.percentile(bl, [2.5, 97.5]), np.percentile(ba, [2.5, 97.5])
        L.append(f"{tag}: AUROC M0 {au(P0, ix0):.3f}, M1 {au(P1, ix0):.3f}; gain {dau:+.3f} [{qa[0]:+.3f}, {qa[1]:+.3f}]; "
                 f"log-likelihood gain per topic {dll:+.4f} [{ql[0]:+.4f}, {ql[1]:+.4f}]")
        if tag == "real":
            verdict = ("useful gain" if ql[0] > 0 and dau >= 0.03 else "no useful gain" if qa[0] >= -0.03 and qa[1] <= 0.03
                       else "not demonstrated")
            L.append(f"VERDICT (registered rule): {verdict}")
        return P0, P1
    run(F, "real")
    for ctl in ("corp", "swap"):
        Fc = F.copy()
        for t in topo:
            Fc[t] = F[f"{t}_{ctl}"]
        run(Fc, f"control {ctl} (report only)")
    u = mannwhitneyu(F.longbar[y == 1], F.longbar[y == 0], alternative="two-sided")
    L.append(f"longest bar, ambiguous vs faceted: AUROC {u.statistic / (y.sum() * (len(y) - y.sum())):.3f}, "
             f"Mann-Whitney U = {u.statistic:.0f}, n1 = {y.sum()}, n2 = {len(y) - y.sum()}, p = {u.pvalue:.3g} (two-sided)")
    from concurrent.futures import ProcessPoolExecutor
    obs = _gain(F, y, m0, topo, cos); rng = np.random.default_rng(0)
    perms = [rng.permutation(y) for _ in range(int(nperm))]                  # drawn in order, seed 0, then farmed out
    with ProcessPoolExecutor(int(os.environ.get("P2_WORKERS", "7"))) as ex:
        ge = sum(g >= obs for g in ex.map(_gain, *zip(*[(F, yp, m0, topo, cos) for yp in perms]), chunksize=8))
    L.append(f"permutation test of the log-likelihood gain (one repetition, {nperm} permutations, seed 0): observed "
             f"{obs:+.4f}, p = {(1 + ge) / (1 + int(nperm)):.3g} (one-sided; reported, not used to decide)")
    (out / "result.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    {"bm25": bm25, "corpus_a": corpus_a, "prep": prep, "feats": feats, "fit": fit}[sys.argv[1]](*sys.argv[2:])
