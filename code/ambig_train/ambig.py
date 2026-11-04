# ambig.py -- written on day 4 at 10:21. Human-labelled check on AmbigNQ train (docs/ambig_train_plan.md). Three steps:
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
    B = pd.read_parquet(Path(real) / "chunks.parquet"); keep = np.ones(len(B), bool) if os.environ.get("FULL") == "1" else B.in_corpus.to_numpy()  # FULL=1 (added day 6): every chunk of the gold pages, not the first 5
    E = np.load(Path(real) / "chunk_emb.npy")[keep].astype(np.float32); titles = B.title[keep].to_numpy()
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    Q = m.encode(qs, batch_size=64, normalize_embeddings=True).astype(np.float32)
    idx = np.stack([np.argsort(-(E @ q), kind="stable")[:K] for q in Q])
    np.savez_compressed(out / "retrieved.npz", Q=Q, idx=idx, label=lab, nwords=np.array([len(q.split()) for q in qs]))
    np.save(out / "corpus_emb.npy", E); np.save(out / "corpus_titles.npy", titles.astype(str))
    print("questions", len(qs), "ambiguous", lab.mean().round(3), "corpus", E.shape, flush=True)


def flip_retrieve(data, real, out):
    """Amendment 1 (11:15): for each ambiguous train question with >= 2 disambiguated questions, three queries:
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
    B = pd.read_parquet(Path(real) / "chunks.parquet"); keep = np.ones(len(B), bool) if os.environ.get("FULL") == "1" else B.in_corpus.to_numpy()  # FULL=1 (added day 6): every chunk of the gold pages, not the first 5
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


def load(out):  # 11:02: worker initializer (macOS spawns workers, so globals are not inherited)
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


if __name__ == "__main__":
    {"retrieve": retrieve, "feats": feats, "auroc": auroc, "flip_retrieve": flip_retrieve, "flip_auroc": flip_auroc}[sys.argv[1]](*sys.argv[2:])
