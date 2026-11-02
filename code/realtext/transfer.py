# transfer.py -- registered transfer test (docs/transfer_plan.md): does a threshold on a topological summary carry over
# between embedding models better than one on a cosine statistic? Clean features, fixed query length,
# leave-document-out arm: label 1 = size-matched C250 (lodo_dens Q250|C250), 0 = C750 (lodo Q250|C750). For each
# corpus, ordered model pair (s, t) and statistic: orientation and threshold maximising balanced accuracy (BA) on s,
# applied to t; transfer loss = best BA on t - BA on t with s's threshold. Median loss over pairs, 95% interval from a
# bootstrap over documents (B = 1000, seed 0). AUROC alongside. Uses every model whose feature table is present.
# Usage: [OUTROOT=out_clean] python transfer.py -> $RT_WORK/$OUTROOT/transfer.csv
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("RT_WORK", str(Path(__file__).resolve().parent / "work"))) / os.environ.get("OUTROOT", "out_clean")
CORP = {"nobel": ROOT, "turing": ROOT / "turing", "fields": ROOT / "fields", "gutenberg": ROOT / "gutenberg", "wiki": ROOT / "wiki"}
MODELS = ["openai-3-small", "openai-3-large", "vertex-005", "modernbert-embed", "bge-base", "gemini-001", "openai-ada-002", "bge-m3", "qwen3-0.6b"]
STATS = {"top1": "cosine", "mean50": "cosine", "nn_pair_cos": "cosine", "std50": "cosine", "slope50": "cosine", "nqc": "cosine",
         "w1": "persistence", "longbar": "persistence", "ltmax": "persistence", "R": "arrangement"}
B = 1000


def best_threshold(x, y, w):
    """Orientation sign and threshold maximising weighted balanced accuracy; predict 1 if sign*x > thr."""
    best = (-1.0, 1, 0.0)
    for sgn in (1, -1):
        v = sgn * x; o = np.argsort(v); vs, ys, ws = v[o], y[o], w[o]
        p_tot, n_tot = (ws * ys).sum(), (ws * (1 - ys)).sum()
        # threshold between vs[k-1] and vs[k]: predicted 0 for first k items
        tn = np.r_[0, np.cumsum(ws * (1 - ys))]; fn = np.r_[0, np.cumsum(ws * ys)]
        ba = 0.5 * (tn / n_tot + (p_tot - fn) / p_tot)
        k = int(np.argmax(ba))
        thr = (vs[k - 1] + vs[k]) / 2 if 0 < k < len(vs) else (vs[0] - 1 if k == 0 else vs[-1] + 1)
        if ba[k] > best[0]:
            best = (float(ba[k]), sgn, float(thr))
    return best


def ba_at(x, y, w, sgn, thr):
    pred = (sgn * x > thr).astype(float)
    return 0.5 * ((w * pred * y).sum() / (w * y).sum() + (w * (1 - pred) * (1 - y)).sum() / (w * (1 - y)).sum())


def auroc(x, y):
    r = pd.Series(x).rank().to_numpy(); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    rows = []
    for corp, p in CORP.items():
        D = {}
        for m in MODELS:
            f = p / f"features_{m}.csv.gz"
            f = f if f.exists() else f.with_suffix(".xz")   # shipped tables are .csv.xz
            if not f.exists():
                continue
            F = pd.read_csv(f, usecols=["variant", "cell", "query_id", "lecture", "top1", "mean50", "std50", "slope50", "nn_pair_cos", "w1", "ltmax", "longbar", "R"])
            a = F[(F.variant == "lodo_dens") & (F.cell == "Q250|C250")].assign(y=1.0)
            b = F[(F.variant == "lodo") & (F.cell == "Q250|C750")].assign(y=0.0)
            X = pd.concat([a, b]).assign(nqc=lambda d: d.std50 / d.mean50)
            D[m] = X.set_index(["query_id", "y"]).sort_index()
        mods = list(D)
        keys = D[mods[0]].index
        for m in mods:  # same queries in every model
            assert D[m].index.equals(keys), (corp, m)
        y = keys.get_level_values("y").to_numpy(float)
        docs = D[mods[0]].lecture.to_numpy(); ud = np.unique(docs); di = np.searchsorted(ud, docs)
        rng = np.random.default_rng(0)
        Wb = np.vstack([np.ones(len(ud))] + [rng.multinomial(len(ud), np.full(len(ud), 1 / len(ud))) for _ in range(B)])[:, di]
        for s in STATS:
            losses = np.zeros((B + 1, len(mods) * (len(mods) - 1)))
            for b in range(B + 1):
                w = Wb[b]; fit = {m: best_threshold(D[m][s].to_numpy(float), y, w) for m in mods}
                j = 0
                for ms in mods:
                    for mt in mods:
                        if ms == mt:
                            continue
                        xt = D[mt][s].to_numpy(float)
                        losses[b, j] = fit[mt][0] - ba_at(xt, y, w, fit[ms][1], fit[ms][2]); j += 1
            med = np.median(losses, axis=1)
            rows.append(dict(corpus=corp, stat=s, kind=STATS[s], n_models=len(mods), median_loss=med[0], lo=np.percentile(med[1:], 2.5),
                             hi=np.percentile(med[1:], 97.5), auroc_median=float(np.median([auroc(D[m][s].to_numpy(float), y) for m in mods]))))
        print(corp, "done", flush=True)
    R = pd.DataFrame(rows); R.to_csv(ROOT / "transfer.csv", index=False)
    print(R.pivot_table(index="stat", columns="corpus", values="median_loss").round(3).to_string())


if __name__ == "__main__":
    main()
