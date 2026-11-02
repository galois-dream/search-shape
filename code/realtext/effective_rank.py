# effective_rank.py -- measurement behind the vMF appendix: the effective rank r = tr(S)/lambda_max(S) of the
# covariance S of the 50 centred unit offsets of a neighbourhood, for 200 random 250-token Nobel chunks (seed 0) used
# as queries against the other 250-token chunks (all text, no exclusion), and the same for the corpus.
# Needs the embeddings. Usage: python effective_rank.py (numpy only) -> $RT_WORK/out/effective_rank.txt
import os
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("RT_WORK", str(Path(__file__).resolve().parent / "work")))  # see README
L = []
for model in ("openai-3-small", "bge-base"):
    P = np.load(ROOT / "emb" / model / "passage_c250.npy").astype(np.float64); P /= np.linalg.norm(P, axis=1, keepdims=True)
    Pc = P - P.mean(0); ev = np.linalg.eigvalsh(Pc.T @ Pc / len(P))[::-1]; cum = np.cumsum(ev) / ev.sum()
    d50, d80, d90 = [int(np.searchsorted(cum, t) + 1) for t in (0.5, 0.8, 0.9)]
    rng = np.random.default_rng(0); rs = []
    for i in rng.choice(len(P), 200, replace=False):
        s = P @ P[i]; s[i] = -np.inf; top = np.argsort(-s)[:50]
        U = P[top] - P[i]; U /= np.linalg.norm(U, axis=1, keepdims=True)
        e = np.linalg.svd(U - U.mean(0), compute_uv=False) ** 2; rs.append(e.sum() / e[0])
    L.append(f"{model}: D={P.shape[1]}, n={len(P)}; corpus effective rank {ev.sum() / ev[0]:.1f}; dims for 50/80/90% of variance "
             f"{d50}/{d80}/{d90}; neighbourhood effective rank median {np.median(rs):.1f} [10th-90th pct {np.percentile(rs, 10):.1f}, {np.percentile(rs, 90):.1f}]")
(ROOT / "out" / "effective_rank.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))
