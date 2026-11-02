# vmf_elongation.py -- traces the vMF appendix's real-data claim: on real neighbourhoods vmf2 tracks how much of the
# spread lies along ONE direction, not a count of meanings. For 120 random Nobel queries in each main cell (Q250|C750
# and Q750|C250; seed 0; text-embedding-3-small; clean text; overlapping chunks excluded, as features.py's primary
# arm): vmf2_r5 (vmf.stats), the leading principal direction's share of the top-5 variance of the 50 centred unit
# offsets, and their effective rank. Spearman correlations within each cell and pooled. Needs the embeddings.
# Usage: CLEAN=1 OUTROOT=out_clean python vmf_elongation.py -> $OUTROOT/vmf_elongation.txt
import numpy as np
from scipy.stats import spearmanr
from features import OUT, offsets
from hdp_checks import neighbourhoods
from vmf import stats

rows = []
for qg, cg in ((250, 750), (750, 250)):
    for q, C, cos in neighbourhoods("openai-3-small", qg, cg, 120):
        U = offsets(q, C)
        ev = np.linalg.svd(U - U.mean(0), compute_uv=False) ** 2
        rows.append((f"Q{qg}|C{cg}", stats(U, seed=0, rs=(5,))["vmf2_r5"], ev[0] / ev[:5].sum(), ev.sum() / ev[0]))
L = [f"# vmf_elongation.txt -- {__import__('time').strftime('%Y-%m-%d %H:%M %Z')}; text-embedding-3-small, Nobel, clean, 120 queries per cell"]
for cell in ("Q250|C750", "Q750|C250", "pooled"):
    R = [r for r in rows if cell == "pooled" or r[0] == cell]
    v, sh, er = (np.array([r[k] for r in R]) for k in (1, 2, 3))
    L.append(f"{cell}: n={len(R)}; Spearman(vmf2, top-direction share of top-5 variance) = {spearmanr(v, sh)[0]:+.2f}; "
             f"Spearman(vmf2, effective rank) = {spearmanr(v, er)[0]:+.2f}; median effective rank {np.median(er):.1f}")
(OUT / "vmf_elongation.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))
