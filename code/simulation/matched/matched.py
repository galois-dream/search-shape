"""Matched-cosine simulation (docs/matched_plan.md: part 1 with amendments 1 and 2). Neighbourhoods whose 50 cosines to
the query are IDENTICAL across classes, differing only in how the bearings are arranged: one meaning, two meanings
(angle phi), three meanings (120 deg) or a ring. Within a meaning, bearings spread in a random p-dimensional subspace of
q-perp (scale s), plus zero-mean isotropic Gaussian noise over all of q-perp (scale tau, matched to the main
simulation's noise-to-signal ratio sigma*sqrt(D)/sqrt(n) at D = 256, n = 16: tau = 0.4 for sigma = 0.1 and 0.8 for
sigma = 0.2). Level A: same s for both classes (bearing cosine to its centre ~0.6 without noise). Level B: the
one-meaning spread is widened until both classes have the same mean pairwise cosine among the 50 neighbours
(coherence). Statistics per neighbourhood: cosine statistics, coherence, median nearest-neighbour cosine, Gram
second-eigenvalue share, vmf2 (r = 5, code/realtext/vmf.py), W1(H0), L_max(H0), LT_max(H1), largest nearest-neighbour
distance, and the yardsticks used to place real text (effective rank, participation ratio, top-5 variance share).
Usage (needs ripser, scipy):
  python matched.py --check        construction diagnostics only (no AUROC of any statistic is computed)
  python matched.py --shard k/14   cells with index = k mod 14 (the run behind the paper: k = 0..13, in parallel)
  python matched.py                all 63 cells in one process
Output: $MATCHED_RESULTS (default ./results)/matched.csv or matched_shard<k>of<n>.csv.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from ripser import ripser

sys.path.insert(0, os.environ.get("VMF_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "realtext")))
from vmf import stats as vmf_stats  # noqa: E402  (vmf2_r5, eig2)

K, N_PER = 50, 40                     # neighbours per neighbourhood; samples per class per seed
SEEDS = range(5)
PS = [2, 5, 20, 100, None]            # None = all of q-perp (D - 1)
TAUS = {"0": 0.0, "s0.1": 0.4, "s0.2": 0.8}
RHO = 0.6                             # Level A: bearing cosine to its centre, noise-free
S_BASE = np.sqrt(1 / RHO ** 2 - 1)


def basis(rng, D, k):
    """k orthonormal vectors in q-perp (q = e1)."""
    G = rng.standard_normal((D, k)); G[0] = 0
    Q, _ = np.linalg.qr(G)
    return Q.T


def bearings(rng, D, kind, p, s, tau, phi_deg=60):
    """K unit bearings in q-perp for one neighbourhood."""
    m1, w = basis(rng, D, 2)
    if kind == "one":
        centres = np.repeat(m1[None], K, 0)
    elif kind == "two":
        m2 = np.cos(np.radians(phi_deg)) * m1 + np.sin(np.radians(phi_deg)) * w
        centres = np.vstack([np.repeat(m1[None], K // 2, 0), np.repeat(m2[None], K - K // 2, 0)])
    elif kind == "three":
        ang = np.radians([0, 120, 240]); idx = np.arange(K) % 3
        centres = np.cos(ang[idx])[:, None] * m1 + np.sin(ang[idx])[:, None] * w
    elif kind == "ring":
        a = rng.uniform(0, 2 * np.pi, K)
        centres = np.cos(a)[:, None] * m1 + np.sin(a)[:, None] * w
    pdim = D - 1 if p is None else p
    Bp = basis(rng, D, pdim)                                   # spread subspace
    xi = rng.standard_normal((K, pdim)) @ Bp / np.sqrt(pdim)   # expected norm 1
    eta = rng.standard_normal((K, D)); eta[:, 0] = 0; eta /= np.sqrt(D - 1)
    T = centres + s * xi + tau * eta
    return T / np.linalg.norm(T, axis=1, keepdims=True)


def cloud(cos, T):
    """Neighbours c_i = cos_i q + sin_i t_i, q = e1."""
    C = np.sqrt(1 - cos ** 2)[:, None] * T
    C[:, 0] = cos
    return C


def coherence(C):
    G = C @ C.T; iu = np.triu_indices(len(C), 1)
    return float(G[iu].mean())


def match_level_b(rng_state, D, kind, p, tau, cos, target):
    """Widen the one-meaning spread until its coherence equals target (same random draws at every s)."""
    lo, hi = 0.0, 20.0
    for _ in range(40):
        mid = (lo + hi) / 2
        rng = np.random.default_rng(rng_state)
        c = coherence(cloud(cos, bearings(rng, D, "one", p, mid, tau)))
        lo, hi = (mid, hi) if c > target else (lo, mid)
    return (lo + hi) / 2


def features(C):
    q = np.zeros(C.shape[1]); q[0] = 1
    U = C - q; U /= np.linalg.norm(U, axis=1, keepdims=True)
    dg = ripser(U, maxdim=1)["dgms"]
    h0 = dg[0][:, 1]; h0 = h0[np.isfinite(h0)]; lt = dg[1][:, 1] - dg[1][:, 0]
    D2 = np.sqrt(np.maximum(2 - 2 * U @ U.T, 0)); np.fill_diagonal(D2, np.inf)
    cos = C[:, 0]; srt = np.sort(cos)[::-1]
    Uc = U - U.mean(0); ev = np.linalg.eigvalsh(Uc @ Uc.T / len(U))[::-1]   # same nonzero eigenvalues as the covariance
    ev = np.clip(ev, 0, None)
    G = C @ C.T; np.fill_diagonal(G, -np.inf)  # amendment 2 (Q1/Q4): median nearest-neighbour cosine
    out = dict(top1=srt[0], mean_cos=cos.mean(), std_cos=cos.std(), slope_cos=np.polyfit(np.arange(1, len(srt) + 1), srt, 1)[0],
               nn_med=float(np.median(G.max(1))),
               coherence=coherence(C), w1=float(np.mean(h0 / 2)), lmax=float(h0.max()), ltmax=float(lt.max() / 2) if len(lt) else 0.0,
               max_nn=float(D2.min(1).max()), eff_rank=float(ev.sum() / ev[0]), part_ratio=float(ev.sum() ** 2 / (ev ** 2).sum()),
               top5_share=float(ev[:5].sum() / ev.sum()))
    v = vmf_stats(U, seed=0, rs=(5,)); out["vmf2"] = v["vmf2_r5"]; out["eig2"] = v["eig2"]
    return out


def cells():
    """(comparison, level, p, tau_name, phi, D) for the plan's cells."""
    for level in ("A", "B"):
        for p in PS:
            for tn in TAUS:
                yield ("one_vs_two", level, p, tn, 60, 768)
    for phi in (30, 90):
        yield ("one_vs_two", "B", 5, "0", phi, 768)
    yield ("one_vs_two", "B", 5, "0", 60, 1536)
    for comp in ("one_vs_three", "ring_vs_one", "ring_vs_two"):
        for p in PS:
            for tn in ("0", "s0.1"):
                yield (comp, "B", p, tn, 60, 768)


def run(check, shard=None):
    rows = []
    for ci, (comp, level, p, tn, phi, D) in enumerate(cells()):
        if shard is not None and ci % shard[1] != shard[0]:  # --shard k/n runs cells with index = k mod n
            continue
        if check and not (comp == "one_vs_two" and tn in ("0", "s0.1") and p in (5, None) and D == 768):
            continue
        a_kind, b_kind = {"one_vs_two": ("one", "two"), "one_vs_three": ("one", "three"), "ring_vs_one": ("ring", "one"), "ring_vs_two": ("ring", "two")}[comp]
        tau = TAUS[tn]
        for seed in SEEDS:
            for i in range(N_PER if not check else 3):
                base = np.random.default_rng((ci, seed, i))
                cos = np.sort(base.uniform(0.55, 0.75, K))[::-1]           # shared by both classes
                st_b = (ci, seed, i, 2)
                Cb = cloud(cos, bearings(np.random.default_rng(st_b), D, b_kind, p, S_BASE if level == "A" else 0.5 * S_BASE, tau, phi))
                if level == "B" and a_kind == "one":
                    s_a = match_level_b((ci, seed, i, 1), D, "one", p, tau, cos, coherence(Cb))
                elif level == "B" and b_kind == "one":                  # ring vs one: widen... the ring side is fixed, one side tuned
                    Ca0 = cloud(cos, bearings(np.random.default_rng((ci, seed, i, 1)), D, a_kind, p, 0.5 * S_BASE, tau, phi))
                    s_b = match_level_b(st_b, D, "one", p, tau, cos, coherence(Ca0))
                    Cb = cloud(cos, bearings(np.random.default_rng(st_b), D, "one", p, s_b, tau, phi)); s_a = 0.5 * S_BASE
                else:  # ring vs two at Level B: both tight; coherence is NOT matched for this comparison (plan names no matching)
                    s_a = S_BASE if level == "A" else 0.5 * S_BASE
                Ca = cloud(cos, bearings(np.random.default_rng((ci, seed, i, 1)), D, a_kind, p, s_a, tau, phi))
                for lab, C in ((a_kind, Ca), (b_kind, Cb)):
                    f = features(C) if not check else dict(coherence=coherence(C), mean_cos=C[:, 0].mean(), top1=C[:, 0].max())
                    rows.append(dict(comparison=comp, level=level, p=p or D - 1, tau=tn, phi=phi, D=D, seed=seed, i=i, cls=lab, **f))
        print(f"cell {ci} {comp} {level} p={p} tau={tn} phi={phi} D={D}", flush=True)
    R = pd.DataFrame(rows)
    if check:
        g = R.groupby(["level", "p", "tau", "cls"]).agg(coh=("coherence", "mean"), mean_cos=("mean_cos", "mean"), top1=("top1", "mean")).round(4)
        print(g.to_string())
        return
    out = os.environ.get("MATCHED_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")); os.makedirs(out, exist_ok=True)
    R.to_csv(os.path.join(out, "matched.csv" if shard is None else f"matched_shard{shard[0]}of{shard[1]}.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--shard")
    a = ap.parse_args(); run(a.check, tuple(map(int, a.shard.split("/"))) if a.shard else None)
