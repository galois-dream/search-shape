# hdp_checks.py -- three checks from Vershynin, High-Dimensional Probability, 2nd ed., on the clean real-text
# neighbourhoods (CLEAN=1, OUTROOT=out_clean).
#   conc  (Thm 5.1.3; sec. 5.2.4, eq. 5.8): under the random-bearing control each feature is a Lipschitz function of 50
#         random directions, so it barely varies between draws. For 200 Nobel Q250|C750 queries (seed 0), keep the real
#         cosines and draw 20 sets of random bearings in D' = 64 ... 4096 dimensions; report the spread (s.d.) of each
#         feature over draws per query, its median, and the spread of the REAL feature over queries. Expected: control
#         s.d. falls like 1/sqrt(D').
#   jl    (Thm 5.3.1): project q and its 50 retrieved neighbours (retrieval done in full dimension) with a random
#         Gaussian map to m = 16 ... 1024 dimensions, recompute the features, and report the largest change of any
#         pairwise offset distance (eps_D) and of each feature, per query (median, max), plus the main-contrast AUROC
#         (Q750|C250 vs Q250|C750) at each m. Deterministic check: |d L_max| <= eps_D and |d W1| <= eps_D / 2.
#   dk    (Thm 4.1.15): from vmf_<model>.csv.gz (no recomputation), the share of real neighbourhoods whose second Gram
#         eigenvalue share (eig2) exceeds the 95th percentile of its 20 corpus-bearing control draws; 5% expected if
#         real neighbourhoods carry no split beyond what the control produces. All corpora and models present,
#         leave-document-out arm, Q250|C750 and size-matched Q250|C250.
# conc and jl need the embeddings. Usage: CLEAN=1 OUTROOT=out_clean python hdp_checks.py conc|jl <model> ;
# python hdp_checks.py dk. Writes $RT_WORK/$OUTROOT/hdp_<mode>[_<model>].csv
import os
import sys
import numpy as np
import pandas as pd

MODE = sys.argv[1] if __name__ == "__main__" else "conc"  # importable (vmf_elongation.py)
if MODE in ("conc", "jl"):
    from features import load_model, offsets, ph, unit, OUT, K
else:
    from pathlib import Path
    OUT = Path(os.environ.get("RT_WORK", str(Path(__file__).resolve().parent / "work"))) / os.environ.get("OUTROOT", "out_clean")
NQ = 200


def neighbourhoods(model, qg, cg, n, seed=0):
    """Top-K neighbours of n random queries (overlapping chunks excluded, as features.run_cell's primary arm)."""
    meta, P, Q = load_model(model)
    qm, cm = meta[qg], meta[cg]
    S = Q[qg] @ P[cg].T
    same = qm.lecture.to_numpy(dtype=object)[:, None] == cm.lecture.to_numpy(dtype=object)[None, :]
    ov = (qm.start.values[:, None] < cm.end.values[None, :]) & (cm.start.values[None, :] < qm.end.values[:, None])
    S = np.where(same & ov, -np.inf, S)
    rows = np.sort(np.random.default_rng(seed).choice(len(qm), min(n, len(qm)), replace=False))
    for i in rows:
        idx = np.argsort(-S[i], kind="stable")[:K]
        yield Q[qg][i], P[cg][idx], S[i, idx]


def random_bearings(cos, Dp, rng):
    """The random-bearing control in Dp dimensions: same cosines to q = e1, bearings uniform on q's orthogonal sphere."""
    V = rng.standard_normal((len(cos), Dp)); V[:, 0] = 0; V = unit(V)
    C = np.sqrt(1 - cos ** 2)[:, None] * V; C[:, 0] = cos
    q = np.zeros(Dp); q[0] = 1
    return q, C


def conc(model):
    rows = []
    for j, (q, C, cos) in enumerate(neighbourhoods(model, 250, 750, NQ)):
        real = ph(offsets(q, C))
        rng = np.random.default_rng((7, j))
        for Dp in (64, 256, 1024, 4096):
            F = np.array([ph(offsets(*random_bearings(cos, Dp, rng))) for _ in range(20)])
            rows.append(dict(model=model, q=j, D=Dp, w1_real=real[0], ltmax_real=real[1], longbar_real=real[2],
                             **{f"{f}_ctrl_mean": F[:, k].mean() for k, f in enumerate(("w1", "ltmax", "longbar"))},
                             **{f"{f}_ctrl_sd": F[:, k].std(ddof=1) for k, f in enumerate(("w1", "ltmax", "longbar"))}))
    R = pd.DataFrame(rows); R.to_csv(OUT / f"hdp_conc_{model}.csv", index=False)
    g = R.groupby("D")
    print(f"{model}: median s.d. over 20 random-bearing draws, by dimension (real feature's s.d. over queries alongside)")
    print(pd.DataFrame({f: g[f"{f}_ctrl_sd"].median() for f in ("w1", "longbar", "ltmax")}).assign(
        w1_sd_x_sqrtD=lambda d: d.w1 * np.sqrt(d.index), longbar_sd_x_sqrtD=lambda d: d.longbar * np.sqrt(d.index)).round(5).to_string())
    print("real s.d. over queries:", {f: round(R.drop_duplicates("q")[f"{f}_real"].std(), 4) for f in ("w1", "longbar", "ltmax")})


def jl(model):
    MS = (16, 32, 64, 128, 256, 512, 1024)
    rows = []
    for qg, cg in ((250, 750), (750, 250)):
        for j, (q, C, cos) in enumerate(neighbourhoods(model, qg, cg, NQ)):
            U = offsets(q, C); Dfull = np.sqrt(np.maximum(2 - 2 * U @ U.T, 0)); full = ph(U)
            row = dict(model=model, cell=f"Q{qg}|C{cg}", q=j, w1=full[0], ltmax=full[1], longbar=full[2])
            for m in MS:
                Rm = np.random.default_rng((11, m)).standard_normal((len(q), m)) / np.sqrt(m)  # one map per m, all queries
                Up = offsets(q @ Rm, C @ Rm); Dp = np.sqrt(np.maximum(2 - 2 * Up @ Up.T, 0)); f = ph(Up)
                row.update({f"epsD_{m}": float(np.abs(Dp - Dfull).max()), f"w1_{m}": f[0], f"ltmax_{m}": f[1], f"longbar_{m}": f[2]})
            rows.append(row)
    R = pd.DataFrame(rows); R.to_csv(OUT / f"hdp_jl_{model}.csv", index=False)
    a, b = R[R.cell == "Q750|C250"], R[R.cell == "Q250|C750"]
    auc = lambda x, y: (np.asarray(x)[:, None] > np.asarray(y)[None, :]).mean() + 0.5 * (np.asarray(x)[:, None] == np.asarray(y)[None, :]).mean()
    out = []
    for m in MS:
        e = R[f"epsD_{m}"]
        out.append(dict(m=m, epsD_med=e.median(), epsD_max=e.max(),
                        dLmax_med=(R[f"longbar_{m}"] - R.longbar).abs().median(), dW1_med=(R[f"w1_{m}"] - R.w1).abs().median(),
                        dLT_med=(R[f"ltmax_{m}"] - R.ltmax).abs().median(),
                        bound_Lmax_ok=bool(((R[f"longbar_{m}"] - R.longbar).abs() <= e + 1e-9).all()),
                        bound_W1_ok=bool(((R[f"w1_{m}"] - R.w1).abs() <= e / 2 + 1e-9).all()),
                        auc_w1=auc(a[f"w1_{m}"], b[f"w1_{m}"]), auc_longbar=auc(a[f"longbar_{m}"], b[f"longbar_{m}"])))
    print(f"{model} (D = {len(q)}): full-dimension AUROC w1 {auc(a.w1, b.w1):.3f}, longbar {auc(a.longbar, b.longbar):.3f}")
    print(pd.DataFrame(out).round(4).to_string(index=False))


def dk():
    rows = []
    for vf in sorted(OUT.glob("**/vmf_*.csv.gz")):
        if "k100" in vf.name:
            continue
        corpus = "nobel" if vf.parent == OUT else vf.parent.name
        V = pd.read_csv(vf)
        for (var, cell), g in V[V.variant.isin(["lodo", "lodo_dens"])].groupby(["variant", "cell"]):
            ctrl = g[[f"eig2_enull{k:02d}" for k in range(20)]].to_numpy()
            rows.append(dict(corpus=corpus, model=g.model.iloc[0], variant=var, cell=cell, n=len(g),
                             share_above_ctrl95=float(np.mean(g.eig2.to_numpy() > np.percentile(ctrl, 95, axis=1))),
                             eig2_real_med=g.eig2.median(), eig2_ctrl_med=float(np.median(ctrl))))
    R = pd.DataFrame(rows); R.to_csv(OUT / "hdp_dk.csv", index=False)
    print(R.pivot_table(index=["corpus", "model"], columns="cell", values="share_above_ctrl95").round(3).to_string())


if __name__ == "__main__":
    {"conc": conc, "jl": jl, "dk": dk}[MODE](*sys.argv[2:])
